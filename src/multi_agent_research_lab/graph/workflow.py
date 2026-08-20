"""LangGraph workflow: hub-spoke supervisor orchestration.

Every worker returns to the supervisor; the supervisor is the single decision
point (routing + stop conditions). Agent internals stay in `agents/`.
"""

from collections.abc import Callable
from time import perf_counter

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        # One shared LLM client: connection reuse + single place for retry/timeout.
        self._llm = llm

    def build(self) -> StateGraph:
        """Create the LangGraph graph: supervisor hub, worker spokes."""

        llm = self._llm if self._llm is not None else LLMClient()
        supervisor = SupervisorAgent()
        workers: dict[str, BaseAgent] = {
            "researcher": ResearcherAgent(llm=llm),
            "analyst": AnalystAgent(llm=llm),
            "writer": WriterAgent(llm=llm),
            "critic": CriticAgent(),
        }

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._as_node(supervisor))
        for name, agent in workers.items():
            graph.add_node(name, self._as_node(agent))
            graph.add_edge(name, "supervisor")

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {name: name for name in workers} | {ROUTE_DONE: END},
        )
        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and convert the result back to ResearchState."""

        settings = get_settings()
        state.started_at = perf_counter()
        compiled = self.build().compile()
        with trace_span("workflow.run", {"query": state.request.query}) as span:
            # Each iteration = supervisor + at most one worker; +4 headroom for the stop path.
            result = compiled.invoke(
                state, config={"recursion_limit": 2 * settings.max_iterations + 4}
            )
        final_state = ResearchState.model_validate(result)
        final_state.add_trace_event("workflow.done", span)
        return final_state

    @staticmethod
    def _as_node(agent: BaseAgent) -> Callable[[ResearchState], dict]:
        def node(state: ResearchState) -> dict:
            with trace_span(f"agent.{agent.name}") as span:
                updated = agent.run(state)
            updated.add_trace_event(f"agent.{agent.name}.span", span)
            return updated.model_dump()

        return node
