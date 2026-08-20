"""Analyst agent: turn research notes into structured analysis."""

from multi_agent_research_lab.agents.base import BaseAgent, format_sources
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are an analysis agent. Given research notes and the underlying sources, produce:\n"
    "1. Key claims (each with its source labels).\n"
    "2. Points of agreement and disagreement between sources.\n"
    "3. Weak or low-trust evidence (e.g. synthetic benchmark documents) explicitly flagged.\n"
    "Only use information from the notes and sources. Keep source labels [S#] intact."
)


class AnalystAgent(BaseAgent):
    """Extracts key claims, compares viewpoints, and flags weak evidence."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.research_notes:
                raise LabError("no research_notes to analyze")
            llm = self._llm or LLMClient()
            response = llm.complete(
                _SYSTEM_PROMPT,
                (
                    f"Query: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes}\n\n"
                    f"Sources:\n{format_sources(state.sources)}"
                ),
            )
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst.done", {"analysis_chars": len(response.content)})
        except LabError as exc:
            state.errors.append(f"analyst: {exc}")
            state.add_trace_event("analyst.error", {"error": str(exc)})
            # Fallback: pass raw notes through so the writer still has material.
            if state.analysis_notes is None and state.research_notes:
                state.analysis_notes = state.research_notes
        return state
