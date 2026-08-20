"""Researcher agent: retrieve sources and produce cited research notes."""

from multi_agent_research_lab.agents.base import BaseAgent, format_sources
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a research agent. You receive a query and a set of retrieved sources "
    "labeled [S1], [S2], ... Write concise research notes (bullet points) that ONLY "
    "use information present in the sources. Every bullet must cite at least one "
    "source label. Flag synthetic/benchmark sources as lower-trust. Do not invent facts."
)


class ResearcherAgent(BaseAgent):
    """Finds sources and produces `research_notes` with citations."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self._llm = llm
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            sources = self._search.search(
                state.request.query, max_results=state.request.max_sources
            )
            if not sources:
                raise LabError("search returned no sources")
            state.sources = sources

            llm = self._llm or LLMClient()
            response = llm.complete(
                _SYSTEM_PROMPT,
                f"Query: {state.request.query}\n\nSources:\n{format_sources(sources)}",
            )
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "num_sources": len(sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "researcher.done",
                {"num_sources": len(sources), "notes_chars": len(response.content)},
            )
        except LabError as exc:
            state.errors.append(f"researcher: {exc}")
            state.add_trace_event("researcher.error", {"error": str(exc)})
        return state
