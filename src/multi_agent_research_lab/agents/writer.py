"""Writer agent: synthesize the final cited answer."""

from multi_agent_research_lab.agents.base import BaseAgent, format_sources
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer. Using the analysis and sources provided, write a clear, "
    "well-structured answer (~500 words) for a technical audience. Requirements:\n"
    "- Every substantive claim cites a source label like [S1].\n"
    "- Do not introduce facts absent from the analysis/sources.\n"
    "- End with a 'Sources' list mapping each cited label to its title.\n"
    "- Note explicitly when evidence is synthetic or weak."
)


class WriterAgent(BaseAgent):
    """Synthesizes `final_answer` with citations."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.analysis_notes:
                raise LabError("no analysis_notes to write from")
            llm = self._llm or LLMClient()
            response = llm.complete(
                _SYSTEM_PROMPT,
                (
                    f"Query: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Analysis:\n{state.analysis_notes}\n\n"
                    f"Sources:\n{format_sources(state.sources)}"
                ),
            )
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writer.done", {"answer_chars": len(response.content)})
        except LabError as exc:
            state.errors.append(f"writer: {exc}")
            state.add_trace_event("writer.error", {"error": str(exc)})
        return state
