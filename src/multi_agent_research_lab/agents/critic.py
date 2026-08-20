"""Critic agent: deterministic citation-coverage check on the final answer.

Kept rule-based (no LLM call) so verification is cheap, reproducible, and cannot
hallucinate. Extendable with an LLM fact-check pass if needed.
"""

import re

from multi_agent_research_lab.agents.base import BaseAgent, source_label
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

_CITATION_RE = re.compile(r"\[S(\d+)\]")


def compute_citation_coverage(state: ResearchState) -> float | None:
    """Fraction of retrieved sources actually cited in the final answer."""

    if not state.final_answer or not state.sources:
        return None
    cited = {f"S{m}" for m in _CITATION_RE.findall(state.final_answer)}
    available = {source_label(i) for i in range(len(state.sources))}
    return len(cited & available) / len(available)


class CriticAgent(BaseAgent):
    """Checks citation coverage and dangling citations on `final_answer`."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        coverage = compute_citation_coverage(state)
        dangling: list[str] = []
        if state.final_answer:
            available = {source_label(i) for i in range(len(state.sources))}
            dangling = sorted(
                {f"S{m}" for m in _CITATION_RE.findall(state.final_answer)} - available
            )
            if dangling:
                state.errors.append(f"critic: dangling citations {dangling}")

        verdict = (
            f"citation_coverage={coverage:.0%}" if coverage is not None else "no answer to check"
        )
        if dangling:
            verdict += f"; dangling citations: {', '.join(dangling)}"
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=verdict,
                metadata={"citation_coverage": coverage, "dangling_citations": dangling},
            )
        )
        state.add_trace_event("critic.done", {"citation_coverage": coverage, "dangling": dangling})
        return state
