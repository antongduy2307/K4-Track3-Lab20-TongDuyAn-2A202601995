"""Unit tests for routing policy, citation coverage, and offline search.

Replaces the skeleton TODO-guard test (implementation is now done).
All tests run offline: no LLM/API calls.
"""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.critic import CriticAgent, compute_citation_coverage
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


def _state(**kwargs) -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), **kwargs)


def _source(title: str = "Doc") -> SourceDocument:
    return SourceDocument(title=title, snippet="snippet")


class TestSupervisorRouting:
    def test_routes_researcher_when_no_sources(self) -> None:
        assert SupervisorAgent().decide(_state()) == ROUTE_RESEARCHER

    def test_routes_analyst_after_research(self) -> None:
        state = _state(sources=[_source()], research_notes="notes")
        assert SupervisorAgent().decide(state) == ROUTE_ANALYST

    def test_routes_writer_after_analysis(self) -> None:
        state = _state(sources=[_source()], research_notes="notes", analysis_notes="analysis")
        assert SupervisorAgent().decide(state) == ROUTE_WRITER

    def test_routes_critic_once_then_done(self) -> None:
        state = _state(
            sources=[_source()],
            research_notes="notes",
            analysis_notes="analysis",
            final_answer="answer [S1]",
        )
        assert SupervisorAgent().decide(state) == ROUTE_CRITIC
        state.route_history.append(ROUTE_CRITIC)
        assert SupervisorAgent().decide(state) == ROUTE_DONE

    def test_max_iterations_guard(self) -> None:
        state = _state(iteration=get_settings().max_iterations)
        assert SupervisorAgent().decide(state) == ROUTE_DONE
        assert any("max_iterations" in e for e in state.errors)

    def test_error_fallback_stops(self) -> None:
        state = _state(errors=["e1", "e2", "e3"])
        assert SupervisorAgent().decide(state) == ROUTE_DONE

    def test_run_records_route_and_trace(self) -> None:
        state = SupervisorAgent().run(_state())
        assert state.route_history == [ROUTE_RESEARCHER]
        assert state.iteration == 1
        assert state.trace[0]["name"] == "supervisor.route"


class TestCritic:
    def test_citation_coverage_partial(self) -> None:
        state = _state(sources=[_source("A"), _source("B")], final_answer="claim [S1].")
        assert compute_citation_coverage(state) == 0.5

    def test_citation_coverage_none_without_answer(self) -> None:
        assert compute_citation_coverage(_state(sources=[_source()])) is None

    def test_dangling_citation_flagged(self) -> None:
        state = _state(sources=[_source()], final_answer="claim [S9].")
        state = CriticAgent().run(state)
        assert any("dangling" in e for e in state.errors)


class TestOfflineSearch:
    def test_returns_ranked_sources(self) -> None:
        results = SearchClient().search(
            "single agent vs multi-agent architectures for research tasks", max_results=3
        )
        assert len(results) == 3
        assert all(r.snippet for r in results)
        assert results[0].metadata["topic_file"].startswith("01_")
