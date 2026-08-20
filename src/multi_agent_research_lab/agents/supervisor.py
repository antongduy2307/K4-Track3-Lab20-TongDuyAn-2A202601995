"""Supervisor / router.

Rule-based routing policy: deterministic, cheap (no LLM call), easy to trace.
The next route is appended to `state.route_history`; the workflow reads the
last entry to decide the next node.
"""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"

_MAX_ERRORS = 3


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {"route": route, "iteration": state.iteration, "errors": len(state.errors)},
        )
        return state

    def decide(self, state: ResearchState) -> str:
        settings = get_settings()

        # Guard 1: iteration budget (prevents infinite routing loops).
        if state.iteration >= settings.max_iterations:
            state.errors.append(f"max_iterations={settings.max_iterations} reached, forcing stop")
            return ROUTE_DONE

        # Guard 2: global wall-clock timeout.
        if (
            state.started_at is not None
            and perf_counter() - state.started_at > settings.timeout_seconds
        ):
            state.errors.append(
                f"timeout_seconds={settings.timeout_seconds} exceeded, forcing stop"
            )
            return ROUTE_DONE

        # Guard 3: too many worker failures -> stop instead of thrashing.
        if len(state.errors) >= _MAX_ERRORS:
            return ROUTE_DONE

        # Happy path: fill missing fields in pipeline order.
        if not state.sources or state.research_notes is None:
            return ROUTE_RESEARCHER
        if state.analysis_notes is None:
            return ROUTE_ANALYST
        if state.final_answer is None:
            return ROUTE_WRITER
        # Bonus: run the critic exactly once after the writer.
        if ROUTE_CRITIC not in state.route_history:
            return ROUTE_CRITIC
        return ROUTE_DONE
