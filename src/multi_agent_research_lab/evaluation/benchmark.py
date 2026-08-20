"""Benchmark single-agent vs multi-agent runs."""

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.agents.critic import compute_citation_coverage
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. Score the answer to the research query on a 0-10 scale "
    "considering: relevance, structure, citation discipline, and absence of invented facts. "
    "Reply with ONLY a number between 0 and 10."
)


def compute_cost(state: ResearchState) -> float | None:
    """Sum per-agent LLM cost recorded in agent_results metadata."""

    costs = [
        r.metadata.get("cost_usd")
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    return sum(costs) if costs else None


def judge_quality(state: ResearchState, llm: LLMClient) -> float | None:
    """LLM-as-judge quality score (0-10). Best-effort: returns None on failure."""

    if not state.final_answer:
        return None
    try:
        response = llm.complete(
            _JUDGE_SYSTEM_PROMPT,
            f"Query: {state.request.query}\n\nAnswer:\n{state.final_answer}",
        )
        return max(0.0, min(10.0, float(response.content.strip().split()[0])))
    except Exception as exc:  # noqa: BLE001 - judging must never fail the benchmark
        logger.warning("quality judge failed: %s", exc)
        return None


def run_benchmark(
    run_name: str, query: str, runner: Runner, judge_llm: LLMClient | None = None
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Run one query through a runner and compute the full metric set."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a crashed run is a data point, not a crash
        logger.error("run %s failed: %s", run_name, exc)
        return None, BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=perf_counter() - started,
            failure_rate=1.0,
            notes=f"crashed: {exc}",
        )
    latency = perf_counter() - started

    failed = state.final_answer is None
    notes = f"routes: {' -> '.join(state.route_history) or 'single-shot'}"
    if state.errors:
        notes += f" | errors: {len(state.errors)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=compute_cost(state),
        quality_score=judge_quality(state, judge_llm) if judge_llm else None,
        citation_coverage=compute_citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
        notes=notes,
    )
    return state, metrics
