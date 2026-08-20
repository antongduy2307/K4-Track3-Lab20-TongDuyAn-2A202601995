"""Command-line entrypoint for the lab."""

import json
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

_BASELINE_SYSTEM_PROMPT = (
    "You are a research assistant doing everything alone: research, analysis, writing. "
    "Using ONLY the provided sources, answer the query in ~500 words for a technical "
    "audience. Cite sources with labels like [S1] and end with a 'Sources' list."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if configure_tracing(settings):
        console.print(
            f"[dim]LangSmith tracing on -> https://smith.langchain.com "
            f"(project: {settings.langsmith_project})[/dim]"
        )


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_single_agent(query: str) -> ResearchState:
    """Single-agent baseline: same retrieval, one LLM call does everything."""

    from multi_agent_research_lab.agents.base import format_sources

    state = ResearchState(request=_parse_query(query))
    state.sources = SearchClient().search(state.request.query, state.request.max_sources)
    response = LLMClient().complete(
        _BASELINE_SYSTEM_PROMPT,
        f"Query: {state.request.query}\n\nSources:\n{format_sources(state.sources)}",
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.SUPERVISOR,  # single agent wears every hat
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "mode": "single-agent-baseline",
            },
        )
    )
    state.add_trace_event("baseline.done", {"answer_chars": len(response.content)})
    return state


def run_multi_agent(query: str) -> ResearchState:
    """Multi-agent workflow runner."""

    state = ResearchState(request=_parse_query(query))
    return MultiAgentWorkflow().run(state)


def _export_trace(state: ResearchState, name: str) -> None:
    store = LocalArtifactStore()
    path = store.write_text(
        f"traces/{name}.json", json.dumps(state.trace, indent=2, ensure_ascii=False)
    )
    console.print(f"[dim]trace exported: {path}[/dim]")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    try:
        state = run_single_agent(query)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    _export_trace(state, "baseline")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    try:
        result = run_multi_agent(query)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(f"[dim]routes: {' -> '.join(result.route_history)}[/dim]")
    if result.errors:
        console.print(Panel.fit("\n".join(result.errors), title="Errors", style="yellow"))
    _export_trace(result, "multi_agent")


@app.command()
def benchmark(
    queries: Annotated[
        list[str], typer.Option("--query", "-q", help="Research query (repeatable)")
    ],
    judge: Annotated[bool, typer.Option(help="Score quality with an LLM judge")] = True,
) -> None:
    """Benchmark single-agent vs multi-agent on the given queries."""

    _init()
    judge_llm = LLMClient() if judge else None
    all_metrics = []
    for i, query in enumerate(queries, start=1):
        runners = (("single-agent", run_single_agent), ("multi-agent", run_multi_agent))
        for run_name, runner in runners:
            console.print(f"[bold]running {run_name} on query {i}...[/bold]")
            state, metrics = run_benchmark(f"{run_name}-q{i}", query, runner, judge_llm)
            all_metrics.append(metrics)
            if state is not None:
                _export_trace(state, f"benchmark_{run_name}_q{i}")

    report = render_markdown_report(all_metrics)
    report += "\nQueries:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1)) + "\n"
    path = LocalArtifactStore().write_text("benchmark_report.md", report)
    console.print(Panel.fit(report, title="Benchmark"))
    console.print(f"[green]report written: {path}[/green]")


if __name__ == "__main__":
    app()
