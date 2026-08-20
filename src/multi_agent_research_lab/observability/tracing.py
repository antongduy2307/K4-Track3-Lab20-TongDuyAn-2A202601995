"""Tracing hooks.

Two layers:
1. `trace_span` — local JSON spans stored in `ResearchState.trace` (always on).
2. LangSmith — enabled via `configure_tracing()` when LANGSMITH_API_KEY is set:
   LangGraph runs are traced automatically through langchain-core, and the
   OpenAI client is wrapped so LLM calls nest under the graph run.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings

logger = logging.getLogger(__name__)

_langsmith_enabled = False


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing if an API key is configured. Returns enabled state."""

    global _langsmith_enabled
    if not settings.langsmith_api_key:
        logger.info("LangSmith disabled (no LANGSMITH_API_KEY); using local JSON traces only")
        _langsmith_enabled = False
        return False
    # langchain-core/langgraph read these environment variables at run time.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # legacy alias, harmless to set both
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    _langsmith_enabled = True
    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s)",
        settings.langsmith_project,
        settings.langsmith_endpoint,
    )
    return True


def langsmith_enabled() -> bool:
    return _langsmith_enabled


def wrap_llm_provider_client(client: Any) -> Any:
    """Wrap an OpenAI SDK client so its calls appear as child runs in LangSmith."""

    if not _langsmith_enabled:
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the pipeline
        logger.warning("could not wrap OpenAI client for LangSmith: %s", exc)
        return client


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Local span recorded into state; nests into LangSmith when enabled."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    langsmith_cm = None
    if _langsmith_enabled:
        try:
            from langsmith.run_helpers import trace as langsmith_trace

            langsmith_cm = langsmith_trace(name=name, run_type="chain", inputs=attributes or {})
            langsmith_cm.__enter__()
        except Exception as exc:  # noqa: BLE001 - tracing must never break the pipeline
            logger.warning("LangSmith span failed for %s: %s", name, exc)
            langsmith_cm = None
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        if langsmith_cm is not None:
            try:
                langsmith_cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LangSmith span close failed for %s: %s", name, exc)
