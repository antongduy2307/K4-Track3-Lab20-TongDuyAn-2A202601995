"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
Retry, timeout, and token/cost accounting all live here rather than inside agents.
"""

import logging
import time
from dataclasses import dataclass

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Unknown models fall back to None cost.
_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None, max_retries: int = 3) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AgentExecutionError("OPENAI_API_KEY is not configured in .env")
        # Import lazily so unit tests without the SDK/key can still import the module.
        from openai import OpenAI

        from multi_agent_research_lab.observability.tracing import wrap_llm_provider_client

        self._client = wrap_llm_provider_client(OpenAI(api_key=settings.openai_api_key))
        self.model = model or settings.openai_model
        self.timeout_seconds = settings.timeout_seconds
        self.max_retries = max_retries

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry + backoff and token/cost accounting."""

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    timeout=self.timeout_seconds,
                )
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else None
                output_tokens = usage.completion_tokens if usage else None
                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=self._estimate_cost(input_tokens, output_tokens),
                )
            except Exception as exc:  # noqa: BLE001 - retry any transport/provider error
                last_error = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise AgentExecutionError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        pricing = _PRICING_PER_1M.get(self.model)
        if pricing is None or input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000
