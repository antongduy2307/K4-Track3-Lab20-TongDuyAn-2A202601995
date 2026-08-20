"""Base agent contract and shared prompt helpers."""

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.core.state import ResearchState


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""


def source_label(index: int) -> str:
    """Stable citation label for the source at `index` (0-based)."""

    return f"S{index + 1}"


def format_sources(sources: list[SourceDocument]) -> str:
    """Render sources as a labeled block usable in prompts: [S1], [S2], ..."""

    blocks: list[str] = []
    for i, doc in enumerate(sources):
        synthetic = " (synthetic benchmark document)" if doc.metadata.get("is_synthetic") else ""
        url = f" | url: {doc.url}" if doc.url else ""
        blocks.append(f"[{source_label(i)}] {doc.title}{synthetic}{url}\n{doc.snippet}")
    return "\n\n".join(blocks)
