"""Search client abstraction for ResearcherAgent.

Implemented as an offline search over `ai_agent_offline_research_corpus_v2/`:
pick the best-matching topic file via the manifest, then rank that topic's
embedded source documents and knowledge articles by keyword overlap.
No network access or API key required.
"""

import csv
import json
import re
from pathlib import Path

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "will",
        "can",
        "could",
        "should",
        "would",
        "what",
        "when",
        "where",
        "which",
        "whom",
        "does",
        "about",
        "into",
        "over",
        "under",
        "between",
    ]
)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


class SearchClient:
    """Offline corpus search client."""

    def __init__(self, corpus_dir: Path = Path("ai_agent_offline_research_corpus_v2")) -> None:
        self.corpus_dir = corpus_dir

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return the top corpus documents relevant to the query."""

        manifest_path = self.corpus_dir / "manifest.csv"
        if not manifest_path.exists():
            raise AgentExecutionError(f"Offline corpus not found at {self.corpus_dir}")

        topic_file = self._best_topic_file(manifest_path, query)
        payload = json.loads(topic_file.read_text(encoding="utf-8"))
        knowledge_base = payload.get("knowledge_base", {})

        query_tokens = _tokens(query)
        candidates: list[tuple[float, SourceDocument]] = []

        for doc in knowledge_base.get("source_documents", []):
            text = doc.get("full_text", "")
            candidates.append(
                (
                    self._score(query_tokens, doc.get("title", ""), text),
                    SourceDocument(
                        title=doc.get("title", "untitled"),
                        url=doc.get("provenance_url"),
                        snippet=text[:700],
                        metadata={
                            "citation_label": doc.get("citation_label"),
                            "is_synthetic": doc.get("is_synthetic", False),
                            "recommended_weight": doc.get("recommended_weight"),
                            "kind": "source_document",
                            "topic_file": topic_file.name,
                        },
                    ),
                )
            )

        for article in knowledge_base.get("knowledge_articles", []):
            text = article.get("content", "")
            candidates.append(
                (
                    self._score(query_tokens, article.get("title", ""), text),
                    SourceDocument(
                        title=article.get("title", "untitled"),
                        url=None,
                        snippet=text[:700],
                        metadata={
                            "citation_label": article.get("article_id"),
                            "is_synthetic": True,
                            "kind": "knowledge_article",
                            "topic_file": topic_file.name,
                        },
                    ),
                )
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in candidates[:max_results]]

    def _best_topic_file(self, manifest_path: Path, query: str) -> Path:
        query_tokens = _tokens(query)
        best_row: dict[str, str] | None = None
        best_score = -1.0
        with manifest_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                score = len(query_tokens & _tokens(row["title"] + " " + row["filename"]))
                if score > best_score:
                    best_score = score
                    best_row = row
        if best_row is None:
            raise AgentExecutionError("Corpus manifest is empty")
        return self.corpus_dir / "topics" / best_row["filename"]

    @staticmethod
    def _score(query_tokens: set[str], title: str, body: str) -> float:
        title_overlap = len(query_tokens & _tokens(title))
        body_overlap = len(query_tokens & _tokens(body[:2000]))
        return title_overlap * 2.0 + body_overlap
