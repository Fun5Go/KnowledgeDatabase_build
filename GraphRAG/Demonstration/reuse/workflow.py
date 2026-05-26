"""End-to-end workflow for 'Knowledge Reuse for New Product Development'."""

from __future__ import annotations

from typing import Any

from .context_expander import ContextExpander
from .historical_failure_retriever import HistoricalFailureRetriever
from .query_builder import normalize_user_query
from .reuse_synthesizer import ReuseSynthesizer
from .schemas import ReuseRecommendation


class KnowledgeReuseWorkflow:
    """Coordinate query building, historical failure retrieval, graph expansion, and synthesis."""

    def __init__(
        self,
        retriever: HistoricalFailureRetriever | None = None,
        expander: ContextExpander | None = None,
        synthesizer: ReuseSynthesizer | None = None,
    ) -> None:
        """Wire the workflow components."""
        self.retriever = retriever or HistoricalFailureRetriever()
        self.expander = expander or ContextExpander()
        self.synthesizer = synthesizer or ReuseSynthesizer()

    def run(
        self,
        raw_query: str,
        query_type: str = "failure_mode",
        product_context: dict[str, Any] | None = None,
        top_k: int = 10,
        max_depth: int = 2,
    ) -> ReuseRecommendation:
        """Run the full demonstration workflow for one new-product query."""
        raise NotImplementedError

    def retrieve(self, raw_query: str, query_type: str = "failure_mode", top_k: int = 10) -> dict[str, Any]:
        """Run only query normalization and historical failure retrieval."""
        raise NotImplementedError

    def expand(self, candidate_ids: list[str], max_depth: int = 2) -> dict[str, Any]:
        """Run only graph-context expansion for selected historical failures."""
        raise NotImplementedError

    def synthesize(self, payload: dict[str, Any]) -> ReuseRecommendation:
        """Run only reuse synthesis from precomputed candidates and context."""
        raise NotImplementedError

