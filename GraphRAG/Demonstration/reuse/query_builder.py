"""Convert a new-product FMEA prompt into retrieval-ready queries."""

from __future__ import annotations

from typing import Any

from .schemas import NewProductQuery


def normalize_user_query(raw_query: str, query_type: str, product_context: dict[str, Any] | None = None) -> NewProductQuery:
    """Normalize free text, query type, and optional product context."""
    raise NotImplementedError


def build_failure_search_text(query: NewProductQuery) -> str:
    """Create the semantic search text used to retrieve historical failures."""
    raise NotImplementedError


def build_context_search_filters(query: NewProductQuery) -> dict[str, Any]:
    """Build optional KG filters such as discipline, document type, or product family."""
    raise NotImplementedError


def expand_query_terms(query: NewProductQuery) -> list[str]:
    """Generate controlled expansion terms for failure modes, causes, and effects."""
    raise NotImplementedError

