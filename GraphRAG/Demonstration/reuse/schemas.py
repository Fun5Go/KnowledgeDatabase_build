"""Data contracts for the new-product knowledge-reuse demonstration.

This module should hold lightweight typed schemas for inputs, intermediate
retrieval results, graph context, and final reuse recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NewProductQuery:
    """User query describing a new product FMEA concern."""

    query_id: str
    query_type: str
    text: str
    product_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalFailureCandidate:
    """A semantically similar historical failure returned from the KG."""

    rank: int
    node_id: str
    labels: list[str]
    title: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SupportingContext:
    """Development-document context connected to a historical failure."""

    failure_node_id: str
    context_nodes: list[dict[str, Any]]
    context_paths: list[dict[str, Any]]


@dataclass
class ReuseRecommendation:
    """Actionable reuse output for the new-product FMEA workflow."""

    query_id: str
    historical_failures: list[HistoricalFailureCandidate]
    supporting_context: list[SupportingContext]
    suggested_risks: list[dict[str, Any]]
    suggested_controls: list[dict[str, Any]]
    suggested_specifications_or_tests: list[dict[str, Any]]


def build_output_schema() -> dict[str, Any]:
    """Return the expected JSON-like output shape for reports and evaluation."""
    raise NotImplementedError

