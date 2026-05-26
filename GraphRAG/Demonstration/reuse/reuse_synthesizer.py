"""Synthesize retrieved failures and graph context into FMEA reuse suggestions."""

from __future__ import annotations

from .schemas import HistoricalFailureCandidate, NewProductQuery, ReuseRecommendation, SupportingContext


class ReuseSynthesizer:
    """Transform evidence into proposed risks, controls, specifications, and tests."""

    def suggest_relevant_risks(
        self,
        query: NewProductQuery,
        candidates: list[HistoricalFailureCandidate],
        contexts: list[SupportingContext],
    ) -> list[dict]:
        """Suggest new-product failure risks grounded in historical failures."""
        raise NotImplementedError

    def suggest_controls(
        self,
        query: NewProductQuery,
        candidates: list[HistoricalFailureCandidate],
        contexts: list[SupportingContext],
    ) -> list[dict]:
        """Suggest prevention, detection, or mitigation controls from reusable evidence."""
        raise NotImplementedError

    def suggest_specifications_or_tests(
        self,
        query: NewProductQuery,
        contexts: list[SupportingContext],
    ) -> list[dict]:
        """Suggest targeted specifications, verification checks, or acceptance tests."""
        raise NotImplementedError

    def build_recommendation(
        self,
        query: NewProductQuery,
        candidates: list[HistoricalFailureCandidate],
        contexts: list[SupportingContext],
    ) -> ReuseRecommendation:
        """Assemble the final reuse recommendation object."""
        raise NotImplementedError

