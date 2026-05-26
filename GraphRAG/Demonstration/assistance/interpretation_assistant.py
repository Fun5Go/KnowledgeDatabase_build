"""Interpret FMEA fields using linked document evidence."""

from __future__ import annotations

from .schemas import FMEAReviewItem, InterpretationResult, LinkedEvidence


class InterpretationAssistant:
    """Explain abbreviated or domain-specific FMEA text in context."""

    def interpret_item(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> InterpretationResult:
        """Build a contextual interpretation for all populated FMEA fields."""
        raise NotImplementedError

    def explain_failure_attribution(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> dict[str, str]:
        """Explain why a failure mode, cause, or effect is attributed to the item."""
        raise NotImplementedError

    def explain_control_measure(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> str:
        """Explain a control measure with support from specifications, constraints, or tests."""
        raise NotImplementedError

    def resolve_abbreviations(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> dict[str, str]:
        """Map detected abbreviations or domain terms to contextual meanings."""
        raise NotImplementedError

