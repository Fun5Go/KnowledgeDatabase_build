"""Detect inconsistencies between FMEA text and connected document evidence."""

from __future__ import annotations

from .schemas import FMEAReviewItem, LinkedEvidence, ReviewFinding


class ConsistencyChecker:
    """Check consistency between FMEA content and linked KG context."""

    def check_field_consistency(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> list[ReviewFinding]:
        """Detect contradictions between FMEA fields and linked evidence."""
        raise NotImplementedError

    def check_control_consistency(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> list[ReviewFinding]:
        """Check whether controls align with design constraints and verification evidence."""
        raise NotImplementedError

    def check_chain_completeness(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> list[ReviewFinding]:
        """Check whether the failure chain has gaps or weakly connected steps."""
        raise NotImplementedError

    def merge_findings(self, finding_groups: list[list[ReviewFinding]]) -> list[ReviewFinding]:
        """Deduplicate and rank review findings by severity and evidence strength."""
        raise NotImplementedError

