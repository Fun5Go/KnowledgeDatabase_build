"""Check whether the FMEA failure chain is sufficiently supported."""

from __future__ import annotations

from .schemas import FMEAReviewItem, LinkedEvidence, ReviewFinding


class ReviewChecker:
    """Review support coverage for failure chains, controls, and verification."""

    def review_failure_chain_support(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> list[ReviewFinding]:
        """Assess whether function, mode, cause, effect, and controls have supporting evidence."""
        raise NotImplementedError

    def check_design_support(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> tuple[str, list[ReviewFinding]]:
        """Check whether design information supports the stated failure chain."""
        raise NotImplementedError

    def check_verification_support(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> tuple[str, list[ReviewFinding]]:
        """Check whether test or acceptance evidence supports the stated controls."""
        raise NotImplementedError

    def identify_missing_evidence(self, item: FMEAReviewItem, evidence: list[LinkedEvidence]) -> list[ReviewFinding]:
        """Identify unsupported FMEA fields and propose what evidence should be inspected."""
        raise NotImplementedError

