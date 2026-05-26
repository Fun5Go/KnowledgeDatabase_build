"""End-to-end workflow for 'FMEA Assistant for Interpretation and Review'."""

from __future__ import annotations

from typing import Any

from .consistency_checker import ConsistencyChecker
from .evidence_linker import EvidenceLinker
from .fmea_text_parser import parse_fmea_row
from .interpretation_assistant import InterpretationAssistant
from .review_checker import ReviewChecker
from .schemas import AssistanceReview


class FMEAAssistanceWorkflow:
    """Coordinate FMEA parsing, evidence linking, interpretation, and review."""

    def __init__(
        self,
        evidence_linker: EvidenceLinker | None = None,
        interpreter: InterpretationAssistant | None = None,
        review_checker: ReviewChecker | None = None,
        consistency_checker: ConsistencyChecker | None = None,
    ) -> None:
        """Wire assistant workflow components."""
        self.evidence_linker = evidence_linker or EvidenceLinker()
        self.interpreter = interpreter or InterpretationAssistant()
        self.review_checker = review_checker or ReviewChecker()
        self.consistency_checker = consistency_checker or ConsistencyChecker()

    def run(self, fmea_row: dict[str, Any], top_k: int = 10) -> AssistanceReview:
        """Run the full assistant workflow for one FMEA row."""
        raise NotImplementedError

    def interpret_only(self, fmea_row: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
        """Retrieve evidence and produce interpretation without review findings."""
        raise NotImplementedError

    def review_only(self, fmea_row: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
        """Retrieve evidence and produce support/consistency findings."""
        raise NotImplementedError

    def inspect_evidence(self, fmea_row: dict[str, Any], field_name: str = "", top_k: int = 10) -> dict[str, Any]:
        """Return linked evidence for direct user inspection."""
        item = parse_fmea_row(fmea_row)
        raise NotImplementedError

