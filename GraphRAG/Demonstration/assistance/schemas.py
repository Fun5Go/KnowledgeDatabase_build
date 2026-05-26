"""Data contracts for the FMEA interpretation and review assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FMEAReviewItem:
    """One FMEA row or selected FMEA field that needs interpretation/review."""

    item_id: str
    function: str = ""
    failure_mode: str = ""
    failure_cause: str = ""
    failure_effect: str = ""
    control_measure: str = ""
    detection_measure: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkedEvidence:
    """Supporting graph evidence linked to an FMEA item or field."""

    evidence_id: str
    evidence_type: str
    source_node_id: str
    source_label: str
    text: str
    relation_path: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpretationResult:
    """Contextual interpretation of abbreviated or domain-specific FMEA text."""

    item_id: str
    normalized_terms: dict[str, str]
    field_explanations: dict[str, str]
    supporting_evidence: list[LinkedEvidence]
    unresolved_terms: list[str] = field(default_factory=list)


@dataclass
class ReviewFinding:
    """One review or consistency finding grounded in KG evidence."""

    finding_id: str
    severity: str
    category: str
    message: str
    evidence: list[LinkedEvidence] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class AssistanceReview:
    """Final assistant output for one FMEA item."""

    item: FMEAReviewItem
    interpretation: InterpretationResult
    findings: list[ReviewFinding]
    design_support_status: str
    verification_support_status: str


def build_assistance_output_schema() -> dict[str, Any]:
    """Return the expected JSON-like output schema for the assistant."""
    raise NotImplementedError

