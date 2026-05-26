"""End-to-end workflow for 'FMEA Assistant for Interpretation and Review'."""

from __future__ import annotations

from typing import Any

from .consistency_checker import ConsistencyChecker
from .evidence_linker import EvidenceLinker
from .fmea_graph_traversal import ALLOWED_ATTRIBUTE_CHUNK_RELATION_TYPES
from .fmea_text_parser import parse_failure_node_input, parse_fmea_row
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

    def build_failure_pre_llm_text_package(
        self,
        failure_id: str,
        chunk_limit_per_attribute: int = 20,
    ) -> dict[str, Any]:
        """Return element/function/cause/mode/effect and attribute-linked chunks.

        This is the semi workflow before LLM interpretation. It accepts a
        Failure id and returns a structured text package that can be shown
        directly or passed into a later LLM prompt.
        """
        item = parse_failure_node_input(failure_id)
        package = self.evidence_linker.fmea_traversal.build_failure_pre_llm_package(
            failure_id=item.metadata["failure_id"],
            allowed_relation_types=ALLOWED_ATTRIBUTE_CHUNK_RELATION_TYPES,
            chunk_limit_per_attribute=chunk_limit_per_attribute,
        )
        package["input_item"] = {
            "item_id": item.item_id,
            "input_type": item.metadata.get("input_type", ""),
            "llm_task": item.metadata.get("llm_task", {}),
        }
        package["text_report"] = render_failure_pre_llm_text_package(package)
        return package

    def interpret_failure_id(
        self,
        failure_id: str,
        chunk_limit_per_attribute: int = 20,
    ) -> dict[str, Any]:
        """Build the evidence package and run LLM interpretation for one Failure."""
        package = self.build_failure_pre_llm_text_package(
            failure_id=failure_id,
            chunk_limit_per_attribute=chunk_limit_per_attribute,
        )
        if not should_call_llm_for_failure_package(package):
            package["interpretation_skipped"] = {
                "status": "insufficient_context",
                "reason": "No linked failure summary or attribute chunk evidence was retrieved.",
            }
            package["interpretation_report"] = render_interpretation_skipped_report(package)
            return package
        interpretation = self.interpreter.interpret_failure_package(package)
        package["interpretation"] = interpretation
        package["interpretation_report"] = render_failure_interpretation_report(interpretation)
        return package


def render_failure_pre_llm_text_package(package: dict[str, Any]) -> str:
    """Render the pre-LLM package into readable plain text."""
    if not package.get("found"):
        return f"Failure not found: {package.get('failure_id', '')}"

    summary = package.get("failure_summary", {})
    lines = [
        f"Failure ID: {package.get('failure_id', '')}",
        "",
        "Element:",
        *_render_node_lines(summary.get("elements", [])),
        "",
        "Function:",
        *_render_node_lines(summary.get("functions", [])),
        "",
        "Cause:",
        *_render_node_lines(summary.get("causes", [])),
        "",
        "Mode:",
        *_render_node_lines(summary.get("modes", [])),
        "",
        "Effect:",
        *_render_node_lines(summary.get("effects", [])),
        "",
        "Relevant Chunks Under Failure Attributes:",
    ]
    attribute_chunks = package.get("attribute_chunks", {})
    for attribute_type in ("cause", "mode", "effect"):
        lines.append("")
        lines.append(f"{attribute_type.title()} chunks:")
        chunks = attribute_chunks.get(attribute_type, [])
        if not chunks:
            lines.append("- No linked chunks found for allowed relationship types.")
            continue
        for index, chunk_record in enumerate(chunks, start=1):
            chunk = chunk_record.get("chunk", {})
            paired_chunk = chunk_record.get("paired_chunk", {})
            relationship = chunk_record.get("relationship", {})
            lines.append(f"- [{index}] {paired_chunk.get('display_name') or chunk.get('name', '')}")
            lines.append(f"  Attribute: {chunk_record.get('attribute_text', '')}")
            lines.append(f"  Relationship: {relationship.get('relation_name') or relationship.get('type', '')}")
            if relationship.get("support_capability"):
                lines.append(f"  Support: {relationship.get('support_capability')}")
            if relationship.get("evidence_span"):
                lines.append(f"  Evidence span: {relationship.get('evidence_span')}")
            text = paired_chunk.get("display_text") or chunk.get("text", "")
            if text:
                lines.append(f"  Chunk text: {text}")
    return "\n".join(lines)


def should_call_llm_for_failure_package(package: dict[str, Any]) -> bool:
    """Return False when the retrieved context is too empty for grounded LLM interpretation."""
    if not package.get("found"):
        return False
    summary = package.get("failure_summary", {})
    has_failure_nodes = any(
        summary.get(key)
        for key in ("elements", "functions", "causes", "modes", "effects")
    )
    attribute_chunks = package.get("attribute_chunks", {})
    has_chunks = any(attribute_chunks.get(key) for key in ("cause", "mode", "effect"))
    return bool(has_failure_nodes or has_chunks)


def render_interpretation_skipped_report(package: dict[str, Any]) -> str:
    """Render a concise report when LLM interpretation is skipped."""
    skipped = package.get("interpretation_skipped", {})
    return "\n".join(
        [
            f"Failure ID: {package.get('failure_id', '')}",
            "",
            "LLM interpretation skipped.",
            f"Status: {skipped.get('status', 'insufficient_context')}",
            f"Reason: {skipped.get('reason', '')}",
        ]
    )


def _render_node_lines(nodes: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return ["- Not available"]
    return [
        f"- {node.get('text', '')} ({node.get('semantic_id', '')})".rstrip()
        for node in nodes
    ]


def render_failure_interpretation_report(interpretation: dict[str, Any]) -> str:
    """Render LLM interpretation into readable plain text."""
    support = interpretation.get("evidence_support_assessment", {})
    how = interpretation.get("how_it_happens", {})
    why = interpretation.get("why_explanation", {})
    what = interpretation.get("what_is_affected", {})
    lines = [
        f"Failure ID: {interpretation.get('failure_id', '')}",
        "",
        "Causal Chain Summary:",
        str(interpretation.get("causal_chain_summary", "")),
        "",
        "Why:",
        str(why.get("answer", "")),
        f"Confidence: {why.get('confidence', '')}",
        "",
        "What Is Affected:",
        str(what.get("answer", "")),
        f"Confidence: {what.get('confidence', '')}",
        "",
        "How It Happens:",
        str(how.get("answer", "")),
        f"Confidence: {how.get('confidence', '')}",
        "",
        "Evidence Support Assessment:",
        f"Overall: {support.get('overall_status', '')}",
        f"Cause: {support.get('cause_support', '')}",
        f"Mode: {support.get('mode_support', '')}",
        f"Effect: {support.get('effect_support', '')}",
        f"Rationale: {support.get('rationale', '')}",
    ]
    gaps = interpretation.get("missing_or_weak_evidence", [])
    if gaps:
        lines.extend(["", "Missing Or Weak Evidence:"])
        for gap in gaps:
            if isinstance(gap, dict):
                lines.append(f"- {gap.get('gap', '')} | Needed: {gap.get('needed_evidence', '')}")
            else:
                lines.append(f"- {gap}")
    notes = interpretation.get("review_notes", [])
    if notes:
        lines.extend(["", "Review Notes:"])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)
