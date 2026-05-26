"""Retrieve linked specifications, design constraints, and test evidence."""

from __future__ import annotations

from typing import Any

from .fmea_graph_traversal import FMEAGraphTraversal
from .schemas import FMEAReviewItem, LinkedEvidence


class EvidenceLinker:
    """Graph access layer for FMEA-to-document evidence retrieval."""

    def __init__(self, graph_retriever: Any | None = None, fmea_traversal: FMEAGraphTraversal | None = None) -> None:
        """Store a Neo4j/GraphRAG access object for later implementation."""
        self.graph_retriever = graph_retriever
        self.fmea_traversal = fmea_traversal or FMEAGraphTraversal()

    def close(self) -> None:
        """Close owned graph resources."""
        self.fmea_traversal.close()

    def retrieve_failure_node_context(self, failure_id: str) -> dict[str, Any]:
        """Retrieve linked cause/mode/effect/function/chunk context for a Failure node."""
        return self.fmea_traversal.get_failure_entity_context(failure_id)

    def retrieve_attribute_context(
        self,
        *,
        semantic_id: str = "",
        text: str = "",
        attribute_type: str = "unknown",
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Retrieve attribute node text and neighboring Failure chains."""
        return self.fmea_traversal.get_attribute_context(
            semantic_id=semantic_id,
            text=text,
            attribute_type=attribute_type,
            top_k=top_k,
        )

    def retrieve_direct_links(self, item: FMEAReviewItem) -> list[LinkedEvidence]:
        """Retrieve evidence directly connected to the FMEA item in the KG."""
        input_type = item.metadata.get("input_type", "")
        if input_type == "failure_node":
            context = self.retrieve_failure_node_context(item.metadata.get("failure_id", item.item_id))
            return linked_evidence_from_failure_context(context)
        if input_type == "failure_attribute":
            context = self.retrieve_attribute_context(
                semantic_id=item.metadata.get("semantic_id", ""),
                text=item.raw_text,
                attribute_type=item.metadata.get("attribute_type", "unknown"),
            )
            return linked_evidence_from_attribute_context(context)
        raise NotImplementedError("Direct-link retrieval is only implemented for failure_node and failure_attribute inputs.")

    def retrieve_field_evidence(self, item: FMEAReviewItem, field_name: str, top_k: int = 10) -> list[LinkedEvidence]:
        """Retrieve evidence linked to a specific FMEA field such as cause or control."""
        raise NotImplementedError

    def retrieve_design_constraints(self, item: FMEAReviewItem, top_k: int = 10) -> list[LinkedEvidence]:
        """Retrieve linked design descriptions, constraints, or rationales."""
        raise NotImplementedError

    def retrieve_verification_evidence(self, item: FMEAReviewItem, top_k: int = 10) -> list[LinkedEvidence]:
        """Retrieve linked verification, test, or acceptance evidence."""
        raise NotImplementedError

    def build_contextual_evidence_view(self, evidence: list[LinkedEvidence]) -> dict[str, list[LinkedEvidence]]:
        """Group evidence into specs, design constraints, rationale, tests, and acceptance info."""
        raise NotImplementedError


def linked_evidence_from_failure_context(context: dict[str, Any]) -> list[LinkedEvidence]:
    """Convert raw Failure traversal output into assistant evidence objects."""
    evidence: list[LinkedEvidence] = []
    failure_id = context.get("failure_id", "")
    for field_name, evidence_type in [
        ("elements", "element"),
        ("functions", "function"),
        ("causes", "cause"),
        ("modes", "mode"),
        ("effects", "effect"),
        ("controls", "control"),
        ("actions", "action"),
        ("documents", "document"),
    ]:
        for index, node in enumerate(context.get(field_name, []), start=1):
            properties = node.get("properties", node) if isinstance(node, dict) else {}
            labels = node.get("labels", []) if isinstance(node, dict) else []
            text = properties.get("text") or properties.get("name") or properties.get("file_name") or ""
            evidence.append(
                LinkedEvidence(
                    evidence_id=f"{failure_id}:{evidence_type}:{index}",
                    evidence_type=evidence_type,
                    source_node_id=properties.get("semantic_id") or properties.get("failure_id") or properties.get("file_name") or "",
                    source_label=labels[0] if labels else evidence_type,
                    text=text,
                    metadata={"properties": properties},
                )
            )
    for index, chunk in enumerate(context.get("connected_chunks", []), start=1):
        properties = chunk.get("properties", {})
        labels = chunk.get("labels", [])
        evidence.append(
            LinkedEvidence(
                evidence_id=f"{failure_id}:chunk:{index}",
                evidence_type="chunk",
                source_node_id=chunk.get("node_id", ""),
                source_label=labels[0] if labels else "Chunk",
                text=properties.get("text") or properties.get("page_content") or properties.get("name") or "",
                relation_path=[
                    {
                        "nodes": chunk.get("path_nodes", []),
                        "relationships": chunk.get("path_relationships", []),
                    }
                ],
                metadata={"properties": properties},
            )
        )
    return evidence


def linked_evidence_from_attribute_context(context: dict[str, Any]) -> list[LinkedEvidence]:
    """Convert raw attribute traversal output into assistant evidence objects."""
    evidence: list[LinkedEvidence] = []
    for index, node in enumerate(context.get("attribute_nodes", []), start=1):
        evidence.append(
            LinkedEvidence(
                evidence_id=f"attribute:{index}",
                evidence_type="attribute",
                source_node_id=node.get("semantic_id", ""),
                source_label=node.get("label", ""),
                text=node.get("text", ""),
                metadata={"properties": node.get("properties", {})},
            )
        )
    for index, failure_context in enumerate(context.get("neighbor_failures", []), start=1):
        failure = failure_context.get("failure", {})
        evidence.append(
            LinkedEvidence(
                evidence_id=f"neighbor_failure:{index}",
                evidence_type="neighbor_failure",
                source_node_id=failure_context.get("failure_id", ""),
                source_label="Failure",
                text=failure.get("name") or failure_context.get("failure_id", ""),
                metadata={"failure_context": failure_context},
            )
        )
    return evidence
