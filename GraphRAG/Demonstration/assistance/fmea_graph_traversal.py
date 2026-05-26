"""Neo4j traversals for FMEA assistant inputs.

The label and relationship names follow `KnowledgeGraph/construction_v2_FMEA.py`.
This module retrieves the concrete Failure/attribute node text that the parser
describes and the LLM later explains.
"""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


ATTRIBUTE_LABELS = {
    "element": "Element",
    "function": "Function",
    "mode": "Mode",
    "effect": "Effect",
    "cause": "Cause",
    "detection": "Detection",
    "prevention": "Prevention",
    "action": "Action",
}

ALLOWED_ATTRIBUTE_CHUNK_RELATION_TYPES = {
    "control_or_mitigation",
    "detection_or_reporting",
    "causal_mechanism",
    "condition_match",
    "consequence_or_effect",
    "design_specification",
    "trigger_or_context",
    "nominal_context_only",
}


class FMEAGraphTraversal:
    """Read-only Neo4j traversal helper for FMEA assistant demonstrations."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def run_query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a read query and return records as dictionaries."""
        records, _, _ = self.driver.execute_query(
            cypher,
            database_=self.database,
            **params,
        )
        return [record.data() for record in records]

    def get_failure_entity_context(self, failure_id: str) -> dict[str, Any]:
        """Retrieve cause/mode/effect/function/element/control text for one Failure.

        Use this for input style 1, for example:
        `DFMEA6011160042R01__R125`.
        """
        rows = self.run_query(
            """
            MATCH (f:Failure {failure_id:$failure_id})
            OPTIONAL MATCH (f)-[:HAS_ELEMENT]->(element:Element)
            OPTIONAL MATCH (f)-[:HAS_FUNCTION]->(function:Function)
            OPTIONAL MATCH (f)-[:HAS_CAUSE]->(cause:Cause)
            OPTIONAL MATCH (f)-[:HAS_MODE]->(mode:Mode)
            OPTIONAL MATCH (f)-[:HAS_EFFECT]->(effect:Effect)
            OPTIONAL MATCH (f)-[:HAS_CONTROLS]->(control)
            WHERE control:Detection OR control:Prevention
            OPTIONAL MATCH (f)-[:HAS_ACTION]->(action:Action)
            OPTIONAL MATCH (document:Document)-[:HAS_Failure]->(f)
            RETURN
                properties(f) AS failure,
                collect(DISTINCT properties(element)) AS elements,
                collect(DISTINCT properties(function)) AS functions,
                collect(DISTINCT properties(cause)) AS causes,
                collect(DISTINCT properties(mode)) AS modes,
                collect(DISTINCT properties(effect)) AS effects,
                collect(DISTINCT {
                    labels: labels(control),
                    properties: properties(control)
                }) AS controls,
                collect(DISTINCT properties(action)) AS actions,
                collect(DISTINCT properties(document)) AS documents
            """,
            failure_id=failure_id,
        )
        if not rows:
            return {"failure_id": failure_id, "found": False}
        context = rows[0]
        context["failure_id"] = failure_id
        context["found"] = True
        context["causal_paths"] = self.get_failure_causal_paths(failure_id)
        context["connected_chunks"] = self.get_failure_connected_chunks(failure_id)
        return _drop_empty_collections(context)

    def build_failure_pre_llm_package(
        self,
        failure_id: str,
        allowed_relation_types: set[str] | None = None,
        chunk_limit_per_attribute: int = 20,
    ) -> dict[str, Any]:
        """Build the semi-workflow output for one Failure before LLM input.

        The package contains the Failure's element/function/cause/mode/effect
        text and relevant chunk evidence connected to the cause, mode, and
        effect attribute nodes through allowed relationship types.
        """
        context = self.get_failure_entity_context(failure_id)
        if not context.get("found"):
            return {
                "failure_id": failure_id,
                "found": False,
                "failure_summary": {},
                "attribute_chunks": {
                    "cause": [],
                    "mode": [],
                    "effect": [],
                },
            }

        allowed = allowed_relation_types or ALLOWED_ATTRIBUTE_CHUNK_RELATION_TYPES
        attribute_chunks = {
            "cause": self.get_chunks_for_failure_attributes(
                failure_id=failure_id,
                attribute_label="Cause",
                failure_relationship="HAS_CAUSE",
                allowed_relation_types=allowed,
                limit=chunk_limit_per_attribute,
            ),
            "mode": self.get_chunks_for_failure_attributes(
                failure_id=failure_id,
                attribute_label="Mode",
                failure_relationship="HAS_MODE",
                allowed_relation_types=allowed,
                limit=chunk_limit_per_attribute,
            ),
            "effect": self.get_chunks_for_failure_attributes(
                failure_id=failure_id,
                attribute_label="Effect",
                failure_relationship="HAS_EFFECT",
                allowed_relation_types=allowed,
                limit=chunk_limit_per_attribute,
            ),
        }

        return {
            "failure_id": failure_id,
            "found": True,
            "failure_summary": {
                "failure": _drop_embedding(context.get("failure", {})),
                "elements": _text_nodes(context.get("elements", [])),
                "functions": _text_nodes(context.get("functions", [])),
                "causes": _text_nodes(context.get("causes", [])),
                "modes": _text_nodes(context.get("modes", [])),
                "effects": _text_nodes(context.get("effects", [])),
            },
            "causal_paths": _compact_causal_paths(context.get("causal_paths", [])),
            "attribute_chunks": attribute_chunks,
            "allowed_relation_types": sorted(allowed),
        }

    def get_failure_causal_paths(self, failure_id: str) -> list[dict[str, Any]]:
        """Retrieve explicit Cause -> Mode -> Effect paths attached to a Failure."""
        return self.run_query(
            """
            MATCH (f:Failure {failure_id:$failure_id})
            OPTIONAL MATCH path = (cause:Cause)<-[:HAS_CAUSE]-(f)-[:HAS_MODE]->(mode:Mode)-[:LEADS_TO]->(effect:Effect)<-[:HAS_EFFECT]-(f)
            WHERE (cause)-[:CAUSES]->(mode)
            RETURN
                properties(cause) AS cause,
                properties(mode) AS mode,
                properties(effect) AS effect,
                [rel IN relationships(path) | type(rel)] AS path_relationships
            """,
            failure_id=failure_id,
        )

    def get_failure_connected_chunks(self, failure_id: str, max_depth: int = 3) -> list[dict[str, Any]]:
        """Retrieve chunk-like document nodes connected to a Failure or its attributes.

        The document KG may use different chunk labels, so this traversal accepts
        any neighboring node whose label contains `Chunk`.
        """
        return self.run_query(
            """
            MATCH (f:Failure {failure_id:$failure_id})
            MATCH path = (f)-[*1..3]-(chunk)
            WHERE any(label IN labels(chunk) WHERE label CONTAINS "Chunk")
            RETURN DISTINCT
                elementId(chunk) AS node_id,
                labels(chunk) AS labels,
                properties(chunk) AS properties,
                [node IN nodes(path) | {
                    id: elementId(node),
                    labels: labels(node),
                    text: coalesce(
                        properties(node)["text"],
                        properties(node)["name"],
                        properties(node)["section_tag"],
                        ""
                    )
                }] AS path_nodes,
                [rel IN relationships(path) | type(rel)] AS path_relationships
            LIMIT 50
            """,
            failure_id=failure_id,
        )

    def get_chunks_for_failure_attributes(
        self,
        *,
        failure_id: str,
        attribute_label: str,
        failure_relationship: str,
        allowed_relation_types: set[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks linked to a Failure's Cause/Mode/Effect attribute nodes."""
        if attribute_label not in {"Cause", "Mode", "Effect"}:
            raise ValueError(f"Unsupported attribute_label: {attribute_label!r}")
        if failure_relationship not in {"HAS_CAUSE", "HAS_MODE", "HAS_EFFECT"}:
            raise ValueError(f"Unsupported failure_relationship: {failure_relationship!r}")

        allowed = sorted(allowed_relation_types or ALLOWED_ATTRIBUTE_CHUNK_RELATION_TYPES)
        rows = self.run_query(
            f"""
            MATCH (failure:Failure {{failure_id:$failure_id}})-[:{failure_relationship}]->(attribute:{attribute_label})
            MATCH path = (chunk)-[rel]-(attribute)
            WHERE any(label IN labels(chunk) WHERE label CONTAINS "Chunk")
            OPTIONAL MATCH (chunk)-[:RATIONALE_FOR]->(spec_from_rationale)
            WHERE any(label IN labels(spec_from_rationale) WHERE label CONTAINS "Chunk")
            OPTIONAL MATCH (rationale_for_spec)-[:RATIONALE_FOR]->(chunk)
            WHERE any(label IN labels(rationale_for_spec) WHERE label CONTAINS "RationaleChunk")
            WITH
                attribute,
                chunk,
                rel,
                path,
                spec_from_rationale,
                rationale_for_spec,
                type(rel) AS relationship_type,
                properties(rel) AS rel_props
            WITH
                attribute,
                chunk,
                rel,
                path,
                spec_from_rationale,
                rationale_for_spec,
                relationship_type,
                rel_props,
                coalesce(
                    rel_props.relation_type,
                    rel_props.primary_relation,
                    rel_props.type,
                    rel_props.name,
                    relationship_type
                ) AS relation_name
            WHERE relation_name IN $allowed_relation_types
               OR relationship_type IN $allowed_relation_types
            RETURN DISTINCT
                attribute.semantic_id AS attribute_semantic_id,
                coalesce(attribute.text, attribute.name, "") AS attribute_text,
                elementId(chunk) AS chunk_node_id,
                labels(chunk) AS chunk_labels,
                coalesce(
                    properties(chunk)["text"],
                    properties(chunk)["raw_text"],
                    properties(chunk)["page_content"],
                    properties(chunk)["objectives"],
                    properties(chunk)["name"],
                    ""
                ) AS chunk_text,
                coalesce(
                    properties(chunk)["name"],
                    properties(chunk)["chunk_id"],
                    properties(chunk)["qd_id"],
                    elementId(chunk)
                ) AS chunk_name,
                relationship_type,
                relation_name,
                rel_props,
                CASE
                    WHEN spec_from_rationale IS NULL THEN null
                    ELSE {{
                        node_id: elementId(spec_from_rationale),
                        labels: labels(spec_from_rationale),
                        name: coalesce(properties(spec_from_rationale)["name"], elementId(spec_from_rationale)),
                        text: coalesce(
                            properties(spec_from_rationale)["text"],
                            properties(spec_from_rationale)["raw_text"],
                            properties(spec_from_rationale)["page_content"],
                            properties(spec_from_rationale)["objectives"],
                            properties(spec_from_rationale)["name"],
                            ""
                        )
                    }}
                END AS spec_from_rationale,
                CASE
                    WHEN rationale_for_spec IS NULL THEN null
                    ELSE {{
                        node_id: elementId(rationale_for_spec),
                        labels: labels(rationale_for_spec),
                        name: coalesce(properties(rationale_for_spec)["name"], elementId(rationale_for_spec)),
                        text: coalesce(
                            properties(rationale_for_spec)["text"],
                            properties(rationale_for_spec)["raw_text"],
                            properties(rationale_for_spec)["page_content"],
                            properties(rationale_for_spec)["objectives"],
                            properties(rationale_for_spec)["name"],
                            ""
                        )
                    }}
                END AS rationale_for_spec,
                [node IN nodes(path) | {{
                    id: elementId(node),
                    labels: labels(node),
                    semantic_id: coalesce(properties(node)["semantic_id"], ""),
                    name: coalesce(properties(node)["name"], ""),
                    text: coalesce(
                        properties(node)["text"],
                        properties(node)["raw_text"],
                        properties(node)["page_content"],
                        properties(node)["objectives"],
                        properties(node)["name"],
                        ""
                    )
                }}] AS path_nodes,
                [path_rel IN relationships(path) | type(path_rel)] AS path_relationships
            LIMIT $limit
            """,
            failure_id=failure_id,
            allowed_relation_types=allowed,
            limit=limit,
        )
        formatted_rows = [_format_attribute_chunk_row(row, attribute_label.lower()) for row in rows]
        return _merge_attribute_chunk_relationships(formatted_rows)

    def get_attribute_context(
        self,
        *,
        semantic_id: str = "",
        text: str = "",
        attribute_type: str = "unknown",
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Retrieve node text and neighboring failures for one attribute input.

        Use this for input style 2. It supports direct `semantic_id` lookup or
        text lookup across Cause/Mode/Effect when the label is unknown.
        """
        labels = _labels_for_attribute_type(attribute_type)
        attribute_nodes = (
            self.find_attribute_by_semantic_id(semantic_id, labels)
            if semantic_id
            else self.find_attribute_by_text(text, labels, top_k=top_k)
        )
        failure_contexts = []
        for node in attribute_nodes:
            failure_contexts.extend(
                self.get_failures_for_attribute(
                    semantic_id=node["semantic_id"],
                    label=node["label"],
                    top_k=top_k,
                )
            )
        return {
            "input": {
                "semantic_id": semantic_id,
                "text": text,
                "attribute_type": attribute_type,
            },
            "attribute_nodes": attribute_nodes,
            "neighbor_failures": _deduplicate_by_key(failure_contexts, "failure_id"),
        }

    def find_attribute_by_semantic_id(self, semantic_id: str, labels: list[str] | None = None) -> list[dict[str, Any]]:
        """Find attribute node text by semantic_id."""
        labels = labels or list(ATTRIBUTE_LABELS.values())
        rows: list[dict[str, Any]] = []
        for label in labels:
            rows.extend(
                self.run_query(
                    f"""
                    MATCH (n:{label} {{semantic_id:$semantic_id}})
                    RETURN
                        $label AS label,
                        n.semantic_id AS semantic_id,
                        coalesce(n.text, n.name, "") AS text,
                        properties(n) AS properties
                    """,
                    semantic_id=semantic_id,
                    label=label,
                )
            )
        return rows

    def find_attribute_by_text(self, text: str, labels: list[str] | None = None, top_k: int = 20) -> list[dict[str, Any]]:
        """Find attribute nodes by exact or case-insensitive contained text."""
        labels = labels or ["Cause", "Mode", "Effect"]
        rows: list[dict[str, Any]] = []
        for label in labels:
            rows.extend(
                self.run_query(
                    f"""
                    MATCH (n:{label})
                    WHERE toLower(coalesce(n.text, n.name, "")) CONTAINS toLower($text)
                    RETURN
                        $label AS label,
                        n.semantic_id AS semantic_id,
                        coalesce(n.text, n.name, "") AS text,
                        properties(n) AS properties
                    LIMIT $top_k
                    """,
                    text=text,
                    label=label,
                    top_k=top_k,
                )
            )
        return rows[:top_k]

    def get_failures_for_attribute(self, semantic_id: str, label: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Traverse from one attribute node to its connected Failure rows."""
        if label not in ATTRIBUTE_LABELS.values():
            raise ValueError(f"Unsupported FMEA attribute label: {label!r}")
        rel = _failure_relationship_for_label(label)
        if not rel:
            return []
        rows = self.run_query(
            f"""
            MATCH (attribute:{label} {{semantic_id:$semantic_id}})
            MATCH (failure:Failure)-[:{rel}]->(attribute)
            OPTIONAL MATCH (failure)-[:HAS_ELEMENT]->(element:Element)
            OPTIONAL MATCH (failure)-[:HAS_FUNCTION]->(function:Function)
            OPTIONAL MATCH (failure)-[:HAS_CAUSE]->(cause:Cause)
            OPTIONAL MATCH (failure)-[:HAS_MODE]->(mode:Mode)
            OPTIONAL MATCH (failure)-[:HAS_EFFECT]->(effect:Effect)
            OPTIONAL MATCH (document:Document)-[:HAS_Failure]->(failure)
            RETURN
                failure.failure_id AS failure_id,
                properties(failure) AS failure,
                collect(DISTINCT properties(element)) AS elements,
                collect(DISTINCT properties(function)) AS functions,
                collect(DISTINCT properties(cause)) AS causes,
                collect(DISTINCT properties(mode)) AS modes,
                collect(DISTINCT properties(effect)) AS effects,
                collect(DISTINCT properties(document)) AS documents
            LIMIT $top_k
            """,
            semantic_id=semantic_id,
            top_k=top_k,
        )
        return [_drop_empty_collections(row) for row in rows]


def _labels_for_attribute_type(attribute_type: str) -> list[str]:
    normalized = attribute_type.lower().replace("failure_", "")
    if normalized in ATTRIBUTE_LABELS:
        return [ATTRIBUTE_LABELS[normalized]]
    if normalized == "semantic_id":
        return list(ATTRIBUTE_LABELS.values())
    return ["Cause", "Mode", "Effect"]


def _failure_relationship_for_label(label: str) -> str:
    return {
        "Element": "HAS_ELEMENT",
        "Function": "HAS_FUNCTION",
        "Mode": "HAS_MODE",
        "Cause": "HAS_CAUSE",
        "Effect": "HAS_EFFECT",
        "Detection": "HAS_CONTROLS",
        "Prevention": "HAS_CONTROLS",
        "Action": "HAS_ACTION",
    }.get(label, "")


def _drop_empty_collections(payload: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in payload.items():
        if isinstance(value, list):
            value = [item for item in value if item not in ({}, None)]
        clean[key] = value
    return clean


def _text_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        text = node.get("text") or node.get("name") or ""
        if not text and not node.get("semantic_id"):
            continue
        output.append(
            {
                "semantic_id": node.get("semantic_id", ""),
                "text": text,
                "properties": _drop_embedding(node),
            }
        )
    return output


def _drop_embedding(node: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in node.items() if key != "embedding"}


def _compact_causal_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for path in paths:
        if not isinstance(path, dict):
            continue
        compact.append(
            {
                "cause": _node_name(path.get("cause", {})),
                "mode": _node_name(path.get("mode", {})),
                "effect": _node_name(path.get("effect", {})),
                "path_relationships": path.get("path_relationships", []),
            }
        )
    return compact


def _node_name(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    return str(node.get("name") or node.get("text") or "")


def _format_attribute_chunk_row(row: dict[str, Any], attribute_type: str) -> dict[str, Any]:
    rel_props = row.get("rel_props") or {}
    relation_name = row.get("relation_name", "") or row.get("relationship_type", "")
    chunk = {
        "node_id": row.get("chunk_node_id", ""),
        "labels": row.get("chunk_labels", []),
        "name": row.get("chunk_name", ""),
        "text": row.get("chunk_text", ""),
    }
    spec_from_rationale = row.get("spec_from_rationale") or {}
    rationale_for_spec = row.get("rationale_for_spec") or {}
    paired_chunk = _build_spec_rationale_pair(chunk, spec_from_rationale, rationale_for_spec)
    return {
        "attribute_type": attribute_type,
        "attribute_semantic_id": row.get("attribute_semantic_id", ""),
        "attribute_text": row.get("attribute_text", ""),
        "chunk": chunk,
        "paired_chunk": paired_chunk,
        "relationship": {
            "type": row.get("relationship_type", ""),
            "types": [row.get("relationship_type", "")] if row.get("relationship_type") else [],
            "relation_name": relation_name,
            "relation_names": [relation_name] if relation_name else [],
            "properties": rel_props,
            "evidence_span": rel_props.get("evidence_span", ""),
            "evidence_spans": [rel_props.get("evidence_span", "")] if rel_props.get("evidence_span") else [],
            "support_capability": rel_props.get("support_capability", ""),
            "support_capabilities": [rel_props.get("support_capability", "")] if rel_props.get("support_capability") else [],
            "justification": rel_props.get("justification", ""),
            "justifications": [rel_props.get("justification", "")] if rel_props.get("justification") else [],
        },
        "path": {
            "nodes": row.get("path_nodes", []),
            "relationships": row.get("path_relationships", []),
        },
    }


def _merge_attribute_chunk_relationships(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        chunk = record.get("chunk", {})
        pair = record.get("paired_chunk", {})
        key = (
            record.get("attribute_semantic_id", ""),
            record.get("attribute_type", ""),
            pair.get("pair_id") or chunk.get("node_id") or chunk.get("name", ""),
        )
        if key not in merged:
            merged[key] = record
            continue

        target = merged[key]
        source_rel = record.get("relationship", {})
        target_rel = target.setdefault("relationship", {})
        _append_unique(target_rel, "relation_names", source_rel.get("relation_name"))
        _append_unique(target_rel, "evidence_spans", source_rel.get("evidence_span"))
        _append_unique(target_rel, "support_capabilities", source_rel.get("support_capability"))
        _append_unique(target_rel, "justifications", source_rel.get("justification"))
        _append_unique(target_rel, "types", source_rel.get("type"))
        target_rel["relation_name"] = "; ".join(target_rel.get("relation_names", []))
        target_rel["evidence_span"] = " | ".join(target_rel.get("evidence_spans", []))
        target_rel["support_capability"] = "; ".join(target_rel.get("support_capabilities", []))
        target_rel["justification"] = " | ".join(target_rel.get("justifications", []))
        target_rel["type"] = "; ".join(target_rel.get("types", [target_rel.get("type", "")]))
        _merge_pair_content(target.setdefault("paired_chunk", {}), record.get("paired_chunk", {}))
        target.setdefault("paths", [target.get("path", {})])
        target["paths"].append(record.get("path", {}))

    return list(merged.values())


def _build_spec_rationale_pair(
    chunk: dict[str, Any],
    spec_from_rationale: dict[str, Any],
    rationale_for_spec: dict[str, Any],
) -> dict[str, Any]:
    chunk_is_rationale = _is_rationale_chunk(chunk)
    if chunk_is_rationale:
        rationale = chunk
        specification = spec_from_rationale
    else:
        specification = chunk
        rationale = rationale_for_spec

    pair_id = (
        specification.get("node_id")
        or rationale.get("node_id")
        or chunk.get("node_id")
        or chunk.get("name", "")
    )
    return {
        "pair_id": pair_id,
        "specification": _compact_chunk(specification),
        "rationale": _compact_chunk(rationale),
        "display_name": _pair_display_name(specification, rationale, chunk),
        "display_text": _pair_display_text(specification, rationale, chunk),
    }


def _merge_pair_content(target: dict[str, Any], source: dict[str, Any]) -> None:
    if not target or not source:
        return
    for key in ("specification", "rationale"):
        if not target.get(key, {}).get("text") and source.get(key, {}).get("text"):
            target[key] = source[key]
    if not target.get("display_text") and source.get("display_text"):
        target["display_text"] = source["display_text"]
    if not target.get("display_name") and source.get("display_name"):
        target["display_name"] = source["display_name"]


def _is_rationale_chunk(chunk: dict[str, Any]) -> bool:
    labels = chunk.get("labels", [])
    name = str(chunk.get("name", "")).lower()
    return any("RationaleChunk" in str(label) for label in labels) or name.endswith("_rationale")


def _compact_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    if not chunk:
        return {"node_id": "", "labels": [], "name": "", "text": ""}
    return {
        "node_id": chunk.get("node_id", ""),
        "labels": chunk.get("labels", []),
        "name": chunk.get("name", ""),
        "text": chunk.get("text", ""),
    }


def _pair_display_name(specification: dict[str, Any], rationale: dict[str, Any], fallback: dict[str, Any]) -> str:
    spec_name = specification.get("name", "")
    rationale_name = rationale.get("name", "")
    if spec_name and rationale_name:
        return f"{spec_name} + {rationale_name}"
    return spec_name or rationale_name or fallback.get("name", "")


def _pair_display_text(specification: dict[str, Any], rationale: dict[str, Any], fallback: dict[str, Any]) -> str:
    parts = []
    if specification.get("text"):
        parts.append(f"Specification: {specification['text']}")
    if rationale.get("text"):
        parts.append(f"Rationale: {rationale['text']}")
    return "\n".join(parts) or fallback.get("text", "")


def _append_unique(target: dict[str, Any], key: str, value: Any) -> None:
    if not value:
        return
    values = target.setdefault(key, [])
    if value not in values:
        values.append(value)


def _deduplicate_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen = set()
    deduplicated = []
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(row)
    return deduplicated
