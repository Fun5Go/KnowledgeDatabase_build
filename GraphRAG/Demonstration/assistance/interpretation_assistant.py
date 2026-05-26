"""Interpret FMEA fields using linked document evidence."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from GraphRAG.llm_init import configure_langsmith, get_llm_backend

from .schemas import FMEAReviewItem, InterpretationResult, LinkedEvidence


LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-5.2")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
USE_PLACEHOLDER_LLM = os.getenv("GRAPHRAG_ASSISTANCE_USE_PLACEHOLDER", "0").lower() in {
    "1",
    "true",
    "yes",
}
LANGSMITH_PROJECT_NAME = configure_langsmith("interpretation")


FAILURE_INTERPRETATION_PROMPT = """You are an engineering FMEA assistant.

Interpret one Failure entity using only the provided graph context and linked
chunk evidence. Do not invent product behavior, missing components, test
results, or undocumented causal links. If evidence is weak or absent, say so.

Required reasoning:
1. Explain WHY the failure happens from linked cause evidence.
2. Explain WHAT is affected from linked mode/effect/function/element evidence.
3. Explain HOW the cause-mode-effect chain propagates, including any missing
   intermediate logic that is implied but not directly evidenced.
4. Assess whether the linked evidence supports the Failure entity and the
   causal chain.

Return one valid JSON object matching this schema:
{output_schema_json}

Input package:
{payload_json}
"""


def build_failure_interpretation_output_schema() -> dict[str, Any]:
    """Return the required LLM output schema for one Failure interpretation."""
    return {
        "failure_id": "string",
        "causal_chain_summary": "short summary of cause -> mode -> effect",
        "why_explanation": {
            "answer": "why the failure happens",
            "evidence_used": ["chunk names or attribute ids"],
            "confidence": "high | medium | low",
        },
        "what_is_affected": {
            "answer": "what element/function/system behavior/customer or requirement is affected",
            "evidence_used": ["chunk names or attribute ids"],
            "confidence": "high | medium | low",
        },
        "how_it_happens": {
            "answer": "stepwise causal propagation",
            "explicit_steps": ["steps directly supported by graph/evidence"],
            "inferred_steps": ["steps inferred from the chain but not directly evidenced"],
            "confidence": "high | medium | low",
        },
        "evidence_support_assessment": {
            "overall_status": "supported | partially_supported | weakly_supported | unsupported",
            "cause_support": "supported | partially_supported | weakly_supported | unsupported",
            "mode_support": "supported | partially_supported | weakly_supported | unsupported",
            "effect_support": "supported | partially_supported | weakly_supported | unsupported",
            "rationale": "brief grounded assessment",
        },
        "missing_or_weak_evidence": [
            {
                "gap": "what is missing or weak",
                "needed_evidence": "specification | design | test | causal explanation | other",
            }
        ],
        "review_notes": ["concise notes useful for an engineer reviewing the FMEA row"],
    }


def build_failure_interpretation_prompt(package: dict[str, Any]) -> str:
    """Build the LLM prompt from the pre-LLM Failure package."""
    payload = build_compact_failure_interpretation_payload(package)
    return FAILURE_INTERPRETATION_PROMPT.format(
        output_schema_json=json.dumps(build_failure_interpretation_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(payload, indent=2, ensure_ascii=False, default=str),
    )


def build_compact_failure_interpretation_payload(package: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the LLM needs from the pre-LLM package."""
    summary = package.get("failure_summary", {})
    return {
        "failure_id": package.get("failure_id", ""),
        "failure_summary": {
            "element": _node_texts(summary.get("elements", [])),
            "function": _node_texts(summary.get("functions", [])),
            "cause": _node_texts(summary.get("causes", [])),
            "mode": _node_texts(summary.get("modes", [])),
            "effect": _node_texts(summary.get("effects", [])),
        },
        "causal_paths": _compact_causal_paths(package.get("causal_paths", [])),
        "attribute_chunks": {
            "cause": _compact_chunk_records(package.get("attribute_chunks", {}).get("cause", [])),
            "mode": _compact_chunk_records(package.get("attribute_chunks", {}).get("mode", [])),
            "effect": _compact_chunk_records(package.get("attribute_chunks", {}).get("effect", [])),
        },
        "allowed_relation_types": package.get("allowed_relation_types", []),
        "llm_task": package.get("input_item", {}).get("llm_task", {}),
    }


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response."""
    text = text.strip()
    if not text:
        raise ValueError("LLM returned an empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in LLM response.")
    return json.loads(match.group(0))


class InterpretationAssistant:
    """Explain abbreviated or domain-specific FMEA text in context."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        use_placeholder: bool = USE_PLACEHOLDER_LLM,
    ) -> None:
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self.use_placeholder = use_placeholder
        self.llm = None

    def interpret_failure_package(self, package: dict[str, Any]) -> dict[str, Any]:
        """Interpret a pre-LLM Failure package with why/what/how/support output."""
        if self.use_placeholder:
            return placeholder_failure_interpretation(package)
        prompt = build_failure_interpretation_prompt(package)
        content = self._invoke_model(prompt)
        return normalize_failure_interpretation(extract_json(content), package)

    def _invoke_model(self, prompt: str) -> str:
        if self.llm is None:
            self.llm = get_llm_backend(
                backend=self.backend,
                model=self.model,
                temperature=self.temperature,
                json_mode=self.json_mode,
            )
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

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


def normalize_failure_interpretation(result: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    """Add defaults so downstream code receives a predictable JSON object."""
    schema = build_failure_interpretation_output_schema()
    normalized = {
        "failure_id": str(result.get("failure_id") or package.get("failure_id", "")),
        "causal_chain_summary": str(result.get("causal_chain_summary", "")),
        "why_explanation": _normalize_answer_block(result.get("why_explanation")),
        "what_is_affected": _normalize_answer_block(result.get("what_is_affected")),
        "how_it_happens": _normalize_how_block(result.get("how_it_happens")),
        "evidence_support_assessment": _normalize_support_block(result.get("evidence_support_assessment")),
        "missing_or_weak_evidence": result.get("missing_or_weak_evidence") if isinstance(result.get("missing_or_weak_evidence"), list) else [],
        "review_notes": result.get("review_notes") if isinstance(result.get("review_notes"), list) else [],
    }
    normalized["_schema"] = schema
    return normalized


def placeholder_failure_interpretation(package: dict[str, Any]) -> dict[str, Any]:
    """Deterministic offline fallback for testing the workflow without an LLM."""
    summary = package.get("failure_summary", {})
    causes = _node_texts(summary.get("causes", []))
    modes = _node_texts(summary.get("modes", []))
    effects = _node_texts(summary.get("effects", []))
    attribute_chunks = package.get("attribute_chunks", {})
    cause_chunks = attribute_chunks.get("cause", [])
    mode_chunks = attribute_chunks.get("mode", [])
    effect_chunks = attribute_chunks.get("effect", [])
    return normalize_failure_interpretation(
        {
            "failure_id": package.get("failure_id", ""),
            "causal_chain_summary": " -> ".join(
                part for part in [
                    "; ".join(causes),
                    "; ".join(modes),
                    "; ".join(effects),
                ] if part
            ),
            "why_explanation": {
                "answer": "Placeholder: linked causes and cause-related chunks should explain why the failure occurs.",
                "evidence_used": _chunk_names(cause_chunks),
                "confidence": _confidence_from_chunks(cause_chunks),
            },
            "what_is_affected": {
                "answer": "Placeholder: linked effects, functions, and elements indicate what is affected.",
                "evidence_used": _chunk_names(effect_chunks),
                "confidence": _confidence_from_chunks(effect_chunks),
            },
            "how_it_happens": {
                "answer": "Placeholder: the graph chain should be interpreted as cause -> mode -> effect.",
                "explicit_steps": [step for step in [", ".join(causes), ", ".join(modes), ", ".join(effects)] if step],
                "inferred_steps": [],
                "confidence": _confidence_from_chunks(cause_chunks + mode_chunks + effect_chunks),
            },
            "evidence_support_assessment": {
                "overall_status": _support_status(cause_chunks + mode_chunks + effect_chunks),
                "cause_support": _support_status(cause_chunks),
                "mode_support": _support_status(mode_chunks),
                "effect_support": _support_status(effect_chunks),
                "rationale": "Placeholder assessment based on whether linked chunks were retrieved.",
            },
            "missing_or_weak_evidence": [],
            "review_notes": ["Placeholder output. Set GRAPHRAG_ASSISTANCE_USE_PLACEHOLDER=0 to call the configured LLM."],
        },
        package,
    )


def _normalize_answer_block(value: Any) -> dict[str, Any]:
    block = value if isinstance(value, dict) else {}
    return {
        "answer": str(block.get("answer", "")),
        "evidence_used": block.get("evidence_used") if isinstance(block.get("evidence_used"), list) else [],
        "confidence": _normalize_confidence(block.get("confidence")),
    }


def _normalize_how_block(value: Any) -> dict[str, Any]:
    block = value if isinstance(value, dict) else {}
    return {
        "answer": str(block.get("answer", "")),
        "explicit_steps": block.get("explicit_steps") if isinstance(block.get("explicit_steps"), list) else [],
        "inferred_steps": block.get("inferred_steps") if isinstance(block.get("inferred_steps"), list) else [],
        "confidence": _normalize_confidence(block.get("confidence")),
    }


def _normalize_support_block(value: Any) -> dict[str, str]:
    block = value if isinstance(value, dict) else {}
    return {
        "overall_status": _normalize_support_status(block.get("overall_status")),
        "cause_support": _normalize_support_status(block.get("cause_support")),
        "mode_support": _normalize_support_status(block.get("mode_support")),
        "effect_support": _normalize_support_status(block.get("effect_support")),
        "rationale": str(block.get("rationale", "")),
    }


def _normalize_confidence(value: Any) -> str:
    text = str(value or "").lower()
    return text if text in {"high", "medium", "low"} else "low"


def _normalize_support_status(value: Any) -> str:
    text = str(value or "").lower()
    allowed = {"supported", "partially_supported", "weakly_supported", "unsupported"}
    return text if text in allowed else "unsupported"


def _node_texts(nodes: list[dict[str, Any]]) -> list[str]:
    return [str(node.get("text", "")) for node in nodes if isinstance(node, dict) and node.get("text")]


def _compact_chunk_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for record in records:
        chunk = record.get("chunk", {})
        paired_chunk = record.get("paired_chunk", {})
        relationship = record.get("relationship", {})
        compact.append(
            {
                "attribute_text": record.get("attribute_text", ""),
                "chunk_name": paired_chunk.get("display_name") or chunk.get("name", ""),
                "chunk_text": paired_chunk.get("display_text") or chunk.get("text", ""),
                "specification_chunk": paired_chunk.get("specification", {}),
                "rationale_chunk": paired_chunk.get("rationale", {}),
                "relationship": relationship.get("relation_name") or relationship.get("type", ""),
                "evidence_span": relationship.get("evidence_span", ""),
                "support_capability": relationship.get("support_capability", ""),
                "justification": relationship.get("justification", ""),
            }
        )
    return compact


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


def _chunk_names(records: list[dict[str, Any]]) -> list[str]:
    return [
        record.get("paired_chunk", {}).get("display_name") or record.get("chunk", {}).get("name", "")
        for record in records
        if isinstance(record, dict)
        and (record.get("paired_chunk", {}).get("display_name") or record.get("chunk", {}).get("name"))
    ]


def _confidence_from_chunks(records: list[dict[str, Any]]) -> str:
    if not records:
        return "low"
    strong = [
        record for record in records
        if str(record.get("relationship", {}).get("support_capability", "")).lower() == "strong"
    ]
    return "high" if strong else "medium"


def _support_status(records: list[dict[str, Any]]) -> str:
    if not records:
        return "unsupported"
    strong = [
        record for record in records
        if str(record.get("relationship", {}).get("support_capability", "")).lower() == "strong"
    ]
    return "supported" if strong else "partially_supported"
