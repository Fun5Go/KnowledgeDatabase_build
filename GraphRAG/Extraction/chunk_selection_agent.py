from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from DocLLM.LLMs.llm_init import configure_langsmith, get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


LANGSMITH_PROJECT_NAME = "FaiureChunkSelection"

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "chunk_selection_prompt.txt"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-5.2")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
MAX_PROMPT_INPUT_TOKENS = int(os.getenv("GRAPHRAG_EXTRACTION_MAX_INPUT_TOKENS", "8000"))
USE_PLACEHOLDER_LLM = os.getenv("GRAPHRAG_EXTRACTION_USE_PLACEHOLDER", "0").lower() in {
    "1",
    "true",
    "yes",
}
ALLOWED_EVIDENCE_LABELS = {"control", "cause", "specification"}
ALLOWED_SUPPORT_CAPABILITIES = {"weak", "moderate", "strong"}


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the chunk-selection prompt template from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM response."""

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


def build_output_schema() -> dict[str, Any]:
    """Return the expected agent response schema."""

    return {
        "analysis_id": "string",
        "query_type": "function_mode | cause",
        "top_chunks": [
            {
                "rank": "integer retrieval rank copied from the selected candidate chunk",
                "label": "string",
                "name": "string",
                "raw_text": "string copied exactly from the selected candidate chunk text",
                "evidence_span": "exact substring copied from raw_text",
                "evidence_label": "control | cause | specification",
                "support_capability": "weak | moderate | strong",
                "justification": "short explanation",
            }
        ],
        "affected_objects": ["string"],
        "causal_subjects": ["string"],
    }


def build_agent_payload(
    query_result: dict[str, Any],
    analysis_item: dict[str, Any] | None = None,
    max_chunk_text_length: int = 1800,
    max_prompt_input_tokens: int = MAX_PROMPT_INPUT_TOKENS,
) -> dict[str, Any]:
    """Build the LLM input from the GraphRAG query result."""

    analysis_item = analysis_item or {}
    evidence = order_evidence_by_connected_groups(
        query_result.get("evidence", []),
        query_result.get("connected_evidence_groups", []),
    )
    group_id_by_node_id = build_connected_group_id_by_node_id(
        query_result.get("connected_evidence_groups", [])
    )
    query_type = normalize_text(analysis_item.get("query_type"))

    payload = {
        "analysis_id": normalize_text(analysis_item.get("analysis_id")),
        "query_type": query_type,
        "query": build_llm_query_payload(analysis_item),
        "candidate_chunks": [
            normalize_candidate_chunk(
                item,
                index,
                max_chunk_text_length,
                connected_group_id=group_id_by_node_id.get(normalize_text(item.get("node_id"))),
            )
            for index, item in enumerate(evidence, start=1)
        ],
        "connected_chunk_groups": normalize_connected_chunk_groups(
            query_result.get("connected_evidence_groups", []),
            evidence,
        ),
    }
    return enforce_prompt_input_token_limit(payload, max_prompt_input_tokens)


def build_llm_query_payload(analysis_item: dict[str, Any]) -> dict[str, str]:
    """Build the compact query payload shown to the LLM."""

    query_type = normalize_text(analysis_item.get("query_type"))
    if query_type == "cause":
        return {
            "Failure cause": normalize_text(analysis_item.get("query_cause")),
            "Discipline": normalize_text(analysis_item.get("cause_discipline")),
        }

    return {
        "Function": normalize_text(analysis_item.get("function_text")),
        "Failure mode": normalize_text(analysis_item.get("query_mode")),
    }


def enforce_prompt_input_token_limit(
    payload: dict[str, Any],
    max_prompt_input_tokens: int,
) -> dict[str, Any]:
    """Limit the query payload placed into the LLM prompt."""

    if max_prompt_input_tokens <= 0:
        return payload

    candidate_chunks = list(payload.get("candidate_chunks", []))
    limited_payload = dict(payload)
    limited_payload["candidate_chunks"] = []
    limited_payload["connected_chunk_groups"] = {}

    for chunk in candidate_chunks:
        trial_chunk = dict(chunk)
        trial_payload = dict(limited_payload)
        trial_payload["candidate_chunks"] = limited_payload["candidate_chunks"] + [trial_chunk]
        trial_payload["connected_chunk_groups"] = filter_connected_chunk_groups(
            payload.get("connected_chunk_groups", []),
            trial_payload["candidate_chunks"],
        )
        trial_tokens = estimate_json_tokens(trial_payload)
        if trial_tokens <= max_prompt_input_tokens:
            limited_payload["candidate_chunks"].append(trial_chunk)
            limited_payload["connected_chunk_groups"] = trial_payload["connected_chunk_groups"]
            continue

        remaining_tokens = max_prompt_input_tokens - estimate_json_tokens(limited_payload)
        if remaining_tokens <= 80:
            break

        trial_chunk["text"] = trim_text_by_tokens(trial_chunk.get("text", ""), remaining_tokens)
        trial_payload["candidate_chunks"] = limited_payload["candidate_chunks"] + [trial_chunk]
        trial_payload["connected_chunk_groups"] = filter_connected_chunk_groups(
            payload.get("connected_chunk_groups", []),
            trial_payload["candidate_chunks"],
        )
        if estimate_json_tokens(trial_payload) <= max_prompt_input_tokens:
            limited_payload["candidate_chunks"].append(trial_chunk)
            limited_payload["connected_chunk_groups"] = trial_payload["connected_chunk_groups"]
        break

    limited_payload["prompt_input_token_threshold"] = max_prompt_input_tokens
    limited_payload["prompt_input_token_estimate"] = estimate_json_tokens(limited_payload)
    return limited_payload


def order_evidence_by_connected_groups(
    evidence: Any,
    connected_groups: Any,
) -> list[dict[str, Any]]:
    """Place retrieved chunks from the same graph component next to each other."""

    if not isinstance(evidence, list):
        return []

    evidence_items = [item for item in evidence if isinstance(item, dict)]
    evidence_by_id = {
        normalize_text(item.get("node_id")): item
        for item in evidence_items
        if normalize_text(item.get("node_id"))
    }
    if not isinstance(connected_groups, list) or not evidence_by_id:
        return evidence_items

    ordered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in connected_groups:
        if not isinstance(group, dict):
            continue
        group_ids = [
            normalize_text(node_id)
            for node_id in group.get("node_ids", [])
            if normalize_text(node_id) in evidence_by_id
        ]
        group_ids.sort(
            key=lambda node_id: normalize_int(evidence_by_id[node_id].get("retrieval_rank")) or 0
        )
        for node_id in group_ids:
            if node_id in seen_ids:
                continue
            ordered.append(evidence_by_id[node_id])
            seen_ids.add(node_id)

    for item in evidence_items:
        node_id = normalize_text(item.get("node_id"))
        if node_id and node_id in seen_ids:
            continue
        ordered.append(item)
    return ordered


def build_connected_group_id_by_node_id(connected_groups: Any) -> dict[str, int]:
    """Map only RELATED/IMPLEMENT graph-grouped chunks to their prompt group id."""

    if not isinstance(connected_groups, list):
        return {}

    group_id_by_node_id: dict[str, int] = {}
    for group in connected_groups:
        if not isinstance(group, dict) or not group.get("has_relationships"):
            continue
        relationships = [
            relationship
            for relationship in group.get("relationships", [])
            if isinstance(relationship, dict)
            and normalize_text(relationship.get("relationship")) in {"RELATED", "IMPLEMENT"}
        ]
        if not relationships:
            continue

        group_id = normalize_int(group.get("group_id"))
        if group_id is None:
            continue
        for node_id in group.get("node_ids", []):
            node_id = normalize_text(node_id)
            if node_id:
                group_id_by_node_id[node_id] = group_id
    return group_id_by_node_id


def normalize_connected_chunk_groups(
    connected_groups: Any,
    evidence: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build compact graph context for the LLM without duplicating chunk text."""

    if not isinstance(connected_groups, list):
        return {}

    chunk_by_node_id = {
        normalize_text(item.get("node_id")): {
            "rank": normalize_int(item.get("retrieval_rank")) or index,
            "name": normalize_text(item.get("name")),
        }
        for index, item in enumerate(evidence, start=1)
        if normalize_text(item.get("node_id"))
    }
    groups: dict[str, dict[str, Any]] = {}
    for group in connected_groups:
        if not isinstance(group, dict) or not group.get("has_relationships"):
            continue

        node_ids = [
            normalize_text(node_id)
            for node_id in group.get("node_ids", [])
            if normalize_text(node_id) in chunk_by_node_id
        ]
        if len(node_ids) < 2:
            continue

        chunks = [chunk_by_node_id[node_id] for node_id in node_ids]
        relationships = []
        for relationship in group.get("relationships", []):
            if not isinstance(relationship, dict):
                continue
            source_id = normalize_text(relationship.get("source_id"))
            target_id = normalize_text(relationship.get("target_id"))
            if source_id not in chunk_by_node_id or target_id not in chunk_by_node_id:
                continue
            relationships.append(
                {
                    "source_rank": chunk_by_node_id[source_id]["rank"],
                    "source_name": chunk_by_node_id[source_id]["name"],
                    "relationship": normalize_text(relationship.get("relationship")),
                    "target_rank": chunk_by_node_id[target_id]["rank"],
                    "target_name": chunk_by_node_id[target_id]["name"],
                }
            )

        if relationships:
            group_id = normalize_int(group.get("group_id")) or len(groups) + 1
            groups[str(group_id)] = {
                "chunks": chunks,
                "relationships": relationships,
            }

    return groups


def filter_connected_chunk_groups(
    connected_groups: Any,
    candidate_chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(connected_groups, dict):
        return {}

    available_ranks = {
        normalize_int(chunk.get("retrieval rank"))
        for chunk in candidate_chunks
        if normalize_int(chunk.get("retrieval rank")) is not None
    }
    filtered_groups: dict[str, dict[str, Any]] = {}
    for group_id, group in connected_groups.items():
        if not isinstance(group, dict):
            continue
        chunks = [
            chunk
            for chunk in group.get("chunks", [])
            if isinstance(chunk, dict) and normalize_int(chunk.get("rank")) in available_ranks
        ]
        relationships = [
            relationship
            for relationship in group.get("relationships", [])
            if isinstance(relationship, dict)
            and normalize_int(relationship.get("source_rank")) in available_ranks
            and normalize_int(relationship.get("target_rank")) in available_ranks
        ]
        if len(chunks) >= 2 and relationships:
            filtered_group = dict(group)
            filtered_group["chunks"] = chunks
            filtered_group["relationships"] = relationships
            filtered_groups[normalize_text(group_id)] = filtered_group
    return filtered_groups


def normalize_candidate_chunk(
    item: dict[str, Any],
    retrieval_rank: int,
    max_text_length: int,
    connected_group_id: int | None = None,
) -> dict[str, Any]:
    """Keep only stable, prompt-friendly candidate fields."""

    chunk = {
        "retrieval rank": normalize_int(item.get("retrieval_rank")) or retrieval_rank,
        "label": normalize_text(item.get("label")),
        "name": build_candidate_name_with_reason(item, max_text_length),
        "section_tag": normalize_text(item.get("section_tag")),
        "text": trim_text(item.get("text"), max_text_length),
    }
    if connected_group_id is not None:
        chunk["connected_group_id"] = connected_group_id
    return chunk


def build_candidate_name_with_reason(item: dict[str, Any], max_text_length: int) -> str:
    name = normalize_text(item.get("name"))
    rationale_texts = [
        normalize_text(text)
        for text in item.get("rationale_texts", [])
        if normalize_text(text)
    ]
    if not rationale_texts:
        return name

    reason_text = " | ".join(rationale_texts)
    reason_text = trim_text(reason_text, max(160, max_text_length // 3))
    if not name:
        return f"Reason: {reason_text}"
    return f"{name}\nReason: {reason_text}"


@traceable(
    run_type="prompt",
    name="graphrag_chunk_selection_prompt_builder",
    tags=["graphrag", "extraction", "chunk-selection", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_prompt(payload: dict[str, Any], template_text: str) -> str:
    """Render the chunk-selection prompt."""

    return template_text.format(
        output_schema_json=json.dumps(build_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
    )


class ChunkSelectionAgent:
    """LLM agent that selects retrieved chunks supporting a query mode/cause."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        use_placeholder: bool = USE_PLACEHOLDER_LLM,
        prompt_template_path: Path = PROMPT_TEMPLATE_PATH,
    ) -> None:
        self.prompt_template_text = load_prompt_template(prompt_template_path)
        self.use_placeholder = use_placeholder
        self.llm = None if use_placeholder else get_llm_backend(
            backend=backend,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    def select_chunks(
        self,
        query_result: dict[str, Any],
        analysis_item: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select chunks from a GraphRAG query result."""

        payload = build_agent_payload(query_result=query_result, analysis_item=analysis_item)
        if self.use_placeholder:
            return self._placeholder_response(payload)

        prompt = build_prompt(payload, self.prompt_template_text)
        content = self._invoke_model(prompt)
        return normalize_selection_response(extract_json(content), payload)

    @traceable(
        run_type="llm",
        name="graphrag_chunk_selection_llm_call",
        tags=["graphrag", "extraction", "chunk-selection", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        """Execute the real LLM call."""

        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Offline fallback that selects the highest ranked chunks."""

        candidates = payload.get("candidate_chunks", [])

        return {
            "analysis_id": normalize_text(payload.get("analysis_id")),
            "query_type": payload.get("query_type", ""),
            "top_chunks": [
                {
                    "rank": item.get("retrieval rank", index),
                    "label": item.get("label", ""),
                    "name": item.get("name", ""),
                    "raw_text": item.get("text", ""),
                    "evidence_span": shortest_placeholder_span(item.get("text", "")),
                    "evidence_label": placeholder_evidence_label(item.get("text", "")),
                    "support_capability": placeholder_support_capability(index),
                    "justification": "Placeholder selected this chunk from the highest-ranked retrieval results.",
                }
                for index, item in enumerate(candidates, start=1)
            ],
            "affected_objects": [],
            "causal_subjects": [],
        }


def normalize_selection_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the LLM response to the required selected-chunk output contract."""

    chunks = response.get("top_chunks") or response.get("selected_chunks") or []
    if not isinstance(chunks, list):
        chunks = []

    normalized_chunks: list[dict[str, Any]] = []
    candidate_by_key = build_candidate_lookup(payload)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        label = normalize_text(chunk.get("label"))
        name = normalize_text(chunk.get("name"))
        rank = normalize_int(chunk.get("rank"))
        if rank is None:
            rank = normalize_int(chunk.get("retrieval rank"))
        candidate = candidate_by_key.get(("rank", normalize_text(rank))) if rank is not None else None
        if rank is None:
            candidate = candidate_by_key.get(("label_name", label, name))
            if candidate:
                rank = normalize_int(candidate.get("retrieval rank"))
        if rank is None:
            continue
        if candidate is None:
            candidate = candidate_by_key.get(("label_name", label, name))

        if candidate:
            label = normalize_text(candidate.get("label"))
            name = normalize_text(candidate.get("name"))

        raw_text = normalize_text(
            candidate.get("text")
            if candidate
            else chunk.get("raw_text") or chunk.get("raw" + " text") or chunk.get("text")
        )
        evidence_label = normalize_evidence_label(chunk.get("evidence_label"))
        evidence_span = normalize_evidence_span(chunk.get("evidence_span"), raw_text)
        normalized_chunks.append(
            {
                "rank": rank,
                "label": label,
                "name": name,
                "raw_text": raw_text,
                "evidence_span": evidence_span,
                "evidence_label": evidence_label,
                "support_capability": normalize_support_capability(chunk.get("support_capability")),
                "justification": normalize_text(chunk.get("justification") or chunk.get("reason")),
            }
        )

    selected_texts = [chunk["raw_text"] for chunk in normalized_chunks]
    return {
        "analysis_id": normalize_text(
            response.get("analysis_id") or payload.get("analysis_id")
        ),
        "query_type": normalize_text(response.get("query_type") or payload.get("query_type")),
        "top_chunks": normalized_chunks,
        "affected_objects": normalize_top_level_terms(response.get("affected_objects"), selected_texts),
        "causal_subjects": normalize_top_level_terms(response.get("causal_subjects"), selected_texts),
    }


def build_candidate_lookup(payload: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Map candidate identifiers to candidate payloads."""

    candidate_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for candidate in payload.get("candidate_chunks", []):
        if not isinstance(candidate, dict):
            continue
        rank = normalize_int(candidate.get("retrieval rank"))
        label_name_key = (
            "label_name",
            normalize_text(candidate.get("label")),
            normalize_text(candidate.get("name")),
        )
        candidate_by_key[label_name_key] = candidate
        if rank is not None:
            candidate_by_key[("rank", normalize_text(rank))] = candidate
    return candidate_by_key


def normalize_evidence_label(value: Any) -> str:
    """Normalize selected-chunk evidence labels."""

    evidence_label = normalize_text(value).lower()
    if evidence_label == "reason":
        return "cause"
    if evidence_label in ALLOWED_EVIDENCE_LABELS:
        return evidence_label
    return "specification"


def normalize_support_capability(value: Any) -> str:
    """Normalize selected-chunk support strength."""

    capability = normalize_text(value).lower()
    if capability in ALLOWED_SUPPORT_CAPABILITIES:
        return capability
    return "weak"


def normalize_evidence_span(value: Any, raw_text: str) -> str:
    """Return an exact selected span, allowing same-text joins with ellipses."""

    span = normalize_text(value)
    if span and evidence_span_is_valid(span, raw_text):
        return span
    return shortest_placeholder_span(raw_text)


def evidence_span_is_valid(span: str, raw_text: str) -> bool:
    """Check that every span segment is copied exactly from the chunk text."""

    if not span or not raw_text:
        return False
    parts = [part.strip() for part in span.split(" ... ")]
    if not parts or any(not part for part in parts):
        return False
    return all(part in raw_text for part in parts)


def normalize_top_level_terms(value: Any, selected_texts: Sequence[str]) -> list[str]:
    """Deduplicate top-level noun phrases and keep only terms present in selected text."""

    if not isinstance(value, list):
        return []

    selected_text = "\n".join(selected_texts).lower()
    terms: list[str] = []
    seen_terms: set[str] = set()
    for item in value:
        term = normalize_text(item)
        if not term:
            continue
        if term.lower() not in selected_text:
            continue
        dedupe_key = term.lower()
        if dedupe_key in seen_terms:
            continue
        seen_terms.add(dedupe_key)
        terms.append(term)
    return terms


def normalize_int(value: Any) -> int | None:
    """Normalize integer-like values."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def placeholder_support_capability(rank: int) -> str:
    """Assign simple support labels for offline placeholder output."""

    if rank == 1:
        return "strong"
    if rank == 2:
        return "moderate"
    return "weak"


def placeholder_evidence_label(text: Any) -> str:
    """Assign a simple evidence label for offline placeholder output."""

    lowered = normalize_text(text).lower()
    control_terms = (
        "detect",
        "protect",
        "prevent",
        "mitigat",
        "shutdown",
        "reset",
        "watchdog",
        "derat",
        "debounce",
        "filter",
        "error",
        "recover",
        "diagnos",
    )
    cause_terms = (
        "because",
        "risk",
        "damage",
        "safety",
        "reliability",
        "unstable",
        "avoid",
    )
    if any(term in lowered for term in control_terms):
        return "control"
    if any(term in lowered for term in cause_terms):
        return "cause"
    return "specification"


def shortest_placeholder_span(text: Any, max_length: int = 240) -> str:
    """Use an exact prefix as a valid fallback evidence span."""

    normalized = normalize_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip()


def normalize_text(value: Any) -> str:
    """Normalize values to safe strings."""

    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def estimate_json_tokens(value: Any) -> int:
    """Estimate tokens for a JSON payload without requiring a tokenizer package."""

    text = json.dumps(json_safe(value), ensure_ascii=False)
    return estimate_text_tokens(text)


def json_safe(value: Any) -> Any:
    """Convert values such as NumPy scalars into JSON-serializable objects."""

    if isinstance(value, dict):
        return {normalize_text(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return normalize_text(value)


def estimate_text_tokens(text: Any) -> int:
    """Approximate token count for prompt budgeting."""

    normalized = normalize_text(text)
    if not normalized:
        return 0
    return max(1, len(re.findall(r"\w+|[^\s\w]", normalized, flags=re.UNICODE)))


def trim_text_by_tokens(text: Any, max_tokens: int) -> str:
    """Trim text using the same lightweight token estimate."""

    normalized = normalize_text(text)
    if estimate_text_tokens(normalized) <= max_tokens:
        return normalized

    approx_chars = max(0, max_tokens * 4)
    if approx_chars <= 3:
        return ""
    return trim_text(normalized, approx_chars)


def trim_text(text: Any, max_length: int) -> str:
    """Trim long text for LLM payloads."""

    normalized = normalize_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    """Deduplicate while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
