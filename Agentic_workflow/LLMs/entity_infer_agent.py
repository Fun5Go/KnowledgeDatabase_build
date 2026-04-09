import json
import re
from typing import Any, Dict, List, Optional

from langsmith import traceable

from .failure_schema import Support_type
from .llm_init import get_llm_backend


class ChunkTextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in LLM response.")

    return json.loads(match.group(0))


class FMEAEntityInferenceAgent:
    """
    Infer FMEA entities from GraphRAG sentence evidence plus structure-analysis context.

    This agent is meant to consume outputs from:
    - FMEASentenceRetrieverV2.query_mode_support(...)
    - FMEASentenceRetrieverV2.query_effect_support(...)

    The prompt preserves linked graph evidence such as:
    - TSChunk <-IMPLEMENT-> FSChunk
    - TSChunk <-DETAILED_BY-> QDChunk
    - RationaleChunk -RATIONALE_FOR-> TSChunk/FSChunk/QDChunk
    """

    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        self.llm = get_llm_backend(
            backend=backend,
            model=model,
            temperature=0.0,
            json_mode=True,
        )

    def _format_graph_evidence_block(self, item: Dict[str, Any]) -> Dict[str, Any]:
        query_item = item.get("query_item", {})
        evidence_blocks = []

        for rank, evidence in enumerate(item.get("evidence", []), start=1):
            primary = evidence.get("primary_sentence", {})
            linked_context = []

            for support in evidence.get("supporting_context", []):
                linked_context.append({
                    "relation_path": (
                        f"{primary.get('label', '')}"
                        f" -{support.get('relationship', '')}-> "
                        f"{support.get('label', '')}"
                    ),
                    "linked_label": support.get("label", ""),
                    "linked_text": support.get("text", ""),
                    "linked_node_id": support.get("node_id", ""),
                })

            evidence_blocks.append({
                "rank": rank,
                "primary_label": primary.get("label", ""),
                "primary_text": primary.get("text", ""),
                "primary_node_id": primary.get("node_id", ""),
                "retrieval_score": primary.get("score", 0.0),
                "score_breakdown": evidence.get("score_breakdown", {}),
                "linked_context": linked_context,
            })

        return {
            "query_type": query_item.get("query_type", ""),
            "target_text": query_item.get("target_text", ""),
            "function_text": query_item.get("function_text", ""),
            "element_text": query_item.get("element_text", ""),
            "product_text": query_item.get("product_text", ""),
            "positive_target_variants": query_item.get("positive_target_variants", []),
            "evidence_blocks": evidence_blocks,
        }

    def _build_entity_inference_payload(
        self,
        structure_context: Dict[str, Any],
        support_result: Dict[str, Any],
        support_key: str,
    ) -> List[Dict[str, Any]]:
        payload = []

        for item in support_result.get(support_key, []):
            evidence_item = self._format_graph_evidence_block(item)
            target_text = evidence_item.get("target_text", "")
            query_type = evidence_item.get("query_type", "")

            payload.append({
                "query_type": query_type,
                "target_text": target_text,
                "structure_context": {
                    "product_text": structure_context.get("product_text", ""),
                    "element_text": structure_context.get("element_text", ""),
                    "function_text": structure_context.get("function_text", ""),
                    "candidate_modes": structure_context.get("candidate_modes", []),
                    "candidate_effects": structure_context.get("candidate_effects", []),
                    "candidate_causes": structure_context.get("candidate_causes", []),
                },
                "retrieved_evidence": evidence_item,
            })

        return payload

    def _build_mode_support_function_payload(
        self,
        structure_context: Dict[str, Any],
        mode_support_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence_pool: List[Dict[str, Any]] = []
        evidence_rank_by_node_id: Dict[str, int] = {}
        evidence_index_by_signature: Dict[tuple, int] = {}
        modes_payload: List[Dict[str, Any]] = []

        for item in mode_support_result.get("mode_support", []):
            query_item = item.get("query_item", {})
            mode_text = query_item.get("target_text", "")
            evidence_ranks: List[int] = []

            for evidence in item.get("evidence", []):
                primary = evidence.get("primary_sentence", {})
                node_id = primary.get("node_id", "")
                if not node_id:
                    continue

                if node_id not in evidence_rank_by_node_id:
                    evidence_entry = {}
                    seen_links = set()
                    relation_groups: Dict[str, List[str]] = {}

                    for support in evidence.get("supporting_context", []):
                        relation = support.get("relationship", "")
                        linked_text = support.get("text", "")
                        linked_label = support.get("label", "")
                        link_key = (relation, linked_label, linked_text)
                        if link_key in seen_links:
                            continue
                        seen_links.add(link_key)
                        relation_groups.setdefault(relation, [])
                        if linked_text:
                            relation_groups[relation].append(linked_text)

                    primary_label = primary.get("label", "")
                    if primary_label == "FSChunk":
                        evidence_entry["function_requirement"] = primary.get("text", "")
                    elif primary_label == "QDChunk":
                        evidence_entry["qd_text"] = primary.get("text", "")
                    elif primary_label == "TSChunk":
                        evidence_entry["ts_text"] = primary.get("text", "")
                    elif primary_label == "RationaleChunk":
                        evidence_entry["rationale_text"] = primary.get("text", "")

                    for relation, texts in relation_groups.items():
                        if relation == "RATIONALE_FOR":
                            if primary_label == "FSChunk":
                                evidence_entry["has_rationale"] = texts
                            elif primary_label == "RationaleChunk":
                                evidence_entry["rationale_for"] = texts
                        elif relation == "DETAILED_BY":
                            if primary_label == "TSChunk":
                                evidence_entry["detailed_by_qd"] = texts
                            elif primary_label == "QDChunk":
                                evidence_entry["verified_ts"] = texts
                        elif relation == "IMPLEMENT":
                            if primary_label == "FSChunk":
                                evidence_entry["implemented_by_ts"] = texts
                            elif primary_label == "TSChunk":
                                evidence_entry["implements_fs"] = texts

                    signature = self._build_evidence_signature(evidence_entry)
                    if signature in evidence_index_by_signature:
                        rank = evidence_index_by_signature[signature]
                        merged = self._merge_evidence_entries(
                            evidence_pool[rank - 1],
                            evidence_entry,
                        )
                        merged["evidence_rank"] = rank
                        evidence_pool[rank - 1] = merged
                    else:
                        rank = len(evidence_pool) + 1
                        evidence_entry["evidence_rank"] = rank
                        evidence_pool.append(evidence_entry)
                        evidence_index_by_signature[signature] = rank

                    evidence_rank_by_node_id[node_id] = rank

                evidence_ranks.append(evidence_rank_by_node_id[node_id])

            modes_payload.append({
                "mode_text": mode_text,
                "positive_target_variants": query_item.get("positive_target_variants", []),
                "evidence_ranks": sorted(set(evidence_ranks)),
            })

        return {
            "query_type": "mode",
            "structure_context": {
                "element_text": structure_context.get("element_text", ""),
                "function_text": structure_context.get("function_text", ""),
                "candidate_effects": structure_context.get("candidate_effects", []),
                "candidate_causes": structure_context.get("candidate_causes", []),
            },
            "modes": modes_payload,
            "shared_evidence_pool": evidence_pool,
        }

    def _build_entity_inference_prompt(self, grouped_payload: List[Dict[str, Any]]) -> str:
        schema = {
            "results": [
                {
                    "query_type": "mode or effect",
                    "target_text": "string",
                    "inferred_entities": [
                        {
                            "entity_type": "mode or effect or cause",
                            "entity_text": "string",
                            "selected": True,
                            "support_type": "complete_entity",
                            "confidence": "high",
                            "reason": "short explanation grounded in the retrieved sentences and relation paths",
                            "supporting_evidence_ranks": [1, 2],
                        }
                    ],
                    "global_reasoning": "short summary grounded in the graph-linked evidence"
                }
            ]
        }

        return f"""
You are an FMEA reasoning agent.

Task:
1. Infer which FMEA entities are supported by the retrieved graph-linked sentence evidence.
2. Use both the structure-analysis context and the retrieved sentence evidence.
3. Treat relation paths as meaningful evidence links. For example:
   - TSChunk -DETAILED_BY-> QDChunk
   - FSChunk -IMPLEMENT-> TSChunk
   - RationaleChunk -RATIONALE_FOR-> FSChunk
4. Prefer direct evidence in the primary sentence.
5. Use linked context to strengthen or weaken the interpretation, but do not over-trust generic rationale text.
6. Only select entities from the provided structure-analysis candidates.
7. If the evidence only partially supports an entity, use:
   - complete_entity
   - composed_from_multiple
   - partial_pattern
   - no_direct_gt
8. Keep reasoning concise and engineering-grounded.

Interpretation rule:
- If query_type is "mode", infer which mode/effect/cause candidates are best supported by the evidence for this function context.
- If query_type is "effect", infer which effect/mode/cause candidates are best supported by the evidence for this function context.

Input:
{json.dumps(grouped_payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    def _build_mode_support_function_prompt(self, payload: Dict[str, Any]) -> str:
        schema = {
            "function_text": "string",
            "results": [
                {
                    "mode_text": "string",
                    "selected_effects": [
                        {
                            "effect_text": "string",
                            "selected": True,
                            "support_type": "complete_entity",
                            "confidence": "high",
                            "reason": "short explanation grounded in the shared evidence pool",
                            "supporting_evidence_ranks": [1, 2],
                        }
                    ],
                    "selected_causes": [
                        {
                            "cause_text": "string",
                            "selected": True,
                            "support_type": "partial_pattern",
                            "confidence": "medium",
                            "reason": "short explanation grounded in the shared evidence pool",
                            "supporting_evidence_ranks": [2],
                        }
                    ],
                    "global_reasoning": "short summary for this mode under the function"
                }
            ]
        }

        return f"""
You are an FMEA reasoning agent.

Task:
1. Work at the function level with one shared evidence pool.
2. Each mode belongs to the same function context. Different modes may reference the same evidence ranks.
3. For EACH mode, infer which effect candidates and cause candidates are supported.
4. Use the shared evidence pool first, then use the linked text fields inside each evidence block.
5. Read the evidence blocks as simplified document structure, for example:
   - function_requirement: text
   - has_rationale: text list
   - ts_text: text
   - detailed_by_qd: text list
   - qd_text: text
   - verified_ts: text list
6. Prefer direct support from the main text field in each evidence block.
7. Use linked text to strengthen the reasoning, but do not over-trust generic rationale text.
8. Only select from the provided effect and cause candidates.
9. Keep reasoning concise and engineering-grounded.

Support type choices:
- complete_entity
- composed_from_multiple
- partial_pattern
- no_direct_gt

Input:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    @staticmethod
    def _build_evidence_signature(evidence_entry: Dict[str, Any]) -> tuple:
        texts: List[str] = []
        for key, value in evidence_entry.items():
            if key == "evidence_rank":
                continue
            if isinstance(value, str) and value.strip():
                texts.append(ChunkTextNormalizer.normalize(value))
            elif isinstance(value, list):
                texts.extend(
                    ChunkTextNormalizer.normalize(item)
                    for item in value
                    if isinstance(item, str) and item.strip()
                )
        return tuple(sorted(set(texts)))

    @staticmethod
    def _merge_evidence_entries(
        left: Dict[str, Any],
        right: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if key == "evidence_rank":
                continue
            if key not in merged:
                merged[key] = value
                continue
            if isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = sorted(set(merged[key] + value))
        return merged

    def _fallback_entity_inference(self, grouped_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in grouped_payload:
            query_type = item.get("query_type", "")
            target_text = item.get("target_text", "")
            ctx = item.get("structure_context", {})

            if query_type == "mode":
                candidate_pool = ctx.get("candidate_modes", [])
                entity_type = "mode"
            else:
                candidate_pool = ctx.get("candidate_effects", [])
                entity_type = "effect"

            selected = []
            if target_text in candidate_pool:
                selected.append({
                    "entity_type": entity_type,
                    "entity_text": target_text,
                    "selected": True,
                    "support_type": "partial_pattern",
                    "confidence": "low",
                    "reason": "Fallback matched the query target to the structure-analysis candidate list.",
                    "supporting_evidence_ranks": [1],
                })

            results.append({
                "query_type": query_type,
                "target_text": target_text,
                "inferred_entities": selected,
                "global_reasoning": "Fallback used query target identity only.",
            })

        return results

    def _clean_entity_inference_result(
        self,
        parsed: Dict[str, Any],
        grouped_payload: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_support_types = {
            "complete_entity",
            "composed_from_multiple",
            "partial_pattern",
            "no_direct_gt",
        }
        valid_confidences = {"high", "medium", "low"}

        payload_lookup = {
            (item.get("query_type", ""), item.get("target_text", "")): item
            for item in grouped_payload
        }

        cleaned = []
        for item in parsed.get("results", []):
            key = (item.get("query_type", ""), item.get("target_text", ""))
            payload_item = payload_lookup.get(key)
            if not payload_item:
                continue

            ctx = payload_item.get("structure_context", {})
            valid_entities = set(
                ctx.get("candidate_modes", [])
                + ctx.get("candidate_effects", [])
                + ctx.get("candidate_causes", [])
            )

            cleaned_entities = []
            for entity in item.get("inferred_entities", []):
                entity_text = entity.get("entity_text", "")
                support_type = entity.get("support_type", "no_direct_gt")
                confidence = entity.get("confidence", "low")

                if entity_text not in valid_entities:
                    continue
                if support_type not in valid_support_types:
                    support_type = "no_direct_gt"
                if confidence not in valid_confidences:
                    confidence = "low"

                cleaned_entities.append({
                    "entity_type": entity.get("entity_type", ""),
                    "entity_text": entity_text,
                    "selected": bool(entity.get("selected", True)),
                    "support_type": support_type,
                    "confidence": confidence,
                    "reason": entity.get("reason", ""),
                    "supporting_evidence_ranks": entity.get("supporting_evidence_ranks", []),
                })

            cleaned.append({
                "query_type": key[0],
                "target_text": key[1],
                "inferred_entities": cleaned_entities,
                "global_reasoning": item.get("global_reasoning", ""),
            })

        returned_keys = {(item["query_type"], item["target_text"]) for item in cleaned}
        for payload_item in grouped_payload:
            key = (payload_item.get("query_type", ""), payload_item.get("target_text", ""))
            if key not in returned_keys:
                cleaned.append({
                    "query_type": key[0],
                    "target_text": key[1],
                    "inferred_entities": [],
                    "global_reasoning": "Model did not return this item.",
                })

        return cleaned

    def _fallback_mode_support_function_inference(
        self,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results = []
        valid_effects = set(payload.get("structure_context", {}).get("candidate_effects", []))

        for mode_item in payload.get("modes", []):
            mode_text = mode_item.get("mode_text", "")
            selected_effects = []
            if mode_text in valid_effects:
                selected_effects.append({
                    "effect_text": mode_text,
                    "selected": True,
                    "support_type": "partial_pattern",
                    "confidence": "low",
                    "reason": "Fallback matched the mode text to an effect candidate exactly.",
                    "supporting_evidence_ranks": mode_item.get("evidence_ranks", [])[:1],
                })

            results.append({
                "mode_text": mode_text,
                "selected_effects": selected_effects,
                "selected_causes": [],
                "global_reasoning": "Fallback used exact text identity only.",
            })

        return results

    def _clean_mode_support_function_result(
        self,
        parsed: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        valid_support_types = {
            "complete_entity",
            "composed_from_multiple",
            "partial_pattern",
            "no_direct_gt",
        }
        valid_confidences = {"high", "medium", "low"}
        valid_effects = set(payload.get("structure_context", {}).get("candidate_effects", []))
        valid_causes = set(payload.get("structure_context", {}).get("candidate_causes", []))
        valid_modes = {item.get("mode_text", "") for item in payload.get("modes", [])}
        valid_ranks = {item.get("evidence_rank") for item in payload.get("shared_evidence_pool", [])}

        cleaned = []
        for item in parsed.get("results", []):
            mode_text = item.get("mode_text", "")
            if mode_text not in valid_modes:
                continue

            cleaned_effects = []
            for effect in item.get("selected_effects", []):
                effect_text = effect.get("effect_text", "")
                if effect_text not in valid_effects:
                    continue
                support_type = effect.get("support_type", "no_direct_gt")
                confidence = effect.get("confidence", "low")
                if support_type not in valid_support_types:
                    support_type = "no_direct_gt"
                if confidence not in valid_confidences:
                    confidence = "low"
                cleaned_effects.append({
                    "effect_text": effect_text,
                    "selected": bool(effect.get("selected", True)),
                    "support_type": support_type,
                    "confidence": confidence,
                    "reason": effect.get("reason", ""),
                    "supporting_evidence_ranks": [
                        rank for rank in effect.get("supporting_evidence_ranks", [])
                        if rank in valid_ranks
                    ],
                })

            cleaned_causes = []
            for cause in item.get("selected_causes", []):
                cause_text = cause.get("cause_text", "")
                if cause_text not in valid_causes:
                    continue
                support_type = cause.get("support_type", "no_direct_gt")
                confidence = cause.get("confidence", "low")
                if support_type not in valid_support_types:
                    support_type = "no_direct_gt"
                if confidence not in valid_confidences:
                    confidence = "low"
                cleaned_causes.append({
                    "cause_text": cause_text,
                    "selected": bool(cause.get("selected", True)),
                    "support_type": support_type,
                    "confidence": confidence,
                    "reason": cause.get("reason", ""),
                    "supporting_evidence_ranks": [
                        rank for rank in cause.get("supporting_evidence_ranks", [])
                        if rank in valid_ranks
                    ],
                })

            cleaned.append({
                "mode_text": mode_text,
                "selected_effects": cleaned_effects,
                "selected_causes": cleaned_causes,
                "global_reasoning": item.get("global_reasoning", ""),
            })

        returned_modes = {item.get("mode_text", "") for item in cleaned}
        for mode_item in payload.get("modes", []):
            mode_text = mode_item.get("mode_text", "")
            if mode_text not in returned_modes:
                cleaned.append({
                    "mode_text": mode_text,
                    "selected_effects": [],
                    "selected_causes": [],
                    "global_reasoning": "Model did not return this mode.",
                })

        return cleaned

    @traceable(
        run_type="chain",
        name="fmea_infer_entities_from_mode_support",
        tags=["fmea", "entity_infer", "mode_support"],
    )
    def infer_entities_from_mode_support(
        self,
        mode_support_result: Dict[str, Any],
        structure_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        grouped_payload = self._build_mode_support_function_payload(
            structure_context=structure_context,
            mode_support_result=mode_support_result,
        )

        if not grouped_payload.get("modes"):
            return []

        prompt = self._build_mode_support_function_prompt(grouped_payload)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)
            return self._clean_mode_support_function_result(parsed, grouped_payload)
        except Exception as exc:
            print(f"[WARN] entity inference from mode support failed, fallback used: {exc}")
            return self._fallback_mode_support_function_inference(grouped_payload)

    @traceable(
        run_type="chain",
        name="fmea_infer_entities_from_effect_support",
        tags=["fmea", "entity_infer", "effect_support"],
    )
    def infer_entities_from_effect_support(
        self,
        effect_support_result: Dict[str, Any],
        structure_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        grouped_payload = self._build_entity_inference_payload(
            structure_context=structure_context,
            support_result=effect_support_result,
            support_key="effect_support",
        )

        if not grouped_payload:
            return []

        prompt = self._build_entity_inference_prompt(grouped_payload)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)
            return self._clean_entity_inference_result(parsed, grouped_payload)
        except Exception as exc:
            print(f"[WARN] entity inference from effect support failed, fallback used: {exc}")
            return self._fallback_entity_inference(grouped_payload)


def pretty_print_entity_inference(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("FMEA ENTITY INFERENCE")
    print("=" * 80)

    for item in results:
        if "mode_text" in item:
            print(f"\nMode: {item.get('mode_text', '')}")
            for effect in item.get("selected_effects", []):
                print(f"  - Effect     : {effect.get('effect_text', '')}")
                print(f"    Selected   : {effect.get('selected', True)}")
                print(f"    SupportType: {effect.get('support_type', '')}")
                print(f"    Confidence : {effect.get('confidence', '')}")
                print(f"    Reason     : {effect.get('reason', '')}")
                print(f"    Evidence   : {effect.get('supporting_evidence_ranks', [])}")
            for cause in item.get("selected_causes", []):
                print(f"  - Cause      : {cause.get('cause_text', '')}")
                print(f"    Selected   : {cause.get('selected', True)}")
                print(f"    SupportType: {cause.get('support_type', '')}")
                print(f"    Confidence : {cause.get('confidence', '')}")
                print(f"    Reason     : {cause.get('reason', '')}")
                print(f"    Evidence   : {cause.get('supporting_evidence_ranks', [])}")
            print(f"  Summary: {item.get('global_reasoning', '')}")
            continue

        print(f"\nQuery Type: {item.get('query_type', '')}")
        print(f"Target    : {item.get('target_text', '')}")
        for entity in item.get("inferred_entities", []):
            print(f"  - Entity Type : {entity.get('entity_type', '')}")
            print(f"    Entity Text : {entity.get('entity_text', '')}")
            print(f"    Selected    : {entity.get('selected', True)}")
            print(f"    SupportType : {entity.get('support_type', '')}")
            print(f"    Confidence  : {entity.get('confidence', '')}")
            print(f"    Reason      : {entity.get('reason', '')}")
            print(f"    Evidence    : {entity.get('supporting_evidence_ranks', [])}")
        print(f"  Summary: {item.get('global_reasoning', '')}")

    print("=" * 80)
