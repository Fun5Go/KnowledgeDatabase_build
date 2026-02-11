import json
from pathlib import Path
from typing import List

from kb_structure import (
    EightDFailureKB,
    Sentence, SentenceKB,
    MaintenanceTag,
    FileMeta, FileMetaStore,
    evaluate_failure,
    EightDFailureEntity
)
from datetime import datetime
from collections import defaultdict
import hashlib
from dataclasses import dataclass, field, asdict
import re


def parse_maintenance_tag(raw: dict | None) -> MaintenanceTag:
    raw = raw or {}
    return MaintenanceTag(
        review_status=raw.get("review_status", "pending"),
        version=raw.get("Version", "V0"),
        last_updated=raw.get("last_updated", ""),
        supersedes=raw.get("supersedes"),
    )

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def to_year(released) -> int | None:
    if not released:
        return None

    s = str(released).strip()
    if not s or s.lower() in {"unknown", "n/a", "na", "none", "null", "-"}:
        return None

    # normalize common ISO variants
    s = s.replace("Z", "+00:00")

    # if it's just a year like "2019"
    if len(s) == 4 and s.isdigit():
        return int(s)

    try:
        return datetime.fromisoformat(s).year
    except ValueError:
        # fallback: extract leading year if present, e.g. "2019-01-07 ..."
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
        return None
    
def make_semantic_id(field_type: str, text: str) -> str:
    norm = normalize_text(text)
    h = hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]
    return f"{field_type}:{h}"

def collect_semantic(
    semantic_map: dict,
    *,
    semantic_id: str,
    field_type: str,
    text: str,
    failure_id: str,
    source_type: str,
):
    node = semantic_map.setdefault(
        semantic_id,
        {
            "semantic_id": semantic_id,
            "field_type": field_type,
            "text": text,
            "failure_ids": [],
            "source_type": source_type,
        },
    )
    if failure_id not in node["failure_ids"]:
        node["failure_ids"].append(failure_id)

# =========================================================
# Ingest 8D JSON (already JSON format)
# =========================================================
def ingest_8d_json(
    json_path: Path,
    failure_kb: EightDFailureKB,
    sentence_kb: SentenceKB,
    meta_kb: FileMetaStore,
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents", [])
    if not docs:
        raise ValueError("Missing documents")

    doc0 = docs[0]
    case_id = doc0.get("file_name")
    product_name = doc0.get("product_name")
    product_pn_id = doc0.get("productPnId")
    product_domain = doc0.get("product_domain")

    # =====================================================
    # META
    # =====================================================
    file_meta = FileMeta(
        file_name=case_id,
        source_type="8D",
        released=doc0.get("released_date"),
        productPnID=product_pn_id,
        productName=product_name,
        product_domain=product_domain,
        version=doc0.get("version"),
    )
    meta_kb.add(file_meta)

    released_year = to_year(file_meta.released)

    failure = data["failure"]
    failure_id = failure["failure_ID"]
    used_sentence_ids: set[str] = set()
    # =====================================================
    # Semantic accumulator (merge later)
    # =====================================================
    semantic_nodes: dict[str, dict] = {}

    # =====================================================
    # Sentence KB (failure sentences)
    # =====================================================
    supporting_sentence_ids: List[str] = []

    for ent in failure.get("supporting_entities", []):
        sid = ent["sentence_id"]
        used_sentence_ids.add(sid)

        s = Sentence(
            id=sid,
            text=ent["text"],
            source_section=ent.get("source_section", ""),
            case_id=case_id,
            annotations=ent.get("annotations", {}),
            failure_id=failure_id,
            cause_id=None,
            sentence_role="failure_sentence",
            product_domain=product_domain,
            productPnID=product_pn_id,
            released_year=released_year,
        )
        sentence_kb.add(
            sentence=s,
            failure_id=failure_id,
            sentence_role="failure_sentence",
        )
        supporting_sentence_ids.append(s.id)

    # =====================================================
    # Collect failure semantic
    # =====================================================
    mode_text = failure.get("failure_mode")
    element_text = failure.get("failure_element")
    effect_text = failure.get("failure_effect")

    mode_id = make_semantic_id("mode", mode_text) if mode_text else None
    element_id = make_semantic_id("element", element_text) if element_text else None
    effect_id = make_semantic_id("effect", effect_text) if effect_text else None

    if mode_id:
        collect_semantic(
            semantic_nodes,
            semantic_id=mode_id,
            field_type="mode",
            text=mode_text,
            failure_id=failure_id,
            source_type="8D",
        )

    if element_id:
        collect_semantic(
            semantic_nodes,
            semantic_id=element_id,
            field_type="element",
            text=element_text,
            failure_id=failure_id,
            source_type="8D",
        )

    if effect_id:
        collect_semantic(
            semantic_nodes,
            semantic_id=effect_id,
            field_type="effect",
            text=effect_text,
            failure_id=failure_id,
            source_type="8D",
        )

    # =====================================================
    # Root causes (semantic + sentence)
    # =====================================================

    cause_entries = []

    for cause in failure.get("root_causes", []) or []:

        cause_text = (cause.get("failure_cause") or "").strip()
        confidence = cause.get("confidence")

        if not cause_text:
            continue

        cause_semantic_id = make_semantic_id("cause", cause_text)

        collect_semantic(
            semantic_nodes,
            semantic_id=cause_semantic_id,
            field_type="cause",
            text=cause_text,
            failure_id=failure_id,
            source_type="8D",
        )

        cause_sentence_ids: List[str] = []

        for ent in cause.get("supporting_entities", []):

            sid = ent["sentence_id"]
            used_sentence_ids.add(sid)
            s = Sentence(
                id=sid,
                text=ent["text"],
                source_section=ent.get("source_section", ""),
                case_id=case_id,
                annotations=ent.get("annotations", {}),
                failure_id=failure_id,
                cause_id=cause_semantic_id,
                sentence_role="cause_sentence",
                product_domain=product_domain,
                productPnID=product_pn_id,
                released_year=released_year,
            )

            sentence_kb.add(
                sentence=s,
                failure_id=failure_id,
                sentence_role="cause_sentence",
                cause_id=cause_semantic_id,
            )

            cause_sentence_ids.append(s.id)

        cause_entries.append({
            "cause_semantic_id": cause_semantic_id,
            "cause_text": cause_text,
            "confidence": confidence,
            "supporting_sentence_ids": cause_sentence_ids,
        })
        for ent in data.get("selected_sentences", []):

            sid = ent["sentence_id"]

            if sid in used_sentence_ids:
                continue  

            s = Sentence(
                id=sid,
                text=ent["text"],
                source_section=ent.get("source_section", ""),
                case_id=case_id,
                annotations=ent.get("annotations", {}),
                failure_id=failure_id,
                cause_id=None,
                sentence_role="other",
                product_domain=product_domain,
                productPnID=product_pn_id,
                released_year=released_year,
            )

            sentence_kb.add(
                sentence=s,
                failure_id=failure_id,
                sentence_role="other",
            )


    # =====================================================
    # Save 8D Failure Entity
    # =====================================================

    entity = EightDFailureEntity(
        failure_id=failure_id,
        mode_id=mode_id,
        element_id=element_id,
        effect_id=effect_id,
        failure_mode_text=mode_text,
        failure_element_text=element_text,
        failure_effect_text=effect_text,
        cause_ids=cause_entries,
        discipline=failure.get("discipline"),
        supporting_sentence_ids=supporting_sentence_ids,
        fmea_type=failure.get("failure_level"),
        source_type="8D",
        productPnID=product_pn_id,
        product_domain=product_domain,
        released_year=released_year,
    )

    failure_kb.upsert_failure_entity(entity)

    # =====================================================
    # 🔥 FLUSH semantic nodes (MERGE WITH EXISTING KB)
    # =====================================================

    for node in semantic_nodes.values():

        semantic_id = node["semantic_id"]
        new_failure_ids = set(node["failure_ids"])

        existing_node = failure_kb.field_store.get(semantic_id)

        if existing_node:
            # ----- MERGE -----
            merged_ids = set(existing_node.get("failure_ids", []))
            merged_ids.update(new_failure_ids)

            failure_kb.upsert_semantic_node(
                semantic_id=semantic_id,
                field_type=existing_node["field_type"],  # keep old
                text=existing_node["text"],              # keep old
                failure_ids=list(merged_ids),
                source_type=existing_node.get("source_type"),
            )

        else:
            # ----- NEW NODE -----
            failure_kb.upsert_semantic_node(
                semantic_id=node["semantic_id"],
                field_type=node["field_type"],
                text=node["text"],
                failure_ids=list(new_failure_ids),
                source_type=node["source_type"],
            )

    print(f"[OK] {json_path.name} ingested (8D)")