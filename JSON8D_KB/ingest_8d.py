import json
from pathlib import Path
from typing import List,Union

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
    failure_id: Union[str, List[str]],
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

    # ---- normalize failure_id ----
    if isinstance(failure_id, list):
        new_ids = failure_id
    else:
        new_ids = [failure_id]

    for fid in new_ids:
        if fid not in node["failure_ids"]:
            node["failure_ids"].append(fid)

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

    def make_failure_uid(fid: str, cid: str | None) -> str:
        return f"{fid}__{cid}" if cid else fid

    used_sentence_ids: set[str] = set()
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
    # Root causes FIRST → build UID list
    # =====================================================

    cause_entries = []
    all_failure_uids: List[str] = []

    mode_text = failure.get("failure_mode")
    element_text = failure.get("failure_element")
    effect_text = failure.get("failure_effect")

    mode_id = make_semantic_id("mode", mode_text) if mode_text else None
    element_id = make_semantic_id("element", element_text) if element_text else None
    effect_id = make_semantic_id("effect", effect_text) if effect_text else None

    for cause in failure.get("root_causes", []) or []:

        cause_text = (cause.get("failure_cause") or "").strip()
        confidence = cause.get("confidence")

        if not cause_text:
            continue

        cause_semantic_id = make_semantic_id("cause", cause_text)
        failure_uid = make_failure_uid(failure_id, cause_semantic_id)
        all_failure_uids.append(failure_uid)

        # ------------------------------
        # Collect cause semantic (bind only to this UID)
        # ------------------------------
        collect_semantic(
            semantic_nodes,
            semantic_id=cause_semantic_id,
            field_type="cause",
            text=cause_text,
            failure_id=failure_uid,
            source_type="8D",
        )

        # ------------------------------
        # Cause supporting sentences
        # ------------------------------
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

        # ------------------------------
        # Save cause-level entity (UID key)
        # ------------------------------
        entity = EightDFailureEntity(
            failure_id=failure_id,
            file_name=case_id,
            mode_id=mode_id,
            element_id=element_id,
            effect_id=effect_id,
            failure_mode_text=failure.get("failure_mode"),
            failure_element_text=failure.get("failure_element"),
            failure_effect_text=failure.get("failure_effect"),
            cause_id=cause_semantic_id,
            failure_cause_text=cause_text,
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
    # Bind shared semantic fields to ALL cause-level failures
    # =====================================================

    if mode_id:
        collect_semantic(
            semantic_nodes,
            semantic_id=mode_id,
            field_type="mode",
            text=mode_text,
            failure_id=all_failure_uids,
            source_type="8D",
        )

    if element_id:
        collect_semantic(
            semantic_nodes,
            semantic_id=element_id,
            field_type="element",
            text=element_text,
            failure_id=all_failure_uids,
            source_type="8D",
        )

    if effect_id:
        collect_semantic(
            semantic_nodes,
            semantic_id=effect_id,
            field_type="effect",
            text=effect_text,
            failure_id=all_failure_uids,
            source_type="8D",
        )

    # =====================================================
    # Other selected sentences
    # =====================================================
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
    # FLUSH semantic nodes
    # =====================================================

    for node in semantic_nodes.values():
        failure_kb.upsert_semantic_node(
            semantic_id=node["semantic_id"],
            field_type=node["field_type"],
            text=node["text"],
            failure_ids=node["failure_ids"],
            source_type="8D",
        )

    print(f"[OK] {json_path.name} ingested (8D, cause-level UID mode)")
