from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
from .PPL_score import ChainPPLEvaluator
import json
from .entity import structure_input_motorcontrol, structure_input_powertrain
BASE_DIR = Path(__file__).resolve().parent


# -------------------------
# Scoring config
# -------------------------
FIELD_WEIGHTS = {
    "mode": 1.2,
    "cause": 1.0,
    "effect": 1.0,
}

COMPLETE_CHAIN_BONUS = 1.4
GRAPH_CONNECTION_WEIGHT = 0.3
GRAPH_EXPAND_DEFAULT_SIM = 0.6


def generate_graph_inferred_chains_from_structure(
    persist_dir: Union[str, Path],
    structure_input: Dict[str, Any],
    min_count: Optional[int] = None,
    min_similarity: float = 0.75,
    source_type: Optional[str] = None,
    top_n: int = 50,
    top_k_per_field: int = 5,
    save_query_json: bool = True,
) -> List[Dict[str, Any]]:

    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    field_store = getattr(kb, "field_store", {}) or {}
    edge_store = getattr(kb, "edge_store", {}) or {}

    mode_to_cause = edge_store.get("mode_to_cause", {}) or {}
    mode_to_effect = edge_store.get("mode_to_effect", {}) or {}

    def _safe(x):
        return "" if x is None else str(x).strip()

    def _id_to_text(sid):
        if not sid:
            return ""
        return _safe((field_store.get(sid) or {}).get("text"))

    query_match_map = {}

    # -------------------------
    # semantic retrieval
    # -------------------------
    def semantic_nodes(text, field):
        text = _safe(text)
        if not text:
            return []

        res = query_semantic_kb(
            persist_dir,
            text,
            field_type=field,
            n_results=top_k_per_field,
            min_count=min_count,
            source_type=source_type,
        ) or {}

        docs = (res.get("documents") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []
        ids = (res.get("ids") or [[]])[0] or []

        out = []

        for doc, meta, dist, sid in zip(docs, metas, dists, ids):
            try:
                sim = 1.0 - float(dist)
            except Exception:
                continue

            if sim < min_similarity:
                continue

            semantic_id = _safe(sid) or _safe((meta or {}).get("semantic_id"))

            node = {
                "semantic_id": semantic_id,
                "text": _id_to_text(semantic_id) or _safe(doc),
                "similarity": float(sim),
                "query_text": text,
            }

            out.append(node)

        query_match_map[text] = {
            "matched": [
                {
                    "matched_text": n["text"],
                    "semantic_id": n["semantic_id"],
                    "similarity": round(n["similarity"], 4),
                }
                for n in out
            ]
        }

        return out

    def dedup_keep_best(nodes):
        best = {}
        for n in nodes:
            sid = n["semantic_id"]
            if sid not in best or n["similarity"] > best[sid]["similarity"]:
                best[sid] = n
        return list(best.values())

    # =========================================================
    # MAIN
    # =========================================================
    all_results = []

    for node in structure_input.get("nodes", []):

        node_id = _safe(node.get("element_id"))
        failure_element_text = _safe(node.get("failure_element"))

        modes_txt = node.get("modes") or []
        causes_txt = node.get("causes") or []
        effects_txt = node.get("effects") or []

        candidate_modes = dedup_keep_best(
            [n for t in modes_txt for n in semantic_nodes(t, "mode")]
        )
        candidate_causes = dedup_keep_best(
            [n for t in causes_txt for n in semantic_nodes(t, "cause")]
        )
        candidate_effects = dedup_keep_best(
            [n for t in effects_txt for n in semantic_nodes(t, "effect")]
        )

        mode_map = {n["semantic_id"]: n for n in candidate_modes}
        cause_map = {n["semantic_id"]: n for n in candidate_causes}
        effect_map = {n["semantic_id"]: n for n in candidate_effects}

        inferred_chains = []
        seen_keys = set()

        # ⭐ 父链记录
        full_chain_mc = set()
        full_chain_me = set()

        # ======================================================
        # 1️⃣ 完整三字段链（父链）
        # ======================================================
        for mode_id, mode_node in mode_map.items():

            sim_mode = mode_node["similarity"]
            q_mode = mode_node["query_text"]

            connected_causes = list((mode_to_cause.get(mode_id) or {}).keys())
            connected_effects = list((mode_to_effect.get(mode_id) or {}).keys())

            for cause_id in connected_causes:
                for effect_id in connected_effects:

                    key = ("MCE", mode_id, cause_id, effect_id)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    cause_node = cause_map.get(cause_id)
                    effect_node = effect_map.get(effect_id)

                    hit_count = 1

                    if cause_node:
                        sim_cause = cause_node["similarity"]
                        q_cause = cause_node["query_text"]
                        hit_count += 1
                    else:
                        sim_cause = GRAPH_EXPAND_DEFAULT_SIM
                        q_cause = "[Graph expanded]"

                    if effect_node:
                        sim_effect = effect_node["similarity"]
                        q_effect = effect_node["query_text"]
                        hit_count += 1
                    else:
                        sim_effect = GRAPH_EXPAND_DEFAULT_SIM
                        q_effect = "[Graph expanded]"

                    if hit_count < 2:
                        continue

                    score = (
                        sim_mode * FIELD_WEIGHTS["mode"]
                        + sim_cause * FIELD_WEIGHTS["cause"]
                        + sim_effect * FIELD_WEIGHTS["effect"]
                    )

                    score += 2 * GRAPH_CONNECTION_WEIGHT

                    if hit_count == 3:
                        score *= COMPLETE_CHAIN_BONUS

                    inferred_chains.append({
                        "node_id": node_id,
                        "failure_element": failure_element_text,
                        "mode": _id_to_text(mode_id),
                        "cause": _id_to_text(cause_id),
                        "effect": _id_to_text(effect_id),
                        "mode_id": mode_id,
                        "cause_id": cause_id,
                        "effect_id": effect_id,
                        "query_mode": q_mode,
                        "query_cause": q_cause,
                        "query_effect": q_effect,
                        "graph_connections": 2,
                        "score": round(score, 6),
                    })

                    full_chain_mc.add((mode_id, cause_id))
                    full_chain_me.add((mode_id, effect_id))

        # ======================================================
        # 2️⃣ mode → effect（二字段）
        # ======================================================
        for mode_id, mode_node in mode_map.items():

            sim_mode = mode_node["similarity"]
            q_mode = mode_node["query_text"]

            connected_effects = list((mode_to_effect.get(mode_id) or {}).keys())

            for effect_id in connected_effects:

                if (mode_id, effect_id) in full_chain_me:
                    continue  # ⭐ 父链已存在，过滤子链

                effect_node = effect_map.get(effect_id)
                if not effect_node:
                    continue  # 必须双命中

                key = ("ME", mode_id, effect_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                sim_effect = effect_node["similarity"]
                q_effect = effect_node["query_text"]

                score = (
                    sim_mode * FIELD_WEIGHTS["mode"]
                    + sim_effect * FIELD_WEIGHTS["effect"]
                )

                score += GRAPH_CONNECTION_WEIGHT

                inferred_chains.append({
                    "node_id": node_id,
                    "failure_element": failure_element_text,
                    "mode": _id_to_text(mode_id),
                    "cause": "",
                    "effect": _id_to_text(effect_id),
                    "mode_id": mode_id,
                    "cause_id": None,
                    "effect_id": effect_id,
                    "query_mode": q_mode,
                    "query_cause": "",
                    "query_effect": q_effect,
                    "graph_connections": 1,
                    "score": round(score, 6),
                })

        # ======================================================
        # 3️⃣ mode → cause（二字段）
        # ======================================================
        for mode_id, mode_node in mode_map.items():

            sim_mode = mode_node["similarity"]
            q_mode = mode_node["query_text"]

            connected_causes = list((mode_to_cause.get(mode_id) or {}).keys())

            for cause_id in connected_causes:

                if (mode_id, cause_id) in full_chain_mc:
                    continue  # ⭐ 父链已存在

                cause_node = cause_map.get(cause_id)
                if not cause_node:
                    continue

                key = ("MC", mode_id, cause_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                sim_cause = cause_node["similarity"]
                q_cause = cause_node["query_text"]

                score = (
                    sim_mode * FIELD_WEIGHTS["mode"]
                    + sim_cause * FIELD_WEIGHTS["cause"]
                )

                score += GRAPH_CONNECTION_WEIGHT

                inferred_chains.append({
                    "node_id": node_id,
                    "failure_element": failure_element_text,
                    "mode": _id_to_text(mode_id),
                    "cause": _id_to_text(cause_id),
                    "effect": "",
                    "mode_id": mode_id,
                    "cause_id": cause_id,
                    "effect_id": None,
                    "query_mode": q_mode,
                    "query_cause": q_cause,
                    "query_effect": "",
                    "graph_connections": 1,
                    "score": round(score, 6),
                })

        inferred_chains.sort(key=lambda x: x["score"], reverse=True)
        all_results.extend(inferred_chains[:top_n])

    if save_query_json:
        save_path = persist_dir / "query_match_log.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(query_match_map, f, indent=2, ensure_ascii=False)

    return all_results

def print_inferred_structure(results, top_n=None):

    if not results:
        print("No inferred chains.")
        return

    from collections import defaultdict

    def fmt_src(src: str) -> str:
        if not src:
            return ""
        return "✓" if src == "semantic" else "⤴"  # semantic=命中, graph_expand=补齐

    grouped = defaultdict(list)
    for r in results:
        grouped[r.get("node_id", "UNKNOWN")].append(r)

    for node_id, chains in grouped.items():

        chains = sorted(chains, key=lambda x: x.get("score", 0), reverse=True)
        if top_n:
            chains = chains[:top_n]

        print("=" * 90)
        print(f"NODE: {node_id}")
        print(f"Total Chains: {len(chains)}")
        print("=" * 90)

        for idx, c in enumerate(chains, 1):

            score = float(c.get("score", 0.0))
            print(f"\n[{idx}] Score: {score:.6f}   (edges: {c.get('graph_connections', 0)})")

            mode_src = c.get("mode_source", "semantic")  # 兼容旧结果：默认 semantic
            cause_src = c.get("cause_source", "")
            effect_src = c.get("effect_source", "")

            # ------------------ Mode ------------------
            print(f"  Mode   {fmt_src(mode_src)}: {c.get('mode','')}")
            print(f"    ↳ ID    : {c.get('mode_id')}")
            print(f"    ↳ Query : {c.get('query_mode','')}")

            # ------------------ Cause ------------------
            if c.get("cause_id") or c.get("cause"):
                print(f"  Cause  {fmt_src(cause_src)}: {c.get('cause','')}")
                print(f"    ↳ ID    : {c.get('cause_id')}")
                print(f"    ↳ Query : {c.get('query_cause','')}")
            else:
                print(f"  Cause     : (none)")

            # ------------------ Effect ------------------
            if c.get("effect_id") or c.get("effect"):
                print(f"  Effect {fmt_src(effect_src)}: {c.get('effect','')}")
                print(f"    ↳ ID    : {c.get('effect_id')}")
                print(f"    ↳ Query : {c.get('query_effect','')}")
            else:
                print(f"  Effect    : (none)")

if  __name__ == "__main__":


    from pprint import pprint

    # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\failure_kb"
    )

    # -----------------------------------------------------
    # 2) Structure Input
    # -----------------------------------------------------
    structure_input = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "Power train",
                "modes": [
                    "Incorrect",
                    "No pulses seen",
                    "No voltage applied",
                    "Incorrect torque applied",
                    "Not enough torque",
                    "Motor breaks/overheats (e.g. resulting in demagnetisation)",
                    "Unstable regulation",
                    "High loss in torque transfer",
                    "Gear train breaks/wears out",
                    "Transmission ratio drifts",
                    "creates too much noise"
                ],
                "causes": [
                    "Gears loose on motor shaft (slips)",
                    "External force on spline",
                    "Motor can not provide enough torque",
                    "Too much friction in gear train",
                    "Gears material/design choice",
                    "Manufacturing tolerances of gears",
                    "Lubrication choice (e.g. degradation)",
                    "Motor design (temperature spec, actuation length/duty cycle)",
                    "Encoder circuit crosstalk",
                    "HW cannot supply enough power",
                    "ADC measurements incorrect (incl. bandwidth)",
                    "Wrong motor driver dimension (current rating etc.)",
                    "Overcurrent detection incorrect (threshold etc.)",
                    "Incorrect control loop (bandwidth)",
                    "Motor not shorted while device is not powered",
                    "Control parameters incorrect",
                    "Thermal protection fails (e.g. I2T)"
                ],
                "effects": [
                    "Does not shift gear",
                    "Incorrect gear shift",
                    "Incorrect cadence (offset)",
                    "Unstable cadence setting",
                    "Incorrect cadence (fixed gear ratio)",
                    "Incorrect ratio (offset)",
                    "Unstable ratio setting",
                    "Does not enter limp home mode",
                    "Sets wrong gear ratio",
                    "Gear ratio drifts when battery is empty",
                    "Firmware update not possible/fails",
                    "Device bricked",
                    "Update takes too much time (>5 minutes)",
                    "Too much noise",
                ]
            }
        ]
    }

    # structure_input = {
    #     "product_domain": "motor_drives",
    #     "nodes": [
    #         {
    #             "element_id": "E1",
    #             "failure_element": "Motor control",
    #             "modes": [
    #                 "Component break-down",
    #                 "Unbalanced motor currents",
    #                 "Incorrect interpretation zero-crossing",
    #                 "Soft start too long",
    #                 "No detection",
    #                 "Welded relay",
    #                 "Relay cannot close",
    #                 "False turn-on / turn-off"
    #             ],
    #             "causes": [
    #                 "Cooling insufficient",
    #                 "Compressor vibrations",
    #                 "(Starting) Motor current too high for chosen components",
    #                 "Overvoltage due to motor disconnect",
    #                 "Under Voltage due to incorrect triggering",
    #                 "Live switching of relays",
    #                 "Priority zero-crossing interrupt too low",
    #                 "Open loop control",
    #                 "No (correctly designed) snubber design",
    #                 "Too high dT junction as a result of power cycling of component"
    #             ],
    #             "effects": [
    #                 "Motor cannot start",
    #                 "Overcurrent towards motor",
    #                 "Motor starts without soft start",
    #             ]
    #         }
    #     ]
    # }
    FIELD_WEIGHTS = {
    "element":0.2,
    "mode": 1.2,
    "cause": 1.0,
    "effect": 1.0,
}
    results_graph = generate_graph_inferred_chains_from_structure(persist_dir=KB_PATH,structure_input=structure_input_powertrain,
                                                                  min_similarity=0.45,top_k_per_field=30)
    print_inferred_structure(results_graph,top_n=25)