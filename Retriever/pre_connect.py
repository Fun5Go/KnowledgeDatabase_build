from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np

# -----------------------------
# Data structure
# -----------------------------
@dataclass
class CandidateChain:
    cause: str
    mode: str
    effect: str
    sim_cm: float
    sim_me: float
    score: float
@dataclass
class CandidatePair:
    left: str
    right: str
    sim: float

# -----------------------------
# Helpers
# -----------------------------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n

def _batch_cosine_sim(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # vec: (d,), mat: (n,d), both normalized
    return mat @ vec  # (n,)


# -----------------------------
# Main: internal chain generation
# -----------------------------
def generate_internal_candidate_chains(
    structure_input: Dict[str, Any],
    *,
    node_index: int = 0,
    # combination controls
    top_k_modes_per_cause: int = 5,
    top_k_effects_per_mode: int = 5,
    beam_width_cm: int = 60,
    beam_width_final: int = 80,
    # thresholds
    min_sim_cm: float = 0.28,
    min_sim_me: float = 0.28,
    # weights
    w_cm: float = 1.0,
    w_me: float = 1.0,
    # embedding model
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[CandidateChain]:
    """
    Generate likely Cause->Mode->Effect chains purely inside the structure input.

    Scoring:
        score = w_cm * sim(cause, mode) + w_me * sim(mode, effect)

    Returns top `beam_width_final` candidates.
    """
    nodes = structure_input.get("nodes") or []
    if not nodes:
        return []

    if node_index < 0 or node_index >= len(nodes):
        raise IndexError(f"node_index {node_index} out of range (nodes={len(nodes)})")

    node = nodes[node_index]
    causes  = [str(c).strip() for c in (node.get("causes") or []) if str(c).strip()]
    modes   = [str(m).strip() for m in (node.get("modes") or []) if str(m).strip()]
    effects = [str(e).strip() for e in (node.get("effects") or []) if str(e).strip()]

    if not causes or not modes or not effects:
        return []

    # Lazy import to keep module lightweight
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(st_model_name)

    def embed(texts: List[str]) -> np.ndarray:
        # returns (n, d) numpy array
        return st.encode(texts, normalize_embeddings=True)

    # --- Embeddings (normalized)
    cause_emb = _l2_normalize(embed(causes))
    mode_emb  = _l2_normalize(embed(modes))
    eff_emb   = _l2_normalize(embed(effects))

    # -----------------------------
    # Step A: Cause -> Mode beams
    # -----------------------------
    cm_beams: List[Tuple[str, str, float]] = []  # (cause, mode, sim_cm)
    for i, cause in enumerate(causes):
        sims = _batch_cosine_sim(cause_emb[i], mode_emb)  # (num_modes,)
        top_idx = np.argsort(-sims)[:top_k_modes_per_cause]
        for j in top_idx:
            s = float(sims[j])
            if s < min_sim_cm:
                continue
            cm_beams.append((cause, modes[int(j)], s))

    cm_beams.sort(key=lambda x: x[2], reverse=True)
    cm_beams = cm_beams[:beam_width_cm]

    if not cm_beams:
        return []

    # -----------------------------
    # Step B: (Cause, Mode) -> Effect beams
    # -----------------------------
    mode_to_idx = {m: idx for idx, m in enumerate(modes)}
    chains: List[CandidateChain] = []

    for cause, mode, sim_cm in cm_beams:
        m_idx = mode_to_idx.get(mode)
        if m_idx is None:
            continue

        sims_me = _batch_cosine_sim(mode_emb[m_idx], eff_emb)
        top_e_idx = np.argsort(-sims_me)[:top_k_effects_per_mode]

        for k in top_e_idx:
            sim_me = float(sims_me[k])
            if sim_me < min_sim_me:
                continue

            score = (w_cm * sim_cm) + (w_me * sim_me)
            chains.append(
                CandidateChain(
                    cause=cause,
                    mode=mode,
                    effect=effects[int(k)],
                    sim_cm=sim_cm,
                    sim_me=sim_me,
                    score=score,
                )
            )

    chains.sort(key=lambda x: x.score, reverse=True)
    return chains[:beam_width_final]


def print_candidate_chains(chains: List[CandidateChain], top_n: int = 20) -> None:
    if not chains:
        print("No candidate chains generated. Try lowering min_sim_cm/min_sim_me.")
        return

    n = min(top_n, len(chains))
    print(f"Generated {len(chains)} candidate chains. Showing top {n}:\n")
    for i, ch in enumerate(chains[:n], 1):
        print(f"Rank {i} | score={ch.score:.4f} | sim(C,M)={ch.sim_cm:.4f} | sim(M,E)={ch.sim_me:.4f}")
        print(f"  Cause : {ch.cause}")
        print(f"  Mode  : {ch.mode}")
        print(f"  Effect: {ch.effect}")
        print("-" * 90)

def generate_internal_candidate_pairs(
    structure_input: Dict[str, Any],
    *,
    node_index: int = 0,
    top_k_modes_per_cause: int = 5,
    top_k_effects_per_mode: int = 5,
    beam_width_cm: int = 80,
    beam_width_me: int = 80,
    min_sim_cm: float = 0.25,
    min_sim_me: float = 0.25,
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[CandidatePair], List[CandidatePair]]:
    """
    Return:
      - cause_mode_pairs: top cause->mode pairs
      - mode_effect_pairs: top mode->effect pairs
    """

    nodes = structure_input.get("nodes") or []
    if not nodes:
        return [], []

    node = nodes[node_index]
    causes  = [str(c).strip() for c in (node.get("causes") or []) if str(c).strip()]
    modes   = [str(m).strip() for m in (node.get("modes") or []) if str(m).strip()]
    effects = [str(e).strip() for e in (node.get("effects") or []) if str(e).strip()]

    if not causes or not modes or not effects:
        return [], []

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(st_model_name)

    def embed(texts: List[str]) -> np.ndarray:
        return st.encode(texts, normalize_embeddings=True)

    cause_emb = _l2_normalize(embed(causes))
    mode_emb  = _l2_normalize(embed(modes))
    eff_emb   = _l2_normalize(embed(effects))

    # -----------------------------
    # Cause -> Mode
    # -----------------------------
    cm_pairs: List[CandidatePair] = []
    for i, cause in enumerate(causes):
        sims = _batch_cosine_sim(cause_emb[i], mode_emb)
        top_idx = np.argsort(-sims)[:top_k_modes_per_cause]
        for j in top_idx:
            s = float(sims[j])
            if s < min_sim_cm:
                continue
            cm_pairs.append(CandidatePair(left=cause, right=modes[int(j)], sim=s))

    cm_pairs.sort(key=lambda x: x.sim, reverse=True)
    cm_pairs = cm_pairs[:beam_width_cm]

    # -----------------------------
    # Mode -> Effect
    # -----------------------------
    me_pairs: List[CandidatePair] = []
    for i, mode in enumerate(modes):
        sims = _batch_cosine_sim(mode_emb[i], eff_emb)
        top_idx = np.argsort(-sims)[:top_k_effects_per_mode]
        for k in top_idx:
            s = float(sims[k])
            if s < min_sim_me:
                continue
            me_pairs.append(CandidatePair(left=mode, right=effects[int(k)], sim=s))

    me_pairs.sort(key=lambda x: x.sim, reverse=True)
    me_pairs = me_pairs[:beam_width_me]

    return cm_pairs, me_pairs


# -----------------------------
# Printing
# -----------------------------
def print_pairs(title: str, pairs: List[CandidatePair], top_n: int = 20) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if not pairs:
        print("No pairs generated. Try lowering min_sim threshold.")
        return

    n = min(top_n, len(pairs))
    for i, p in enumerate(pairs[:n], 1):
        print(f"Rank {i} | sim={p.sim:.4f}")
        print(f"  LEFT : {p.left}")
        print(f"  RIGHT: {p.right}")
        print("-" * 90)



if __name__ == "__main__":
    structure_input = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "",
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
                    "Update takes too much time (>5 minutes)"
                ]
            }
        ]
    }
    # result = generate_internal_candidate_pairs(structure_input=structure_input)

    cm_pairs, me_pairs = generate_internal_candidate_pairs(
        structure_input=structure_input,
        top_k_modes_per_cause=5,
        top_k_effects_per_mode=5,
        beam_width_cm=50,
        beam_width_me=50,
        min_sim_cm=0.22,
        min_sim_me=0.22,
    )

    print_pairs("Cause → Mode (Top)", cm_pairs, top_n=40)
    print_pairs("Mode → Effect (Top)", me_pairs, top_n=10)


  