from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
import re
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


# -----------------------------
# Utilities
# -----------------------------
_WS = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """Lightweight normalization for de-dup / matching only."""
    s = str(s or "").strip().lower()
    s = _WS.sub(" ", s)
    return s

def l2_normalize(E: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize."""
    E = np.asarray(E, dtype=np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E / n

def cosine_sim_vector_to_matrix(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    v: (d,) normalized
    M: (n,d) normalized
    return: (n,) cosine sims
    """
    return (M @ v).astype(np.float32)

def topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    """Fast top-k indices for 1D array."""
    k = int(min(k, x.shape[0]))
    if k <= 0:
        return np.array([], dtype=np.int64)
    # argpartition then sort within top-k
    idx = np.argpartition(-x, kth=k-1)[:k]
    idx = idx[np.argsort(-x[idx])]
    return idx.astype(np.int64)


# -----------------------------
# Standard candidate generation
# -----------------------------
def generate_internal_candidate_pairs_standard(
    structure_input: Dict[str, Any],
    *,
    node_index: int = 0,

    # retrieval width
    top_k_modes_per_cause: int = 10,
    top_k_causes_per_mode: int = 10,
    top_k_effects_per_mode: int = 10,
    top_k_modes_per_effect: int = 10,

    # global beam caps
    beam_width_cm: int = 120,
    beam_width_me: int = 120,

    # to avoid beam domination (per-left quota)
    per_cause_quota: int = 3,
    per_mode_quota_for_me: int = 3,

    # similarity thresholds (should be calibrated)
    min_sim_cm: float = 0.35,
    min_sim_me: float = 0.35,

    # model
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[CandidatePair], List[CandidatePair]]:
    """
    Standardized candidate edge generation with:
      - normalization + de-dup
      - reciprocal top-k filtering
      - per-left quota to prevent beam domination
      - global beam truncation

    Return:
      cm_pairs: Cause -> Mode
      me_pairs: Mode  -> Effect
    """
    nodes = structure_input.get("nodes") or []
    if not nodes:
        return [], []
    if node_index < 0 or node_index >= len(nodes):
        return [], []

    node = nodes[node_index]
    raw_causes  = [str(c).strip() for c in (node.get("causes") or []) if str(c).strip()]
    raw_modes   = [str(m).strip() for m in (node.get("modes") or []) if str(m).strip()]
    raw_effects = [str(e).strip() for e in (node.get("effects") or []) if str(e).strip()]

    if not raw_causes or not raw_modes or not raw_effects:
        return [], []

    # 1) Normalize + de-dup but keep original text (first occurrence)
    def dedup_keep_first(texts: List[str]) -> Tuple[List[str], List[str]]:
        norm2orig: Dict[str, str] = {}
        for t in texts:
            nt = normalize_text(t)
            if not nt:
                continue
            if nt not in norm2orig:
                norm2orig[nt] = t
        norms = list(norm2orig.keys())
        origs = [norm2orig[n] for n in norms]
        return norms, origs

    cause_norms, causes  = dedup_keep_first(raw_causes)
    mode_norms,  modes   = dedup_keep_first(raw_modes)
    eff_norms,   effects = dedup_keep_first(raw_effects)

    if not causes or not modes or not effects:
        return [], []

    # 2) Embed (single normalization: either here OR later; do it once)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(st_model_name)

    # normalize_embeddings=True already does L2 norm in sentence-transformers
    C = np.asarray(st.encode(causes,  normalize_embeddings=True), dtype=np.float32)  # (nc, d)
    M = np.asarray(st.encode(modes,   normalize_embeddings=True), dtype=np.float32)  # (nm, d)
    E = np.asarray(st.encode(effects, normalize_embeddings=True), dtype=np.float32)  # (ne, d)

    # (Optional safety) If you don't trust normalize_embeddings, uncomment:
    # C = l2_normalize(C); M = l2_normalize(M); E = l2_normalize(E)

    # 3) Precompute reciprocal neighbor sets
    # --- Cause -> Mode candidates
    cause_to_modes: List[List[int]] = []
    for i in range(C.shape[0]):
        sims = cosine_sim_vector_to_matrix(C[i], M)
        idx = topk_indices(sims, top_k_modes_per_cause)
        idx = [int(j) for j in idx if float(sims[j]) >= min_sim_cm]
        cause_to_modes.append(idx)

    mode_to_causes: List[List[int]] = []
    for j in range(M.shape[0]):
        sims = cosine_sim_vector_to_matrix(M[j], C)
        idx = topk_indices(sims, top_k_causes_per_mode)
        idx = [int(i) for i in idx if float(sims[i]) >= min_sim_cm]
        mode_to_causes.append(idx)

    # --- Mode -> Effect candidates
    mode_to_effects: List[List[int]] = []
    for j in range(M.shape[0]):
        sims = cosine_sim_vector_to_matrix(M[j], E)
        idx = topk_indices(sims, top_k_effects_per_mode)
        idx = [int(k) for k in idx if float(sims[k]) >= min_sim_me]
        mode_to_effects.append(idx)

    eff_to_modes: List[List[int]] = []
    for k in range(E.shape[0]):
        sims = cosine_sim_vector_to_matrix(E[k], M)
        idx = topk_indices(sims, top_k_modes_per_effect)
        idx = [int(j) for j in idx if float(sims[j]) >= min_sim_me]
        eff_to_modes.append(idx)

    # 4) Build reciprocal pairs with de-dup
    def build_reciprocal_pairs_left_to_right(
        left_texts: List[str],
        right_texts: List[str],
        left_emb: np.ndarray,
        right_emb: np.ndarray,
        left_to_right: List[List[int]],
        right_to_left: List[List[int]],
        *,
        min_sim: float,
        per_left_quota: int,
        beam_width: int,
    ) -> List[CandidatePair]:
        pairs: List[CandidatePair] = []
        seen = set()

        # per-left quota first (coverage), then global beam
        for i, rlist in enumerate(left_to_right):
            if not rlist:
                continue
            # score all reciprocal candidates
            scored: List[Tuple[float, int]] = []
            for j in rlist:
                # reciprocal check
                if i not in right_to_left[j]:
                    continue
                sim = float(right_emb[j] @ left_emb[i])  # since normalized
                if sim < min_sim:
                    continue
                key = (normalize_text(left_texts[i]), normalize_text(right_texts[j]))
                if key in seen:
                    continue
                scored.append((sim, j))

            if not scored:
                continue

            scored.sort(key=lambda x: x[0], reverse=True)
            for sim, j in scored[: max(0, int(per_left_quota))]:
                seen.add((normalize_text(left_texts[i]), normalize_text(right_texts[j])))
                pairs.append(CandidatePair(left=left_texts[i], right=right_texts[j], sim=float(sim)))

        # if still too many, global sort + cut
        pairs.sort(key=lambda x: x.sim, reverse=True)
        return pairs[: int(beam_width)]

    cm_pairs = build_reciprocal_pairs_left_to_right(
        left_texts=causes,
        right_texts=modes,
        left_emb=C,
        right_emb=M,
        left_to_right=cause_to_modes,
        right_to_left=mode_to_causes,
        min_sim=min_sim_cm,
        per_left_quota=per_cause_quota,
        beam_width=beam_width_cm,
    )

    me_pairs = build_reciprocal_pairs_left_to_right(
        left_texts=modes,
        right_texts=effects,
        left_emb=M,
        right_emb=E,
        left_to_right=mode_to_effects,
        right_to_left=eff_to_modes,
        min_sim=min_sim_me,
        per_left_quota=per_mode_quota_for_me,
        beam_width=beam_width_me,
    )

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


@dataclass
class RankedEffect:
    failure_cause: str
    failure_mode: str
    failure_effect: str
    sim_qe: float     # sim(query, effect)
    sim_me: float     # sim(mode, effect)
    sim_cm: float     # sim(cause, mode) (0 if no cause)
    score: float

def rank_effects_by_mode_cause_embedding(
    structure_input: Dict[str, Any],
    *,
    mode_text: str,
    cause_text: Optional[str] = None,
    node_index: int = 0,
    top_n: int = 10,
    min_sim_qe: float = 0.0,
    # weights
    w_qe: float = 1.0,   # query -> effect
    w_me: float = 0.3,   # mode  -> effect (stabilize when query noisy)
    w_cm: float = 0.2,   # cause -> mode  (gate/prior when cause exists)
    # model
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[RankedEffect]:
    """
    Rank SA effects given (mode, optional cause) by embedding similarity.

    If cause_text is provided (non-empty):
      query = "cause ; mode"
      score = w_qe*sim(query,effect) + w_me*sim(mode,effect) + w_cm*sim(cause,mode)

    Else:
      query = "mode"
      score = w_qe*sim(mode,effect)   (and sim_me == sim_qe; sim_cm = 0)

    Always loops over effects (vectorized), returns top_n.
    """
    nodes = structure_input.get("nodes") or []
    if not nodes:
        return []
    if node_index < 0 or node_index >= len(nodes):
        raise IndexError(f"node_index {node_index} out of range (nodes={len(nodes)})")

    node = nodes[node_index]
    effects = [str(e).strip() for e in (node.get("effects") or []) if str(e).strip()]
    if not effects:
        return []

    mode_text = (mode_text or "").strip()
    cause_text = (cause_text or "").strip()

    if not mode_text and not cause_text:
        return []

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(st_model_name)

    def emb_one(t: str) -> np.ndarray:
        v = st.encode([t], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)[0]  # (d,)

    def emb_many(ts: List[str]) -> np.ndarray:
        v = st.encode(ts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)  # (n,d)

    # embeddings
    E = emb_many(effects)                 # (ne,d)
    m_vec = emb_one(mode_text) if mode_text else None
    c_vec = emb_one(cause_text) if cause_text else None

    # query vector
    if cause_text and mode_text:
        q_vec = emb_one(f"{cause_text} ; {mode_text}")
    elif mode_text:
        q_vec = emb_one(mode_text)
    else:
        q_vec = emb_one(cause_text)  # fallback (rare)

    # sims
    sim_qe = (E @ q_vec).astype(np.float32)             # (ne,)
    sim_me = (E @ m_vec).astype(np.float32) if m_vec is not None else sim_qe.copy()
    sim_cm = float(c_vec @ m_vec) if (c_vec is not None and m_vec is not None) else 0.0

    # score (vector)
    if cause_text and mode_text:
        score = w_qe * sim_qe + w_me * sim_me + w_cm * sim_cm
    else:
        score = w_qe * sim_qe  # == sim(mode,effect) if mode exists

    # filter + top-n
    idx = np.where(sim_qe >= float(min_sim_qe))[0]
    if idx.size == 0:
        return []

    # take top_n among idx
    k = min(int(top_n), int(idx.size))
    # argpartition on subset
    sub = score[idx]
    top_local = np.argpartition(-sub, kth=k-1)[:k]
    top_idx = idx[top_local]
    top_idx = top_idx[np.argsort(-score[top_idx])]

    out: List[RankedEffect] = []
    for i in top_idx.tolist():
        out.append(RankedEffect(
            failure_cause=cause_text,
            failure_mode=mode_text,
            failure_effect=effects[i],
            sim_qe=float(sim_qe[i]),
            sim_me=float(sim_me[i]),
            sim_cm=float(sim_cm),
            score=float(score[i]),
        ))
    return out


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
                    "Update takes too much time (>5 minutes)",
                    "Too much noise",
                ]
            }
        ]
    }
    # result = generate_internal_candidate_pairs(structure_input=structure_input)

    # cm_pairs, me_pairs = generate_internal_candidate_pairs_standard(
    #     structure_input=structure_input,
    #     top_k_modes_per_cause=5,
    #     top_k_effects_per_mode=5,
    #     beam_width_cm=50,
    #     beam_width_me=50,
    #     min_sim_cm=0.22,
    #     min_sim_me=0.22,
    # )

    # print_pairs("Cause → Mode (Top)", cm_pairs, top_n=20)
    # print_pairs("Mode → Effect (Top)", me_pairs, top_n=20)

    top = rank_effects_by_mode_cause_embedding(
    structure_input,
    mode_text="Not enough torque",
    cause_text="Control parameters incorrect",
    top_n=10,
    min_sim_qe=0.15,
)
    for i, r in enumerate(top, 1):
        print(i, r.failure_effect, "score=", round(r.score, 4), "sim_qe=", round(r.sim_qe, 4), "sim_cm=", round(r.sim_cm, 4))


  