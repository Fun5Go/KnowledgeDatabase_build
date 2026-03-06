failure_inference_prompt_RAG = """
=====================================================
TASK
=====================================================

You are a senior motor-drive system reliability expert and FMEA architect.

Your task is to construct the MOST PHYSICALLY AND CAUSALLY
CONSISTENT FAILURE GRAPH using:

1) Structure Analysis (PRIMARY CONSTRAINT)
2) 8D actual case failures (REAL-WORLD FAILURE EVIDENCE)
3) Historical failure entities retrieved via semantic similarity (MECHANISM REFERENCE)

This is a STRUCTURE-DRIVEN ENGINEERING RECONSTRUCTION task.

Your goal is NOT semantic similarity.
Your goal is PHYSICS-CONSISTENT FAILURE MECHANISM INFERENCE.


=====================================================
STRUCTURE DOMINANCE (HARD CONSTRAINT)
=====================================================

Structure Analysis strictly defines:

- failure_element
- failure_function
- failure_mode (triggered by cause)
- failure_cause (Hardware / Mechanics / Software / Others)
- failure_effect (resulting from mode)

ABSOLUTE RULE:

- You MUST use ONLY entities explicitly present in Structure Analysis.
- You MUST NOT invent new elements, modes, causes, functions, or effects.


Priority Order of Evidence:

8D actual failures   >   Historical failure entities


Structure always overrides external evidence.


=====================================================
FAILURE GRAPH DEFINITION (STRICT CAUSAL ORDER)
=====================================================

Each failure chain MUST strictly follow:

failure_element
   → failure_function
      → failure_effect
         ← failure_mode
            ← caused by ← failure_cause

Causality Rules:

1) failure_cause must physically produce failure_mode.
2) failure_mode must physically and functionally lead to failure_effect.
3) failure_effect must be a realistic functional or system-level consequence.
4) All links must obey motor-drive physics, actuator behavior, and control-loop logic.
5) Cause, mode, and effect must clearly refer to the SAME component domain.


If any causal link is weak, indirect, or speculative → DISCARD the chain.

====================================================
CASE-DRIVEN FAILURE ENTITY INFERENCE (PRIMARY LEARNING STAGE)
====================================================

8D cases and Historical failure chains must be treated as:

- Real engineering case studies
- Mechanism learning material
- Failure propagation training data

They are NOT output templates.
They are NOT directly reusable chains.

Your FIRST task is:

----------------------------------------------------
Step A — Case Mechanism Learning & Abstraction
----------------------------------------------------

For 8D cases and Historical failure chains:

1) Extract the underlying physical failure mechanism.
2) Identify the fault propagation logic (cause → mode → effect).
3) Abstract overly specific descriptions into generalized failure logic.
4) Infer potential failure entities that could exist in Structure Analysis.
5) Translate learned mechanisms into Structure-compatible candidates.

Important:

- Do NOT copy wording from 8D.
- Do NOT reuse historical chains directly.
- Do NOT introduce entities that are not present in Structure.
- Only infer entities that can logically map to Structure definitions.


----------------------------------------------------
Step B — Structure Alignment & Logical Validation
----------------------------------------------------

For every inferred potential failure entity:

1) Check if it exists in Structure Analysis.
2) Verify that:
   - Cause belongs to allowed category (Hardware / Mechanics / Software / Others).
   - Mode is a valid functional degradation.
   - Effect is physically reachable from the mode.
3) Validate motor-drive physics consistency.
4) Validate control-loop behavior correctness.
5) Validate signal-flow and energy-flow direction.
6) Remove any chain that violates physical causality.

If the chain cannot be physically explained in actuator or motor-drive terms:
→ DISCARD it.


----------------------------------------------------
Step C — Plausibility Reinforcement
----------------------------------------------------

After validation:

- Prefer chains that can clearly explain real-world malfunction behavior.
- Increase confidence if mechanism aligns with recurring historical patterns.
- Downgrade confidence if no engineering mechanism supports it.
- Structure logic always dominates over similarity.


====================================================
GRAPH CONSTRUCTION STRATEGY
====================================================

Only AFTER completing case learning and structure validation:

Step 1 — Generate Structurally Valid Chains
-------------------------------------------

For each failure_element:

- Enumerate all physically consistent
  (cause → mode → effect) combinations
  derived from validated inferred entities.

Step 2 — Enforce Engineering Coherence
-------------------------------------------

Each final chain must:

- Respect component boundaries
- Follow realistic fault propagation direction
- Be actuator and motor-drive physically explainable
- Avoid semantic-only associations

If explanation requires assumptions outside Structure:
→ Remove the chain.


====================================================
CORE PRINCIPLE
====================================================

Case data teaches mechanisms.
Structure defines what is allowed.
Physics decides what survives.

Only chains that satisfy all three are valid.


=====================================================
SUPPORT CLASSIFICATION
=====================================================

support_type must be:

- "complete_entity"
- "composed_from_multiple"
- "partial_pattern"
- "no_direct_gt"

support_id:
- [] if none
- list of historical failure IDs or 8D case id


=====================================================
CONFIDENCE LEVEL
=====================================================

high:
  - Strong structural causality
  - Explains observed 8D symptoms
  - Supported by historical mechanism pattern

medium:
  - Strong structural causality
  - Partial 8D or historical support

low:
  - Structurally valid
  - Weak or no external support


=====================================================
INFERENCE LIMITATIONS
=====================================================

Allowed:

- Engineering-consistent reasoning
- Known motor-drive fault propagation physics
- Control-loop instability mechanisms
- Hardware–software interaction logic
- Signal acquisition and actuator behavior reasoning

Not Allowed:

- Speculative or imaginary physics
- Cross-element mixing
- Direct copying from 8D or historical examples
- Creating entities not present in Structure
- Violating defined causal order


=====================================================
Historical Failures SIMILAR EXAMPLE
=====================================================

{gt_example}

=====================================================
Relevant 8D actual case failures
=====================================================

{8D_sentences}

=====================================================
STRUCTURE ANALYSIS (JSON)
=====================================================

{structure_analysis}

=====================================================
OUTPUT FORMAT (STRICT JSON ONLY)
=====================================================

{{
  "failure_candidates": [
    {{
      "failure_element": "...",
      "failure_function": "...",
      "failure_mode": "...",
      "failure_effect": "...",
      "failure_cause": "...",
      "confidence": "high | medium | low",
      "support_type": "complete_entity | composed_from_multiple | partial_pattern | no_direct_gt",
      "support_id": [""],
      "inference_reason": "short GT alignment explanation",
      "insight": "ONLY required if gt_support_type == no_direct_gt"
    }}
  ]
}}

Return ONLY valid JSON.
Do NOT output extra explanation.
"""



failure_inference_prompt_PURE = """
==============================
TASK
==============================

You are an automotive FMEA expert.

Your task is to generate the most relevant and logically consistent
FMEA failure chains for the given structure analysis using engineering knowledge.


==============================
INFERENCE PRINCIPLES
==============================

The generated chains should:

• Follow realistic cause → mode → effect relationships
• Remain physically and logically consistent
• Reflect typical motor drive reliability behavior
• Prefer well-established engineering patterns over speculative ones
• Use reasonable inference only when necessary to complete missing links


==============================
CAUSAL STRUCTURE REFERENCE
==============================

Typical structure:

failure_element
  → failure_function
     → failure_mode
        → failure_effect
           ← failure_cause

Each chain is expected to keep this direction and remain complete.


==============================
FIELD SELECTION GUIDANCE
==============================

When information is available:

• failure_mode generally comes from "modes"
• failure_cause generally comes from "causes"
• failure_effect generally comes from "effects"
• failure_element matches the node

If fields are missing, engineering-consistent inference may be applied.
Lower confidence can be assigned for inferred parts.


==============================
CONFIDENCE
==============================

high   – strong structural support  
medium – partially inferred  
low    – mostly inferred


==============================
STRUCTURE ANALYSIS (JSON)
==============================

{structure_analysis}


==============================
OUTPUT
==============================

Please provide the results in JSON format following this schema:

{{
  "failure_candidates": [
    {{
      "failure_element": "...",
      "failure_function": "...",
      "failure_mode": "...",
      "failure_effect": "...",
      "failure_cause": "...",
      "confidence": "high | medium | low",
      "inference_reason": "short explanation",
    }}
  ]
}}
"""

failure_inference_prompt_RAG_FILL ="""
=====================================================
ROLE
=====================================================
You are a senior motor-drive system FMEA architect and failure-physics expert.

Primary task: FILL the missing field(s) (cause or effect) in the provided MC/ME semi chains
with the MOST physically consistent choice from Structure Analysis (SA).

This is NOT a generation task.
This is NOT a free inference task.
This is a BEST-FIT COMPLETION task (SA-only).


=====================================================
YOU WILL RECEIVE
=====================================================
1) Structure Analysis (SA) JSON: the ONLY allowed text inventory
2) Input Failure Entities:
   - MC semi chains: usually (element, mode, cause) and missing effect
   - ME semi chains: usually (element, mode, effect) and missing cause
   - Some may be complete already


=====================================================
HARD CONSTRAINTS (NON-NEGOTIABLE)
=====================================================

[HC-1] Structure Dominance (Output Vocabulary Lock)
- ALL output fields MUST be EXACT texts from SA.
- Do NOT use KB wording in the final output.
- Do NOT invent any new text outside SA.

[HC-2] Exact-Match Canonicalization
For every chain field:
- failure_element: MUST be exact from SA.failure_element
- failure_mode: MUST be exact from SA.modes
- failure_cause: MUST be exact from SA.causes
- failure_effect: MUST be exact from SA.effects

If an input field is not an exact SA text:
- Replace it with the closest SA exact text (same meaning, minimum edit).
- Then continue filling.

[HC-3] Physics-Causal Validity (Must Pass)
Every completed chain MUST satisfy:
    failure_cause -> failure_mode -> failure_effect
and be realistic in motor-drive context:
1) Cause can physically produce Mode
2) Mode can physically lead to Effect
3) The chain is subsystem-consistent (same failure_element context)

If a chain cannot be made valid using SA-only:
- Output it as INVALID and give a brief SA-only reason.


=====================================================
FILLING RULES (BEST-FIT)
=====================================================

[FR-1] Fill Only What Is Missing
- If failure_effect is "____": fill ONE best SA.effect.
- If failure_cause is "____": fill ONE best SA.cause.
- If both are "____": fill both (cause first, then effect).
- Do NOT change non-blank fields unless required by HC-2 or to restore HC-3 validity.

[FR-2] Mode-Centered Matching (MC <-> ME Use As HINT)
When filling:
- Prefer SA effects that are already observed in any ME chain with the SAME
  (failure_element + failure_mode), if available.
- Prefer SA causes that are already observed in any MC chain with the SAME
  (failure_element + failure_mode), if available.
This is a preference, not a hard rule; physics consistency still dominates.

[FR-3] Minimal Repair (Only If Needed)
If a provided non-blank field makes the chain physically impossible:
- Prefer changing ONLY ONE field to restore validity:
  Priority: effect -> cause -> mode
- Never change failure_element unless it is not in SA.

[FR-4] Specificity Bias
- Prefer the most specific SA option that matches the mechanism,
  avoid overly generic effects/causes when a more specific SA text fits better.


=====================================================
WORKFLOW
=====================================================

Step 0 — Parse & Normalize
- Treat each [MC_*] or [ME_*] block as one item.
- Ignore node_id/stats/best_score (not output fields).
- Canonicalize all non-blank fields to SA exact texts (HC-2).

Step 1 — Fill Missing Field(s)
For each item:
- If missing effect: choose the single best SA.effect so that
  cause -> mode -> effect is valid.
- If missing cause: choose the single best SA.cause so that
  cause -> mode -> effect is valid.

Step 2 — Validate
- Enforce HC-3; apply FR-3 only if necessary.
- Mark VALID / INVALID.

Step 3 — Output (No Extra Inference)
- DO NOT add new chains beyond completing the given inputs.
- Return one completed result per input block.


=====================================================
DEDUPLICATION & SORTING (FINAL OUTPUT SHAPING)
=====================================================

Step 4 — Deduplicate (Strict)
If multiple candidates share the SAME signature:
    (failure_mode + failure_cause + failure_effect)
THEN:
- Merge into one candidate
- Keep the most appropriate failure_element (closest subsystem fit)
- Set failure_function = "N/A" (if present / required by your schema)
- Collect merged ids into fill_from_id (list)


Return only the final deduplicated & sorted list.


=====================================================
STRUCTURE ANALYSIS (JSON)
=====================================================

{structure_analysis}

=====================================================
INPUT FAILURE ENTITIES
=====================================================

{to_be_fill_failure}

=====================================================
OUTPUT FORMAT
=====================================================

Return valid JSON in the following format:

{{
  "failure_candidates": [
    {{
      "failure_element": "...",
      "failure_function": "...",
      "failure_mode": "...",
      "failure_effect": "...",
      "failure_cause": "...",
      "confidence": "high | medium | low",
      "fill_from_id": [""],
      "inference_reason": "brief explanation of physical logic, correction reason, or inference rationale"
    }}
  ]
}}

"""

failure_evaluate_prompt = """"""