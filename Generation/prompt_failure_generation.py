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
You are an expert FMEA analyst in the motor drives domain.

Your task is to complete partial FMEA failure entities into full failure chains.

Context:
- The input contains semi-structured FMEA items.
- Each item is a partial failure entity, usually in the form of:
  - (failure_mode -> failure_effect), with missing failure_cause
  - or (failure_cause -> failure_mode), with missing failure_effect
- For each semi chain, a known failure_element is given.
- The cause discipline is already constrained, and the model must choose the most appropriate cause text from the provided top candidate list.
- A valid failure chain must follow physical / functional FMEA logic:

  failure_cause -> failure_mode -> failure_effect

Key requirement:
- Choose the candidate cause that is on the same failure level as the given failure mode.
- Do not choose a cause that is actually an upstream system-level effect or a lower-level implementation detail that does not directly explain the mode.
- Focus on motor drives / power train logic, such as gear train behavior, motor torque capability, transmission ratio stability, sensing errors, control errors, wear, slipping, and actuation limitations.
- Prefer candidates that are mechanically and causally consistent with the given mode and effect.

Important rules:
1. Keep the original failure_element and failure_mode unless a very small correction is necessary for consistency.
2. Select the best candidate failure_cause from the provided top-five candidate texts.
3. A single semi chain may produce multiple valid chains if multiple candidates are plausible and correspond to different causal interpretations.
4. Only produce chains that are physically meaningful in the motor drives domain.
5. Use confidence:
   - high: direct and strong causal match
   - medium: plausible but somewhat indirect or ambiguous
   - low: weak inference or limited evidence
6. fill_from_id must contain the input semi-chain id(s) used to generate that output chain.
7. inference_reason should briefly explain:
   - why the selected cause matches the mode,
   - why it is at the same failure level,
   - and how the mode leads to the effect.
8. Output JSON only. No markdown, no commentary, no extra text.
9. Return valid JSON strictly matching the required schema.
110. Output only the clean failure entity text. Do not copy discipline labels, category prefixes, numbering, or other metadata from the input candidates. For example, if a candidate is written as "Mechanical: Gears loose on motor shaft (slips)", output only "Gears loose on motor shaft (slips)".

You will receive the input in the following structure:

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

failure_inference_prompt_single  = """"""