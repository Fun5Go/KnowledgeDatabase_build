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

failure_inference_prompt_RAG_FILL = """
=====================================================
TASK
=====================================================
You are a senior motor-drive system FMEA architect and failure-physics expert.

Your task is to COMPLETE semi failure chains using ONLY the candidate options
provided in each chain.

This is a constrained selection task.
You MUST select exactly ONE candidate from the provided candidate list.

--------------------------------------------------
INPUT STRUCTURE
--------------------------------------------------

Each item is a SEMI CHAIN describing a failure relationship.

Two types exist:

MC chain:
    (element + mode + cause) are given
    effect is missing

ME chain:
    (element + mode + effect) are given
    cause is missing

Each chain contains a list of candidate options:

MC chains:
    ppl_candidates_effect

ME chains:
    ppl_candidates_cause

You MUST select from that list only.


--------------------------------------------------
STRICT CONSTRAINTS
--------------------------------------------------

1. Candidate Restriction (MANDATORY)
   - You MUST select ONE option from the provided candidate list.
   - Do NOT invent new causes or effects.

2. Structure Analysis Vocabulary Lock
   - All final fields must match exactly the texts from Structure Analysis.

3. Physics Causality Check
   Every completed chain must satisfy:

        failure_cause → failure_mode → failure_effect

   The chain must be physically plausible in a motor-drive system.

4. Element Consistency
   Cause, mode and effect must be consistent with the subsystem
   described by failure_element.


--------------------------------------------------
SELECTION PRIORITY
--------------------------------------------------

When choosing between candidates use the following order:

1️⃣ Physical plausibility of the mechanism
2️⃣ Consistency with the failure mode semantics
3️⃣ Consistency with the subsystem (Power train)
4️⃣ Candidate ranking statistics (coh, f_nll)

Statistics are only a weak hint.
They must NOT override physics.


--------------------------------------------------
CHAIN COMPLETION RULES
--------------------------------------------------

For each chain:

MC chain:
    select ONE candidate from ppl_candidates_effect
    assign it to failure_effect

ME chain:
    select ONE candidate from ppl_candidates_cause
    assign it to failure_cause

For each semi chain, if all candidates are not suitable pick the other from the SA list

Do NOT modify:
- failure_element
- failure_mode
- existing cause/effect fields

=====================================================
INPUT FAILURE ENTITIES (SEMI CHAIN)
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