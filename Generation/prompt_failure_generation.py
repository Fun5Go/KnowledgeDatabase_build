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
OUTPUT COUNT REQUIREMENT (MANDATORY)
=====================================================

You MUST output exactly 20 failure_candidates.

If more candidates are possible, select the best 20 by:

1) Strongest structural causality
2) Strongest 8D explanation capability
3) Strongest historical mechanism alignment


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

=====================================================
OUTPUT COUNT REQUIREMENT (MANDATORY)
=====================================================

You MUST output exactly 20 failure_candidates.

- If more candidates are possible, select the best 20 by:
  1) strongest structural causality
  2) highest GT support
  3) highest diversity (different mode/cause/effect)

- If fewer than 15 valid candidates exist using Structure Analysis:
  output as many as possible and explain in inference_reason why no more valid chains exist.

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
ROLE
=====================================================
You are a senior motor-drive system FMEA architect and failure-physics expert.

Task: REVIEW, CORRECT, COMPLETE, and EXPAND the provided "Semi-filled Failure Entity" entries.
This is a STRUCTURE-DRIVEN reconstruction task for motor-drive systems.

=====================================================
HARD CONSTRAINTS (NON-NEGOTIABLE)
=====================================================

[HC-1] Structure Dominance
- ALL final output fields MUST come ONLY from the provided Structure Analysis (SA) texts.
- Do NOT use the Knowledge Base (KB) wording in the final output.
- Do NOT invent any new text outside SA.

[HC-2] Field Exact-Match Rule
For every output candidate:
- failure_element: MUST be EXACT text from SA (closest match if input is not exact).
- failure_function: MUST remain EXACT as provided in input; if deduplicated -> "N/A".
- failure_mode: MUST be EXACT mode text from SA.
- failure_cause: MUST be EXACT cause text from SA.
- failure_effect: MUST be EXACT effect text from SA.

[HC-3] Physics-Causal Validity
Every chain MUST satisfy:
    failure_cause -> failure_mode -> failure_effect
Validate:
1) Cause can physically produce Mode in motor-drive context
2) Mode can physically lead to Effect
3) Chain is coherent, realistic, and not abstract

If invalid:
- Replace cause/mode/effect using other SA texts to make it physically correct.
- If no valid correction exists using SA only: mark INVALID and explain why.

=====================================================
CORE OBJECTIVE
=====================================================

Step 1 — Replace KB / Repair Inputs
For each input failure entity:
- Replace any non-SA fields with SA-exact texts.
- Correct physically wrong chains by selecting better SA cause/mode/effect.
- Prioritize physics correctness over similarity.

Step 2 — Deduplicate (Strict)
If multiple candidates share the SAME:
    (failure_mode + failure_cause + failure_effect)
THEN:
- Merge into one candidate
- Keep the most logical failure_element
- Set failure_function = "N/A"
- Append merged original ids into fill_from_id (list)

Step 3 — Expand & Diversify (Inference)
After processing all inputs:
- Infer additional realistic failure chains using ONLY SA texts
- Must be different combinations than existing output
- Must NOT duplicate any existing (mode + cause + effect)
- fill_from_id MUST be empty for inferred chains
- Must follow strong causal motor-drive physics (not random mixing)

=====================================================
OUTPUT COUNT REQUIREMENT
=====================================================
You MUST output EXACTLY 20 failure_candidates.

If more than 20 are possible:
Select best 20 by:
1) strongest physical causality
2) strongest support by provided 8D evidence (if aligns)
3) diversity across mode/cause/effect

If fewer than 15 valid candidates exist using SA only:
Output as many as possible and explain in inference_reason why more cannot be formed.

=====================================================
8D REAL-WORLD EVIDENCE (GROUNDING ONLY)
=====================================================
Use these as REAL symptoms/mechanism hints to improve plausibility,
BUT you STILL MUST output ONLY SA-exact texts.

# 8D Case 1: 8D6782170310R02 - Motor noise
- Element: motor control algorithm
- Mode: high frequency noise during idle
- Causes:
  - incorrect stall handling in traffic light mode
  - motor control algorithm tuning error
  - hardware error in current measurement circuit

# 8D Case 2: 8D6782170362R01 - Material Debris
- Element: gear assembly
- Mode: material debris in gear
- Effect: device runs very noisily
- Causes:
  - flash pressed out during dowel pin assembly
  - flash formation at gear cover dowel holes

# 8D Case 3: 8D6782170329R02 - App/Integration config resets
- Element: Hub Interface and mobile app integration
- Mode: configuration settings not retained after power cycle
- Effect: max cadence resets to 90 RPM; configuration incomplete; workflow disrupted
- Causes:
  - incorrect initialization sequence from external flash
  - temporary variant misassignment during startup
  - asynchronous notification and data retrieval conflict
  - notification list extension bug on reconnect


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