failure_inference_prompt_RAG = """
=====================================================
TASK
=====================================================

You are a senior FMEA domain expert.

Your task is to construct the MOST PHYSICALLY AND LOGICALLY
CONSISTENT FAILURE GRAPH based on the provided Structure Analysis and the ground truth examples which is semantic similar to the query.

This is a STRUCTURE-DRIVEN reconstruction task.

The PRIMARY objective is:

→ Use Structure Analysis to generate the most technically coherent
  and causally valid FMEA failure chains.

Ground Truth (GT) examples are IMPORTANT to infer your reasoning:
They serve as validation and enhancement references,


=====================================================
CORE PRINCIPLE (STRUCTURE DOMINANCE)
=====================================================

1. Structure Analysis defines:
   - available elements
   - possible failure modes
   - possible causes
   - possible effects

2. ALL generated failure chains MUST use only Structure fields.

3. Your first priority is:
   - Physical correctness
   - Motor drive engineering logic
   - Causal consistency

4. GT examples are used to:
   - Infer the possible failure entity
   - Validate plausibility
   - Reinforce known patterns
   - Increase confidence level

If GT contradicts strong structural logic:
→ Follow Structure.


=====================================================
FAILURE GRAPH OBJECTIVE
=====================================================

You must construct a failure graph consisting of:

failure_element
   → failure_function
      → failure_mode
         → failure_effect
            ← caused by ← failure_cause

Requirements:

- Cause MUST physically produce Mode
- Mode MUST realistically lead to Effect
- All chains must be motor-drive related
- Avoid trivial or redundant combinations
- Prefer physically meaningful chains over syntactic matches


=====================================================
GRAPH CONSTRUCTION STRATEGY
=====================================================

Step 1 — Structure-First Chain Generation
-----------------------------------------

For each element in Structure:

- Evaluate all possible (cause → mode → effect) combinations.
- Select only those that are physically consistent.
- Discard illogical chains.

Step 2 — Optimize Graph Coherence
----------------------------------

- Avoid duplicate (mode + cause + effect) combinations.
- Avoid overly similar chains.
- Prefer diverse but realistic failure mechanisms.

Step 3 — GT Validation (Secondary)
-----------------------------------

Compare generated chains against GT examples.

If a chain matches a GT entity:
    → Mark as "complete_entity"
If partially aligned:
    → "partial_pattern"
If composed from multiple GT:
    → "composed_from_multiple"
If no GT alignment:
    → "no_direct_gt"

GT must NEVER override structural physics.


=====================================================
GT SUPPORT CLASSIFICATION
=====================================================

"gt_support_type" must be one of:

- "complete_entity"
- "composed_from_multiple"
- "partial_pattern"
- "no_direct_gt"

support_failure_id:
- [] if no GT
- list of GT IDs if aligned


=====================================================
CONFIDENCE LEVEL
=====================================================

- "high"
    → Strong structural logic + direct GT match

- "medium"
    → Strong structural logic + partial GT support

- "low"
    → Structurally valid but no GT reference


=====================================================
INFERENCE LIMITATIONS
=====================================================

Allowed:
- Engineering-consistent reasoning
- Cross-element physics reuse
- Known motor-drive failure mechanisms

NOT allowed:
- Speculative or unrealistic behavior
- Violating physical causality
- Random structure combinations

=====================================================
GROUND TRUTH SIMILAR EXAMPLE
=====================================================

{gt_example}

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
      "gt_support_type": "complete_entity | composed_from_multiple | partial_pattern | no_direct_gt",
      "support_failure_id": [""],
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