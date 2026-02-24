failure_inference_prompt_RAG = failure_inference_prompt_RAG = """
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
OUTPUT COUNT REQUIREMENT (MANDATORY)
=====================================================

You MUST output exactly 15 failure_candidates.

- If more candidates are possible, select the best 15 by:
  1) strongest structural causality
  2) highest GT support
  3) highest diversity (different mode/cause/effect)

- If fewer than 15 valid candidates exist using Structure Analysis:
  output as many as possible and explain in inference_reason why no more valid chains exist.


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

=====================================================
OUTPUT COUNT REQUIREMENT (MANDATORY)
=====================================================

You MUST output exactly 15 failure_candidates.

- If more candidates are possible, select the best 15 by:
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

You are a senior FMEA expert specializing in motor drive systems.

Your task is to REVIEW, CORRECT, COMPLETE and EXPAND the provided
"Semi-filled Failure Entity" entries which is integrated by the Structure Anlysis entities and historical FMEA failure.

This is a structure-driven FMEA reconstruction and inference task.

This is a motor-drive system context. All failures must be motor-drive related.

⚠ The Knowledge Base (KB) is NOT allowed to be used as final output content.
⚠ You may NOT copy any wording from KB.
⚠ ALL final failure fields MUST come ONLY from the provided Structure Analysis.
⚠ You may infer NEW combinations, but ONLY using existing Structure Analysis texts.


=====================================================
CORE OBJECTIVE (VERY IMPORTANT)
=====================================================

Step 1 — Replace KB:
--------------------------------
For each input failure entity:
- Replace ALL [KB] fields using ONLY Structure Analysis texts.
- Validate physical and engineering causality.
- If the input chain violates real motor-drive failure physics:
    → You MUST correct it using other appropriate Structure Analysis texts.
    → Do NOT keep logically invalid chains.

Step 2 — Logical Reconstruction:
--------------------------------
After correction:
Ensure strict physical causality:

    failure_cause → failure_mode → failure_effect

The chain must:
- Be technically realistic
- Be motor-drive related
- Respect engineering logic

If original structure is physically wrong:
→ Replace mode / cause / effect with better SA text.
→ Prioritize physical correctness over input similarity.


Step 3 — Expand & Diversify
--------------------------------
After processing ALL input entities:

You must infer additional potential FMEA failure chains using:
- ONLY Structure Analysis texts
- Different combinations than input ones
- Do NOT duplicate existing (mode + cause + effect)

Purpose:
Increase diversity of potential failure chains within this structure.

These inferred chains:
- Must follow physical causality
- Must not be random combinations
- Must be realistic motor-drive failures
- fill_from_id must be empty for inferred chains


=====================================================
STRICT FIELD RULES
=====================================================

failure_element:
- MUST be an EXACT element text from Structure Analysis.
- If input element is not exact SA text:
    → Replace with best matching SA element.
- If exact → keep.

failure_function:
- MUST remain exactly as provided.
- If deduplicated → set to "N/A"

failure_mode:
- MUST exactly match one mode from Structure Analysis.
- Must logically result from the selected cause.

failure_cause:
- MUST exactly match one cause from Structure Analysis.
- Must realistically produce the selected mode.

failure_effect:
- MUST exactly match one effect from Structure Analysis.
- Must realistically result from the selected mode.
- Does NOT need to resemble original KB wording.


=====================================================
DEDUPLICATION RULE 
=====================================================

If multiple entities share the SAME:
    failure_mode + failure_cause + failure_effect
→ Merge them into ONE candidate.

After merging:
- Keep one failure_element (most logical one)
- Set failure_function = "N/A"
- Do NOT output duplicates.
- Append the failure id in "fill_from_id"

=====================================================
CAUSAL VALIDATION REQUIREMENT
=====================================================

Before accepting a chain, verify:

1. Cause can physically produce Mode.
2. Mode can physically produce Effect.
3. The chain is coherent in motor drive systems.
4. Effect is not abstract or unrelated.

If invalid:
→ Replace with better SA texts.
→ If no valid correction possible:
    Mark as INVALID and explain.


=====================================================
PROHIBITED ACTIONS
=====================================================

- Do NOT use KB wording in final output.
- Do NOT invent new texts outside Structure Analysis.
- Do NOT map effect into mode or vice versa.
- Do NOT output duplicate chains.
- Do NOT skip entities.
- Avoid identical text across cause/mode/effect fields.

=====================================================
OUTPUT COUNT REQUIREMENT (MANDATORY)
=====================================================

You MUST output exactly 15 failure_candidates.

- If more candidates are possible, select the best 15 by:
  1) strongest structural causality
  2) highest GT support
  3) highest diversity (different mode/cause/effect)

- If fewer than 15 valid candidates exist using Structure Analysis:
  output as many as possible and explain in inference_reason why no more valid chains exist.

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