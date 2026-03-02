failure_inference_prompt_RAG = """
=====================================================
TASK
=====================================================

You are a senior motor-drive system reliability expert and FMEA architect.

Your task is to construct the MOST PHYSICALLY AND CAUSALLY
CONSISTENT FAILURE GRAPH using:

1) Structure Analysis (PRIMARY CONSTRAINT)
2) 8D actual case sentences (REAL-WORLD EVIDENCE)
3) Ground Truth similar examples (REFERENCE PATTERNS)

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
- If something appears in 8D but not in Structure → it can only be used as reasoning evidence.


Priority Order:

Structure Physics  >  8D Evidence  >  GT Similarity

If 8D or GT contradict structural logic:
→ Follow Structure.


=====================================================
FAILURE GRAPH DEFINITION (STRICT CAUSAL ORDER)
=====================================================

Each failure chain MUST strictly follow:

failure_element
   → failure_function
      → failure_mode
         → failure_effect
            ← caused by ← failure_cause

Causality rules:

1) failure_cause physically produces failure_mode
2) failure_mode physically leads to failure_effect
3) failure_effect must be a realistic system-level or functional consequence
4) All links must obey motor-drive engineering logic


If any causal link is weak or speculative → discard the chain.


=====================================================
HOW TO USE 8D SENTENCES
=====================================================

8D sentences represent observed field symptoms.

You must:

1) Extract observable symptoms from 8D sentences.
2) Interpret what physical malfunction could explain that symptom.
3) Map that malfunction to a VALID Structure-based:
      cause → mode → effect chain.

IMPORTANT:

- 8D describes WHAT happened.
- Structure defines WHAT is possible.
- You must infer HOW it happened.
- 8D must NEVER introduce new failure entities.


If a candidate chain explains an 8D symptom:
→ Increase its confidence.

If it does not:
→ It is still allowed if structurally strong.


=====================================================
GRAPH CONSTRUCTION STRATEGY
=====================================================

Step 1 — Generate Structurally Valid Chains
-------------------------------------------

For each failure_element:

- Enumerate possible (cause → mode → effect) combinations.
- Keep ONLY physically consistent combinations.
- Discard semantically matched but physically weak chains.

Step 2 — Enforce Engineering Plausibility
-----------------------------------------

For each candidate:

- Check actuator behavior
- Check motor-drive control logic
- Check signal / sensor / power relationships
- Ensure realistic fault propagation

If mechanism explanation is unclear → remove it.

Step 3 — Align with 8D Evidence
--------------------------------

- Does this chain explain one or more observed symptoms?
- If yes → increase confidence
- If partially → medium confidence
- If no → low confidence (but allowed)

Step 4 — Validate Against GT (Secondary)
-----------------------------------------

Use GT examples ONLY to:

- Confirm known failure patterns
- Increase plausibility confidence

Never copy GT directly without structural validation.


=====================================================
GT SUPPORT CLASSIFICATION
=====================================================

gt_support_type must be:

- "complete_entity"
- "composed_from_multiple"
- "partial_pattern"
- "no_direct_gt"

support_failure_id:
- [] if none
- list of GT IDs if aligned


=====================================================
CONFIDENCE LEVEL
=====================================================

high:
  - Strong structural causality
  - Explains 8D symptom
  - Supported by GT

medium:
  - Strong structural causality
  - Partially supported by 8D or GT

low:
  - Structurally valid
  - Weak or no external support


=====================================================
INFERENCE LIMITATIONS
=====================================================

Allowed:
- Engineering-consistent reasoning
- Known motor-drive failure propagation mechanisms
- Control-loop and actuator reasoning
- Hardware-software interaction logic

Not Allowed:
- Speculative physics
- Random structure combinations
- GT copying
- Creating new entities not in Structure
- Violating causal order


=====================================================
OUTPUT COUNT REQUIREMENT (MANDATORY)
=====================================================

You MUST output exactly 25 failure_candidates.

- If more candidates are possible, select the best 25 by:
  1) strongest structural causality
  2) strongest 8D explanation capability
  3) highest GT support
  4) highest diversity (different mode/cause/effect)



=====================================================
GROUND TRUTH SIMILAR EXAMPLE
=====================================================

{gt_example}

=====================================================
Relevant 8D actual case sentences
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

You MUST output exactly 25 failure_candidates.

- If more candidates are possible, select the best 25 by:
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

You MUST output exactly 25 failure_candidates.

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

failure_evaluate_prompt = """"""