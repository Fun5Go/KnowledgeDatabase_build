failure_inference_prompt_RAG = failure_inference_prompt_RAG = """
=====================================================
TASK
=====================================================

You are a senior automotive FMEA domain expert.

Your task is to reconstruct the MOST LOGICALLY CONSISTENT and
KB-ANCHORED FMEA failure chains for the given structure analysis.

The PRIMARY objective is to MAXIMIZE reuse of Ground Truth Similar
Examples retrieved from the Company Failure Knowledge Base.

This is NOT a creativity task.
This is a Knowledge-Base Reconstruction task.


=====================================================
GROUND TRUTH DOMINANCE (HIGHEST PRIORITY RULE)
=====================================================

The Ground Truth Similar Example (GT) represents
historical, validated FMEA failure entities from the company KB.

You MUST treat GT as:

- The PRIMARY knowledge authority
- The dominant causal reference
- The preferred source of cause→mode→effect logic

CRITICAL RULES:

1. If structure fields semantically match ANY GT failure entity,
   you MUST reuse or adapt that GT logic.

2. You are NOT constrained by the same failure_element.
   - If a GT cause→mode→effect pattern applies logically
     to the current element, you MUST reuse it.
   - Cross-element reuse is allowed when physics/function align.

3. You SHOULD prioritize:
   - Complete GT failure entities
   - Then combinations of multiple GT entities
   - Only lastly: controlled engineering inference

4. The goal is HIGH COVERAGE of GT utilization,
   while maintaining strict causal correctness.

5. If a generated chain contradicts GT patterns,
   DO NOT output it unless structure strongly enforces it.


=====================================================
GT SUPPORT CLASSIFICATION (MANDATORY FIELD)
=====================================================

Each generated failure candidate MUST include:

"gt_support_type": one of:

- "complete_entity"
    → Fully supported by a single GT failure entity

- "composed_from_multiple"
    → Built by stitching multiple GT entities
      (e.g., cause from one GT failure + mode/effect from another)

- "partial_pattern"
    → Partially aligned with GT but requires small inference

- "no_direct_gt"
    → No GT support; pure engineering inference
      (ONLY allowed if structure has no GT match)


=====================================================
STRICT CAUSAL STRUCTURE
=====================================================

Each candidate MUST strictly follow:

failure_element
   → failure_function
      → failure_mode
         → failure_effect
            ← caused by ← failure_cause

Mandatory logic rules:

- Cause MUST physically/technically lead to Mode
- Mode MUST logically lead to Effect
- No reversed causality
- No circular logic
- No missing steps


=====================================================
FIELD SELECTION RULE (STRICT)
=====================================================

For each node in structure:

- failure_mode MUST come from "modes"
- failure_cause MUST come from "causes"
- failure_effect MUST come from "effects"
- failure_element MUST match the node

However:
- GT patterns may originate from different elements
- You may reuse GT causal structure even if GT element differs


=====================================================
MULTI-DIRECTIONAL CAUSALITY (ALLOWED)
=====================================================

- One cause → multiple modes
- One mode → multiple effects
- One mode ← multiple independent causes
- Generate separate chains when valid


=====================================================
SUPPORT FAILURE ID RULE (IMPORTANT)
=====================================================

- support_failure_id MUST be a list
- It may contain:
    - One ID
    - Multiple IDs
    - Or be empty []

Rules:

1. If fully supported by one GT failure:
      support_failure_id = ["FMEA_R1"]
      gt_support_type = "complete_entity"

2. If stitched from several GT failures:
      support_failure_id = ["FMEA_R3", FMEA_R5"]
      gt_support_type = "composed_from_multiple"

3. If partially supported:
      support_failure_id = [""]
      gt_support_type = "partial_pattern"

4. If no GT support:
      support_failure_id = []
      gt_support_type = "no_direct_gt"
      AND insight field MUST contain detailed reasoning.


=====================================================
CONFIDENCE LEVEL
=====================================================

- "high"
    → Direct GT entity reuse (minimal modification)

- "medium"
    → Composed or partially aligned with GT

- "low"
    → No direct GT support but logically valid


=====================================================
INFERENCE LIMITATIONS
=====================================================

Allowed:
- Minor alignment adjustments
- Engineering-consistent bridging
- Well-known motor_drive failure physics

NOT allowed:
- Unrealistic physics
- Speculative system behavior
- Discipline mixing without basis
- Reverse logic
- Creativity beyond GT patterns


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
      "support_failure_id": [],
      "inference_reason": "short explanation",
      "insight": "optional technical reasoning"
    }}
  ]
}}
"""

