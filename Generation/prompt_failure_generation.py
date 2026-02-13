failure_inference_prompt = """
==============================
TASK
==============================

You are an automotive FMEA expert.

Your task is to generate the MOST RELEVANT and LOGICALLY CONSISTENT
FMEA failure chains for the given structure analysis.

You MUST use the Ground Truth Similar Example as the PRIMARY knowledge source.

==============================
GROUND TRUTH PRIORITY (CRITICAL)
==============================

The Ground Truth Similar Example is retrieved from the Failure Knowledge Base (KB)
and represents historical, real FMEA failure chains that are semantically closest
to the current structure.

You MUST treat the Ground Truth Example as the PRIMARY and MOST TRUSTWORTHY
knowledge source for this task.

This means:
- Prefer reusing and adapting GT causal patterns over creating new ones.
- When a structure item can be mapped to one or more GT patterns, you MUST map it.
- Only use engineering inference to bridge small gaps needed for alignment.
- If a chain contradicts GT patterns, do NOT output it (unless structure strongly proves otherwise).
- Aim to maximize the utilization of relevant GT patterns (high coverage), while keeping correctness.

The goal is NOT creativity. The goal is KB-anchored reconstruction of likely failure chains
based on historical similar failures.

==============================
STRICT INFERENCE CONSTRAINTS
==============================

1. Ground Truth Example is the dominant reference from historical KB failures (MANDATORY):
   - The GT example contains historical similar failures stored in the knowledge base.
   - It defines the most reliable cause → mode → effect logic for this context.
   - You MUST prioritize GT patterns whenever they are semantically applicable.
   - Do NOT ignore GT patterns: if a structure node matches a GT pattern, you MUST reuse/adapt it.
   - Use inference ONLY as a light bridge to align structure fields with GT logic, not to invent new chains.

2. Multi-directional Causality is allowed:
   - A single failure cause may lead to multiple failure modes.
   - A single failure mode may have multiple independent causes (e.g., HW, SW, thermal, mechanical).
   - Generate separate failure chains for each valid pairing.

3. Controlled engineering inference is allowed:
   - Minor logical completion
   - Reasonable domain-consistent inference
   - Alignment with known motor_drive behavior

4. DO NOT:
   - Invent unrealistic physics
   - Create speculative system behavior
   - Reverse cause/effect direction
   - Create circular logic
   - Introduce unrelated disciplines without support

5. If structure alignment is weak:
   - Either do not generate the chain
   - Or assign lower confidence


==============================
VALID FMEA CAUSAL STRUCTURE
==============================

Each candidate must strictly follow:

failure_element
   → failure_function
      → failure_mode
         → failure_effect
            ← caused by ← failure_cause

Mandatory rules:
- Cause MUST logically lead to Mode
- Mode MUST logically lead to Effect
- No reversed logic
- No missing links


==============================
FIELD SELECTION RULE
==============================

For each node in structure:

- failure_mode MUST come from "modes"
- failure_cause MUST come from "causes"
- failure_effect MUST come from "effects"
- failure_element MUST match the node

Do NOT invent new phrases unless strongly supported by:
- GT pattern
- Or well-established motor_drive engineering knowledge


==============================
SUPPORT FAILURE ID RULE
==============================

- Use ONLY failure IDs from the Ground Truth Example if applicable.
- A candidate may reference multiple IDs.
- If the chain is strongly aligned with GT → include its ID(s).
- If NO GT ID directly supports the chain:
    - Leave support_failure_id as empty list []
    - Provide full technical reasoning inside "insight"
    - Clearly explain why the inferred relationship is valid.


==============================
CONFIDENCE LEVEL
==============================

- "high"   → structure closely matches GT pattern
- "medium" → mostly aligned with minor adjustment
- "low"    → inferred without direct GT support but logically valid


==============================
INFERENCE REASON
==============================

Brief explanation of:
- How structure aligns with GT pattern
- Why cause → mode → effect chain is valid

Keep concise and engineering-focused.


==============================
GROUND TRUTH SIMILAR EXAMPLE
==============================

{gt_example}


==============================
STRUCTURE ANALYSIS (JSON)
==============================

{structure_analysis}


==============================
OUTPUT FORMAT (STRICT JSON ONLY)
==============================

{{
  "failure_candidates": [
    {{
      "failure_element": "...",
      "failure_function": "...",
      "failure_mode": "...",
      "failure_effect": "...",
      "failure_cause": "...",
      "confidence": "high | medium | low",
      "support_failure_id": [""],
      "inference_reason": "short explanation",
      "insight": "only required if no GT support; provide full technical reasoning"
    }}
  ]
}}

Return ONLY valid JSON.
Do not output any extra explanation.

"""
