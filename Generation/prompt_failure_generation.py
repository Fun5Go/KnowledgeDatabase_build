failure_inference_prompt = """
You are given:
1) Multiple Ground Truth Similar Example retrieved from the Failure Knowledge Base.
   This example is the most semantically similar failure chain to the current structure.
   It should be treated as the PRIMARY reference pattern.

2) A Structure Analysis describing failure elements and possible failure modes,
   failure effects, and failure causes. Their relationships are not correctly matched currently.
Your task:
Infer the MOST RELEVANT and LOGICALLY CONSISTENT FMEA failure chains
for the given Structure Analysis accroding to the ground truth example

----------------------------------------
CRITICAL INFERENCE PRINCIPLES
----------------------------------------

1. The Ground Truth Similar Examples are your MAIN guidance.
   - Prioritize structural similarity to the example.
   - Reuse its causal logic pattern if applicable.
   - Do NOT ignore the example.

2. DO NOT over-infer or force new relationships.
   - No aggressive reasoning.
   - No speculative failure physics.
   - No creative expansion beyond what is structurally supported.

3. Only perform LIGHT inference:
   - Minor logical corrections of mismatched cause/effect
   - Small completion of missing function wording if clearly implied
   - Alignment of structure terms to the GT example pattern

4. If the Structure Analysis does not strongly support a chain,
   you should:
   - Avoid generating it
   - Or assign lower confidence

----------------------------------------
FMEA LOGIC REQUIREMENT
----------------------------------------

Each candidate must follow correct FMEA causality:

   failure_element
      → failure_function
         → failure_mode
            → failure_effect
               ← caused by ← failure_cause

Cause must logically lead to Mode.
Mode must logically lead to Effect.
No reversed or circular logic.

----------------------------------------
support_failure_id RULE
----------------------------------------

- Use the failure IDs from the Ground Truth Example that support your reasoning.
- If adapting from the GT example, include its ID.
- Do NOT fabricate IDs.
- A failure candidate can be supported by multiple IDs.

----------------------------------------
confidence RULE
----------------------------------------

- "high" → structure closely matches GT example
- "medium" → mostly aligned with light adjustment
- "low" → weak alignment but still plausible

----------------------------------------
inference_reason RULE
----------------------------------------

Briefly explain:
- How the structure aligns with the GT example
- Why the causal chain is valid
Keep concise and engineering-focused.

----------------------------------------
Ground Truth Similar Example:
{gt_example}

----------------------------------------
Structure Analysis:
{structure_analysis}

----------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)

{{
  "failure_candidates": [
    {{
      "failure_element": "...",
      "failure_function": "...",
      "failure_mode": "...",
      "failure_effect": "...",
      "failure_cause": "...",
      "confidence": "high | medium | low",
      "support_failure_id": ["id_xxx"],
      "inference_reason": "technical reasoning"
    }}
  ]
}}

Return ONLY valid JSON.
Do not output any extra explanation.
"""
