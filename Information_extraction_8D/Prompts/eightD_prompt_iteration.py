iter_prompt_1 = """
From the following 8D sections (D2–D4), select sentences that contain
high engineering value.
High-value signals include:
- failure symptoms or abnormal behavior
- failure modes or effects
- validated or suspected root causes
- containment 
- verification or diagnostic evidence

Return JSON with:
- selected_sentences:
    - sentence (verbatim)
    - source: D2 | D3 | D4
    - signal_type: symptom | failure | cause | action | evidence
    - confidence (0.0–1.0)

D2:
{d2}

D3:
{d3}

D4:
{d4}
"""

iter_prompt_2 = """
Task: Iteration 2 – consolidate failures from extracted text signals.

From the given signals, identify distinct technical failures and output them
in the STRICT JSON format below.

Rules:
- Use short engineering phrases for all fields
- Only `inferred_insight` may be a full sentence
- Do NOT invent information
- If unclear, set the field to null

Failure definition:
- a global system_name (ONE for the entire document)
- system_element: concrete component or subsystem (not full product)
- failure_mode: how the element fails (technical, not symptom)
- failure_effect: consequence of the failure (or null)
- root_cause: validated or most likely technical cause
- discipline_type: "HW" | "ESW" | "MCH" | "Other"

Text entities:
- Attach one or more supporting_entities per failure
- Copy text EXACTLY from the signals (no paraphrasing)
- source_section: D2 | D3 | D4 | D5 | D6
- entity_type: symptom | cause | action | observation | context

Input signals:
{signals}

Output (STRICT JSON):
{{
  "system_name": "",
  "failures": [
    {{
      "system_element": "",
      "failure_mode": "",
      "failure_effect": "",
      "discipline_type":  (Should be one of followings)               
            "HW"  → hardware-related cause   
            "ESW" → embedded software/firmware cause  
            "MCH" → mechanical cause  
            "Other" → none of the above ",
      "root_cause": "",
      "supporting_entities": [
        {{
          "text": "",
          "source_section": "",
          "entity_type": ""
        }}
      ],
      "inferred_insight": ""
    }}
  ]
}}
"""