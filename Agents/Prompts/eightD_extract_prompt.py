D2_prompt = """
You are an expert reliability and systems engineer. Extract structured D2 information from the following 8D report text.

The D2 section may contain multiple distinct failures.
Your goal is to:
- Identify the overall System_name ONCE for the entire D2 section.
- Identify all failures, each with its own System_element, mode, effect, and raw context.
- Combine global symptoms into one field: problem_symptoms.

----------------------------------------
RULES
----------------------------------------
1. Determine a SINGLE System_name for the entire D2 section (broad subsystem).
2. Identify every separate failure described in the text.
3. Assign a unique failure_id in order of appearance: "F1", "F2", ...
4. For each failure, extract ONLY:
      - system_element
      - failure_mode
      - failure_effect (if stated clearly; otherwise null)
      - raw_context (exact text describing that failure)
5. DO NOT include System_name inside each failure.
6. The top-level object MUST follow this schema:

{{
  "system_name": "",
  "problem_symptoms": "",
  "failures": [
    {{
      "failure_id": "F1",
      "system_element": "",
      "failure_mode": "",
      "failure_effect": "",
      "raw_context": ""
    }}
  ],
  "raw_context": "<<Full original D2 text>>"
}}

7. If any field cannot be determined, set it to null.
8. Do NOT merge separate failures.
9. Output MUST be strictly valid JSON only.

----------------------------------------
D2 TEXT:
{content}

Now extract all failures and return ONLY JSON.
"""

D4_prompt = """
You are an expert root cause analysis engineer. Extract structured D4 information from the following 8D report text.

The D4 section may contain multiple root causes, each corresponding to a different failure from D2.

----------------------------------------
RULES
----------------------------------------
1. Identify all root causes described in the text.
2. According to D2 info to infer the root causes corresponding to the elements and failure modes
3. Assign a failure_id to each cause such as "F1", "F2", etc.
   - If the D2 failure_id mapping is known, use it.
   - If not known, assign in order of appearance (F1, F2, F3…).
4. For each root cause, extract:
      - Should fully follow the elements and failure modes
      - failure_id: The ID of the corresponding failure.
      - discipline_type:
            "HW"  → Hardware
            "ESW" → Embedded software / firmware
            "MCH" → Mechanics
            "Other" → Anything not in HW/ESW/MCH
      - root_cause: A one-sentence engineering cause explanation.
      - raw_context: The exact text segment describing this cause.
5. Never invent causes not explicitly represented in the text.
6. The final output MUST be strictly valid JSON only.

----------------------------------------
OUTPUT FORMAT (strict):
{{
  "root_causes": [
    {{
      "failure_id": "F1",
      "discipline_type": "HW",
      "root_cause": "",
      "raw_context": ""
      "impacted_element:""
    }}
  ],
  "raw_context": "<<Full original D4 text>>"
}}
----------------------------------------
System elements and failures:
{d2_context}

D4 TEXT:
{content}

Now extract ALL root causes and return ONLY JSON.

"""