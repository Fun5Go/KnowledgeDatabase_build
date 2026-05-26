D2_prompt = """
You are a reliability engineer. Extract structured D2 information.

TASK:
Given the D2 text, output:
Return ONLY valid JSON.
{{
  "system_name": "", # Define a electronic subassembly
  "problem_symptoms": "",
  "failures": [
    {{
      "system_element": "",
      "failure_mode": "",  
      "failure_effect": "", 
      "raw_context": ""
    }}
  ],
  "raw_context": "<<Full original D2 text>>"
}}
Field definitions:
- system_element:
  - One of the elements in the system (a subsystem).
  - Choose a concrete subsystem/component, not the whole product.
  - Be more aware of the top paragraph of the D2 text, where system elements are typically introduced.

- failure_mode:
  - The way or manner in which a product could fail.
  - Must be described in technical terms and NOT as a symptom noticeable by the customer.
  - Focus on how the system or subsystem malfunctions.

- failure_effect:
  - The corresponding effects of the failure on the system, function, or user.
  - Describe what happens when this failure mode occurs.


RULES:
- Identify all distinct failures in order of appearance
- For each failure extract:
    • system_element  
    • failure_mode  
    • failure_effect (null if unclear)  
    • raw_context = exact text describing that failure  
- Do NOT merge failures.
- Do NOT repeat system_name inside failures.
- Use null for unknown fields.
- JSON only.

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