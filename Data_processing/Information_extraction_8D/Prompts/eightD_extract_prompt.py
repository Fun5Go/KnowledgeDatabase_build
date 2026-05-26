D2_prompt = """
The D2 section named "Define the problem and symptoms" may contain one or multiple distinct failures.
Your goal is to:
- Identify the overall System_name ONCE for the entire D2 section.
- Identify all failures, each with its own System_element, mode, effect, and raw context.
- Combine global symptoms into one field: problem_symptoms.
Field definitions:
- system_element:
  - One of the elements or components in the system.
  - Choose a concrete phycical component, not the whole product.
  - Be more aware of the top paragraph of the D2 text, where system elements are typically introduced.

- failure_mode:
  - The way or manner in which a product could fail.
  - Must be described in technical terms and NOT as a symptom noticeable by the customer.
  - Focus on how the system or subsystem malfunctions.

- failure_effect:
  - The corresponding effects of the failure on the system, function, or user.
  - Describe what happens when this failure mode occurs.
  (Attention: Each unique system_element must have exactly ONE failure mode and ONE failure effect in the D2 section. And the elements should be different and unique)

----------------------------------------
RULES
----------------------------------------
1. Determine a SINGLE System_name for the entire D2 section (broad subsystem).
2. Identify every separate failure described in the text.
3. For each failure, extract ONLY:
      - system_element
      - failure_mode
      - failure_effect (if stated clearly; otherwise null)
      - raw_context (exact text describing that failure)
4. DO NOT include System_name inside each failure.
5. The top-level object MUST follow this schema:

{{
  "system_name": "",
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

6. If any field cannot be determined, set it to null.
7. Output MUST be strictly valid JSON only.

----------------------------------------
D2 TEXT:
{content}

Now extract all failures and return ONLY JSON.
"""

D4_prompt = """
Your task is to extract **structured D4 root cause information** and map it directly to the **system_element + failure_mode pairs from D2**.
A single root cause must correspond to exactly one (and only one) system_element + failure_mode pair.
If you cannot fine the matchment of root cause return null.

============================================================
RULES & OBJECTIVES
============================================================

1. Identify all **true engineering root causes** described in the D4 section.
   - Focus especially on the **last paragraph(s)** of each technical subsection, 
     where conclusions and causes are usually stated.

2. For each system_element + failure_mode listed in the D2 context:
   - Determine whether D4 provides a cause.
   - If yes → extract ONE concise cause for each system_element + failure_mode pair.

3. For each identified root cause, output:
      - discipline_type:
            "HW"  → hardware-related cause  
            "ESW" → embedded software/firmware cause  
            "MCH" → mechanical cause  
            "Other" → none of the above  
      - impacted_element: system_element from D2  
      - root_cause: concise, single-sentence engineering explanation  
            *Should match the style of failure_mode: short, factual, technical.*  
      - raw_context: exact copy-paste of the text describing the cause 

5. **Do NOT invent root causes.**  
   - Only extract what is explicitly stated OR strongly implied in the D4 text.

6. **Do NOT repeat the failure_mode in the cause.**  
   - The cause must explain WHY the failure happens technically.

7. The final output must be STRICTLY VALID JSON.

============================================================
OUTPUT FORMAT (STRICT)
============================================================
{{
  "root_causes": [
    {{
      "discipline_type": "HW",
      "impacted_element": "",
      "root_cause": "",
      "raw_context": ""
    }}
  ],
  "raw_context": "<<Full original D4 text>>"
}}

============================================================
D2 SYSTEM ELEMENTS & FAILURE MODES
============================================================
{d2_context}

============================================================
D4 TEXT (SOURCE)
============================================================
{content}

Extract ALL root causes and output ONLY valid JSON.

"""