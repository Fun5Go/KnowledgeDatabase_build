Prompt = """
Read the following 8D content (D2 problem definition + D3 interim actions + D4 root cause analysis) as **one single combined engineering document**.
Your task is to extract all technical failures and their validated root causes from the whole text. 
--- D2 Raw ---
{d2}

--- D3 Raw ---
{d3}

--- D4 Raw ---
{d4}


============================================================
OBJECTIVE
============================================================
(Except for **infer_context**, all other fields must be expressed as **short engineering phrases**, not long sentences.)
From the combined 8D content (D2–D4), identify:
- The global system_name (ONE for the entire document)
- Every distinct technical failure, each with:
    - system_element
    - failure_mode
    - failure_effect
    - raw_context (text describing this failure)
- The validated engineering root cause for each failure
- An infer_context field that synthesizes helpful information from anywhere in the text
  (symptoms, containment actions, diagnostic clues, etc.)

============================================================
FAILURE IDENTIFICATION RULES
============================================================
A *failure* must be defined by:
- system_element  
    - A concrete subsystem or physical component decomposed from the system_name
    - Do NOT use the whole product  
- failure_mode  
    - How the system_element malfunctions in technical terms  
    - be described in technical terms and NOT as a symptom
- failure_effect  
    - What happens when the failure occurs  
    - If unclear, set to null  

============================================================
Few-shots examples: 
============================================================


============================================================
FINAL OUTPUT FORMAT (STRICT JSON) 
============================================================
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
      "infer_context": "",
      "raw_context": ""
    }}
  ]
}}

"""



# Extract failures from this 8D documents

# OUTPUT FORMAT (STRICT JSON):
# {{
#   "system_name": "",
#   "failures": [
#     {{
#       "system_element": "",
#       "failure_mode": "",
#       "failure_effect": "",
#       "discipline_type":    (Should be one of followings)               
#             "HW"  → hardware-related cause   
#             "ESW" → embedded software/firmware cause  
#             "MCH" → mechanical cause  
#             "Other" → none of the above ",
#       "root_cause": "",
#       "infer_context": ""
#     }}
#   ]
# }}