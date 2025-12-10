Prompt = """
This is a test stage, just give me a fake example

{{
  "system_name": "", # Define a electronic subassembly
  "failures": [
    {{
      "system_element": "",
      "failure_mode": "",  
      "failure_effect": "", 
      "discipline_type": "",
      "root_cause": "",
      "infer_context": ""
    }}
  ],
}}
,etc.

The raw context below:

D2 raw context:
{d2_text}

D3 raw context:
{d3_text}

D4 raw context:
{d4_text}

"""