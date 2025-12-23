iter_prompt_1 = """
From the following 8D sections (D2,D3,D4), select sentences that contain
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
    - id: S1, S2, ... (sequential in each section)
    - signal_type: symptom | failure | cause | action | evidence
    - confidence (0.0–1.0)

D2:Define problem and symptoms
{d2}

D3: Provide a quick fix / Interim Containment Plan
{d3}

D4:Analyze root cause
{d4}
"""

iter_prompt_2 = """
Task: Consolidate failures from extracted text signals.

From the input signals:
1. Identify ONE distinct technical failure
2. Group all related signals under this failure
3. Extract failure mode, effect, and root cause ONLY if explicitly supported

====================
FAILURE RULES (D2/D3 focused)
====================
- system_name:
    ONE global system name (string or "")
- failure_element:
    Functional system element or subsystem
    (e.g. "power supply", "memory interface", "PCB")
    If it is a manufacturing defect or process step, fill the failure level with "process"
- failure_mode:
    Operational malfunction during use
    (e.g. "intermittent connection", "short circuit", "no boot")
- failure_effect:
    Observable consequence of the failure (or null)
- failure_level:
    One of: "system", "sub_system", "process"


====================
ROOT CAUSE RULES (D4 focused, could be a list of causes with various subsection: e.g. 2.4.1)
====================
- cause_level:
    One of: "design", "process", "component", "software", "test_condition", "unknown"
- failure_cause:
    Root cause of the failure (or null), could be software (algorithm), hardware (e.g. electronics componnet), mechenical, or process related
- discipline_type:
    One of: "HW", "ESW", "MCH", "Other"
- inferred_insight:
    MAY summarize relationships
    MUST NOT add new facts
- confidence:
    low/medium/high

====================
SUPPORTING ENTITIES RULES
====================
- Every failure and every root cause MUST be supported
- text MUST be copied EXACTLY from a signal
- id MUST reference the original signal id
- entity_type:
    symptom | cause | action | observation | context

Input signals:
{signals}

Output (STRICT JSON):
{{
  "system_name": "",
  "failure_element": "",
  "failure_mode": "",
  "failure_effect": "",
  "failure_level": "sub_system",
  "supporting_entities": [  # The support text for failure inference is mostly from D2 and D3
    {{
        "text": "",
        "source_section": "D2 | D3 | D4",
        "id": "",
        "entity_type": ""
    }}
  "root_causes": [
    {{
      "cause_level": "design",
      "failure_cause": "",
      "discipline_type": "HW",
      "supporting_entities": [ # The support text for aause inference is mostly from D4 and D3
        {{
          "text": "",
          "source_section": "D2 | D3 | D4",
          "id": "",
          "entity_type": ""
        }}
      ],
      "inferred_insight": "",
      "confidence": 0.0
    }}
  ]
}}
"""