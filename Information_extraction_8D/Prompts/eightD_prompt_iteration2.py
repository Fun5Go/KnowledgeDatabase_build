
iter_prompt_2 = """
Task: Organize failure-related and cause-related information from extracted text signals.

Primary goal:
- Identify ONE distinct technical failure
- Select the potential valuable signal sentence to identify the failure causes
- NOT to decide the final true root cause

The purpose of this task is information preservation, not final judgment.

====================
GENERAL PRINCIPLES
====================
- Prefer including potentially relevant signals over excluding them
- Root causes in this task represent cause-related information clusters, NOT final conclusions

====================
FAILURE RULES (D2 / D3 focused)
====================
- system_name:
    ONE global system name (string or "")
- failure_element:
    Functional system element or subsystem
    (e.g. "power supply", "motor drive", "PCB")
- failure_mode:
    Operational malfunction during use
    (e.g. "short circuit", "no start", "intermittent operation")
- failure_effect:
    Observable consequence of the failure (or "")
- failure_level:
    One of: "system", "sub_system", "process"

Rules:
- Failure-related information must be supported mainly by signals with:
    entity_type = symptom OR occurrence
- Do NOT infer failure mode or effect beyond what is explicitly stated
- If multiple formulations exist, choose the most general one

====================
ROOT CAUSE RULES (D4 focused)
====================
IMPORTANT:
- root_causes are NOT final conclusions
- They represent organized collections of cause-related information
- Multiple alternative or progressive cause paths are ENCOURAGED

For each root cause:
- cause_level:
    One of: "design", "process", "component", "software", "test_condition", "unknown"
- failure_cause:
    Short engineering label describing the suspected cause mechanism
- discipline_type:
    One of: "HW", "ESW", "MCH", "Other"
-cause_parent:
    ID of the parent cause (if any)
    Should be a chain relation: 
    General example: "component cause: capacitor/ resistor" -> "design cause: input protection not implemented" 
- inferred_insight:
    MAY summarize relationships between signals
    MAY describe progressive or layered relationships
        (e.g. test condition → electrical stress → component damage)
- confidence:
    low | medium | high

Rules:
- Root causes must be supported mainly by signals with:
    entity_type = investigation OR root_cause_evidence
- Signals with assertion_level = suspected represent hypotheses only
- Do NOT collapse multiple hypotheses into one unless explicitly stated

Quantity guidance:
- If cause-related signals exist, generate 1–3 root_causes
- Even partial, competing, or intermediate explanations are valid

====================
OUTPUT REFINEMENT RULES (MANDATORY)
====================
- failure_mode, failure_effect, and failure_cause MUST be:
    - short engineering noun phrases
    - typically 2–6 words
- Do NOT restate evidence text
- Do NOT write full sentences as labels

====================
SUPPORTING ENTITIES RULES
====================
- Every failure and every root cause MUST be supported by one or more signals
- text MUST be copied EXACTLY from the selected sentence text
- id MUST reference the original sentence id
- source_section MUST match the original sentence
- annotations MUST be copied exactly from the input signal

====================
SIGNAL SELECTION GUIDANCE
====================
Include signals if they:
- describe abnormal conditions, deviations, or stress factors
- report measurements, comparisons, or investigations
- mention test setups, test deviations, or environmental conditions
- express hypotheses, suspicions, or possible explanations

Exclusion should be conservative.

====================
INPUT SIGNAL FORMAT
====================
Each signal contains:
- id
- text
- source_section (D2 | D3 | D4)
- entity_type (symptom | condition | occurrence | investigation | root_cause_evidence)
- assertion_level (observed | confirmed | ruled_out | suspected)

Input signals:
{signals}

====================
OUTPUT (STRICT JSON)
====================

{{
  "system_name": "",
  "failure_element": "",
  "failure_mode": "",
  "failure_effect": "",
  "failure_level": "sub_system",
  "supporting_entities": [
    {{
      "id": "",
      "text": "",
      "source_section": "D2 | D3 | D4",
      "annotations":{{
      "entity_type": "symptom | condition | occurrence | investigation | root_cause_evidence",
      "assertion_level": "observed | confirmed | ruled_out | suspected"
      "faithful_score": 
      "faithful_type": 
      }}
    }}
  ],
  "root_causes": [
    {{
      "cause_level": "design | process | component | software | test_condition | unknown",
      "failure_cause": "",
      "discipline_type": "HW | ESW | MCH | Other",
      "supporting_entities": [
        {{
          "id": "",
          "text": "",
          "source_section": "D2 | D3 | D4",
          "annotations":{{
          "entity_type": "symptom | condition | occurrence | investigation | root_cause_evidence",
          "assertion_level": "observed | confirmed | ruled_out | suspected"
          "faithful_score": ""
          "faithful_type": ""
          }}
        }}
      ],
      "inferred_insight": "",
      "confidence": "low | medium | high"
    }}
  ]
}}
"""