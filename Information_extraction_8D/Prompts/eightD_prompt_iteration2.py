
iter_prompt_2= """
Task: Organize failure-related and cause-related information from extracted text signals.

Primary goal:
- Identify ONE distinct technical failure
- Preserve potentially valuable cause-related information
- NOT to determine the final true root cause

This task is for information preservation and structuring, not final judgment.

====================
GENERAL PRINCIPLES
====================
- Prefer inclusion over exclusion
- Multiple parallel or alternative cause paths are VALID (max 4)
- Preserve uncertainty
- Do NOT resolve contradictions

====================
FAILURE RULES (D2 / D3 focused)
====================

Failure definition:
A failure describes WHAT went wrong:
- loss of function
- degraded performance
- abnormal execution of intended function
- pyhsical damage

Failure describes BEHAVIOR, not cause or investigation.

Fields:
- system_name:
    ONE global system name, or ""

- failure_element:
    Functional system element, component group, or subsystem
    Example: "power supply", "motor drive", "input protection circuit", interface)

- failure_mode:
    Short engineering noun phrase describing the functional failure mode
    (2–6 words, no full sentences)
    Examples:
        - "no output voltage"
        - "unexpected shutdown"
        - "overcurrent during operation"
    Explicitly NOT a failure_mode:
      - component damage descriptions
      - investigation outcomes
      - test names or procedures
      - causes or stress mechanisms

- failure_effect:
    Observable consequence resulting from the failure mode
    (system impact, damage, or test outcome)
    If the effect is not explicitly stated as a consequence,set failure_effect = "".
    Do NOT infer system impact.

- failure_level:
    One of:system | sub_system | process

Failure supporting_entities:
- MUST reference signals from section D2 or D3
- MUST describe observed failure behavior
- MUST NOT reference D4-only signals

====================
ROOT CAUSE RULES (D4 focused, FMEA-aligned)
====================
IMPORTANT:
For D4-derived causes, your task is NOT to analyze technical correctness.

You MUST:
- Reflect suspected or investigated cause mechanisms as written
- Treat long investigations or reasoning chains as a single cause candidate
- Avoid decomposing complex investigations into multiple causes unless explicitly stated

If a cause requires deep technical interpretation, keep it coarse and mark confidence = low or medium.


Definition:
Root causes are POTENTIAL cause mechanisms or contributing factors,
not final conclusions.

They answer: "What could have led to the failure?"

General rules:
- Multiple, parallel, or layered cause paths are VALID


For each root cause:
- cause_level:
    One of:
        "design"         (architecture, protection concept, margins)
        "component"      (parts, materials, electronics)
        "process"        (manufacturing, assembly, configuration)
        "software"       (logic, timing, control,algorithm)
        "test_condition" (stress, misuse, environment, deviations)
        "unknown"

- failure_cause:
    Short engineering noun phrase describing the suspected cause mechanism
    (2–6 words, no full sentences)
    Examples:
        - "Incorrect driving of blanking circuit - ESW"
        - "Fault in communication cable (short/open) - Other"
        - "Pre-charge active device failes - HW"
        - "Magnet alignment / magnetization - MCH"

- discipline_type:
    One of: "HW", "ESW", "MCH", "Other"

- cause_parent:
  - Use ONLY if the text EXPLICITLY states a sequential cause relationship
    using causal language such as:
    "leads to", "results in", "causes", "due to", "as a consequence of"
  - Do NOT infer or construct causal chains based on engineering logic
  - If not explicitly stated, leave cause_parent empty

- inferred_insight:(OPTIONAL)
    May summarize relationships between causes or signals
    MUST NOT assert certainty or final judgment

- confidence
  - low | medium | high

Confidence guidance:
- high:
    Supported by confirmed or ruled_out investigation signals
- medium:
    Supported by multiple suspected or observed investigation signals
- low:
    Supported by a single suspected signal or hypothesis

Root cause supporting_entities:
- MUST reference D4 investigation or root_cause_evidence signals
- Do NOT reference  D2 symptom-only signals

====================
OUTPUT REFINEMENT RULES (MANDATORY)
====================
- failure_mode, failure_effect, failure_cause:
  - noun phrases only
  - concise and non-redundant
  - no full sentences

- Do NOT restate evidence text
- Do NOT introduce new information

====================
SUPPORTING ENTITIES RULES
====================
- ONLY reference sentence IDs
- DO NOT add other information to the supporting entities except for sentence IDs
- Only use sentence IDs that appear in the input signals

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
  "failure_level": "",
  "supporting_entities": [
    {{
      "sentence_id": "",
      }}
    }}
  ],
  "root_causes": [
    {{
      "cause_level": "design | process | component | software | test_condition | unknown",
      "failure_cause": "",
      "discipline_type": "HW | ESW | MCH | Other",
      "cause_parent": "",
      "supporting_entities": [
        {{
          "sentence_id": "",
        }}
      ],
      "inferred_insight": "",
      "confidence": "low | medium | high",
    }}
  ]
}}
"""
#- Partial or intermediate causes are preferred over forced completeness