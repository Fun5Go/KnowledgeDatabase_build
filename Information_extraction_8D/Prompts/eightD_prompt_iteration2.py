
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
- Multiple hypotheses and alternative cause paths are encouraged
- Do NOT over-integrate or over-conclude

====================
FAILURE RULES (D2 / D3 focused, DFMEA-aligned)
====================

Failure definition:
- A failure describes WHAT went wrong: a loss, degradation, or abnormal execution of an intended function.
- Failures describe functional behavior, NOT causes, tests, or investigations.

- system_name:
    ONE global system name (string or "")

- failure_element:
    Functional system element, component group, or subsystem
    (e.g. "power supply", "motor drive", "input protection circuit")

- failure_mode:
    Short engineering noun phrase describing the functional failure mode
    (2–6 words, no full sentences)
    Examples:
        - "no output voltage"
        - "unexpected shutdown"
        - "overcurrent during operation"

    NOT allowed:
        - test names or procedures
        - investigation activities
        - pure damage descriptions without functional meaning

- failure_effect:
    Observable consequence resulting from the failure mode
    (system impact, damage, or test outcome)
    Use "" if not explicitly stated.

- failure_level:
    One of: "system", "sub_system", "process"

Rules:
- Failure-related information MUST be supported mainly by signals with:
    entity_type = symptom OR occurrence
- Damage or destruction statements (e.g. "burnt component")
  SHOULD be treated as failure_effect unless a functional loss is stated
- Test setups and investigations MUST NOT define failure_mode
- If multiple descriptions exist, select the most general functional one

====================
ROOT CAUSE RULES (D4 focused, FMEA-aligned)
====================

Definition:
- Root causes represent suspected cause mechanisms, conditions, or contributing factors
- They are FMEA-style "Potential Cause / Mechanism", NOT final conclusions

General rules:
- Multiple, parallel, or layered cause paths are VALID


For each root cause:
- cause_level:
    One of:
        "design"         (architecture, protection concept, margins)
        "component"      (parts, materials, ratings)
        "process"        (manufacturing, assembly, configuration)
        "software"       (logic, timing, control)
        "test_condition" (stress, misuse, environment, deviations)
        "unknown"

- failure_cause:
    Short engineering noun phrase describing the suspected cause mechanism
    (2–6 words, no full sentences)
    Examples:
        - "insufficient input protection"
        - "excessive voltage stress"
        - "improper test setup"

- discipline_type:
    One of: "HW", "ESW", "MCH", "Other"

- cause_parent:
    ID of the parent cause, if applicable
    Used to form causal chains
    (e.g. test condition → electrical stress → component damage)

- inferred_insight:
    OPTIONAL.
    May summarize relationships between causes or signals
    MUST NOT assert certainty or final judgment

- confidence:
    low | medium | high
    Based on signal strength and assertion levels

Rules:
- Root causes MUST be supported mainly by signals with:
    entity_type = investigation OR root_cause_evidence
- Condition-based descriptions (usage, environment, test setup)
  are valid root causes even without a detailed physical mechanism
- Signals with assertion_level = suspected represent hypotheses only
- Do NOT merge distinct hypotheses unless explicitly stated
- Prevention controls, detection methods, and actions
  MUST NOT be modeled as failure modes or failure causes

====================
OUTPUT REFINEMENT RULES (MANDATORY)
====================
- failure_mode, failure_effect, and failure_cause MUST be:
    - short engineering noun phrases
    - concise, non-redundant
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
- describe abnormal behavior, damage, or deviations
- report measurements, investigations, or comparisons
- describe test conditions or environmental stress
- express hypotheses or possible explanations

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
  "failure_level": "",
  "supporting_entities": [
    {{
      "id": "",
      "text": "",
      "source_section": "D2 | D3 | D4",
      "annotations":{{
      "entity_type": "symptom | condition | occurrence | investigation | root_cause_evidence",
      "assertion_level": "observed | confirmed | ruled_out | suspected",
      "faithful_score": ,
      "faithful_type": ,
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
          "id": "",
          "text": "",
          "source_section": "D2 | D3 | D4",
          "annotations":{{
          "entity_type": "symptom | condition | occurrence | investigation | root_cause_evidence",
          "assertion_level": "observed | confirmed | ruled_out | suspected",
          "faithful_score": "",
          "faithful_type": "",
          }}
        }}
      ],
      "inferred_insight": "",
      "confidence": "low | medium | high",
    }}
  ]
}}
"""
#- Partial or intermediate causes are preferred over forced completeness