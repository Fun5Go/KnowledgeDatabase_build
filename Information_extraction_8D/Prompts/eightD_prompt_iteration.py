iter_prompt_1 = """
SELECT FACTUAL SENTENCES from the D2-D4 section in 8D report.
These sentences will be used for a failure knowledge base construction.

--------------------------------------------------
WHAT TO SELECT
--------------------------------------------------

From the following 8D sections (D2,D3,D4), select sentences that contain
high engineering value.
High-value signals include:
- failure symptoms or abnormal behavior: 
    Example: "current trip", "motor swapped", "broken down", "board damage", "unable to start", 
              "endless loop", "phase fails"
- failure modes or effects
- Failure happened condition (external operational or environmental state)
- Occurrence information (frequency, intermittency, quantity)
- validated or suspected root causes
- containment
- verification or diagnostic evidence



--------------------------------------------------
SENTENCE RULES
--------------------------------------------------

- One sentence = one fact
- Split sentences with multiple facts
- Remove references to figures, tables, or manuals
- Do not infer or summarize

Light rephrasing is allowed ONLY to:
- remove pronouns or references
- normalize tense
- simplify wording without adding or removing facts


--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return a JSON array.
Each item must follow this schema exactly:

{{
  "selected_sentences": [
    {{
      "text": "Concise factual sentence",
      "source_section": "D2 | D3 | D4",
      "annotations":{{
      }}
    }}
  ]
}}

--------------------------------------------------
INPUT
--------------------------------------------------
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
3. Extract failure mode, effect, and root cause ONLY if explicitly supported by signals

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
- Failure-related information MUST be inferred mainly from D2/D3 signals
- Do NOT infer failure effect beyond what is explicitly stated

====================
ROOT CAUSE RULES (D4 focused)
====================
! Multiple root causes are allowed !

For each root cause:
- cause_level:
    One of: "design", "process", "component", "software", "test_condition", "unknown"
- failure_cause:
    Specific electronics, software, or mechanical engineering term
    The cause must directly lead to the failure mode
- discipline_type:
    One of: "HW", "ESW", "MCH", "Other"
- inferred_insight:
    MAY summarize relationships between provided signals
    MUST NOT introduce new facts
- confidence:
    low | medium | high

Rules:
- Root causes MUST be inferred mainly from D4 signals
- Signals with assertion_level = suspected indicate hypotheses only
- Do NOT upgrade suspected causes to confirmed without explicit confirmation

====================
OUTPUT REFINEMENT RULES (MANDATORY)
====================
- failure_mode, failure_effect, and failure_cause MUST be short engineering labels
- Use noun phrases, NOT full sentences
- Typical length: 2–6 words
- Do NOT restate evidence text

====================
SUPPORTING ENTITIES RULES
====================
- Every failure and every root cause MUST be supported by one or more signals
- text MUST be copied EXACTLY from the selected sentence text
- id MUST reference the original sentence id
- source_section MUST match the original section (D2 | D3 | D4)

====================
INPUT SIGNAL FORMAT
====================
Each signal contains:
- id
- text
- source_section (D2 | D3 | D4)
- failthness

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
      "source_section": "D2 | D3 | D4"
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
          "source_section": "D4"
        }}
      ],
      "inferred_insight": "",
      "confidence": "low | medium | high"
    }}
  ]
}}
"""