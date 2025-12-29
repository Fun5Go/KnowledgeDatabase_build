iter_prompt_1 = """
SELECT FACTUAL SENTENCES from the D2-D4 section in 8D report.
These sentences will be used for a failure knowledge base and failure entity extraction

--------------------------------------------------
WHAT TO SELECT
--------------------------------------------------

Select atomic factual sentences and assign ONE entity_type per sentence:

1. "symptom" 
   - Observable failure behavior
   - What does not work or behaves incorrectly

2.  "condition"
   - MUST describe an external operational or environmental state
   - Temperature, power cycling, stress, usage state.
   - MUST describe an external state, not software logic or system behavior

3.  "occurrence"
   - Frequency or probability of failure
   - Intermittent vs permanent behavior
   - Ratios such as 1/400x

4.  "investigation" 
   - Diagnostic findings
   - What was investigated and confirmed working
   - What was ruled out as a cause

5. "root_cause_evidence" 
   - Physical or logical evidence of a cause
   - Manufacturing defects, soldering issues
   - Process deviations linked to failure

--------------------------------------------------
ASSERTION LEVEL
--------------------------------------------------

For each sentence, assign an assertion_level describing
how the statement is asserted in the source text.

Use ONLY one of the following values:

- observed     : directly seen or measured
- confirmed    : verified through testing or repetition
- ruled_out    : explicitly stated as not being the cause
- suspected    : hypothesis or correlation, not yet proven

Rules:
- Use “confirmed” ONLY if explicitly stated
- Use “observed” for physical findings unless confirmation is stated
- Do NOT treat assertion_level as probability or final conclusion


--------------------------------------------------
SENTENCE RULES
--------------------------------------------------

- One sentence = one fact
- Split sentences with multiple facts
- Remove references to figures, tables, or manuals
- Light rephrasing allowed only for clarity
- Do not infer or summarize

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return a JSON array.
Each item must follow this schema exactly:

{{
  "selected_sentences": [
    {{
      "entity_type": "symptom | condition | occurrence | investigation | root_cause_evidence",
      "text": "Concise factual sentence",
      "source_section": "D2 | D3 | D4",
      "assertion_level": "observed | confirmed | ruled_out | suspected"
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
- In D4, subsection headers (e.g. "2.4.1") define an analysis topic
  (potential component or failure cause).

- Do NOT select subsection headers as sentences.

- Select factual sentences under each subsection and associate them
  with the nearest preceding subsection header as context.
{d4}
"""

iter_prompt_2 = """
Task: Consolidate failures from extracted text signals.

From the input signals:
1. Identify ONE distinct technical failure
2. Group all related signals under this failure
3. Extract failure mode, effect, and root cause ONLY if explicitly supported by signals

====================
FAILURE RULES (D2/D3 focused)
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
    Observable consequence of the failure (or null)
- failure_level:
    One of: "system", "sub_system", "process"

Rules:
- failure-related information must be supported mainly by signals with entity_type = symptom or occurrence
- Do NOT infer failure mode or effect beyond what is explicitly stated

====================
ROOT CAUSE RULES (D4 focused)
====================
Multiple root causes are allowed.

For each root cause:
- cause_level:
    One of: 'design', 'process', 'test' or 'component'
- failure_cause:
    Root cause of the failure (or null)
    Must be explicitly supported by signals
- discipline_type:
    One of: "HW", "ESW", "MCH", "Other"
- inferred_insight:
    MAY summarize relationships between signals
    MUST NOT introduce new facts
- confidence:
    low | medium | high

Rules:
- Root causes must be supported mainly by signals with entity_type = root_cause_evidence or investigation
- Signals with assertion_level = suspected indicate hypotheses, not confirmed causes
- Do NOT upgrade suspected causes to confirmed without explicit support

====================
OUTPUT REFINEMENT RULES (MANDATORY)
====================
- failure_mode, failure_effect, and failure_cause MUST be short engineering labels
- Use noun phrases, NOT full sentences
- Length: typically 2–6 words
- Do NOT include verbs like "is", "was", "due to", "because", "caused by"
- Do NOT restate evidence text

====================
SUPPORTING ENTITIES RULES
====================
- Every failure and every root cause MUST be supported by one or more signals
- text MUST be copied EXACTLY from the selected sentence text
- id MUST reference the original sentence id
- source_section MUST match the original sentence
- entity_type MUST be copied from the original sentence entity_type
- assertion_level MUST be copied from the original sentence assertion_level

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
      "entity_type": "",
      "assertion_level": ""
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
          "entity_type": "",
          "assertion_level": ""
        }}
      ],
      "inferred_insight": "",
      "confidence": "low | medium | high"
    }}
  ]
}}
"""