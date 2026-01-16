iter_prompt_1 = """
SELECT FACTUAL SENTENCES from the D2–D4 sections of an 8D report.
These sentences will be used for failure knowledge base construction.

--------------------------------------------------
WHAT TO SELECT
--------------------------------------------------

From the following 8D sections (D2, D3, D4), select sentences that contain
high engineering value.

High-value signals include:
- Failure symptoms or abnormal behavior
  (e.g. current trip, unable to start, no output, endless loop)
- Failure modes or observable effects
- Conditions under which the failure occurred
  (environmental, operational, configuration-related)
- Occurrence information
  (frequency, intermittency, quantity, ratios)
- Validated or suspected root causes
- Interim containment or mitigation actions
- Verification, investigation, or diagnostic evidence

--------------------------------------------------
SENTENCE RULES
--------------------------------------------------

- One sentence = one factual statement
- Split sentences that contain multiple independent facts
- Remove references to figures, tables, section numbers, or manuals
- Do NOT infer, conclude, or summarize beyond the original text

Light rephrasing is allowed ONLY to:
- remove pronouns or unclear references
- normalize tense
- simplify wording without adding or removing facts

--------------------------------------------------
ANNOTATION RULES (IMPORTANT)
--------------------------------------------------

For EACH selected sentence, assign annotations as follows:

1) status (MANDATORY for all sentences):
   - "support": the sentence supports or indicates a failure behavior,
                cause, containment, or evidence
   - "exclude": the sentence explicitly rules out, disproves, or shows
                no effect of a suspected cause or mechanism
   - "suspect": the sentence suggests a possible cause or mechanism
                without confirmation

Status must reflect ONLY what is stated in the sentence.
Do NOT upgrade or downgrade certainty.

2) subject (ONLY for D4 sentences):

- subject represents the INVESTIGATION OBJECT,
  not the individual sentence topic.

- All D4 sentences that belong to the same investigation
  MUST share the EXACT SAME subject value.

- subject must be a short, stable noun phrase describing
  the hardware block, software component, or process
  being investigated.

- Do NOT create a new subject variation if the investigation
  object is the same.

Subject naming rules (priority order):

1. If a clear investigation header or label exists in D4
   (e.g. "eMMC", "PMIC power supplies", "SCFW software versions"),
   use that label as subject for all related sentences.

2. If no explicit header exists, choose ONE generic component-level, process-level
   name and reuse it consistently.

3. Prefer shorter, generic names over detailed or derived names.

For D2 and D3 sentences:
- Do NOT add subject (leave it empty)


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
      "status": "support", "exclude","suspect",
      "subject": ""
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