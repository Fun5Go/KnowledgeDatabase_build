# Demonstration: FMEA Assistant for Interpretation and Review

This folder sketches the code structure for the scenario where connected FMEA
and development-document graphs help users interpret and review FMEA content.

## Goal

Given one FMEA row, field, failure attribution, or control measure, the assistant
should:

1. Parse and normalize the FMEA text.
2. Retrieve linked specifications, design constraints, rationales, tests, and
   acceptance evidence from the connected KG.
3. Interpret abbreviated or domain-specific FMEA wording using that evidence.
4. Let users inspect why a failure mode, cause, effect, or control is linked to
   supporting document context.
5. Review whether the failure chain is sufficiently supported by product design
   and verification information.
6. Flag consistency issues and missing support for engineering review.

## Proposed Files

`schemas.py`
: Defines `FMEAReviewItem`, `LinkedEvidence`, `InterpretationResult`,
  `ReviewFinding`, and `AssistanceReview`.

`fmea_text_parser.py`
: Converts a raw FMEA table row or free-text concern into normalized fields,
  then detects abbreviations and domain-specific terms.

`evidence_linker.py`
: Retrieves evidence connected to the FMEA item, including specifications,
  design constraints, rationales, verification evidence, and acceptance
  information.

`interpretation_assistant.py`
: Explains failure attribution, control measures, and abbreviated FMEA terms
  using linked evidence.

`review_checker.py`
: Checks whether the failure chain and controls are sufficiently supported by
  product design and verification information.

`consistency_checker.py`
: Detects contradictions, weak links, or incomplete chains between the FMEA row
  and connected KG evidence.

`report_builder.py`
: Produces structured JSON and readable Markdown review reports.

`workflow.py`
: Coordinates parsing, evidence linking, interpretation, support review, and
  consistency checking.

`main.py`
: Future command-line entry point for running the assistant demonstration.

## Workflow

```text
raw FMEA row / selected field
        |
        v
fmea_text_parser.parse_fmea_row()
        |
        v
EvidenceLinker.retrieve_direct_links()
EvidenceLinker.retrieve_field_evidence()
        |
        v
InterpretationAssistant.interpret_item()
        |
        v
ReviewChecker.review_failure_chain_support()
ConsistencyChecker.check_field_consistency()
        |
        v
report_builder.build_review_json()
report_builder.build_review_markdown()
```

## Implementation Plan

Step 1: Map existing FMEA graph labels and document graph labels to evidence
types.

Step 2: Implement `EvidenceLinker` Cypher traversals for direct links and
field-specific evidence inspection.

Step 3: Implement interpretation with a small rule-based term resolver first,
then optionally add an LLM-assisted explanation layer.

Step 4: Implement review and consistency checks with explicit support criteria
for design evidence, verification evidence, and failure-chain completeness.

