# Demonstration: Knowledge Reuse for New Product Development

This folder sketches the code structure for the demonstration scenario where a
new-product SFMEA team queries the connected KG for historical failure knowledge
and navigates to supporting development-document context.

## Goal

Given a new product concern such as a failure mode, cause, or effect, the
workflow should:

1. Normalize the user query and optional product context.
2. Retrieve semantically similar historical failures from the connected KG.
3. Expand each retrieved failure to connected specifications, design
   descriptions, rationales, verification evidence, and acceptance information.
4. Synthesize reusable engineering guidance:
   - relevant risks to consider earlier,
   - more appropriate prevention/detection controls,
   - targeted specifications, verification checks, or tests.
5. Return both structured JSON and a readable engineering report.

## Proposed Files

`schemas.py`
: Defines shared data contracts: `NewProductQuery`,
  `HistoricalFailureCandidate`, `SupportingContext`, and
  `ReuseRecommendation`.

`query_builder.py`
: Converts raw new-product text into retrieval-ready inputs. It will normalize
  query text, build search strings, create graph filters, and optionally expand
  failure-related terms.

`historical_failure_retriever.py`
: Retrieves semantically similar historical failure modes, causes, or effects.
  Later this should call the existing Neo4j/vector/full-text retrieval utilities
  in `GraphRAG`.

`context_expander.py`
: Traverses graph relationships from selected historical failures to connected
  development documents, then groups the context by document type and exposes
  trace paths for justification.

`reuse_synthesizer.py`
: Converts retrieved failures plus graph context into proposed reusable risks,
  controls, specifications, and tests.

`report_builder.py`
: Produces human-readable Markdown reports and structured JSON outputs.

`workflow.py`
: Coordinates the full demonstration pipeline and also exposes partial stages:
  retrieval only, context expansion only, and synthesis only.

`main.py`
: Future command-line entry point for running the demonstration from a query.

## Workflow

```text
raw new-product query
        |
        v
query_builder.normalize_user_query()
        |
        v
HistoricalFailureRetriever.hybrid_retrieve()
        |
        v
ContextExpander.expand_for_candidates()
        |
        v
ReuseSynthesizer.build_recommendation()
        |
        v
report_builder.build_json_report()
report_builder.build_markdown_report()
```

## Implementation Plan

Step 1: Connect `HistoricalFailureRetriever` to existing GraphRAG retrieval
classes.

Step 2: Define Cypher traversals in `ContextExpander` for specifications,
design descriptions, rationales, verification, and acceptance information.

Step 3: Implement `ReuseSynthesizer` as a rule-based or LLM-assisted evidence
synthesis component.

Step 4: Add `main.py` arguments and save reports under a demonstration output
folder.

