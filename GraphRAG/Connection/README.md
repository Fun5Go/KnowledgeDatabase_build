# GraphRAG Connection

This folder contains the workflow used to connect FMEA failure information with evidence retrieved from technical-document knowledge graphs.

The workflow has two main stages:

1. Rerank retrieved chunks as `support`, `suspect`, or `irrelevant`.
2. Extract exact evidence spans and classify their relationship to the FMEA mode, cause, or effect.

The resulting connections can be validated, evaluated, and written back to Neo4j.

Run the commands below from the project root. The current module imports also require the `GraphRAG` directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\GraphRAG"
```

## Main Scripts

### `main.py`

Main command-line entry point. It builds queries from the configured FMEA structure, retrieves document chunks, runs the Connection workflow, and saves JSON results.

Common commands:

```powershell
# List the available FMEA queries
python -m GraphRAG.Connection.main --list-queries

# Run the complete workflow for one query
python -m GraphRAG.Connection.main --query-number 1 --stage auto

# Run with a prebuilt Connection payload
python -m GraphRAG.Connection.main --input path/to/payload.json --stage auto --output result.json

# Run without an external LLM, for local testing
python -m GraphRAG.Connection.main --query-number 1 --placeholder-llm
```

Use `python -m GraphRAG.Connection.main --help` for all retrieval, rank-range, batch, and output options.

### `connection_workflow.py`

Implements `ConnectionWorkflow`, the two-stage orchestration layer. It supports `rerank`, `extract`, and `auto` modes, batches agent calls, validates outputs, and builds result summaries. This module is normally used by `main.py`.

### `connection_agents.py`

Defines the LLM agents for chunk reranking and evidence-relation extraction. It also normalizes model responses and provides deterministic placeholder agents for tests and offline development.

### `neo4j_graph_connection.py`

Reads completed Connection result files and creates relationships from technical-document chunk nodes to FMEA `Mode`, `Cause`, or `Effect` nodes in Neo4j.

Set `JSON_ROOT` in the script to the result directory, then run:

```powershell
python -m GraphRAG.Connection.neo4j_graph_connection
```

### `validate_results.py`

Validates rerank and extraction result files, including required fields and exact evidence spans.

```powershell
python -m GraphRAG.Connection.validate_results --results-dir GraphRAG/Connection/results
```

Add `--report validation_report.json` to save the validation report.

### `review_retrieval_evaluation.py`

Benchmarks GraphRAG retrieval configurations against reviewed true-positive and false-negative chunks. It can compare dense, sparse, or hybrid retrieval, cross-encoder reranking, and section-tag bonuses.

```powershell
python -m GraphRAG.Connection.review_retrieval_evaluation --help
```

### `test_connection_workflow.py`

Contains workflow, prompt, normalization, and validation tests. The tests use placeholder agents and do not require an LLM call.

```powershell
python -m pytest GraphRAG/Connection/test_connection_workflow.py
```

## Support Modules

- `prompts.py`: Prompt templates for reranking and evidence extraction.
- `schemas.py`: Allowed tags, relation types, support levels, direction values, and output schema definitions.
- `validators.py`: Validation and summary-building helpers for agent outputs.
- `__init__.py`: Package exports for the Connection module.

These modules are imported by the main workflow and are not normally run directly.

## Evaluation Scripts

The `evaluation/` folder contains tools for preparing human-review data and calculating retrieval, reranking, extraction, and relationship metrics.

### `evaluation/build_split_review_json.py`

Converts result files into separate rerank-review and extraction-review JSON files.

```powershell
python -m GraphRAG.Connection.evaluation.build_split_review_json
```

### `evaluation/merge_human_review_results.py`

Merges rerank and extraction outputs into integrated per-query human-review files.

```powershell
python -m GraphRAG.Connection.evaluation.merge_human_review_results
```

### `evaluation/main.py`

Runs LLM judges over human-reviewed chunks and reports conflicts with existing review labels.

```powershell
python -m GraphRAG.Connection.evaluation.main --input-dir path/to/reviews --output-dir path/to/conflicts
```

### `evaluation/support_suspect_judge_agent.py`

LLM judge for chunks tagged as `support` or `suspect`. It checks selection decisions and extracted relation information. This module is used by `evaluation/main.py`.

### `evaluation/irrelevant_judge_agent.py`

LLM judge for chunks tagged as `irrelevant`. It identifies potentially relevant chunks that may have been rejected incorrectly. This module is used by `evaluation/main.py`.

### `evaluation/retrieval_evaluation.py`

Calculates retrieval metrics at configurable values of K from human-review files.

```powershell
python -m GraphRAG.Connection.evaluation.retrieval_evaluation --help
```

### `evaluation/rerank_evaluation.py`

Evaluates the `support`, `suspect`, and `irrelevant` reranking decisions.

```powershell
python -m GraphRAG.Connection.evaluation.rerank_evaluation --help
```

### `evaluation/extract_evaluation.py`

Evaluates whether the workflow selected the correct chunks for evidence extraction.

```powershell
python -m GraphRAG.Connection.evaluation.extract_evaluation --help
```

### `evaluation/relationship_evaluation.py`

Evaluates the correctness and completeness of relationships extracted from selected chunks.

```powershell
python -m GraphRAG.Connection.evaluation.relationship_evaluation --help
```

### `evaluation/final_extract_review_evaluation.py`

Calculates final extraction and relationship metrics from the completed human-review dataset.

```powershell
python -m GraphRAG.Connection.evaluation.final_extract_review_evaluation --help
```

### `evaluation/GroundTruth/build_ground_truth_support_template.py`

Builds a failure-text ground-truth template from support chunks in raw-text and integrated-text rerank results.

```powershell
python -m GraphRAG.Connection.evaluation.GroundTruth.build_ground_truth_support_template --help
```

## Configuration

Create or update the project `.env` file:

```dotenv
LLM_BACKEND=openai
LLM_MODEL=azure/gpt-5.2
OPENAI_API_BASE=http://your-openai-compatible-endpoint/v1
OPENAI_API_KEY=your_api_key

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

Optional LangSmith tracing variables:

```dotenv
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=GraphRAGConnection
```

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install neo4j
```

## Typical Workflow

```powershell
# 1. Run retrieval, reranking, and evidence extraction
python -m GraphRAG.Connection.main --query-number 1 --stage auto

# 2. Validate generated result files
python -m GraphRAG.Connection.validate_results --results-dir GraphRAG/Connection/results

# 3. Review or evaluate the results with scripts in evaluation/

# 4. Write accepted evidence relationships to Neo4j
python -m GraphRAG.Connection.neo4j_graph_connection
```

Many evaluation scripts contain default input and output directories based on the current experiment. Use their `--help` options or update their path constants before running them on a different dataset.
