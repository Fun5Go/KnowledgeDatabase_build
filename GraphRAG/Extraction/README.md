# GraphRAG Extraction

This folder contains the GraphRAG workflow used to retrieve and select technical-document chunks related to FMEA failure modes, causes, and effects.

The main extraction agent reviews retrieved chunks, selects grounded engineering evidence, and classifies it as `control`, `cause`, or `specification`. A separate QD/FAT workflow selects detection and control evidence from qualification documents and factory acceptance tests.

Run the commands below from the project root. The current imports require the `GraphRAG` directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\GraphRAG"
```

## Main Extraction Workflow

### `main.py`

Main command-line entry point. It builds mode and cause queries from the configured FMEA structure, retrieves document chunks, runs the chunk-selection agent, and saves the results as JSON.

```powershell
# List available queries
python -m GraphRAG.Extraction.main --list-queries

# Run one query
python -m GraphRAG.Extraction.main --query-number 1

# Run locally without an external LLM
python -m GraphRAG.Extraction.main --query-number 1 --placeholder-llm

# Save one result file per query
python -m GraphRAG.Extraction.main --save-oneprocess --oneprocess-output-dir GraphRAG/Extraction/top60
```

Use `python -m GraphRAG.Extraction.main --help` for all output and batch options.

### `chunk_selection_agent.py`

Defines `ChunkSelectionAgent`. It prepares prompts, enforces input-token limits, batches candidates, normalizes LLM output, validates exact evidence spans, and provides a placeholder selector for tests and offline development.

### `chunk_selection_prompt.txt`

Prompt template used by `ChunkSelectionAgent` to select evidence and assign evidence labels and support capability.

### `demo_query.py`

Small command-line wrapper for running one extraction query by number.

```powershell
python -m GraphRAG.Extraction.demo_query --list-queries
python -m GraphRAG.Extraction.demo_query 1 --placeholder-llm
```

## QD/FAT Detection-Control Workflow

### `main_qd_detection_control.py`

Retrieves QD and FAT chunks for FMEA modes, causes, and effects, then selects evidence describing detection, verification, or control coverage.

```powershell
# List QD/FAT queries
python -m GraphRAG.Extraction.main_qd_detection_control --list-queries

# Run one query with hybrid retrieval
python -m GraphRAG.Extraction.main_qd_detection_control --query-number 1 --retrieval-mode hybrid --top-k 20

# Test without an external LLM
python -m GraphRAG.Extraction.main_qd_detection_control --query-number 1 --placeholder-llm
```

### `qd_detection_control_agent.py`

Defines the QD/FAT selection agent, including prompt construction, token control, response normalization, and placeholder behavior.

### `qd_detection_control_prompt.txt`

Prompt template used to select QD/FAT detection and control evidence.

### `merge_qd_selected_chunks.py`

Merges selected chunks from per-query QD/FAT result files into one human-review JSON file. It can also evaluate a completed review file.

```powershell
python -m GraphRAG.Extraction.merge_qd_selected_chunks --folder path/to/qd_results --output qd_review.json
python -m GraphRAG.Extraction.merge_qd_selected_chunks --evaluate-review qd_review.json --output qd_evaluation.json
```

## Neo4j Connection Scripts

### `connect_document_failure_graph.py`

Connects selected document chunks to FMEA `Mode` and `Cause` nodes in Neo4j. Relationship types are based on the selected evidence label, and each relationship stores its evidence span, support capability, justification, and trace information.

Run a dry check first, then write the connections:

```powershell
python -m GraphRAG.Extraction.connect_document_failure_graph --input-dir GraphRAG/Extraction --dry-run
python -m GraphRAG.Extraction.connect_document_failure_graph --input-dir GraphRAG/Extraction
```

By default, the script removes legacy `upstream`, `downstream`, and `self` evidence edges. Use `--keep-legacy-edges` to retain them.

### `neo4j_graph_test_connection.py`

Legacy/test connector for QD/FAT results. It creates `test_for` relationships from selected QD/FAT chunks to FMEA nodes.

Set `JSON_ROOT` in the script before running:

```powershell
python -m GraphRAG.Extraction.neo4j_graph_test_connection
```

## Analysis and Evaluation

### `retrieval_evaluation.py`

Builds retrieval ground truth from selected chunks and compares dense, sparse, or hybrid retrieval configurations using recall and ranking metrics.

```powershell
python -m GraphRAG.Extraction.retrieval_evaluation --init-ground-truth
python -m GraphRAG.Extraction.retrieval_evaluation --help
```

### `qd_retrieval_evaluation.py`

Evaluates QD/FAT retrieval coverage at configurable K values using QD review data.

```powershell
python -m GraphRAG.Extraction.qd_retrieval_evaluation --help
```

### `one_step_evaluation.py`

Evaluates aggregate one-step chunk-selection results against integrated human-review labels and writes JSON and CSV metrics.

```powershell
python -m GraphRAG.Extraction.one_step_evaluation --help
```

### `evaluate_oneprocess_results.py`

Evaluates per-query `oneprocess*.json` selection files against the final extraction review ground truth.

```powershell
python -m GraphRAG.Extraction.evaluate_oneprocess_results --help
```

### `find_mode_cause_chunk_overlaps.py`

Finds selected document chunks that occur in both function-mode and cause query results.

```powershell
python -m GraphRAG.Extraction.find_mode_cause_chunk_overlaps --input-dir GraphRAG/Extraction
```

## Tests and Package Files

### `test_chunk_selection_agent.py`

Tests response normalization, evidence-span validation, batching, connected-chunk handling, and placeholder selection.

```powershell
python -m pytest GraphRAG/Extraction/test_chunk_selection_agent.py
```

### `__init__.py`

Defines the Extraction package and lazily exports the QD selected-chunk merge and evaluation helpers.

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

Optional settings:

```dotenv
GRAPHRAG_EXTRACTION_MAX_INPUT_TOKENS=8000
GRAPHRAG_EXTRACTION_USE_PLACEHOLDER=0
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=GraphRAGExtraction
```

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install neo4j
```

Several evaluation scripts use experiment-specific default paths. Run a script with `--help` or update its path constants when using a different result or review dataset.
