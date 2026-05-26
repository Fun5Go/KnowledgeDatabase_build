# JSON FMEA Knowledge Base

This folder converts raw FMEA JSONL / JSON records into a vector-based FMEA knowledge base.

It builds searchable stores for failure modes, causes, sentence-level descriptions, metadata, and related FMEA structures.

## Main Scripts

- `ingest_fmea.py`: Ingests FMEA JSON or JSONL records into knowledge-base stores.
- `build_single.py`: Builds a knowledge base from a single configured input.
- `fmea_batch_process.py`: Batch-oriented FMEA knowledge-base construction.
- `query_fmea.py`: Query helpers for retrieving FMEA failures and causes.
- `sentence_builder.py`: Builds textual sentence representations from structured FMEA chains.
- `kb_structure.py`: Data structures and vector-store wrappers used by the FMEA knowledge base.
- `evaluation/`: Retrieval, semantic, structural, and grouping evaluation scripts.
- `KG/`: Graph-oriented utilities for FMEA knowledge-base data.

## Outputs

Generated vector stores and metadata are written to configured `kb_data` folders.

