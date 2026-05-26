# JSON 8D Knowledge Base

This folder converts raw 8D JSON records into a vector-based 8D knowledge base.

It builds searchable stores for 8D failure descriptions, causes, evidence sentences, and document metadata.

## Main Scripts

- `ingest_8d.py`: Ingests structured 8D JSON records into knowledge-base stores.
- `build_single.py`: Builds a knowledge base from a single configured 8D JSON input.
- `build_sentence_kb.py`: Builds sentence-level vector stores for 8D evidence.
- `main.py`: Main runner for 8D knowledge-base construction.
- `query.py`: Query helpers for retrieving 8D failures, causes, and sentences.
- `kb_structure.py`: Data structures and vector-store wrappers used by the 8D knowledge base.
- `Evaluation/`: Semantic, structural, and visualization evaluation scripts.

## Outputs

Generated vector stores and metadata are written to configured `kb_data` folders.

