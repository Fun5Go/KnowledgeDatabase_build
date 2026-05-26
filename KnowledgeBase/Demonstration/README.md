# Demonstration

This folder is used to demonstrate retrieval and failure reconstruction for the Midway presentation.

It combines knowledge-base retrieval with failure reconstruction prompts to show how candidate failure chains can be generated from structure-analysis input.

## Main Scripts

- `main_retrieval.py`: Runs retrieval-focused demonstrations.
- `main_LLM.py`: Runs LLM-focused demonstration logic.
- `RAG.py`: Combines retrieval context with failure reconstruction generation.
- `retrieval.py`: Retrieval helper functions used by the demos.
- `entity.py`: Entity definitions and utilities for demonstration workflows.

## Outputs

The demo scripts may write candidate failure outputs such as `failure_candidates_RAG.json` or `failure_candidates_pure.json` into this folder.

