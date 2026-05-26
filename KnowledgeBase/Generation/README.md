# Generation

This folder reconstructs failure entities from structure-analysis texts.

The generation workflow uses retrieved knowledge-base context, structure-analysis input, and LLM prompts to build candidate failure chains such as failure mode, failure cause, and failure effect.

## Main Scripts

- `main.py`: Main generation workflow using retrieval and graph-query context.
- `main_single.py`: Single-case generation workflow.
- `LLM_function.py`: LangChain tools and prompt execution for failure reconstruction.
- `prompt_failure_generation.py`: Prompt templates for pure, RAG-based, and fill-in generation modes.
- `failure_schema.py`: Pydantic schemas for structured generation outputs.
- `utils.py`: Formatting, chain construction, and scoring helpers.
- `evaluation/`: Evaluation utilities for generated failure candidates.

