# Failure Classification

This folder contains scripts for identifying the product or failure domain of 8D documents and filtering FMEA records by domain-related keywords.

## Main Components

- `domain_identification.py`: Batch workflow for reading 8D documents, classifying their domain, and writing domain-labeled metadata stores.
- `LLM_TOOLs.py`: LangChain tools for parsing 8D DOCX sections and calling the domain-identification LLM prompt.
- `llm.py`: LLM backend initialization for OpenAI-compatible and local Ollama models.
- `product_schema.py`: Pydantic output schemas for structured domain classification.
- `prompt_identify_classify.py`: Prompt template used by the domain-identification agent.
- `FMEA/keywords_recall.py`: Keyword-based FMEA recall script for extracting motor-drive-related failure records.
- `FMEA/Key_words_list.py`: Keyword lists and regex helpers used by the FMEA recall script.

## Inputs and Outputs

The default batch paths are configured in the `if __name__ == "__main__"` blocks of the scripts:

- 8D source documents are read from the configured raw 8D folder.
- Document metadata is loaded from the configured `8D_with_filename.json` file.
- Domain-labeled JSON outputs are written to the configured metadata store folder.
- FMEA keyword recall reads converted FMEA JSON files and writes JSONL output.

Update these path constants before running the scripts in a different environment.

## Usage

Run domain classification from the repository root:

```bash
python -m Data_processing.Failure_classification.domain_identification
```

Run FMEA keyword recall from the repository root:

```bash
python -m Data_processing.Failure_classification.FMEA.keywords_recall
```

