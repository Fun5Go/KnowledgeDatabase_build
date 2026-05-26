# Information Extraction 8D

This folder contains the 8D document extraction workflow. It parses raw 8D document metadata, extracts important D-sections, selects evidence sentences, builds structured failure chains, and evaluates extraction quality.

## Folder Structure

- `main/`: Main workflows and runnable entry points.
  - `eight_D_agent.py`: Core pipeline for building structured 8D cases from parsed JSON.
  - `build_8d_main.py`: Single-file runner that writes extracted 8D case JSON and selected sentence JSON.
  - `end_to_end.py`: Batch-oriented runner for processing folders of 8D raw metadata JSON files.
  - `llm.py`: LLM backend initialization.
  - `test_demo.py` and `langextract_demo.py`: Experimental/demo scripts.
- `tools/`: Document parsing, section extraction, and sentence normalization utilities.
- `Prompts/`: Prompt templates for D2/D4 extraction, failure identification, and iterative sentence selection.
- `Schemas/`: Pydantic models for 8D cases, sections, failure chains, and selected sentences.
- `Evaluation/`: Evaluation scripts for selected sentence coverage, faithfulness, compression, and extraction quality.

## Inputs and Outputs

The workflow generally expects raw 8D metadata JSON files produced from DOCX parsing. Default paths are defined inside the runner scripts:

- `SENTENCE_OUTPUT_DIR`: Output folder for selected supporting sentences.
- `FAILURE_OUTPUT_DIR`: Output folder for extracted failure-identification results.
- `OUTPUT_DIR`: Output folder for final structured 8D case JSON files.

Update these constants before running the workflow in a new environment.

## Usage

Run a single 8D extraction workflow from the repository root:

```bash
python -m Data_processing.Information_extraction_8D.main.build_8d_main
```

Run the batch workflow:

```bash
python -m Data_processing.Information_extraction_8D.main.end_to_end
```

Run extraction evaluation:

```bash
python -m Data_processing.Information_extraction_8D.Evaluation.main
```

