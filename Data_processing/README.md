# Data Processing

This folder contains the data preparation and document-processing workflows used before retrieval, knowledge-base construction, and downstream FMEA/8D analysis.

## Folder Overview

- `DATA_collection_process/`: Data collection utilities. These scripts deduplicate file metadata and copy source files from their original paths into the project data folders.
- `fmea_to_json/`: Parses FMEA worksheets into structured JSON files. It supports both `.xlsm` DFMEA files and older `.xlsx` FMEA files.
- `Failure_classification/`: Classifies motor-drives-domain failures in 8D and FMEA data. It converts 8D DOCX documents into JSON metadata and converts selected FMEA JSON records into JSONL files.
- `Information_extraction_8D/`: Extracts failure information from 8D JSON files, including evidence sentence selection, failure-chain construction, and extraction evaluation.
- `doc_part_demo/`: Parsing tests and demos for development documents such as specifications, qualification documents, and test reports.

## Import Style

Use the `Data_processing` package path when importing modules from this folder:

```python
from Data_processing.fmea_to_json.common_utils import load_fmea_index
from Data_processing.Information_extraction_8D.main.eight_D_agent import build_8d_case_from_json
```
