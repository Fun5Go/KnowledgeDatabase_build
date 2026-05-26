# FMEA to JSON

This folder converts FMEA Excel files into structured JSON files for downstream retrieval, knowledge-base construction, and analysis.

## Main Scripts

- `run.py`: Batch entry point. It scans the configured input folder, processes `.xlsm` and `.xlsx` files, and writes one JSON file per FMEA workbook.
- `xlsm_parser.py`: Parser for DFMEA `.xlsm` workbooks.
- `xlsx_parser.py`: Parser for older `.xlsx` FMEA workbooks.
- `common_utils.py`: Shared helpers for metadata extraction, discipline parsing, Excel cell handling, and FMEA index loading.

## Inputs and Outputs

The default paths are configured at the top of `run.py`:

- `INPUT_DIR`: Folder containing raw FMEA Excel files.
- `OUTPUT_DIR`: Folder where converted JSON files are written.
- `FMEA_INDEX_PATH`: Metadata index used to enrich parsed FMEA records.

Update these constants before running the batch conversion in a different environment.

## Usage

Run the batch converter from the repository root:

```bash
python -m Data_processing.fmea_to_json.run
```

The script processes only `.xlsm` and `.xlsx` files. Unsupported files in the input folder are skipped.
