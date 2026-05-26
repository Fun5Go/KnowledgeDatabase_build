# DATA Collection Process

This folder contains utility scripts for preparing and collecting source file metadata before downstream FMEA and 8D processing.

## Main Scripts

- `duplicate_remove.py`: Deduplicates file information records so repeated entries are removed before further processing.
- `copy_*.py`: Copies files from their original source paths into the configured project data folders.

## Notes

- The `copy_*.py` scripts use path lists or metadata JSON files as input and write updated file-location metadata after copying.
- Check the path constants inside each script before running them in a new environment.
