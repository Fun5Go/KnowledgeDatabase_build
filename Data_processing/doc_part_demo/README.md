# Document Part Demo

This folder contains parsing tests and demo scripts for development documents.

It is used to experiment with extracting and structuring information from engineering documents such as functional specifications, technical specifications, qualification documents, and test reports.

## Main Scripts

- `KG_specification.py`: Parses functional and technical specification PDFs and prepares requirement chunks for knowledge-graph import.
- `KG_qualification.py`: Parses qualification document PDFs and prepares QD/TST chunks for knowledge-graph import.
- `QD_process.py`: Standalone parser for qualification document test sections.
- `knowledge_base.py`: Experimental document chunking and vector-store workflow.
- `demo.py`: Minimal document loading, splitting, embedding, and retrieval demo.
- `retrieval.py`, `graph_retrieval.py`, `SA_query.py`: Retrieval and graph-query experiments.

## Test Documents

The included PDF files are local parsing test inputs for development and validation. The scripts use paths relative to this folder where possible.
