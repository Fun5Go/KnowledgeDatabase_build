# Document Knowledge Graph

This folder contains the scripts used to build a Neo4j knowledge graph from specification, qualification, and factory acceptance test documents.

The workflow extracts document structure and technical content from PDF/DOCX files, creates embeddings and graph nodes, and links related documents and sections.

## Folder Overview

- `doc_KG/`: Document parsing, knowledge-graph construction, and cross-document linking scripts.

## Scripts

### `doc_KG/KG_specification.py`

Parses Functional Specification (FS) and Technical Specification (TS) documents. It extracts requirements, section hierarchies, tables, figures, and rationale content, then imports them into Neo4j with vector embeddings. It also creates `IMPLEMENT` links from TS requirements to their referenced FS requirements.

Set `FS_PATH`, `ESW_TS_PATH`, and `HW_TS_PATH` at the top of the script, then run:

```powershell
python KG/doc_KG/KG_specification.py
```

### `doc_KG/KG_qualification.py`

Parses Qualification Documents (QD) and Factory Acceptance Test (FAT) documents. It extracts qualification/test records, objectives, preconditions, requirement references, document sections, and test results, then imports them into Neo4j.

Set `ESW_QD_PATH`, `HW_QD_PATH`, `FAT_PATH`, and the document lists at the top of the script, then run:

```powershell
python KG/doc_KG/KG_qualification.py
```

When `DEBUG_SAVE_PARSED_JSON` is enabled, the parsed document data is also saved as JSON. Update `PARSED_JSON_PATH` if necessary.

### `doc_KG/doc_link.py`

Creates links between documents already imported into Neo4j. It connects FAT documents to FS documents with `ACCEPTED`, QD documents to TS documents with `VERIFIED`, and matching section nodes with `SHARED`.

Set the document paths and Neo4j connection values at the top of the script, then run:

```powershell
python KG/doc_KG/doc_link.py
```

Run this script after the specification and qualification graphs have been built.

## Recommended Workflow

```powershell
python KG/doc_KG/KG_specification.py
python KG/doc_KG/KG_qualification.py
python KG/doc_KG/doc_link.py
```

## Configuration

`KG_specification.py` reads the Neo4j connection from the project `.env` file:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

`KG_qualification.py` and `doc_link.py` currently define the Neo4j connection directly in the scripts. Update those values before use.

Install the project dependencies and Neo4j Python driver from the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install neo4j
```

The scripts currently contain absolute document paths. Update them for the local environment before running the workflow.
