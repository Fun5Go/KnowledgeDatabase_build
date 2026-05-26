# Knowledge Base

This folder contains the workflows for building vector-based knowledge bases, querying them, and reconstructing failure entities from retrieval results and structure-analysis text.

## Folder Overview

- `Demonstration/`: Demonstration scripts for retrieval and failure reconstruction used in the Midway presentation.
- `Generation/`: Failure entity reconstruction from structure-analysis texts.
- `JSON_FMEA_KB/`: Converts raw FMEA JSONL / JSON records into a vector-based FMEA knowledge base.
- `JSON8D_KB/`: Converts raw 8D JSON records into a vector-based 8D knowledge base.
- `Retriever/`: Query interfaces for single-query and multi-query retrieval over the knowledge bases.

## Import Style

Use the `KnowledgeBase` package path when importing modules from this folder:

```python
from KnowledgeBase.Retriever.SA_query import build_failure_chains_from_structure
from KnowledgeBase.JSON_FMEA_KB.query_fmea import retrieve_failures
```

