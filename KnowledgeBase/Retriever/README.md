# Retriever

This folder provides the query interface for the vector-based knowledge bases.

It supports both single-query and multi-query retrieval workflows, including semantic retrieval, sentence retrieval, graph-assisted retrieval, and structure-analysis based querying.

## Main Scripts

- `failure_query_tools.py`: Single-query semantic retrieval over the FMEA failure knowledge base.
- `sentence_query_tools.py`: Sentence-level retrieval utilities for 8D and supporting context.
- `SA_query.py`: Structure-analysis query interface for building failure chains.
- `SA_querybyid.py`: ID-based structure-analysis query helpers.
- `graph_query.py`: Graph-assisted retrieval and reranking utilities.
- `KG_KB.py`: Combined knowledge-graph and knowledge-base retrieval helpers.
- `bm25_baseline.py`: BM25 baseline retrieval implementation.
- `PPL_score.py`: Language-model perplexity scoring utilities.
- `evaluation/`: Evaluation scripts for retrieval quality and entity matching.

