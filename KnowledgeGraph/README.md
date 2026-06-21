# Knowledge Graph

This folder contains the scripts used to build, query, and expand a Neo4j knowledge graph from FMEA and 8D data.

FMEA and 8D records share semantic nodes such as `Element`, `Mode`, `Cause`, and `Effect`. This connects FMEA risk analysis with 8D failure evidence and corrective actions.

## Main Scripts

### `construction_v2_FMEA.py`

Builds the current FMEA knowledge graph from a JSONL file. It creates FMEA nodes, relationships, text embeddings, Neo4j constraints, and vector indexes.

Before running, set `JSONL_FILE` in the script to the FMEA JSONL file:

```powershell
python KnowledgeGraph/construction_v2_FMEA.py
```

### `KG_8D_construction.py`

Builds the 8D knowledge graph from a directory of JSON files. It adds failure modes, root causes, supporting sentences, D5 corrective actions, and D6 implementation information to the same Neo4j graph.

Before running, set `JSON_ROOT` in the script to the 8D JSON directory:

```powershell
python KnowledgeGraph/KG_8D_construction.py
```

### `KG_construction.py`

Legacy FMEA/8D vector knowledge-graph builder. It reads the prepared failure and sentence stores, creates vector indexes, and supports merging canonical cause, mode, and effect groups.

Update the input paths at the top of the script before running:

```powershell
python KnowledgeGraph/KG_construction.py
```

### `exporter.py`

Provides `Neo4jFMEAExporter`, which imports `fmea_field_store.json`, `entity_store.json`, and `fmea_edge_store.json` from a persistence directory into Neo4j. This file has no command-line entry point and is intended to be imported:

```python
from KnowledgeGraph.exporter import Neo4jFMEAExporter

exporter = Neo4jFMEAExporter(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    "path/to/persist_dir",
)
try:
    exporter.export_all()
finally:
    exporter.close()
```

## Query Scripts

### `query.py`

Contains basic Neo4j vector-search and graph-query functions, including semantic node search, failure-chain search, mode reasoning, and graph statistics.

Edit the example query in the `__main__` block and run:

```powershell
python KnowledgeGraph/query.py
```

### `SA_query.py`

Maps structured FMEA input to existing KG nodes through semantic retrieval and returns ranked failure candidates with their graph context.

Edit `structure_input_motorcontrol` and the retrieval settings in the `__main__` block, then run:

```powershell
python KnowledgeGraph/SA_query.py
```

## `KG Expand` Scripts

These scripts are experimental workflows for exporting the graph, training link-prediction models, and inferring missing failure relations.

### `KG Expand/get_triples.py`

Exports `CAUSES` and `LEADS_TO` relationships from Neo4j to `triples.tsv`, and exports node types, text, and embeddings to `nodes.tsv`.

Run it from the desired output directory because the TSV output paths are relative:

```powershell
python "KnowledgeGraph/KG Expand/get_triples.py"
```

### `KG Expand/KG_embedding.py`

Trains a PyKEEN knowledge-graph embedding model or uses a trained model to predict possible causes and effects and write them back to Neo4j.

Update the file paths and model settings at the top of the script, then use one of the commands:

```powershell
python "KnowledgeGraph/KG Expand/KG_embedding.py" train
python "KnowledgeGraph/KG Expand/KG_embedding.py" expand
```

### `KG Expand/KG_softmatch.py`

Builds similarity links between semantically close modes, causes, and effects, then propagates possible causes and effects through those links.

Update `JSON_FILE` and the similarity thresholds before running:

```powershell
python "KnowledgeGraph/KG Expand/KG_softmatch.py"
```

### `KG Expand/RGCN_train.py`

Trains a weighted R-GCN with a ComplEx decoder for `Cause -> Mode` and `Mode -> Effect` link prediction. It reads `nodes.tsv` and `triples.tsv` and saves the best model checkpoint.

Update `NODES_FILE`, `TRIPLES_FILE`, `SAVE_PATH`, and the training settings before running:

```powershell
python "KnowledgeGraph/KG Expand/RGCN_train.py"
```

### `KG Expand/RGCN_inference.py`

Loads an R-GCN model and predicts modes for a cause or effects for a mode.

Update the node, triple, and model paths and the test node IDs in `main()`, then run:

```powershell
python "KnowledgeGraph/KG Expand/RGCN_inference.py"
```

### `KG Expand/SA_infer.py`

Combines Neo4j semantic retrieval with the trained weighted R-GCN/ComplEx model. It maps structured FMEA queries to KG nodes and predicts cause-mode-effect links.

Update `NODE_FILE`, `TRIPLE_FILE`, `MODEL_PATH`, and `structure_input_motorcontrol` before running:

```powershell
python "KnowledgeGraph/KG Expand/SA_infer.py"
```

## Neo4j Configuration

The scripts read the Neo4j connection from the project `.env` file:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

Install the project dependencies and the Neo4j Python driver before running the scripts:

```powershell
python -m pip install -r requirements.txt
python -m pip install neo4j
```

Several scripts currently contain absolute input or model paths. Update these paths for the local environment before use.
