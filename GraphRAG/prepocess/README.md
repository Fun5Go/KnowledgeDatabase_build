This folder contains preprocessing scripts for GraphRAG.

Current script:
- `export_chunks_to_json.py`: export `FSChunk` and `TSChunk` nodes from Neo4j into JSON files while keeping `name`, `text`, and linked `rationales`.
- `match_ips3_effects.py`: read exported chunks, split node text and rationale text into sentences, first match `product_function` with relaxed preprocessing, then match `effect` within the corresponding function scope.
- `ips3_effect_taxonomy.json`: editable taxonomy for `product function -> effects` without hardcoded per-effect keyword lists.

Usage:

```powershell
python .\GraphRAG\prepocess\export_chunks_to_json.py
python .\GraphRAG\prepocess\match_ips3_effects.py
```
