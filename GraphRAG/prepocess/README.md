This folder contains preprocessing scripts for GraphRAG.

Current script:
- `export_chunks_to_json.py`: export `FSChunk` and `TSChunk` nodes from Neo4j into JSON files while keeping `name`, `text`, and linked `rationales`.
- `match_ips3_effects.py`: only read `FSChunk` nodes and match `product_function` text against each complete node text. The script does not split text into sentences, and one node can keep multiple function matches.
- `match_element_modes_ts.py`: read `TSChunk` nodes and match a chosen `failure_element -> function -> mode` against complete node text plus rationale text.
- `match_element_causes_ts.py`: read `TSChunk` nodes and match a chosen `failure_element -> discipline -> cause` against complete node text plus rationale text.
- `ips3_effect_taxonomy.json`: editable taxonomy for `product function -> effects` without hardcoded per-effect keyword lists.
- `fmea_mode_cause_taxonomy.json`: taxonomy for `failure_element -> functions -> modes` and `causes -> discipline`.

Usage:

```powershell
python .\GraphRAG\prepocess\export_chunks_to_json.py
python .\GraphRAG\prepocess\match_ips3_effects.py
python .\GraphRAG\prepocess\match_element_modes_ts.py --element-name "Motor control"
python .\GraphRAG\prepocess\match_element_causes_ts.py --element-name "Motor control"
```
