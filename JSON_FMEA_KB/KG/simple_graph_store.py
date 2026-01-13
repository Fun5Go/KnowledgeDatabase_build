from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

from .graph_store import GraphStore


class SimpleGraphStore(GraphStore):
    """
    A lightweight local Knowledge Graph store:
    - Nodes: node_id -> {"label": str, "props": dict}
    - Edges: adjacency with (src, rel, dst) keys and props
    Persists to a single JSON file in persist_dir.
    """

    def __init__(self, persist_dir: Path, filename: str = "fmea_graph_store.json"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.persist_dir / filename

        self._nodes: Dict[str, Dict[str, Any]] = {}
        # key: "src|rel|dst"
        self._edges: Dict[str, Dict[str, Any]] = {}

        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._nodes = data.get("nodes", {})
            self._edges = data.get("edges", {})

    # ---------- helpers ----------
    @staticmethod
    def _edge_key(src_id: str, rel: str, dst_id: str) -> str:
        return f"{src_id}|{rel}|{dst_id}"

    def upsert_node(self, node_id: str, label: str, props: Dict[str, Any]) -> None:
        cur = self._nodes.get(node_id)
        if cur is None:
            self._nodes[node_id] = {"label": label, "props": dict(props)}
        else:
            # merge props
            cur["label"] = label or cur.get("label", "")
            cur_props = cur.get("props", {})
            cur_props.update(props)
            cur["props"] = cur_props

    def upsert_edge(self, src_id: str, rel: str, dst_id: str, props: Optional[Dict[str, Any]] = None) -> None:
        key = self._edge_key(src_id, rel, dst_id)
        cur = self._edges.get(key)
        if cur is None:
            self._edges[key] = {
                "src": src_id,
                "rel": rel,
                "dst": dst_id,
                "props": dict(props or {}),
            }
        else:
            cur_props = cur.get("props", {})
            cur_props.update(props or {})
            cur["props"] = cur_props
            self._edges[key] = cur

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str, rel: Optional[str] = None, direction: str = "out") -> List[str]:
        out: Set[str] = set()
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")

        for e in self._edges.values():
            if direction == "out":
                if e["src"] != node_id:
                    continue
                if rel is not None and e["rel"] != rel:
                    continue
                out.add(e["dst"])
            else:
                if e["dst"] != node_id:
                    continue
                if rel is not None and e["rel"] != rel:
                    continue
                out.add(e["src"])
        return list(out)

    def edges(self, node_id: str, rel: Optional[str] = None, direction: str = "out") -> List[Tuple[str, str, str, Dict[str, Any]]]:
        res: List[Tuple[str, str, str, Dict[str, Any]]] = []
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'")
        for e in self._edges.values():
            if direction == "out":
                if e["src"] != node_id:
                    continue
                if rel is not None and e["rel"] != rel:
                    continue
                res.append((e["src"], e["rel"], e["dst"], e.get("props", {})))
            else:
                if e["dst"] != node_id:
                    continue
                if rel is not None and e["rel"] != rel:
                    continue
                res.append((e["src"], e["rel"], e["dst"], e.get("props", {})))
        return res

    def find_nodes(self, label: str, prop_key: str, prop_value: str) -> List[str]:
        hits: List[str] = []
        for nid, n in self._nodes.items():
            if n.get("label") != label:
                continue
            props = n.get("props", {})
            if str(props.get(prop_key, "")) == str(prop_value):
                hits.append(nid)
        return hits

    def save(self) -> None:
        data = {"nodes": self._nodes, "edges": self._edges}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    # =========================
    # Read-only public helpers
    # =========================
    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        return self._nodes

    @property
    def edges(self) -> Dict[str, Dict[str, Any]]:
        return self._edges

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
        }