from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from .graph_store import GraphStore
from .schema import (
    NODE_FAILURE, NODE_ELEMENT, NODE_MODE, NODE_EFFECT, NODE_CAUSE, NODE_SYSTEM,
    E_HAS_ELEMENT, E_HAS_MODE, E_HAS_EFFECT, E_CAUSE_OF, E_IN_SYSTEM, E_LEADS_TO,
    K_FAILURE_ID, K_CAUSE_ID, K_NAME, K_TEXT, K_SYSTEM
)

from .simple_graph_store import SimpleGraphStore

# --------------------------------------------------
# Helper: get readable text from node
# --------------------------------------------------
def _node_text(g: GraphStore, node_id: str) -> str:
    n = g.get_node(node_id) or {}
    props = n.get("props", {})
    for key in (K_TEXT, K_NAME, K_FAILURE_ID, K_CAUSE_ID, K_SYSTEM):
        if props.get(key):
            return str(props[key])
    return node_id


# --------------------------------------------------
# 1️⃣ Full FMEA chain by failure_id
# --------------------------------------------------
def get_chain_by_failure_id(g: GraphStore, failure_id: str) -> Dict[str, Any]:
    failure_nodes = g.find_nodes(NODE_FAILURE, K_FAILURE_ID, failure_id)
    if not failure_nodes:
        return {"failure_id": failure_id, "found": False}

    f = failure_nodes[0]

    elements = g.neighbors(f, E_HAS_ELEMENT)
    modes = g.neighbors(f, E_HAS_MODE)
    effects = g.neighbors(f, E_HAS_EFFECT)
    systems = g.neighbors(f, E_IN_SYSTEM)

    # Causes come from incoming edges to Mode
    causes: List[str] = []
    for m in modes:
        causes.extend(g.neighbors(m, E_CAUSE_OF, direction="in"))

    return {
        "failure_id": failure_id,
        "found": True,
        "system": [_node_text(g, s) for s in systems],
        "failure_element": [_node_text(g, e) for e in elements],
        "failure_mode": [_node_text(g, m) for m in modes],
        "failure_effect": [_node_text(g, e) for e in effects],
        "failure_cause": [_node_text(g, c) for c in causes],
    }


# --------------------------------------------------
# 2️⃣ Expand all modes under an element (knowledge-level)
# --------------------------------------------------
def expand_modes_under_element(g: GraphStore, element_name: str) -> Dict[str, Any]:
    elements = g.find_nodes(NODE_ELEMENT, K_NAME, element_name)
    if not elements:
        return {"element": element_name, "found": False, "modes": []}

    e = elements[0]
    modes = g.neighbors(e, E_HAS_MODE)

    return {
        "element": element_name,
        "found": True,
        "modes": [_node_text(g, m) for m in modes],
    }


# --------------------------------------------------
# 3️⃣ Get all causes for a given failure mode
# --------------------------------------------------
def get_causes_for_mode(g: GraphStore, mode_text: str) -> Dict[str, Any]:
    modes = g.find_nodes(NODE_MODE, K_TEXT, mode_text)
    if not modes:
        return {"mode": mode_text, "found": False, "causes": []}

    m = modes[0]
    causes = g.neighbors(m, E_CAUSE_OF, direction="in")

    return {
        "mode": mode_text,
        "found": True,
        "causes": [_node_text(g, c) for c in causes],
    }


# --------------------------------------------------
# 4️⃣ Validate Cause → Mode link (consistency check)
# --------------------------------------------------
def validate_cause_mode_link(g: GraphStore, cause_id: str, mode_text: str) -> bool:
    causes = g.find_nodes(NODE_CAUSE, K_CAUSE_ID, cause_id)
    modes = g.find_nodes(NODE_MODE, K_TEXT, mode_text)
    if not causes or not modes:
        return False

    c = causes[0]
    m = modes[0]

    return m in g.neighbors(c, E_CAUSE_OF)


def main():
    graph = SimpleGraphStore(Path("kb_data"))

    # # --------- 1. element -> modes ----------
    # element = "Communication Expansion Interface"
    # print(f"\n[QUERY] Modes under element: {element}")
    # for m in get_modes_under_element(graph, element):
    #     print(" -", m)

    # --------- 2. mode -> causes ----------
    mode = "Controller memory (RAM/Flash) insufficient"
    print(f"\n[QUERY] Causes for mode: {mode}")
    res = get_causes_for_mode(graph, mode)
    for c in res["causes"]:
        print(" -", c)

    # # --------- 3. full chain ----------
    # failure_id = "DFMEA6744190124R07__F1"
    # print(f"\n[QUERY] Full chain for failure_id: {failure_id}")
    # chain = get_full_chain(graph, failure_id)
    # print(chain)


if __name__ == "__main__":
    main()