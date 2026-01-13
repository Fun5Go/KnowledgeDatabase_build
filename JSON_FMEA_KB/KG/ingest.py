
import json
from pathlib import Path
from KG.schema import *
from KG.simple_graph_store import SimpleGraphStore


def ingest_cause_store_json(json_path: Path, graph: SimpleGraphStore):
    """
    Ingest fmea_cause_store.json
    Top-level structure:
      {
        cause_id: { failure_id, failure_mode, failure_element, ... }
      }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON dict for cause store")

    for cause_id, c in data.items():

        # ---------- Failure ----------
        f_node = NodeKey(
            NODE_FAILURE, K_FAILURE_ID, c["failure_id"]
        ).to_id()
        graph.upsert_node(
            f_node,
            NODE_FAILURE,
            {K_FAILURE_ID: c["failure_id"]}
        )

        # ---------- Element ----------
        e_node = NodeKey(
            NODE_ELEMENT, K_NAME, c["failure_element"]
        ).to_id()
        graph.upsert_node(
            e_node,
            NODE_ELEMENT,
            {K_NAME: c["failure_element"]}
        )
        graph.upsert_edge(f_node, E_HAS_ELEMENT, e_node)

        # ---------- Mode ----------
        m_node = NodeKey(
            NODE_MODE, K_TEXT, c["failure_mode"]
        ).to_id()
        graph.upsert_node(
            m_node,
            NODE_MODE,
            {K_TEXT: c["failure_mode"]}
        )
        graph.upsert_edge(f_node, E_HAS_MODE, m_node)
        graph.upsert_edge(e_node, E_HAS_MODE, m_node)

        # ---------- Effect ----------
        ef_node = NodeKey(
            NODE_EFFECT, K_TEXT, c["failure_effect"]
        ).to_id()
        graph.upsert_node(
            ef_node,
            NODE_EFFECT,
            {K_TEXT: c["failure_effect"]}
        )
        graph.upsert_edge(f_node, E_HAS_EFFECT, ef_node)
        graph.upsert_edge(m_node, E_LEADS_TO, ef_node)

        # ---------- Cause ----------
        c_node = NodeKey(
            NODE_CAUSE, K_CAUSE_ID, cause_id
        ).to_id()
        graph.upsert_node(
            c_node,
            NODE_CAUSE,
            {
                K_CAUSE_ID: cause_id,
                K_TEXT: c["failure_cause"],
                "discipline": c.get("discipline"),
                "occurrence": c.get("occurrence"),
                "detection": c.get("detection"),
                "recommended_action": c.get("recommended_action"),
            }
        )
        graph.upsert_edge(c_node, E_CAUSE_OF, m_node)

    graph.save()