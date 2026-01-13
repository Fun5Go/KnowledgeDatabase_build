from pathlib import Path
from KG.simple_graph_store import SimpleGraphStore
from .ingest import ingest_cause_store_json


def resolve_paths():
    base = Path(__file__).resolve().parent

    kb_data = base.parent / "kb_data"
    kb_data.mkdir(parents=True, exist_ok=True)

    json_root = base.parent / "fmea_json_raw"
    return kb_data, json_root


def main():
    kb_data, json_root = resolve_paths()

    cause_path = kb_data / "cause_kb" / "fmea_cause_store.json"
    if not cause_path.exists():
        raise FileNotFoundError(f"JSON not found: {cause_path}")

    graph = SimpleGraphStore(kb_data)

    print(f"[INFO] Ingest single FMEA JSON into KG: {cause_path.name}")

    ingest_cause_store_json(
        json_path=cause_path,
        graph=graph,
    )

    print("[INFO] Done.")
    print(f"[INFO] Nodes : {len(graph.nodes)}")
    print(f"[INFO] Edges : {len(graph.edges)}")


if __name__ == "__main__":
    main()
