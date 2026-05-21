"""GraphRAG extraction agents and orchestration."""

__all__ = [
    "evaluate_selected_chunks_review",
    "merge_selected_chunks_from_folder",
    "write_merged_selected_chunks",
    "write_selected_chunks_review_evaluation",
]


def __getattr__(name: str):
    if name in __all__:
        from .merge_qd_selected_chunks import (
            evaluate_selected_chunks_review,
            merge_selected_chunks_from_folder,
            write_merged_selected_chunks,
            write_selected_chunks_review_evaluation,
        )

        exports = {
            "evaluate_selected_chunks_review": evaluate_selected_chunks_review,
            "merge_selected_chunks_from_folder": merge_selected_chunks_from_folder,
            "write_merged_selected_chunks": write_merged_selected_chunks,
            "write_selected_chunks_review_evaluation": write_selected_chunks_review_evaluation,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
