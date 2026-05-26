"""Build human-readable demonstration outputs."""

from __future__ import annotations

from .schemas import ReuseRecommendation


def build_markdown_report(recommendation: ReuseRecommendation) -> str:
    """Render the reuse recommendation as a concise engineering review report."""
    raise NotImplementedError


def build_json_report(recommendation: ReuseRecommendation) -> dict:
    """Render the reuse recommendation as structured JSON for downstream use."""
    raise NotImplementedError


def save_report(report: str | dict, output_path: str) -> None:
    """Persist a generated report to disk."""
    raise NotImplementedError

