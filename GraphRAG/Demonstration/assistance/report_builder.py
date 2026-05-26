"""Build outputs for FMEA assistant interpretation and review."""

from __future__ import annotations

from .schemas import AssistanceReview


def build_review_json(review: AssistanceReview) -> dict:
    """Render an assistant review as structured JSON."""
    raise NotImplementedError


def build_review_markdown(review: AssistanceReview) -> str:
    """Render an assistant review as a readable engineering review note."""
    raise NotImplementedError


def save_review_report(report: str | dict, output_path: str) -> None:
    """Persist a generated assistant report to disk."""
    raise NotImplementedError

