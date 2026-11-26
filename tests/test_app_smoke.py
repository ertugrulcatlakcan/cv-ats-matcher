"""Smoke tests for app.py using Streamlit AppTest."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from streamlit.testing.v1 import AppTest

from src.ui_helpers import add_items_to_workspace

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _make_pdf_bytes(text: str) -> bytes:
    """Create a small in-memory digital PDF for UI smoke testing."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_app_renders_empty_state_without_exception() -> None:
    """App boots cleanly in empty workspace state with zero runtime exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()

    assert not at.exception
    tab_labels = [tab.label for tab in at.tabs]
    assert "CV Ingestion & Inspector" in tab_labels
    assert "Job Matching (Phase 2)" in tab_labels
    assert "OCR Benchmark" in tab_labels


def test_app_renders_single_and_batch_workspace_states() -> None:
    """App transitions cleanly between 1-file Split View and 2-file Batch Workspace."""
    pdf_a = _make_pdf_bytes(
        "Candidate One Senior Engineer experienced in C++, C#, .NET, and Python 3.12."
    )
    pdf_b = _make_pdf_bytes(
        "Candidate Two Backend Architect skilled in Node.js, Vue.js, and CI/CD pipelines."
    )

    ws_single, _, _, _ = add_items_to_workspace([], [("candidate_one.pdf", pdf_a)])
    at_single = AppTest.from_file(APP_PATH, default_timeout=15)
    at_single.session_state["workspace"] = ws_single
    at_single.session_state["selected_hash"] = ws_single[0].content_hash
    at_single.run()
    assert not at_single.exception

    ws_batch, _, _, _ = add_items_to_workspace(
        ws_single,
        [("candidate_two.pdf", pdf_b)],
    )
    at_batch = AppTest.from_file(APP_PATH, default_timeout=15)
    at_batch.session_state["workspace"] = ws_batch
    at_batch.session_state["selected_hash"] = ws_batch[0].content_hash
    at_batch.run()
    assert not at_batch.exception
