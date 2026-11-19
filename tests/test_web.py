"""Headless test of the Streamlit UI using streamlit.testing.AppTest (skipped if Streamlit is absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_web_app_full_flow(tmp_path: Path, monkeypatch, sample_bank: str, sample_notes: str) -> None:
    monkeypatch.chdir(tmp_path)  # no .env picked up
    monkeypatch.setenv("AQB_PROVIDER", "mock")
    monkeypatch.setenv("AQB_OUTPUT_DIR", str(tmp_path / "outputs"))

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value.endswith("Autonomous Question Bank")

    at.text_area[0].set_value(Path(sample_bank).read_text())
    at.text_area[1].set_value(Path(sample_notes).read_text())
    at.text_area[2].set_value("Keep it short.")
    at.text_input[0].set_value("webtest")
    next(b for b in at.button if "Analyze" in b.label).click()
    at.run()
    assert not at.exception, at.exception

    # Auto-run solved everything; topics, batches and download buttons are shown.
    state_file = tmp_path / "outputs" / "webtest" / "state.json"
    assert state_file.exists()
    markdown = " ".join(m.value for m in at.markdown)
    assert "Batch" in " ".join(e.label for e in at.expander)
    assert at.dataframe, "topics table should render"
    assert "All questions solved" in " ".join(s.value for s in at.success)
    assert (tmp_path / "outputs" / "webtest" / "final_answer_key.docx").exists()
    assert (tmp_path / "outputs" / "webtest" / "final_answer_key.pdf").exists()
    assert "Repeated questions" in markdown or "frequently examined" in markdown.lower()
