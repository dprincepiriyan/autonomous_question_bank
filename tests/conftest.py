"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqb.config import Settings  # noqa: E402

EXAMPLES = ROOT / "examples"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(provider="mock", model="mock", output_dir=tmp_path / "outputs", retry_base_delay=0.0)


@pytest.fixture
def sample_bank() -> str:
    return str(EXAMPLES / "sample_question_bank.txt")


@pytest.fixture
def sample_notes() -> str:
    return str(EXAMPLES / "sample_notes.md")
