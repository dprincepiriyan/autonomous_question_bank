"""Tests for LLM utilities: retries, JSON repair, env loading, config clamping, ingestion."""

from __future__ import annotations

import os

import pytest

from aqb.config import Settings
from aqb.env import load_dotenv
from aqb.ingestion import NoteRetriever, chunk_notes, load_many
from aqb.llm import (
    LLMClient,
    LLMError,
    RetryingLLMClient,
    generate_json_list,
    parse_json_response,
)


class ScriptedClient(LLMClient):
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize(
    "raw",
    [
        '[{"a": 1}]',
        '```json\n[{"a": 1}]\n```',
        'Sure! Here is the JSON:\n```json\n[{"a": 1}]\n```\nHope it helps.',
        '[{"a": 1},]',
        'Result: [{"a": 1}] done',
    ],
)
def test_parse_json_response_repairs_common_issues(raw: str) -> None:
    assert parse_json_response(raw) == [{"a": 1}]


def test_parse_json_keeps_code_fences_inside_answers() -> None:
    raw = '[{"question_id": "Q1", "answer_markdown": "```mermaid\\ngraph TD; A-->B\\n```"}]'
    assert parse_json_response(raw)[0]["answer_markdown"].startswith("```mermaid")


def test_parse_json_escapes_raw_newlines_in_strings() -> None:
    raw = '[{"answer_markdown": "line one\nline two"}]'
    assert parse_json_response(raw)[0]["answer_markdown"] == "line one\nline two"


def test_generate_json_list_reasks_on_invalid_json_and_unwraps_objects() -> None:
    client = ScriptedClient(["not json at all", '{"answers": [{"question_id": "Q1"}]}'])
    assert generate_json_list(client, "PROMPT") == [{"question_id": "Q1"}]
    assert "not valid JSON" in client.prompts[1]


def test_generate_json_list_gives_up() -> None:
    client = ScriptedClient(["nope", "still nope"])
    with pytest.raises(LLMError, match="invalid JSON"):
        generate_json_list(client, "PROMPT")


def test_retrying_client_backs_off_then_succeeds() -> None:
    sleeps: list[float] = []
    inner = ScriptedClient([RuntimeError("429 rate limited"), RuntimeError("503"), "ok"])
    client = RetryingLLMClient(inner, max_retries=3, base_delay=1.0, sleep=sleeps.append)
    assert client.generate("p") == "ok"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_retrying_client_fails_fast_on_auth_errors() -> None:
    inner = ScriptedClient([RuntimeError("Invalid API key provided")])
    client = RetryingLLMClient(inner, max_retries=3, sleep=lambda _: None)
    with pytest.raises(LLMError, match="Non-retryable"):
        client.generate("p")


def test_retrying_client_exhausts_retries() -> None:
    inner = ScriptedClient([RuntimeError("timeout")] * 3)
    client = RetryingLLMClient(inner, max_retries=2, sleep=lambda _: None)
    with pytest.raises(LLMError, match="after 3 attempts"):
        client.generate("p")


def test_load_dotenv(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nAQB_T1=hello\nexport AQB_T2="quoted value"\nAQB_T3=keep # trailing\nAQB_T4=x\n')
    monkeypatch.setenv("AQB_T4", "existing")
    for k in ("AQB_T1", "AQB_T2", "AQB_T3"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(env)
    assert os.environ["AQB_T1"] == "hello"
    assert os.environ["AQB_T2"] == "quoted value"
    assert os.environ["AQB_T3"] == "keep"
    assert os.environ["AQB_T4"] == "existing"


def test_batch_sizes_are_clamped_to_design_limits() -> None:
    s = Settings(batch_size_2m=50, batch_size_13m=1)
    assert (s.batch_size_2m, s.batch_size_13m) == (12, 5)
    s2 = s.with_overrides(batch_size_2m=11, batch_size_13m=6, provider="gemini")
    assert (s2.batch_size_2m, s2.batch_size_13m, s2.model) == (11, 6, "gemini-2.5-flash")


def test_chunking_tracks_pdf_pages_and_retrieval_per_question() -> None:
    text = "[Page 1]\n" + "Paging divides memory into frames. " * 40 + "\n\n[Page 2]\n" + \
           "Segmentation divides memory into segments. " * 40
    chunks = chunk_notes(text, "os.pdf", chunk_size=600, overlap=50)
    assert chunks[0].chunk_id.startswith("os.pdf#p1.")
    assert any(c.chunk_id.startswith("os.pdf#p2.") for c in chunks)

    retriever = NoteRetriever(chunks)
    hits = retriever.retrieve_for_batch(["What is paging?", "What is segmentation?"], per_query=1, max_chunks=4)
    joined = " ".join(h.text for h in hits)
    assert "Paging" in joined and "Segmentation" in joined


def test_load_many_handles_files_and_inline_text(tmp_path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("file content")
    docs = load_many([str(f), "inline notes here", str(f)])
    assert docs[0] == ("file content", "a.txt")
    assert docs[1] == ("inline notes here", "inline-input")
    assert docs[2][1] == "a.txt(2)"


def test_docx_and_pdf_loading(tmp_path) -> None:
    docx = pytest.importorskip("docx")
    pymupdf = pytest.importorskip("pymupdf")
    from aqb.ingestion import load_text_from_path

    d = docx.Document()
    d.add_paragraph("Deadlock requires four conditions.")
    d.save(tmp_path / "n.docx")
    assert "four conditions" in load_text_from_path(str(tmp_path / "n.docx"))[0]

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Round Robin uses a time quantum.")
    pdf.save(tmp_path / "n.pdf")
    text, source = load_text_from_path(str(tmp_path / "n.pdf"))
    assert source == "n.pdf" and "[Page 1]" in text and "time quantum" in text
