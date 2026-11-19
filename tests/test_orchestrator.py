"""End-to-end tests for the orchestrator: batching, retries, resume, verification, export, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aqb.cli import main as cli_main
from aqb.config import Settings
from aqb.llm import LLMError, MockLLMClient
from aqb.orchestrator import AutonomousSolverOrchestrator


def _bank(n2: int, n13: int) -> str:
    lines = ["PART A (2 Marks)"]
    lines += [f"{i}. Define concept {i} of memory paging." for i in range(1, n2 + 1)]
    lines += ["PART B (13 Marks)"]
    lines += [f"{n2 + i}. Explain memory segmentation design {i} in detail." for i in range(1, n13 + 1)]
    return "\n".join(lines)


NOTES = (
    "## Paging\nPaging is a memory concept that divides memory into fixed-size frames. Paging avoids "
    "external fragmentation.\n\n## Segmentation\nSegmentation is a memory design that divides a program "
    "into logical segments. Segmentation supports the user view of memory."
)


def test_controlled_batching_sizes(settings: Settings) -> None:
    orch = AutonomousSolverOrchestrator(MockLLMClient(), settings)
    state = orch.prepare(_bank(25, 11), NOTES, "Use bullet points.", session_id="t1")
    sizes = [(r.marks, len(r.answers)) for r in orch.solve_iter(state)]
    assert sizes == [(2, 10), (2, 10), (2, 5), (13, 5), (13, 5), (13, 1)]
    assert not state.has_pending() and len(state.answers) == 36


def test_batch_size_upper_bounds(settings: Settings) -> None:
    s = settings.with_overrides(batch_size_2m=12, batch_size_13m=6)
    orch = AutonomousSolverOrchestrator(MockLLMClient(), s)
    state = orch.prepare(_bank(25, 13), NOTES, session_id="t2")
    sizes = [(r.marks, len(r.answers)) for r in orch.solve_iter(state)]
    assert sizes == [(2, 12), (2, 12), (2, 1), (13, 6), (13, 6), (13, 1)]


def test_full_run_exports_key_with_topics(settings: Settings, sample_bank: str, sample_notes: str) -> None:
    orch = AutonomousSolverOrchestrator(MockLLMClient(), settings)
    state, key_path = orch.run(
        sample_bank, sample_notes, "short answers", session_id="full", formats=("md", "docx", "pdf")
    )
    text = key_path.read_text()
    assert "## Important Topics to Focus On" in text
    assert "## Part A — 2-Mark Questions" in text and "## Part B — 13-Mark Questions" in text
    assert (key_path.parent / "final_answer_key.docx").exists()
    pdf_path = key_path.parent / "final_answer_key.pdf"
    assert pdf_path.exists() and pdf_path.stat().st_size > 500
    assert len(list((key_path.parent / "batches").glob("batch_*.md"))) == state.batch_count == 4
    # The Linux VFS question is not in the notes -> must be flagged, not hallucinated.
    vfs = next(a for a in state.answers if "Linux kernel" in a.question_text)
    assert vfs.answer_markdown.startswith("[Not found in provided notes]") and vfs.needs_review
    # Topics are lesson sub-topics derived from the notes, ranked by marks.
    assert {t.topic for t in state.important_topics} == {
        "Process Concept", "Threads", "CPU Scheduling", "Process Synchronization", "Deadlocks"
    }


class FlakyMock(MockLLMClient):
    """Fails permanently on the Nth solver call to simulate a crash / quota exhaustion."""

    def __init__(self, fail_on_batch: int) -> None:
        super().__init__()
        self.fail_on_batch = fail_on_batch
        self.solver_calls = 0

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        if "QUESTIONS TO SOLVE IN THIS BATCH:" in prompt:
            self.solver_calls += 1
            if self.solver_calls == self.fail_on_batch:
                raise LLMError("quota exceeded")
        return super().generate(prompt, temperature, json_mode)


def test_resume_after_crash_does_not_redo_work(settings: Settings) -> None:
    orch = AutonomousSolverOrchestrator(FlakyMock(fail_on_batch=3), settings)
    state = orch.prepare(_bank(25, 6), NOTES, session_id="crash")
    with pytest.raises(LLMError):
        for _ in orch.solve_iter(state):
            pass
    saved = json.loads((settings.output_dir / "crash" / "state.json").read_text())
    assert sum(q["status"] == "completed" for q in saved["questions"]) == 20

    fresh = MockLLMClient()
    orch2 = AutonomousSolverOrchestrator(fresh, settings)  # new process, cold caches
    resumed = orch2.load_session("crash")
    results = list(orch2.solve_iter(resumed))
    assert [(r.marks, len(r.answers)) for r in results] == [(2, 5), (13, 5), (13, 1)]
    assert len(resumed.answers) == 31 and not resumed.has_pending()
    assert len({a.question_id for a in resumed.answers}) == 31


class OmittingMock(MockLLMClient):
    """Drops the last question of the first solver reply."""

    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        out = super().generate(prompt, temperature, json_mode)
        if "QUESTIONS TO SOLVE IN THIS BATCH:" in prompt and self.first:
            self.first = False
            return json.dumps(json.loads(out)[:-1])
        return out


def test_omitted_questions_are_retried(settings: Settings) -> None:
    llm = OmittingMock()
    orch = AutonomousSolverOrchestrator(llm, settings)
    state = orch.prepare(_bank(10, 0), NOTES, session_id="omit")
    result = next(orch.solve_iter(state))
    assert len(result.answers) == 10
    assert not any("did not return" in a.answer_markdown for a in result.answers)
    solver_calls = [p for p in llm.calls if "QUESTIONS TO SOLVE IN THIS BATCH:" in p]
    assert len(solver_calls) == 2 and "Q10 (2 Marks" in solver_calls[1]


class BrokenJSONMock(MockLLMClient):
    """Returns garbage for any batch larger than 3 questions -> forces the split-and-retry path."""

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        if "QUESTIONS TO SOLVE IN THIS BATCH:" in prompt:
            n = prompt.split("QUESTIONS TO SOLVE IN THIS BATCH:")[1].count("Marks, Topic")
            if n > 3:
                return "I'm sorry, the output got cut off mid-answer..."
        return super().generate(prompt, temperature, json_mode)


def test_invalid_json_batches_are_split(settings: Settings) -> None:
    orch = AutonomousSolverOrchestrator(BrokenJSONMock(), settings)
    state = orch.prepare(_bank(0, 5), NOTES, session_id="split")
    result = next(orch.solve_iter(state))
    assert len(result.answers) == 5
    assert all(a.citations for a in result.answers)


class RejectingVerifierMock(MockLLMClient):
    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        if "ANSWERS TO VERIFY:" in prompt:
            payload = json.loads(prompt.split("ANSWERS TO VERIFY:", 1)[1])
            return json.dumps([
                {"question_id": p["question_id"], "supported": False,
                 "unsupported_claims": ["invented statistic"], "revised_answer_markdown": "REVISED"}
                for p in payload
            ])
        return super().generate(prompt, temperature, json_mode)


def test_verifier_revises_and_flags_long_answers(settings: Settings) -> None:
    s = settings.with_overrides(verify_long_answers=True)
    orch = AutonomousSolverOrchestrator(RejectingVerifierMock(), s)
    state = orch.prepare(_bank(10, 2), NOTES, session_id="verify")
    results = list(orch.solve_iter(state))
    short, long_ = results
    assert all(a.answer_markdown != "REVISED" for a in short.answers)  # verifier only for long answers
    assert all(a.answer_markdown == "REVISED" and a.needs_review for a in long_.answers)
    assert "invented statistic" in long_.answers[0].review_note


def test_invalid_citations_are_dropped_and_flagged(settings: Settings) -> None:
    class BadCiteMock(MockLLMClient):
        def generate(self, prompt, temperature=0.15, json_mode=False):
            out = super().generate(prompt, temperature, json_mode)
            if "QUESTIONS TO SOLVE IN THIS BATCH:" in prompt:
                items = json.loads(out)
                for it in items:
                    it["citations"] = ["made-up-source#C99"]
                return json.dumps(items)
            return out

    orch = AutonomousSolverOrchestrator(BadCiteMock(), settings)
    state = orch.prepare(_bank(10, 0), NOTES, session_id="cites")
    result = next(orch.solve_iter(state))
    found = [a for a in result.answers if not a.answer_markdown.startswith("[Not found")]
    assert found and all(a.citations == [] and a.needs_review for a in found)


def test_session_listing_and_duplicate_names(settings: Settings) -> None:
    orch = AutonomousSolverOrchestrator(MockLLMClient(), settings)
    orch.prepare(_bank(3, 0), NOTES, session_id="dup")
    with pytest.raises(FileExistsError):
        orch.prepare(_bank(3, 0), NOTES, session_id="dup")
    sessions = AutonomousSolverOrchestrator.list_sessions(settings.output_dir)
    assert sessions == [{"session_id": "dup", "done": 0, "total": 3}]


def test_cli_end_to_end_with_resume(tmp_path: Path, sample_bank: str, sample_notes: str, capsys) -> None:
    out = tmp_path / "cli_out"
    base = ["--provider", "mock", "--output-dir", str(out), "--env-file", str(tmp_path / "none.env")]
    rc = cli_main(base + ["-q", sample_bank, "-n", sample_notes, "-s", "", "--session", "c1", "--max-batches", "1"])
    assert rc == 0
    assert "Continue with: aqb --resume c1" in capsys.readouterr().out

    rc = cli_main(base + ["--resume", "c1", "--docx", "--pdf"])
    captured = capsys.readouterr().out
    assert rc == 0 and "22/22 questions solved" in captured
    assert (out / "c1" / "final_answer_key.docx").exists()
    assert (out / "c1" / "final_answer_key.pdf").exists()

    assert cli_main(base + ["--list-sessions"]) == 0
    assert "c1" in capsys.readouterr().out
