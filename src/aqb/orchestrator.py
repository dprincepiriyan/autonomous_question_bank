"""Stateful orchestrator for the autonomous question-bank solving loop.

Lifecycle:
    prepare()     -> parse question bank, index notes, rank important topics, checkpoint
    solve_iter()  -> generator: one controlled batch per iteration (10-12 short / 5-6 long), checkpoint each
    export()      -> final answer key (Markdown, optional DOCX)
    load_session()-> resume any interrupted session from its checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aqb.config import Settings
from aqb.export import export_docx, export_markdown, export_pdf, is_not_found
from aqb.ingestion import NoteRetriever, chunk_notes, load_many
from aqb.llm import LLMClient, LLMError, generate_json_list
from aqb.models import NoteChunk, Question, SessionState, SolvedAnswer, TopicImportance
from aqb.prompts import (
    build_batch_solver_prompt,
    build_question_parser_prompt,
    build_topic_focus_prompt,
    build_verifier_prompt,
    is_short,
)
from aqb.question_parser import (
    assign_topics_from_notes,
    compute_topic_importance,
    needs_topic_refinement,
    find_repeated_questions,
    heuristic_parse_questions,
    key_concepts,
    renumber,
    split_question_bank,
)

log = logging.getLogger(__name__)

STATE_FILE = "state.json"
NOTES_FILE = "notes_corpus.json"


@dataclass
class BatchResult:
    """Outcome of one solver iteration (one 'reply')."""

    batch_number: int
    marks: int
    answers: list[SolvedAnswer]
    done: int
    total: int

    def to_markdown(self) -> str:
        lines = [f"## Batch {self.batch_number} — {len(self.answers)} × {self.marks}-mark questions", ""]
        for a in self.answers:
            lines.append(f"### {a.question_id}. {a.question_text}" + (" ⚠️" if a.needs_review else ""))
            lines.append("")
            lines.append(a.answer_markdown.strip())
            if a.citations:
                lines += ["", f"<sub>Sources: {', '.join(a.citations)}</sub>"]
            lines.append("")
        return "\n".join(lines)


class SessionNotFoundError(FileNotFoundError):
    pass


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if v]


def _safe_session_id(raw: str | None) -> str:
    if raw:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._")
        if cleaned:
            return cleaned
    return datetime.now().strftime("session_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]


class AutonomousSolverOrchestrator:
    """Drives ingestion, topic prioritization, batch solving, checkpointing, and key export."""

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or Settings.from_env()
        self._on_log = on_log
        self._retrievers: dict[str, NoteRetriever] = {}

    # ------------------------------------------------------------------ utils
    def _log(self, message: str) -> None:
        log.info(message)
        if self._on_log:
            self._on_log(message)

    def session_dir(self, session_id: str) -> Path:
        return self.settings.output_dir / session_id

    def _save_checkpoint(self, state: SessionState) -> None:
        session_dir = self.session_dir(state.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        tmp = session_dir / f"{STATE_FILE}.tmp"
        tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, session_dir / STATE_FILE)

    # ------------------------------------------------------------------ phase 1
    def prepare(
        self,
        question_bank_inputs: str | list[str],
        notes_inputs: str | list[str],
        style_instructions: str = "",
        session_id: str | None = None,
    ) -> SessionState:
        """Parse inputs, rank important topics and create a checkpointed session."""
        qb_docs = load_many(_as_list(question_bank_inputs))
        if not qb_docs or not any(t.strip() for t, _ in qb_docs):
            raise ValueError("Question bank is empty.")
        qb_text = "\n\n".join(t for t, _ in qb_docs)

        notes_docs = [(t, s) for t, s in load_many(_as_list(notes_inputs)) if t.strip()]
        if not notes_docs:
            raise ValueError("Study notes are empty.")

        sid = _safe_session_id(session_id)
        session_dir = self.session_dir(sid)
        if (session_dir / STATE_FILE).exists():
            raise FileExistsError(f"Session '{sid}' already exists. Resume it or choose another name.")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "question_bank.txt").write_text(qb_text, encoding="utf-8")
        (session_dir / NOTES_FILE).write_text(
            json.dumps([{"source": s, "text": t} for t, s in notes_docs], ensure_ascii=False),
            encoding="utf-8",
        )
        retriever = self._build_retriever(notes_docs)
        self._retrievers[sid] = retriever
        self._log(
            f"Indexed {len(notes_docs)} notes file(s) into {len(retriever.chunks)} chunks "
            f"({retriever.total_chars():,} chars)."
        )

        questions = self.parse_questions(qb_text)
        if not questions:
            raise ValueError("No questions could be parsed from the question bank.")
        if needs_topic_refinement(questions) and assign_topics_from_notes(questions, [t for t, _ in notes_docs]):
            self._log("Assigned lesson sub-topics to questions using the notes' section headings.")
        n_short = sum(1 for q in questions if is_short(q.marks))
        self._log(f"Parsed {len(questions)} questions ({n_short} short-answer, {len(questions) - n_short} long-answer).")

        notes_preview = "\n\n".join(t for t, _ in notes_docs)
        state = SessionState(
            session_id=sid,
            style_instructions=style_instructions.strip(),
            questions=questions,
            important_topics=self.analyze_topics(questions, notes_preview),
            repeated_questions=find_repeated_questions(questions),
            key_concepts=key_concepts(questions),
            notes_sources=[s for _, s in notes_docs],
        )
        self._save_checkpoint(state)
        return state

    def _build_retriever(self, notes_docs: list[tuple[str, str]]) -> NoteRetriever:
        chunks: list[NoteChunk] = []
        for text, source in notes_docs:
            chunks.extend(
                chunk_notes(
                    text,
                    source=source,
                    chunk_size=self.settings.chunk_size_chars,
                    overlap=self.settings.chunk_overlap_chars,
                )
            )
        return NoteRetriever(chunks)

    def parse_questions(self, qb_text: str) -> list[Question]:
        """Parse the bank with the LLM (in slices) and fall back to the heuristic parser."""
        mode = self.settings.parse_mode
        heuristic = heuristic_parse_questions(qb_text)
        if mode == "heuristic":
            return heuristic

        try:
            parsed: list[Question] = []
            slices = split_question_bank(qb_text, self.settings.parse_chunk_chars)
            for idx, (slice_text, header) in enumerate(slices, start=1):
                if len(slices) > 1:
                    self._log(f"Parsing question bank slice {idx}/{len(slices)}...")
                items = generate_json_list(
                    self.llm, build_question_parser_prompt(slice_text, header), temperature=0.0
                )
                for item in items:
                    if isinstance(item, dict) and str(item.get("text", "")).strip():
                        item.setdefault("id", "Q0")
                        parsed.append(Question.from_dict({**item, "status": "pending"}))
            parsed = renumber(parsed)
        except LLMError as exc:
            if mode == "llm":
                raise
            self._log(f"LLM question parsing failed ({exc}); using heuristic parser.")
            return heuristic

        if mode == "auto" and heuristic and len(parsed) < 0.7 * len(heuristic):
            self._log(
                f"LLM parser found {len(parsed)} questions but heuristic found {len(heuristic)}; "
                "using heuristic result to avoid dropping questions."
            )
            return heuristic
        return parsed or heuristic

    def analyze_topics(self, questions: list[Question], notes_preview: str) -> list[TopicImportance]:
        """Compute topic weightage in code, then enrich with LLM focus points (best effort)."""
        topics = compute_topic_importance(questions)
        try:
            items = generate_json_list(
                self.llm, build_topic_focus_prompt(topics, questions, notes_preview), temperature=0.1
            )
            by_name = {str(i.get("topic", "")).strip().casefold(): i for i in items if isinstance(i, dict)}
            for t in topics:
                extra = by_name.get(t.topic.casefold())
                if extra:
                    t.focus_points = [str(p) for p in extra.get("focus_points", [])][:6]
                    if extra.get("rationale"):
                        t.rationale = f"{t.rationale} {str(extra['rationale']).strip()}"
        except LLMError as exc:
            self._log(f"Topic focus-point generation skipped ({exc}).")
        return topics

    # ------------------------------------------------------------------ resume
    def load_session(self, session_id: str) -> SessionState:
        path = self.session_dir(session_id) / STATE_FILE
        if not path.exists():
            raise SessionNotFoundError(f"No checkpoint found for session '{session_id}' at {path}")
        return SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def list_sessions(output_dir: Path) -> list[dict]:
        sessions = []
        for state_file in sorted(Path(output_dir).glob(f"*/{STATE_FILE}"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            total = len(data.get("questions", []))
            done = sum(1 for q in data.get("questions", []) if q.get("status") == "completed")
            sessions.append({"session_id": data.get("session_id", state_file.parent.name), "done": done, "total": total})
        return sessions

    def _retriever_for(self, state: SessionState) -> NoteRetriever:
        if state.session_id not in self._retrievers:
            notes_path = self.session_dir(state.session_id) / NOTES_FILE
            if not notes_path.exists():
                raise SessionNotFoundError(f"Notes corpus missing for session '{state.session_id}'.")
            docs = [(d["text"], d["source"]) for d in json.loads(notes_path.read_text(encoding="utf-8"))]
            self._retrievers[state.session_id] = self._build_retriever(docs)
        return self._retrievers[state.session_id]

    # ------------------------------------------------------------------ phase 2
    def next_batch(self, state: SessionState) -> list[Question]:
        """Short answers first in batches of 10-12, then long answers in batches of 5-6.

        A batch always contains a single mark value so answer depth stays consistent.
        """
        pending = state.pending()
        if not pending:
            return []
        short = [q for q in pending if is_short(q.marks)]
        group = short or pending
        target = group[0].marks
        size = self.settings.batch_size_2m if is_short(target) else self.settings.batch_size_13m
        return [q for q in group if q.marks == target][:size]

    def _context_for(self, retriever: NoteRetriever, batch: list[Question]) -> list[NoteChunk]:
        if retriever.total_chars() <= self.settings.full_context_char_limit:
            return retriever.chunks  # small notes: give the model everything, no retrieval misses
        return retriever.retrieve_for_batch(
            [f"{q.topic} {q.text}" for q in batch],
            per_query=self.settings.top_k_per_question,
            max_chunks=self.settings.max_context_chunks,
        )

    def _call_solver(
        self, batch: list[Question], chunks: list[NoteChunk], style: str
    ) -> dict[str, SolvedAnswer]:
        """One LLM call; if the JSON is unusable, split the batch in half and retry each half."""
        prompt = build_batch_solver_prompt(batch, chunks, style)
        try:
            items = generate_json_list(self.llm, prompt, temperature=self.settings.temperature)
        except LLMError as exc:
            if "invalid JSON" in str(exc) and len(batch) > 1:
                mid = len(batch) // 2
                self._log(f"Batch output unusable; splitting {len(batch)} questions into {mid} + {len(batch) - mid}.")
                left = self._call_solver(batch[:mid], chunks, style)
                left.update(self._call_solver(batch[mid:], chunks, style))
                return left
            raise

        valid_chunk_ids = {c.chunk_id for c in chunks}
        by_id = {q.id: q for q in batch}
        results: dict[str, SolvedAnswer] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id", "")).strip()
            q = by_id.get(qid)
            body = str(item.get("answer_markdown", "")).strip()
            if q is None or not body or qid in results:
                continue
            raw_cites = [str(c).strip().strip("[]") for c in item.get("citations", []) or []]
            cites = [c for c in raw_cites if c in valid_chunk_ids]
            results[qid] = SolvedAnswer(
                question_id=qid,
                marks=q.marks,
                question_text=q.text,
                answer_markdown=body,
                citations=cites,
            )
        return results

    def _verify(self, answers: list[SolvedAnswer], chunks: list[NoteChunk]) -> None:
        to_check = [a for a in answers if not is_not_found(a.answer_markdown)]
        if not to_check:
            return
        try:
            items = generate_json_list(self.llm, build_verifier_prompt(to_check, chunks), temperature=0.0)
        except LLMError as exc:
            self._log(f"Verifier skipped for this batch ({exc}).")
            return
        by_id = {a.question_id: a for a in to_check}
        for item in items:
            if not isinstance(item, dict):
                continue
            ans = by_id.get(str(item.get("question_id", "")))
            if ans is None or item.get("supported", True):
                continue
            claims = [str(c) for c in item.get("unsupported_claims", [])]
            revised = str(item.get("revised_answer_markdown", "")).strip()
            if revised:
                ans.answer_markdown = revised
            ans.needs_review = True
            ans.review_note = "Verifier removed unsupported claims: " + "; ".join(claims[:5]) if claims else "Verifier revised answer."

    def solve_batch(self, state: SessionState, batch: list[Question]) -> BatchResult:
        retriever = self._retriever_for(state)
        chunks = self._context_for(retriever, batch)
        solved = self._call_solver(batch, chunks, state.style_instructions)

        missing = [q for q in batch if q.id not in solved]
        if missing:
            self._log(f"Model skipped {', '.join(q.id for q in missing)}; retrying them once.")
            retry_chunks = self._context_for(retriever, missing)
            solved.update(self._call_solver(missing, retry_chunks, state.style_instructions))

        answers: list[SolvedAnswer] = []
        for q in batch:
            ans = solved.get(q.id) or SolvedAnswer(
                question_id=q.id,
                marks=q.marks,
                question_text=q.text,
                answer_markdown="[Solver did not return an answer for this question]",
                needs_review=True,
                review_note="Model omitted this question twice.",
            )
            if is_not_found(ans.answer_markdown):
                ans.needs_review = True
                ans.review_note = ans.review_note or "Not covered by the provided notes."
            elif not ans.citations and not ans.needs_review:
                ans.needs_review = True
                ans.review_note = "Answer has no valid citation to the notes."
            answers.append(ans)

        if self.settings.verify_long_answers and not is_short(batch[0].marks):
            self._verify(answers, chunks)

        for q in batch:
            q.status = "completed"
        state.answers = [a for a in state.answers if a.question_id not in {q.id for q in batch}] + answers
        state.batch_count += 1
        self._save_checkpoint(state)

        done, total = state.progress
        result = BatchResult(state.batch_count, batch[0].marks, answers, done, total)
        batches_dir = self.session_dir(state.session_id) / "batches"
        batches_dir.mkdir(exist_ok=True)
        (batches_dir / f"batch_{state.batch_count:03d}.md").write_text(result.to_markdown(), encoding="utf-8")
        return result

    def solve_iter(self, state: SessionState, max_batches: int | None = None) -> Iterator[BatchResult]:
        """Yield one BatchResult per controlled batch until the queue is empty."""
        count = 0
        while state.has_pending():
            if max_batches is not None and count >= max_batches:
                return
            batch = self.next_batch(state)
            if not batch:
                return
            done, total = state.progress
            self._log(
                f"Solving batch {state.batch_count + 1}: {len(batch)} × {batch[0].marks}-mark "
                f"({done}/{total} done)..."
            )
            yield self.solve_batch(state, batch)
            count += 1

    # ------------------------------------------------------------------ phase 3
    def export(self, state: SessionState, formats: tuple[str, ...] = ("md",)) -> dict[str, Path]:
        session_dir = self.session_dir(state.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {"md": export_markdown(state, session_dir)}
        if "docx" in formats:
            try:
                paths["docx"] = export_docx(state, session_dir)
            except RuntimeError as exc:
                self._log(str(exc))
        if "pdf" in formats:
            try:
                paths["pdf"] = export_pdf(state, session_dir)
            except RuntimeError as exc:
                self._log(str(exc))
        return paths

    # ------------------------------------------------------------------ convenience
    def run(
        self,
        question_bank_input: str | list[str],
        notes_input: str | list[str],
        style_instructions: str = "",
        session_id: str | None = None,
        on_topics_ready: Callable[[list[TopicImportance]], None] | None = None,
        on_batch_complete: Callable[[int, int, list[SolvedAnswer]], None] | None = None,
        formats: tuple[str, ...] = ("md",),
    ) -> tuple[SessionState, Path]:
        """Full pipeline in one call. Returns (final_state, markdown_key_path)."""
        state = self.prepare(question_bank_input, notes_input, style_instructions, session_id)
        if on_topics_ready:
            on_topics_ready(state.important_topics)
        for result in self.solve_iter(state):
            if on_batch_complete:
                on_batch_complete(result.batch_number, result.marks, result.answers)
        return state, self.export(state, formats)["md"]
