"""Stateful orchestrator for the autonomous question-bank solving loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from aqb.config import Settings
from aqb.ingestion import NoteRetriever, chunk_notes, load_text_from_path
from aqb.llm import LLMClient, parse_json_response
from aqb.models import (
    Question,
    SessionState,
    SolvedAnswer,
    TopicImportance,
)
from aqb.prompts import (
    build_batch_solver_prompt,
    build_question_parser_prompt,
    build_topic_analysis_prompt,
)

BatchCallback = Callable[[int, int, list[SolvedAnswer]], None]


class AutonomousSolverOrchestrator:
    """Drives ingestion, topic prioritization, batch solving, checkpointing, and key export."""

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or Settings.from_env()

    def run(
        self,
        question_bank_input: str,
        notes_input: str,
        style_instructions: str = "",
        session_id: str | None = None,
        on_topics_ready: Callable[[list[TopicImportance]], None] | None = None,
        on_batch_complete: BatchCallback | None = None,
    ) -> tuple[SessionState, Path]:
        """Execute the full autonomous loop and return (final_state, answer_key_path)."""
        qb_text, _ = load_text_from_path(question_bank_input)
        notes_text, notes_source = load_text_from_path(notes_input)

        chunks = chunk_notes(
            notes_text,
            source=notes_source,
            chunk_size=self.settings.chunk_size_chars,
            overlap=self.settings.chunk_overlap_chars,
        )
        retriever = NoteRetriever(chunks)

        questions = self._parse_questions(qb_text)
        topics = self._analyze_topics(questions, notes_text)
        if on_topics_ready:
            on_topics_ready(topics)

        sid = session_id or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
        session_dir = self.settings.output_dir / sid
        session_dir.mkdir(parents=True, exist_ok=True)

        state = SessionState(
            session_id=sid,
            style_instructions=style_instructions.strip(),
            questions=questions,
            important_topics=topics,
        )
        self._save_checkpoint(state, session_dir)

        while state.has_pending():
            batch, batch_marks = self._next_batch(state)
            if not batch:
                break

            relevant_chunks = retriever.retrieve(
                [q.text for q in batch],
                top_k=self.settings.top_k_chunks,
            )
            prompt = build_batch_solver_prompt(
                batch=batch,
                chunks=relevant_chunks,
                style_instructions=state.style_instructions,
            )
            raw_reply = self.llm.generate(prompt, temperature=self.settings.temperature)
            solved_batch = self._parse_batch_answers(batch, raw_reply)

            solved_ids = {ans.question_id for ans in solved_batch}
            for q in batch:
                if q.id in solved_ids:
                    q.status = "completed"
                else:
                    # Fallback to prevent infinite loops if LLM omits an ID
                    q.status = "completed"
                    solved_batch.append(
                        SolvedAnswer(
                            question_id=q.id,
                            marks=q.marks,
                            question_text=q.text,
                            answer_markdown="[Solver omitted this question in batch response]",
                        )
                    )

            state.answers.extend(solved_batch)
            state.batch_count += 1
            self._save_checkpoint(state, session_dir)

            if on_batch_complete:
                on_batch_complete(state.batch_count, batch_marks, solved_batch)

        key_path = self._export_answer_key(state, session_dir)
        return state, key_path

    def _parse_questions(self, raw_question_bank: str) -> list[Question]:
        prompt = build_question_parser_prompt(raw_question_bank)
        raw = self.llm.generate(prompt, temperature=0.1)
        parsed = parse_json_response(raw)
        if not isinstance(parsed, list):
            raise ValueError("Question parser did not return a JSON list.")
        return [Question.from_dict(item) for item in parsed]

    def _analyze_topics(
        self,
        questions: list[Question],
        notes_text: str,
    ) -> list[TopicImportance]:
        prompt = build_topic_analysis_prompt(questions, notes_text)
        raw = self.llm.generate(prompt, temperature=0.1)
        parsed = parse_json_response(raw)
        if not isinstance(parsed, list):
            return []
        return [TopicImportance(**item) for item in parsed]

    def _next_batch(self, state: SessionState) -> tuple[list[Question], int]:
        """Select the next controlled batch: 10-12 for 2M, 5-6 for 13M."""
        pending_2m = state.pending_by_marks(2)
        if pending_2m:
            return pending_2m[: self.settings.batch_size_2m], 2

        pending_13m = state.pending_by_marks(13)
        if pending_13m:
            return pending_13m[: self.settings.batch_size_13m], 13

        # Handle any other mark weights cleanly
        remaining = [q for q in state.questions if q.status == "pending"]
        if not remaining:
            return [], 0
        target_marks = remaining[0].marks
        limit = self.settings.batch_size_2m if target_marks <= 4 else self.settings.batch_size_13m
        same_mark = [q for q in remaining if q.marks == target_marks]
        return same_mark[:limit], target_marks

    @staticmethod
    def _parse_batch_answers(batch: list[Question], raw_reply: str) -> list[SolvedAnswer]:
        parsed = parse_json_response(raw_reply)
        if not isinstance(parsed, list):
            raise ValueError("Batch solver did not return a JSON list.")
        by_id = {q.id: q for q in batch}
        results: list[SolvedAnswer] = []
        for item in parsed:
            qid = str(item["question_id"])
            q = by_id.get(qid)
            if q is None:
                continue
            results.append(
                SolvedAnswer(
                    question_id=qid,
                    marks=q.marks,
                    question_text=q.text,
                    answer_markdown=str(item.get("answer_markdown", "")).strip(),
                    citations=[str(c) for c in item.get("citations", [])],
                )
            )
        return results

    @staticmethod
    def _save_checkpoint(state: SessionState, session_dir: Path) -> None:
        state_path = session_dir / "state.json"
        state_path.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _export_answer_key(state: SessionState, session_dir: Path) -> Path:
        lines: list[str] = [
            f"# Final Answer Key ({state.session_id})",
            "",
            f"- **Total Questions Solved:** {len(state.answers)}",
            f"- **Total Batches:** {state.batch_count}",
            "",
            "## Important Topics to Focus On",
            "",
        ]
        if state.important_topics:
            lines.append("| Topic | Priority | Questions | Total Marks | Focus Rationale |")
            lines.append("| :--- | :--- | :---: | :---: | :--- |")
            for t in state.important_topics:
                lines.append(
                    f"| {t.topic} | **{t.priority}** | {t.question_count} | {t.total_marks} | {t.rationale} |"
                )
        else:
            lines.append("_No topic analysis generated._")

        grouped: dict[int, list[SolvedAnswer]] = {}
        for ans in state.answers:
            grouped.setdefault(ans.marks, []).append(ans)

        for marks in sorted(grouped):
            lines.extend(["", f"## {marks}-Mark Questions & Answer Key", ""])
            for ans in grouped[marks]:
                cites = f" *(Sources: {', '.join(ans.citations)})*" if ans.citations else ""
                lines.append(f"### {ans.question_id}: {ans.question_text}{cites}")
                lines.append("")
                lines.append(ans.answer_markdown)
                lines.append("")

        key_path = session_dir / "final_answer_key.md"
        key_path.write_text("\n".join(lines), encoding="utf-8")
        return key_path
