"""Command-line interface for Autonomous Question Bank.

Examples:
    aqb                                   # interactive prompts
    aqb -q bank.pdf -n unit1.pdf unit2.pdf --style "bullet points" --docx
    aqb --list-sessions
    aqb --resume session_20261119_210000_ab12
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aqb.config import Settings
from aqb.env import load_dotenv
from aqb.llm import LLMError, create_llm_client
from aqb.models import SessionState
from aqb.orchestrator import AutonomousSolverOrchestrator, BatchResult, SessionNotFoundError


def _print_topics(state: SessionState) -> None:
    print("\n" + "=" * 60)
    print(" IMPORTANT TOPICS TO FOCUS ON")
    print("=" * 60)
    for i, t in enumerate(state.important_topics, start=1):
        print(f"{i:>2}. [{t.priority:<6}] {t.topic} — {t.question_count} Qs, {t.total_marks} marks")
        for p in t.focus_points:
            print(f"      • {p}")
    if state.key_concepts:
        print("\n Frequently examined: " + ", ".join(c["term"] for c in state.key_concepts[:10]))
    if state.repeated_questions:
        print(" Repeated questions: " + "; ".join(", ".join(g) for g in state.repeated_questions[:8]))
    print("=" * 60 + "\n")


def _print_batch(result: BatchResult) -> None:
    print("\n" + "-" * 60)
    print(f" Batch {result.batch_number}: {len(result.answers)} × {result.marks}-mark  "
          f"[{result.done}/{result.total} done]")
    print("-" * 60)
    for a in result.answers:
        flag = "  ⚠ needs review" if a.needs_review else ""
        print(f"\n{a.question_id}. {a.question_text}{flag}")
        print(a.answer_markdown)
        if a.citations:
            print(f"   (Sources: {', '.join(a.citations)})")


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _ask_multi(prompt: str) -> list[str]:
    """Accept one or more paths separated by commas, or pasted text."""
    raw = _ask(prompt)
    if not raw:
        return []
    parts = [p.strip().strip("'\"") for p in raw.split(",")]
    if all(Path(p).expanduser().is_file() for p in parts):
        return [str(Path(p).expanduser()) for p in parts]
    return [raw]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aqb",
        description="Autonomous Question Bank solver: 10-12 short / 5-6 long answers per batch, grounded in your notes.",
    )
    p.add_argument("-q", "--questions", nargs="+", help="Question bank file(s) (txt/md/pdf/docx) or raw text")
    p.add_argument("-n", "--notes", nargs="+", help="Study notes file(s) (txt/md/pdf/docx/pptx) or raw text")
    p.add_argument("-s", "--style", default=None, help="Optional answer style instructions")
    p.add_argument("--style-file", default=None, help="Read answer style instructions from a file")
    p.add_argument("--session", default=None, help="Name for a new session (default: timestamp)")
    p.add_argument("--resume", metavar="SESSION_ID", help="Resume an interrupted session")
    p.add_argument("--list-sessions", action="store_true", help="List saved sessions and exit")
    p.add_argument("--provider", choices=["gemini", "openai", "mock"], help="LLM provider (default from .env)")
    p.add_argument("--model", help="Model name (default from .env)")
    p.add_argument("--batch-2m", type=int, help="Short-answer batch size (10-12)")
    p.add_argument("--batch-13m", type=int, help="Long-answer batch size (5-6)")
    p.add_argument("--parse-mode", choices=["auto", "llm", "heuristic"], help="Question bank parsing strategy")
    p.add_argument("--verify", action="store_true", help="Run a fact-check pass on long answers")
    p.add_argument("--max-batches", type=int, help="Stop after N batches (resume later)")
    p.add_argument("--docx", action="store_true", help="Also export the final key as .docx")
    p.add_argument("--pdf", action="store_true", help="Also export the final key as .pdf")
    p.add_argument("--output-dir", help="Where sessions are stored (default: outputs/)")
    p.add_argument("--env-file", default=".env", help="Path to .env file")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env().with_overrides(
        provider=args.provider,
        model=args.model,
        batch_size_2m=args.batch_2m,
        batch_size_13m=args.batch_13m,
        parse_mode=args.parse_mode,
        verify_long_answers=True if args.verify else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    if args.list_sessions:
        sessions = AutonomousSolverOrchestrator.list_sessions(settings.output_dir)
        if not sessions:
            print(f"No sessions in {settings.output_dir}/")
        for s in sessions:
            status = "complete" if s["done"] == s["total"] else "in progress"
            print(f"{s['session_id']:<40} {s['done']:>4}/{s['total']:<4} {status}")
        return 0

    try:
        llm = create_llm_client(settings)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    orch = AutonomousSolverOrchestrator(llm, settings, on_log=lambda m: print(f"» {m}"))
    print(f"Autonomous Question Bank  (provider: {settings.provider}, model: {settings.model}, "
          f"batches: {settings.batch_size_2m}×short / {settings.batch_size_13m}×long)")

    try:
        if args.resume:
            state = orch.load_session(args.resume)
            done, total = state.progress
            print(f"Resuming '{state.session_id}' — {done}/{total} already solved.")
        else:
            questions = args.questions or _ask_multi("1. Question bank file path(s) (comma-separated) or paste text: ")
            notes = args.notes or _ask_multi("2. Study notes file path(s) (comma-separated) or paste text: ")
            if args.style_file:
                style = Path(args.style_file).read_text(encoding="utf-8")
            elif args.style is not None:
                style = args.style
            else:
                style = _ask("3. Optional answer style instructions (Enter to skip): ")
            if not questions or not notes:
                print("Error: both a question bank and study notes are required.", file=sys.stderr)
                return 2
            state = orch.prepare(questions, notes, style, session_id=args.session)
            print(f"Session: {state.session_id}")

        _print_topics(state)

        for result in orch.solve_iter(state, max_batches=args.max_batches):
            _print_batch(result)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — resume with:")
        print(f"  aqb --resume {state.session_id}" if "state" in locals() else "  (no session created yet)")
        return 130
    except (LLMError, SessionNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        if "state" in locals():
            print(f"Progress is saved — resume with: aqb --resume {state.session_id}", file=sys.stderr)
        return 1

    formats = ["md"]
    if args.docx:
        formats.append("docx")
    if args.pdf:
        formats.append("pdf")
    paths = orch.export(state, formats=tuple(formats))
    done, total = state.progress
    review = sum(1 for a in state.answers if a.needs_review)
    print(f"\n✔ {done}/{total} questions solved in {state.batch_count} batches ({review} flagged for review).")
    for fmt, path in paths.items():
        print(f"  {fmt.upper()}: {path}")
    if state.has_pending():
        print(f"Stopped early. Continue with: aqb --resume {state.session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
