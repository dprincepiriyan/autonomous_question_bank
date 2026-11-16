"""Interactive CLI for Autonomous Question Bank."""

from __future__ import annotations

import argparse

from aqb.config import Settings
from aqb.llm import create_llm_client
from aqb.models import SolvedAnswer, TopicImportance
from aqb.orchestrator import AutonomousSolverOrchestrator


def _print_topics(topics: list[TopicImportance]) -> None:
    print("\n=== IMPORTANT TOPICS TO FOCUS ON ===")
    for t in topics:
        print(f"- [{t.priority}] {t.topic} ({t.question_count} Qs, {t.total_marks} Marks): {t.rationale}")
    print("=" * 38 + "\n")


def _print_batch(batch_num: int, marks: int, solved: list[SolvedAnswer]) -> None:
    print(f"\n--- Batch {batch_num} ({len(solved)} x {marks}-Mark Questions) ---")
    for ans in solved:
        cites = f" [Sources: {', '.join(ans.citations)}]" if ans.citations else ""
        print(f"\n{ans.question_id}: {ans.question_text}{cites}")
        print(ans.answer_markdown)
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Question Bank Solver (10-12 x 2M, 5-6 x 13M per batch)"
    )
    parser.add_argument("--questions", help="Path to question bank file (or raw text)")
    parser.add_argument("--notes", help="Path to study notes file (or raw text)")
    parser.add_argument("--style", default=None, help="Optional answer formatting instructions")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini, openai, or mock")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.provider:
        settings = Settings(
            provider=args.provider.strip().lower(),
            model=settings.model,
            batch_size_2m=settings.batch_size_2m,
            batch_size_13m=settings.batch_size_13m,
            temperature=settings.temperature,
            top_k_chunks=settings.top_k_chunks,
            output_dir=settings.output_dir,
        )

    print("Welcome to Autonomous Question Bank (AQB)!")
    qb_input = args.questions or input("1. Enter path to Question Bank (or paste text): ").strip()
    notes_input = args.notes or input("2. Enter path to Study Notes/Material (or paste text): ").strip()
    if args.style is not None:
        style_input = args.style
    else:
        style_input = input("3. Optional answer style instructions (press Enter to skip): ").strip()

    llm = create_llm_client(settings)
    orchestrator = AutonomousSolverOrchestrator(llm=llm, settings=settings)

    state, key_path = orchestrator.run(
        question_bank_input=qb_input,
        notes_input=notes_input,
        style_instructions=style_input,
        on_topics_ready=_print_topics,
        on_batch_complete=_print_batch,
    )

    print(f"\nCompleted {len(state.answers)} questions across {state.batch_count} batches.")
    print(f"Final Answer Key saved to: {key_path}")


if __name__ == "__main__":
    main()
