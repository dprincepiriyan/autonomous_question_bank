"""Unit and integration tests for the Autonomous Question Bank orchestrator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqb.config import Settings
from aqb.llm import MockLLMClient
from aqb.orchestrator import AutonomousSolverOrchestrator


class OrchestratorTestCase(unittest.TestCase):
    def test_controlled_batching_and_final_key_generation(self) -> None:
        # 25 x 2-markers + 11 x 13-markers = 36 questions
        lines_2m = [f"Define concept {i} [2M] (topic: Core Basics)" for i in range(1, 26)]
        lines_13m = [
            f"Explain architecture {i} in detail [13M] (topic: System Design)" for i in range(1, 12)
        ]
        question_bank = "\n".join(["Part A (2 Marks)", *lines_2m, "Part B (13 Marks)", *lines_13m])
        notes = (
            "Core Basics covers foundational definitions, properties, and key terms.\n\n"
            "System Design explains layered architectures, state machines, and retrieval pipelines."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                provider="mock",
                batch_size_2m=10,
                batch_size_13m=5,
                output_dir=Path(tmpdir),
            )
            orchestrator = AutonomousSolverOrchestrator(llm=MockLLMClient(), settings=settings)

            observed_batches: list[tuple[int, int]] = []

            def on_batch(batch_num: int, marks: int, solved: list) -> None:
                observed_batches.append((marks, len(solved)))

            state, key_path = orchestrator.run(
                question_bank_input=question_bank,
                notes_input=notes,
                style_instructions="Use bullet points.",
                session_id="test_run",
                on_batch_complete=on_batch,
            )

            # 25 -> 10, 10, 5; 11 -> 5, 5, 1
            self.assertEqual(
                observed_batches,
                [(2, 10), (2, 10), (2, 5), (13, 5), (13, 5), (13, 1)],
            )
            self.assertEqual(len(state.answers), 36)
            self.assertEqual(state.batch_count, 6)
            self.assertFalse(state.has_pending())

            # Topic importance should rank System Design (11*13=143M) above Core Basics (25*2=50M)
            self.assertGreaterEqual(len(state.important_topics), 2)
            self.assertEqual(state.important_topics[0].topic, "System Design")
            self.assertEqual(state.important_topics[0].total_marks, 143)

            # Verify checkpoint & final key exist
            state_file = Path(tmpdir) / "test_run" / "state.json"
            self.assertTrue(state_file.exists())
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["answers"]), 36)

            self.assertTrue(key_path.exists())
            key_text = key_path.read_text(encoding="utf-8")
            self.assertIn("Important Topics to Focus On", key_text)
            self.assertIn("2-Mark Questions & Answer Key", key_text)
            self.assertIn("13-Mark Questions & Answer Key", key_text)


if __name__ == "__main__":
    unittest.main()
