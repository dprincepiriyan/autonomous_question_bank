"""Tests for question-bank parsing, topic ranking, repeated-question detection."""

from __future__ import annotations

from pathlib import Path

from aqb.models import Question
from aqb.question_parser import (
    assign_topics_from_notes,
    compute_topic_importance,
    find_repeated_questions,
    heuristic_parse_questions,
    key_concepts,
    split_question_bank,
    split_notes_sections,
)


def test_heuristic_parses_sample_bank(sample_bank: str) -> None:
    qs = heuristic_parse_questions(Path(sample_bank).read_text())
    assert len(qs) == 22
    assert [q.marks for q in qs].count(2) == 14
    assert [q.marks for q in qs].count(13) == 8
    assert qs[0].text == "Define a process."
    # Title lines before the first numbered question must not become questions.
    assert not any("QUESTION BANK" in q.text for q in qs)
    # Multi-line question with sub-part marks (7) + (6) is merged and summed to 13.
    q16 = qs[15]
    assert "(ii) Describe the context switch" in q16.text and q16.marks == 13


def test_heuristic_part_headers_with_counts_and_or_alternatives() -> None:
    raw = """
    PART – A (5 x 2 = 10 Marks)
    1. What is an algorithm?
    2. Define time complexity.
    PART – B (2 x 16 = 32 Marks)
    3. (a) Explain merge sort with an example.
    (OR)
    (b) Explain quick sort with an example.
    4. Explain Dijkstra's algorithm.
    """
    qs = heuristic_parse_questions(raw)
    assert [q.marks for q in qs] == [2, 2, 16, 16, 16]
    assert "merge sort" in qs[2].text and "quick sort" in qs[3].text


def test_heuristic_unnumbered_lines_and_inline_tags() -> None:
    raw = "2 Marks\nDefine entropy [2M] (topic: Information Theory)\nWhat is a channel? [2M]\n" \
          "13 Marks\nExplain Huffman coding in detail [13M] (topic: Source Coding)"
    qs = heuristic_parse_questions(raw)
    assert [(q.marks, q.topic) for q in qs] == [
        (2, "Information Theory"),
        (2, "General"),
        (13, "Source Coding"),
    ]


def test_unit_headers_become_topics() -> None:
    raw = "UNIT I - Basics\n1. Define X.\nUNIT II: Advanced Topics\n1. Define Y."
    qs = heuristic_parse_questions(raw)
    assert [q.topic for q in qs] == ["Basics", "Advanced Topics"]


def test_split_question_bank_keeps_header_context() -> None:
    lines = ["PART B (13 Marks)"] + [f"{i}. Explain concept number {i} in detail." for i in range(1, 60)]
    slices = split_question_bank("\n".join(lines), max_chars=400)
    assert len(slices) > 2
    assert slices[0][1] == ""
    assert all(header == "PART B (13 Marks)" for _, header in slices[1:])
    rejoined = "\n".join(text for text, _ in slices)
    assert rejoined.count("Explain concept") == 59


def test_topic_importance_is_computed_not_guessed() -> None:
    qs = [Question(f"Q{i}", "x", 2, "A") for i in range(5)] + [Question("Q9", "y", 13, "B")] * 2
    topics = compute_topic_importance(qs)
    assert topics[0].topic == "B" and topics[0].total_marks == 26 and topics[0].priority == "High"
    assert topics[1].topic == "A" and topics[1].total_marks == 10 and topics[1].question_count == 5


def test_repeated_questions_and_key_concepts() -> None:
    qs = [
        Question("Q1", "Define a semaphore.", 2),
        Question("Q2", "What is a semaphore? Define it.", 2),
        Question("Q3", "Explain semaphore based producer consumer problem.", 13),
        Question("Q4", "Explain the producer consumer problem.", 13),
        Question("Q5", "Define deadlock.", 2),
    ]
    groups = find_repeated_questions(qs)
    assert ["Q1", "Q2"] in groups
    terms = [c["term"] for c in key_concepts(qs)]
    assert "producer consumer" in terms


def test_notes_sections_and_topic_assignment(sample_notes: str) -> None:
    text = Path(sample_notes).read_text()
    titles = [t for t, _ in split_notes_sections(text)]
    assert titles == ["Process Concept", "Threads", "CPU Scheduling", "Process Synchronization", "Deadlocks"]

    qs = [
        Question("Q1", "Explain the Banker's algorithm.", 13),
        Question("Q2", "Explain Round Robin scheduling.", 13),
        Question("Q3", "List the benefits of multithreading.", 2),
    ]
    assert assign_topics_from_notes(qs, [text])
    assert [q.topic for q in qs] == ["Deadlocks", "CPU Scheduling", "Threads"]
