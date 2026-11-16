"""Prompt templates for question parsing, topic analysis, and grounded batch solving."""

from __future__ import annotations

from aqb.models import NoteChunk, Question


def build_question_parser_prompt(raw_question_bank: str) -> str:
    """Build prompt to extract all questions into a structured JSON array."""
    return f"""You are a strict academic question-bank parser.
Extract EVERY question from the input below and return ONLY valid JSON (no markdown fences).

Schema for each item:
{{
  "id": "Q1",
  "marks": 2,
  "topic": "Short lesson/topic name",
  "text": "Exact question text"
}}

Rules:
- Preserve every question; do not merge, skip, or invent questions.
- Classify mark weight accurately (e.g., 2 for 2-markers, 13 for 13-markers). If marks are grouped under Part A (2 marks) / Part B (13 marks), apply that mark value to each question in the section.
- Infer a concise topic name for each question.

QUESTION BANK:
{raw_question_bank}
"""


def build_topic_analysis_prompt(questions: list[Question], notes_preview: str) -> str:
    """Build prompt to summarize the most important topics to focus on."""
    q_lines = "\n".join(f"- [{q.id} | {q.marks}M | {q.topic}] {q.text}" for q in questions)
    return f"""You are an academic exam strategist.
Analyze the parsed question bank and study notes preview below.
Identify the most important topics in this lesson that the student should focus on, ranked by total mark weightage and frequency.

Return ONLY valid JSON (no markdown fences) as a list of objects:
[
  {{
    "topic": "Topic Name",
    "question_count": 4,
    "total_marks": 30,
    "priority": "High",
    "rationale": "Why this topic matters and what sub-concepts repeat."
  }}
]

PARSED QUESTIONS:
{q_lines}

NOTES PREVIEW:
{notes_preview[:4000]}
"""


def build_batch_solver_prompt(
    batch: list[Question],
    chunks: list[NoteChunk],
    style_instructions: str,
) -> str:
    """Build a strictly grounded prompt for solving one controlled batch."""
    marks = batch[0].marks if batch else 2
    depth_guidance = (
        "Keep each 2-mark answer concise, accurate, and exam-ready (2-4 sentences or key bullet points)."
        if marks <= 2
        else "Provide a comprehensive, structured 13-mark answer (Introduction/Definition, Core Explanation/Steps, Diagram/Formula if applicable, Advantages/Applications, Conclusion)."
    )
    user_style = style_instructions.strip() or "None provided; use clear academic formatting."

    context_block = "\n\n".join(
        f"[{c.chunk_id}]\n{c.text}" for c in chunks
    ) or "[No note chunks available]"

    questions_block = "\n".join(
        f"- {q.id} ({q.marks} Marks, Topic: {q.topic}): {q.text}" for q in batch
    )

    return f"""You are a precise academic tutor solving a controlled batch of {len(batch)} ({marks}-mark) questions.

GROUNDING & ANTI-HALLUCINATION RULES:
1. Answer ONLY using the PROVIDED STUDY NOTES below.
2. Cite the chunk IDs (e.g., [notes.pdf#C3]) used for each answer.
3. If the provided notes do NOT contain enough information to answer a question, set answer_markdown to "[Not found in provided notes]" and citations to []. Do NOT fabricate facts.
4. {depth_guidance}
5. Follow the user's optional style instructions: {user_style}

RETURN ONLY VALID JSON (no markdown fences) as a list matching the exact batch order:
[
  {{
    "question_id": "Q1",
    "answer_markdown": "...",
    "citations": ["notes.pdf#C1"]
  }}
]

PROVIDED STUDY NOTES:
{context_block}

QUESTIONS TO SOLVE IN THIS BATCH:
{questions_block}
"""
