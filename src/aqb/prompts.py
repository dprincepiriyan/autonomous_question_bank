"""Prompt templates for question parsing, topic analysis, grounded batch solving, and verification.

Each prompt contains a stable marker line (e.g. `QUESTION BANK:`) that the offline
MockLLMClient keys off, so keep those markers intact when editing.
"""

from __future__ import annotations

import json

from aqb.models import NoteChunk, Question, SolvedAnswer, TopicImportance

SHORT_ANSWER_MAX_MARKS = 5


def is_short(marks: int) -> bool:
    return marks <= SHORT_ANSWER_MAX_MARKS


def build_question_parser_prompt(raw_question_bank: str, section_context: str = "") -> str:
    """Build prompt to extract all questions in (a slice of) the bank into a JSON array."""
    context_line = (
        f"CONTEXT: This text continues a larger question bank. The most recent section/unit header "
        f"before this slice was: \"{section_context}\". Apply its mark value / unit to questions "
        f"until a new header appears.\n"
        if section_context
        else ""
    )
    return f"""You are a strict academic question-bank parser.
Extract EVERY question from the input below and return ONLY a valid JSON array (no markdown fences, no commentary).

Schema for each item:
{{"id": "Q1", "marks": 2, "topic": "Short lesson/topic name", "text": "Exact question text"}}

Rules:
- Preserve every question exactly as written; do not merge, skip, paraphrase, or invent questions.
- Ignore headers, instructions, course codes, CO/BT-level columns and page furniture.
- Mark weight: use explicit marks if shown (e.g. "[2M]", "(13)", "(7)+(6)" sums to 13). Otherwise use the section header
  (e.g. "PART A (10 x 2 = 20)" means 2 marks each; "PART B" usually 13 marks; "PART C" usually 15 marks).
- If a long question has "(OR)" alternatives such as 11(a) / 11(b), output each alternative as a separate question.
- Topic: the specific sub-topic inside the lesson that the question tests (e.g. "CPU Scheduling", "Deadlocks"),
  1-4 words. Reuse the exact same topic name for every question on the same sub-topic. Do not use the unit title
  as the topic unless the question is about the unit as a whole.
{context_line}
QUESTION BANK:
{raw_question_bank}
"""


def build_topic_focus_prompt(
    topics: list[TopicImportance],
    questions: list[Question],
    notes_preview: str,
) -> str:
    """Ask the LLM only for *qualitative* focus points; weights are computed in code."""
    by_topic: dict[str, list[Question]] = {}
    for q in questions:
        by_topic.setdefault(q.topic, []).append(q)

    blocks = []
    for t in topics:
        qs = by_topic.get(t.topic, [])
        q_lines = "\n".join(f"  - ({q.marks}M) {q.text[:220]}" for q in qs[:25])
        blocks.append(
            f"### {t.topic} — {t.question_count} questions, {t.total_marks} marks, priority {t.priority}\n{q_lines}"
        )
    topics_block = "\n\n".join(blocks)

    return f"""You are an academic exam strategist helping a student prepare efficiently.
For each topic below, list 2-5 concrete sub-concepts the student must master, based on what the questions
repeatedly ask and what the notes cover (definitions, diagrams, derivations, comparisons, algorithms...).
Do NOT change the topic names, counts, marks, or priorities.

Return ONLY a valid JSON array (no markdown fences):
[{{"topic": "<exact topic name>", "focus_points": ["...", "..."], "rationale": "one sentence on why it matters"}}]

TOPICS TO ANALYZE:
{topics_block}

NOTES PREVIEW:
{notes_preview[:5000]}
"""


def _format_context(chunks: list[NoteChunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks) or "[No note chunks available]"


def build_batch_solver_prompt(
    batch: list[Question],
    chunks: list[NoteChunk],
    style_instructions: str,
) -> str:
    """Build a strictly grounded prompt for solving one controlled batch."""
    marks = batch[0].marks if batch else 2
    if is_short(marks):
        depth_guidance = (
            f"Each answer is worth {marks} marks: keep it concise and exam-ready "
            "(2-4 sentences or 3-5 crisp bullet points). Lead with the definition/key fact."
        )
    else:
        depth_guidance = (
            f"Each answer is worth {marks} marks: write a complete, well-structured exam answer using headings "
            "(Introduction/Definition → Core explanation or steps → Diagram (as a Mermaid code block) or formula "
            "(LaTeX) where the notes support it → Advantages/Applications/Comparison as relevant → Conclusion). "
            "Aim for roughly 400-700 words, without padding."
        )
    user_style = style_instructions.strip() or "None provided; use clear academic formatting."

    questions_block = "\n".join(
        f"- {q.id} ({q.marks} Marks, Topic: {q.topic}): {q.text}" for q in batch
    )

    return f"""You are a precise academic tutor writing the answer key for a controlled batch of {len(batch)} question(s).

GROUNDING & ANTI-HALLUCINATION RULES:
1. Answer ONLY using the PROVIDED STUDY NOTES below. Do not use outside knowledge for facts, numbers, or names.
2. For each answer, list the chunk IDs you used in "citations" (exactly as shown in square brackets, e.g. "notes.pdf#p4.C7").
3. If the notes do NOT contain enough information for a question, set "answer_markdown" to "[Not found in provided notes]"
   followed by one line saying what is missing, and set "citations" to []. Never guess.
4. {depth_guidance}
5. Follow the user's answer style instructions (they override formatting, never the grounding rules): {user_style}
6. Answer every question listed, in the same order, exactly once.

RETURN ONLY A VALID JSON ARRAY (no markdown fences around the JSON). Escape newlines inside strings as \\n.
[{{"question_id": "Q1", "answer_markdown": "...", "citations": ["notes.pdf#C1"]}}]

PROVIDED STUDY NOTES:
{_format_context(chunks)}

QUESTIONS TO SOLVE IN THIS BATCH:
{questions_block}
"""


def build_verifier_prompt(answers: list[SolvedAnswer], chunks: list[NoteChunk]) -> str:
    """Critic pass: check long answers against the notes and repair unsupported claims."""
    payload = [
        {"question_id": a.question_id, "question": a.question_text, "answer_markdown": a.answer_markdown}
        for a in answers
    ]
    return f"""You are a strict fact-checker for an exam answer key.
Compare every answer against the PROVIDED STUDY NOTES. Flag any claim, number, name, formula or step that is
NOT supported by the notes. If an answer contains unsupported content, rewrite it so it only contains supported
content (keep the same structure and style). If everything is supported, return it unchanged.

Return ONLY a valid JSON array (no markdown fences):
[{{"question_id": "Q1", "supported": true, "unsupported_claims": [], "revised_answer_markdown": "..."}}]

PROVIDED STUDY NOTES:
{_format_context(chunks)}

ANSWERS TO VERIFY:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
