"""Exporters for the final answer key (Markdown always; DOCX via python-docx; PDF via PyMuPDF)."""

from __future__ import annotations

import html
import re
from pathlib import Path

from aqb.models import NOT_FOUND_MARKER, SessionState, SolvedAnswer


def _md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(state: SessionState) -> str:
    answers = state.ordered_answers()
    review = [a for a in answers if a.needs_review]
    done, total = state.progress

    lines: list[str] = [
        "# Final Answer Key",
        "",
        f"- **Session:** `{state.session_id}`",
        f"- **Questions solved:** {done} / {total} in {state.batch_count} batch(es)",
        f"- **Answers needing review:** {len(review)}",
    ]
    if state.notes_sources:
        lines.append(f"- **Study material:** {', '.join(state.notes_sources)}")
    if state.style_instructions:
        lines.append(f"- **Answer style:** {state.style_instructions}")

    # ---- Important topics ----
    lines += ["", "## Important Topics to Focus On", ""]
    if state.important_topics:
        lines.append("| # | Topic | Priority | Questions | Total Marks | Why |")
        lines.append("| :-: | :--- | :--- | :-: | :-: | :--- |")
        for i, t in enumerate(state.important_topics, start=1):
            lines.append(
                f"| {i} | {_md_escape_cell(t.topic)} | **{t.priority}** | {t.question_count} | "
                f"{t.total_marks} | {_md_escape_cell(t.rationale)} |"
            )
        focus = [t for t in state.important_topics if t.focus_points]
        if focus:
            lines += ["", "### What to master in each topic", ""]
            for t in focus:
                lines.append(f"**{t.topic}** ({t.priority})")
                lines += [f"- {p}" for p in t.focus_points]
                lines.append("")
    else:
        lines.append("_No topic analysis available._")

    if state.key_concepts:
        lines += ["", "### Most frequently examined concepts", ""]
        lines.append(
            ", ".join(f"`{c['term']}` ({c['questions']} Qs)" for c in state.key_concepts)
        )

    if state.repeated_questions:
        by_id = {q.id: q for q in state.questions}
        lines += ["", "### Repeated questions (high chance of being asked)", ""]
        for group in state.repeated_questions:
            first = by_id.get(group[0])
            label = first.text[:120] if first else group[0]
            lines.append(f"- {', '.join(group)} — _{label}_")

    # ---- Answers grouped by marks ----
    grouped: dict[int, list[SolvedAnswer]] = {}
    for ans in answers:
        grouped.setdefault(ans.marks, []).append(ans)

    part_letter = iter("ABCDEFGHIJ")
    for marks in sorted(grouped):
        lines += ["", f"## Part {next(part_letter)} — {marks}-Mark Questions", ""]
        for ans in grouped[marks]:
            flag = " ⚠️" if ans.needs_review else ""
            lines.append(f"### {ans.question_id}. {ans.question_text}{flag}")
            lines.append("")
            lines.append(ans.answer_markdown.strip())
            if ans.citations:
                lines += ["", f"<sub>Sources: {', '.join(ans.citations)}</sub>"]
            if ans.review_note:
                lines += ["", f"> **Review:** {ans.review_note}"]
            lines.append("")

    if review:
        lines += ["", "## Needs Review", ""]
        lines.append(
            "These answers were not fully supported by the notes or were flagged by the verifier. "
            "Check them against your textbook."
        )
        lines.append("")
        for a in review:
            lines.append(f"- **{a.question_id}** ({a.marks}M): {a.review_note or 'Flagged'}")

    return "\n".join(lines).rstrip() + "\n"


def export_markdown(state: SessionState, session_dir: Path) -> Path:
    path = session_dir / "final_answer_key.md"
    path.write_text(render_markdown(state), encoding="utf-8")
    return path


_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")


def _add_inline_runs(paragraph, text: str) -> None:
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def export_docx(state: SessionState, session_dir: Path) -> Path:
    """Render the Markdown key into a Word document (headings, bullets, tables, code blocks)."""
    try:
        import docx
        from docx.shared import Pt
    except ImportError as exc:
        raise RuntimeError("python-docx is required for DOCX export. Install with: pip install python-docx") from exc

    document = docx.Document()
    md = render_markdown(state)
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            para = document.add_paragraph()
            run = para.add_run("\n".join(code))
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            i += 1
            continue

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip().replace("\\|", "|") for c in lines[i].strip().strip("|").split(" | ")]
                if not all(re.fullmatch(r":?-+:?", c.strip("|").strip() or "-") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                width = max(len(r) for r in rows)
                table = document.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                for r_idx, row in enumerate(rows):
                    for c_idx in range(width):
                        cell_text = row[c_idx] if c_idx < len(row) else ""
                        cell_para = table.cell(r_idx, c_idx).paragraphs[0]
                        _add_inline_runs(cell_para, cell_text)
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            document.add_heading(heading.group(2), level=min(len(heading.group(1)), 4))
        elif re.match(r"^[-*]\s+", stripped):
            _add_inline_runs(document.add_paragraph(style="List Bullet"), re.sub(r"^[-*]\s+", "", stripped))
        elif re.match(r"^\d+[.)]\s+", stripped):
            _add_inline_runs(document.add_paragraph(style="List Number"), re.sub(r"^\d+[.)]\s+", "", stripped))
        elif stripped.startswith(">"):
            para = document.add_paragraph()
            _add_inline_runs(para, stripped.lstrip("> "))
            for run in para.runs:
                run.italic = True
        elif stripped:
            text = re.sub(r"</?sub>", "", stripped)
            _add_inline_runs(document.add_paragraph(), text)
        i += 1

    path = session_dir / "final_answer_key.docx"
    document.save(str(path))
    return path


def _inline_to_html(text: str) -> str:
    """Convert inline Markdown (**bold**, *italic*, `code`) into escaped HTML."""
    out: list[str] = []
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append(f"<strong>{html.escape(part[2:-2])}</strong>")
        elif part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(f"<em>{html.escape(part[1:-1])}</em>")
        else:
            out.append(html.escape(part))
    return "".join(out)


def render_html(state: SessionState) -> str:
    """Render the Markdown answer key into clean HTML suitable for PDF generation."""
    md = render_markdown(state)
    lines = md.splitlines()
    body: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("```"):
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            body.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            i += 1
            continue

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip().replace("\\|", "|") for c in lines[i].strip().strip("|").split(" | ")]
                if not all(re.fullmatch(r":?-+:?", c.strip("|").strip() or "-") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                body.append("<table>")
                for r_idx, row in enumerate(rows):
                    tag = "th" if r_idx == 0 else "td"
                    cells_html = "".join(f"<{tag}>{_inline_to_html(c)}</{tag}>" for c in row)
                    body.append(f"<tr>{cells_html}</tr>")
                body.append("</table>")
            continue

        if re.match(r"^[-*]\s+", stripped):
            body.append("<ul>")
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                item = re.sub(r"^[-*]\s+", "", lines[i].strip())
                body.append(f"<li>{_inline_to_html(item)}</li>")
                i += 1
            body.append("</ul>")
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            body.append("<ol>")
            while i < len(lines) and re.match(r"^\d+[.)]\s+", lines[i].strip()):
                item = re.sub(r"^\d+[.)]\s+", "", lines[i].strip())
                body.append(f"<li>{_inline_to_html(item)}</li>")
                i += 1
            body.append("</ol>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            lvl = min(len(heading.group(1)), 4)
            body.append(f"<h{lvl}>{_inline_to_html(heading.group(2))}</h{lvl}>")
        elif stripped.startswith(">"):
            body.append(f"<blockquote>{_inline_to_html(stripped.lstrip('> '))}</blockquote>")
        elif stripped.startswith("<sub>") and stripped.endswith("</sub>"):
            body.append(f"<p class='cite'>{_inline_to_html(stripped[5:-6])}</p>")
        elif stripped:
            text = re.sub(r"</?sub>", "", stripped)
            body.append(f"<p>{_inline_to_html(text)}</p>")
        i += 1

    css = """
    body { font-family: sans-serif; font-size: 10pt; line-height: 1.45; color: #1a1a1a; }
    h1 { font-size: 18pt; border-bottom: 1.5pt solid #2563eb; padding-bottom: 4pt; color: #0f172a; }
    h2 { font-size: 13.5pt; margin-top: 14pt; color: #1e3a8a; border-bottom: 0.5pt solid #cbd5e1; padding-bottom: 2pt; }
    h3 { font-size: 11pt; margin-top: 10pt; color: #0f172a; }
    h4 { font-size: 10pt; margin-top: 8pt; color: #334155; }
    table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
    th, td { border: 0.5pt solid #94a3b8; padding: 4pt 6pt; text-align: left; }
    th { background-color: #f1f5f9; font-weight: bold; }
    pre { background-color: #f8fafc; border: 0.5pt solid #cbd5e1; padding: 6pt; font-family: monospace; font-size: 8.5pt; }
    code { font-family: monospace; font-size: 9pt; background-color: #f1f5f9; }
    blockquote { border-left: 2.5pt solid #d97706; padding-left: 8pt; color: #78350f; margin: 6pt 0; }
    p.cite { font-size: 8.5pt; color: #475569; font-style: italic; margin-top: 2pt; }
    """
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(body)}</body></html>"


def export_pdf(state: SessionState, session_dir: Path) -> Path:
    """Render the answer key into a paginated A4 PDF using PyMuPDF Story."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF export. Install with: pip install pymupdf") from exc

    path = session_dir / "final_answer_key.pdf"
    story = pymupdf.Story(html=render_html(state))
    mediabox = pymupdf.paper_rect("a4")
    content_rect = mediabox + (42, 42, -42, -42)
    writer = pymupdf.DocumentWriter(str(path))
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(content_rect)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return path


def is_not_found(answer_markdown: str) -> bool:
    return answer_markdown.strip().startswith(NOT_FOUND_MARKER)
