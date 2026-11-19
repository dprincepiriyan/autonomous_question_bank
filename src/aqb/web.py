"""Streamlit web UI for Autonomous Question Bank.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from aqb.config import BATCH_2M_RANGE, BATCH_13M_RANGE, DEFAULT_MODELS, Settings
from aqb.env import load_dotenv
from aqb.export import render_markdown
from aqb.llm import LLMError, create_llm_client
from aqb.orchestrator import AutonomousSolverOrchestrator, BatchResult


# ---------------------------------------------------------------------------- sidebar
def sidebar_settings() -> tuple[Settings, str | None]:
    base = Settings.from_env()
    st.sidebar.header("⚙️ Settings")
    providers = ["gemini", "openai", "mock"]
    provider = st.sidebar.selectbox(
        "LLM provider", providers, index=providers.index(base.provider) if base.provider in providers else 2,
        help="'mock' runs fully offline for a demo (answers are copied from your notes).",
    )
    default_model = base.model if provider == base.provider else DEFAULT_MODELS[provider]
    model = st.sidebar.text_input("Model", value=default_model, disabled=provider == "mock")
    api_key = None
    if provider != "mock":
        api_key = st.sidebar.text_input(
            "API key", type="password", help="Leave empty to use GEMINI_API_KEY / OPENAI_API_KEY from .env"
        ) or None

    st.sidebar.subheader("Batching (anti-hallucination)")
    b2 = st.sidebar.slider("2-mark questions per reply", *BATCH_2M_RANGE, value=base.batch_size_2m)
    b13 = st.sidebar.slider("13-mark questions per reply", *BATCH_13M_RANGE, value=base.batch_size_13m)
    verify = st.sidebar.checkbox(
        "Fact-check long answers (extra LLM call)", value=base.verify_long_answers
    )
    parse_mode = st.sidebar.selectbox(
        "Question parsing", ["auto", "llm", "heuristic"], index=["auto", "llm", "heuristic"].index(base.parse_mode)
    )
    settings = base.with_overrides(
        provider=provider,
        model=model if provider != "mock" else "mock",
        batch_size_2m=b2,
        batch_size_13m=b13,
        verify_long_answers=verify,
        parse_mode=parse_mode,
    )
    return settings, api_key


def sidebar_sessions(settings: Settings) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("📂 Saved sessions")
    sessions = AutonomousSolverOrchestrator.list_sessions(settings.output_dir)
    if not sessions:
        st.sidebar.caption("No sessions yet.")
        return
    labels = {f"{s['session_id']}  ({s['done']}/{s['total']})": s["session_id"] for s in sessions}
    choice = st.sidebar.selectbox("Open session", list(labels))
    if st.sidebar.button("Open / Resume", use_container_width=True):
        st.session_state["session_id"] = labels[choice]
        st.rerun()


# ---------------------------------------------------------------------------- helpers
def save_uploads(files, settings: Settings) -> list[str]:
    if not files:
        return []
    upload_dir = settings.output_dir / "_uploads" / uuid.uuid4().hex[:8]
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        dest = upload_dir / Path(f.name).name
        dest.write_bytes(f.getbuffer())
        paths.append(str(dest))
    return paths


def render_batch(result: BatchResult, expanded: bool = False) -> None:
    flagged = sum(1 for a in result.answers if a.needs_review)
    title = f"Batch {result.batch_number} — {len(result.answers)} × {result.marks}-mark"
    if flagged:
        title += f"  (⚠️ {flagged} to review)"
    with st.expander(title, expanded=expanded):
        st.markdown(result.to_markdown())


def render_topics(state) -> None:
    st.subheader("🎯 Important topics to focus on")
    if state.important_topics:
        st.dataframe(
            [
                {
                    "Topic": t.topic,
                    "Priority": t.priority,
                    "Questions": t.question_count,
                    "Total marks": t.total_marks,
                    "Focus on": "; ".join(t.focus_points),
                }
                for t in state.important_topics
            ],
            use_container_width=True,
            hide_index=True,
        )
    cols = st.columns(2)
    with cols[0]:
        if state.key_concepts:
            st.markdown("**Most frequently examined concepts**")
            st.markdown(", ".join(f"`{c['term']}` ({c['questions']})" for c in state.key_concepts))
    with cols[1]:
        if state.repeated_questions:
            st.markdown("**Repeated questions** (likely to be asked)")
            for g in state.repeated_questions[:10]:
                st.markdown(f"- {', '.join(g)}")


# ---------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Autonomous Question Bank", page_icon="📚", layout="wide")
    load_dotenv()
    settings, api_key = sidebar_settings()
    sidebar_sessions(settings)

    st.title("📚 Autonomous Question Bank")
    st.caption(
        f"Solves your question bank in controlled batches ({settings.batch_size_2m} × 2-mark / "
        f"{settings.batch_size_13m} × 13-mark per reply), grounded strictly in your notes."
    )

    try:
        llm = create_llm_client(settings, api_key=api_key)
    except (RuntimeError, ValueError) as exc:
        st.error(f"LLM setup failed: {exc}")
        st.stop()

    log_box = st.empty()
    orch = AutonomousSolverOrchestrator(llm, settings, on_log=lambda m: log_box.info(m))
    session_id = st.session_state.get("session_id")

    # ---- Step 1: inputs ----
    if not session_id:
        st.header("1️⃣ Provide your material")
        c1, c2 = st.columns(2)
        with c1:
            qb_files = st.file_uploader(
                "Question bank", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
            )
            qb_text = st.text_area("…or paste the question bank", height=150)
        with c2:
            notes_files = st.file_uploader(
                "Study notes / material", type=["pdf", "docx", "pptx", "txt", "md"], accept_multiple_files=True
            )
            notes_text = st.text_area("…or paste notes", height=150)
        style = st.text_area(
            "Optional: how should the answers look?",
            placeholder="e.g. 2-markers in 2-3 lines. 13-markers: intro, diagram, explanation, advantages, conclusion.",
        )
        name = st.text_input("Session name (optional)")

        if st.button("🚀 Analyze & start", type="primary"):
            questions = save_uploads(qb_files, settings) + ([qb_text] if qb_text.strip() else [])
            notes = save_uploads(notes_files, settings) + ([notes_text] if notes_text.strip() else [])
            if not questions or not notes:
                st.warning("Please provide both a question bank and study notes.")
                st.stop()
            try:
                with st.spinner("Parsing question bank and ranking important topics…"):
                    state = orch.prepare(questions, notes, style, session_id=name or None)
            except (LLMError, ValueError, RuntimeError, FileExistsError) as exc:
                st.error(str(exc))
                st.stop()
            st.session_state["session_id"] = state.session_id
            st.session_state["auto_run"] = True
            st.rerun()
        return

    # ---- Step 2: topics + solving loop ----
    try:
        state = orch.load_session(session_id)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.session_state.pop("session_id", None)
        st.stop()

    top = st.columns([3, 1])
    top[0].header(f"Session `{state.session_id}`")
    if top[1].button("➕ New session"):
        st.session_state.pop("session_id", None)
        st.rerun()

    render_topics(state)

    st.header("2️⃣ Answer key")
    done, total = state.progress
    progress = st.progress(done / total if total else 1.0, text=f"{done}/{total} questions solved")

    # Previously solved batches (reconstructed from saved batch replies).
    batches_dir = orch.session_dir(state.session_id) / "batches"
    for path in sorted(batches_dir.glob("batch_*.md")) if batches_dir.exists() else []:
        with st.expander(path.stem.replace("_", " ").title()):
            st.markdown(path.read_text(encoding="utf-8"))

    if state.has_pending():
        c1, c2 = st.columns(2)
        run_all = c1.button("▶️ Solve all remaining", type="primary") or st.session_state.pop("auto_run", False)
        run_one = c2.button("⏭️ Solve next batch only")
        if run_all or run_one:
            try:
                for result in orch.solve_iter(state, max_batches=1 if run_one else None):
                    render_batch(result, expanded=True)
                    progress.progress(result.done / result.total, text=f"{result.done}/{result.total} questions solved")
            except LLMError as exc:
                st.error(f"{exc}\n\nProgress is saved — press **Solve all remaining** to resume.")
                st.stop()
            st.rerun()
        st.caption("Tip: you can stop at any time — every batch is checkpointed and can be resumed later.")
    else:
        log_box.success("All questions solved! 🎉")

    # ---- Step 3: downloads ----
    st.header("3️⃣ Download")
    paths = orch.export(state, formats=("md", "docx", "pdf"))
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "⬇️ Answer key (Markdown)",
        data=render_markdown(state),
        file_name=f"{state.session_id}_answer_key.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if "docx" in paths:
        d2.download_button(
            "⬇️ Answer key (Word)",
            data=paths["docx"].read_bytes(),
            file_name=f"{state.session_id}_answer_key.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    if "pdf" in paths:
        d3.download_button(
            "⬇️ Answer key (PDF)",
            data=paths["pdf"].read_bytes(),
            file_name=f"{state.session_id}_answer_key.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
