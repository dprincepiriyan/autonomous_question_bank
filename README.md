# Autonomous Question Bank (AQB)

An autonomous, stateful agent (CLI + Streamlit web app) that solves academic question banks in controlled batches grounded strictly in your study material, highlights the highest-yield topics and repeated questions to focus on, and exports a complete answer key in **Markdown, Word (`.docx`), and PDF (`.pdf`)**.

## Key Features

- **Controlled Batching (Anti-Hallucination & Anti-Truncation)**
  - **10–12 two-mark (short) questions** per batch reply
  - **5–6 thirteen-mark (long) questions** per batch reply
  - Single mark-weight per batch so answer depth stays consistent
- **Strict Grounding & Verification**
  - Small notes (`≤ 60k` chars) are passed in full; large notes use per-question BM25 retrieval with page-aware chunk citations (`notes.pdf#p4.C7`)
  - Missing topics are flagged `[Not found in provided notes]` instead of fabricated
  - Uncited or omitted answers are automatically retried or marked `⚠️ Needs Review`
  - Optional `--verify` critic pass fact-checks long answers against the source chunks
- **Important Topics & Exam Strategy**
  - Deterministic mark-weightage ranking per lesson sub-topic (derived from unit headers or notes headings)
  - Detects **most frequently examined concepts** and **repeated questions**
- **Resumable & Fault-Tolerant**
  - Atomic checkpoint (`state.json`) + per-batch reply logs (`batches/batch_NNN.md`) after every turn
  - Exponential backoff retries, JSON auto-repair, and automatic batch-splitting on truncated LLM output
  - Resume any interrupted run with `aqb --resume <session_id>` or via the Streamlit sidebar
- **Multi-Format Input & Output**
  - Reads `.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, or pasted text (supports multiple files)
  - Exports `final_answer_key.md`, `final_answer_key.docx`, and `final_answer_key.pdf`

---

## Documentation

- [System Architecture](docs/ARCHITECTURE.md) — Component diagram, module map, and anti-hallucination guardrails.
- [Agentic Workflow](docs/WORKFLOW.md) — End-to-end sequence diagram, batching rules, recovery & resume lifecycle.

---

## Quickstart

### 1. Clone & Install

```bash
git clone git@github.com:dprincepiriyan/autonomous_question_bank.git
cd autonomous_question_bank

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

*(Install only what you need: `.[llm]`, `.[docs]`, `.[web]`, or `.[dev]`.)*

### 2. Configure `.env`

```bash
cp .env.example .env
# Edit .env: set AQB_PROVIDER=gemini (or openai / mock) and your API key.
# .env is loaded automatically at startup.
```

Supports Google Gemini (`GEMINI_API_KEY`), OpenAI (`OPENAI_API_KEY`), any OpenAI-compatible server via `OPENAI_BASE_URL` (Ollama, Groq, OpenRouter), and an offline `mock` provider for testing.

### 3. Run the Web App (Streamlit)

```bash
streamlit run app.py
```

Upload your question bank & study notes, optionally specify answer formatting instructions, inspect ranked **Important Topics**, watch batches solve live, and download the final key as **Markdown, Word, or PDF**.

### 4. Run the CLI

```bash
# Interactive prompts (asks for question bank, notes, and optional style instructions)
aqb

# Non-interactive with sample files (offline mock demo)
aqb -q examples/sample_question_bank.txt \
    -n examples/sample_notes.md \
    --style-file examples/sample_style.txt \
    --provider mock --docx --pdf

# Real LLM run with long-answer fact-checking
aqb -q bank.pdf -n unit1.pdf unit2.pptx \
    -s "Use bullet points and Mermaid diagrams for 13-markers" \
    --provider gemini --verify --docx --pdf

# List and resume interrupted sessions
aqb --list-sessions
aqb --resume <session_id> --docx --pdf
```

---

## Running Tests

```bash
pytest -v
```
