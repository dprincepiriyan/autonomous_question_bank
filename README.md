# Autonomous Question Bank (AQB)

An autonomous, stateful agent that solves academic question banks in controlled batches grounded in your study material, highlights the most important topics to focus on, and compiles a complete final answer key.

## Why Controlled Batching?

Solving an entire question bank in one LLM call leads to output truncation, skipped questions, and hallucinations. AQB avoids this by running a **stateful orchestrator loop**:
- **10–12 two-mark questions** per batch reply
- **5–6 thirteen-mark questions** per batch reply
- **Grounded retrieval** from your notes with chunk citations (or `[Not found in provided notes]` if missing)
- **Checkpointed state** (`state.json`) after every batch

---

## Documentation

- [System Architecture](docs/ARCHITECTURE.md) — Component diagram, module responsibilities, and anti-hallucination design.
- [Agentic Workflow](docs/WORKFLOW.md) — End-to-end sequence diagram, batching logic, and state transitions.

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/dprincepiriyan/autonomous_question_bank.git
cd autonomous_question_bank

# Optional: install LLM & PDF/DOCX extras
pip install -e ".[llm,docs]"
```

### 2. Configure Environment (Optional for `mock` mode)

```bash
cp .env.example .env
# Set AQB_PROVIDER=gemini (or openai / mock) and your API key
```

### 3. Run Interactive CLI

```bash
PYTHONPATH=src python3 -m aqb.cli
```

At startup, the agent prompts you for:
1. **Question Bank** (path to `.txt`, `.md`, `.pdf`, `.docx`, or raw text)
2. **Study Material / Notes** (path to `.txt`, `.md`, `.pdf`, `.docx`, or raw text)
3. **Optional Style Instructions** (e.g., *"Use bullet points, include formulas, keep 2-markers under 40 words"*)

You can also test the full loop offline using the built-in deterministic `mock` provider:

```bash
PYTHONPATH=src python3 -m aqb.cli --provider mock
```

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
