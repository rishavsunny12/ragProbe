# RAGProbe

**Find the questions your RAG pipeline will fail on — before your users do.**

RAGProbe analyzes your chunk corpus topology (the graph of how chunks relate to each
other in embedding space) and generates adversarial questions targeting four structural
failure modes: **multi-hop**, **buried-fact**, **distractor**, and **near-miss boundary**.
It then runs those questions against your RAG pipeline over HTTP, grades the answers,
and produces a regression diff for CI.

Every other eval tool (RAGAS, DeepEval, TruLens) requires you to write test questions.
RAGProbe generates them from your chunk graph. **Zero test authorship required.**

---

## Failure modes

| Mode          | What it targets                                                            |
|---------------|---------------------------------------------------------------------------|
| `multi_hop`   | Distant chunk pairs that must be combined to answer                        |
| `buried_fact` | A fact surrounded by many near-duplicate "distractor" chunks              |
| `distractor`  | Moderately similar chunk pairs where one is a tempting wrong answer        |
| `near_miss`   | High-betweenness "chokepoint" chunks that sit at retrieval rank boundaries |

---

it gives a report of questions and wethere your system passes or not check image below,

<img width="1852" height="1260" alt="Screenshot 2026-06-23 115536" src="https://github.com/user-attachments/assets/e06f0d2d-e1e4-4383-9257-d2989fd1e7a6" />


## Installation

```bash
git clone https://github.com/rishavsunny12/ragProbe.git
cd ragProbe
pip install -e .
# with dev/test extras:
pip install -e ".[dev]"
```

Requires Python 3.11+.

---

## Configure your LLM (do this first)

RAGProbe uses [litellm](https://github.com/BerriAI/litellm) for **question generation**
and **answer grading**. You must configure a model before running `generate`, `run`, or
`calibrate`.

**Indexing (`ragprobe index`) does not use an LLM** — embeddings run locally via
`sentence-transformers` (`all-MiniLM-L6-v2`) with no API key.

### Option A — OpenAI (GPT)

```bash
export OPENAI_API_KEY=sk-...
export RAGPROBE_DEFAULT_LLM=openai/gpt-4o-mini   # optional default
```

### Option B — Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export RAGPROBE_DEFAULT_LLM=anthropic/claude-3-5-sonnet-20241022   # optional default
```

### Option C — Local Ollama (no API key)

```bash
ollama pull llama3
ollama serve
export RAGPROBE_DEFAULT_LLM=ollama/llama3
```

Use the `ollama/` provider (e.g. `ollama/llama3`), **not** `ollama_chat/` — the latter
is significantly slower in litellm for the same model.

### Choosing models per command

| Command | Flag | What it does |
|---------|------|----------------|
| `generate` | `--llm` | Draft and verify adversarial questions |
| `run` | `--grader-llm` | Grade your RAG pipeline's answers |
| `calibrate` | `--llm` and `--grader-llm` | Compare question difficulty sets |

If `RAGPROBE_DEFAULT_LLM` is set, you can omit `--llm` / `--grader-llm`. Otherwise the
default is `openai/gpt-4o-mini` (requires `OPENAI_API_KEY`).

Examples:

```bash
ragprobe generate --llm openai/gpt-4o-mini
ragprobe generate --llm anthropic/claude-3-5-sonnet-20241022
ragprobe generate --llm ollama/llama3

ragprobe run questions.jsonl --pipeline http://localhost:8000/query \
  --grader-llm openai/gpt-4o-mini
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for `openai/*` models |
| `ANTHROPIC_API_KEY` | Required for `anthropic/*` models |
| `RAGPROBE_DEFAULT_LLM` | Default model for `--llm` / `--grader-llm` (litellm string) |
| `RAGPROBE_DB_PATH` | Override SQLite index path (default: `.ragprobe/index.db`) |
| `OLLAMA_API_BASE` | Ollama server URL (default: `http://localhost:11434`) |

### Your RAG pipeline

RAGProbe does **not** configure your RAG app's LLM. It only POSTs questions to the URL
you pass with `--pipeline` (e.g. `http://localhost:8000/query`) and grades the JSON
response. Your pipeline must already be running and accept:

```json
{"query": "your question here"}
```

---

## Quick start

Once your API key (or Ollama) is configured:

```bash
# 1. Embed chunks and build the topology graph (no API key)
ragprobe index ./chunks.jsonl

# 2. Generate adversarial questions from the graph
ragprobe generate --llm openai/gpt-4o-mini

# 3. Run questions against your RAG pipeline and grade answers
ragprobe run .ragprobe/questions.jsonl \
  --pipeline http://localhost:8000/query \
  --grader-llm openai/gpt-4o-mini \
  --output baseline.json

# 4. Later, compare a new run against the baseline to catch regressions
ragprobe run .ragprobe/questions.jsonl \
  --pipeline http://localhost:8000/query \
  --grader-llm openai/gpt-4o-mini \
  --output current.json
ragprobe diff baseline.json current.json --fail-on-regression 5

# 5. (optional) Prove the topology-aware questions are actually harder
ragprobe calibrate --pipeline http://localhost:8000/query \
  --llm openai/gpt-4o-mini \
  --grader-llm openai/gpt-4o-mini
```

For **local Ollama**, add `--timeout 120 --concurrency 1` to `run` — generation can
take 30+ seconds per question.

A minimal fixture corpus ships with the repo for smoke testing:

```bash
ragprobe index tests/fixtures/chunks.jsonl
```

---

## Input format

Chunks are provided as JSONL (one chunk per line):

```jsonl
{"id": "chunk_001", "text": "Enterprise customers receive a 20% discount.", "metadata": {"source": "pricing.md", "page": 3}}
{"id": "chunk_002", "text": "Tier upgrades require 30 days written notice.", "metadata": {"source": "faq.md", "page": 1}}
```

---


## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not integration"
```

---

## License

MIT
