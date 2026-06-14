# System Archaeologist
- Reverse-engineer how any software product works using evidence-based reasoning, powered by **DSPy v2.5**.

## What it does
Enter a product name (e.g. "Cursor", "Duolingo", "Kafka") and get:

- **Evidence** — real web sources via Tavily + LLM knowledge
- **Architecture Hypothesis** — layered component tree
- **Contradictions** — challenges to the hypothesis
- **Final Design** — synthesized architecture + confidence score
- **Mermaid Diagram** — rendered architecture visualization
- **Multi-Agent Debate** — Agent A vs Agent B, judged by an impartial third agent

Everything streams **live** via SSE (Server-Sent Events) as each DSPy stage completes.

## Stack

| Layer      | Technology |
|---|---|
| Reasoning  | DSPy v2.5+ (ChainOfThought, 7 Signatures) |
| LLM        | Ollama → Gemini (fallback) |
| Search     | Tavily API |
| API        | FastAPI + SSE |
| Database   | PostgreSQL (asyncpg) |
| Frontend   | Astro |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL running locally
- Ollama installed: https://ollama.ai

```bash
# Pull the Ollama model
ollama pull llama3.2
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env     # Windows
# cp .env.example .env     # Mac/Linux
# Edit .env — set DATABASE_URL and optionally GOOGLE_API_KEY
```

### 3. Database Setup

```sql
-- Run in psql or pgAdmin
CREATE DATABASE system_archaeologist;
```
The schema (tables + indexes) is created automatically on first startup.

### 4. Start Backend

```bash
cd backend
uvicorn main:app --reload
# API docs at: http://localhost:8000/docs
```

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
# App at: http://localhost:4321
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/analyze/stream?system=Cursor` | **SSE stream** — live stage-by-stage output |
| `POST` | `/api/analyze` | Blocking full analysis, returns JSON |
| `GET` | `/api/analyses` | List recent analyses |
| `GET` | `/api/analyses/{id}` | Get analysis by UUID |
| `GET` | `/api/health` | LLM provider + DB status |

### SSE Event Format

Each event:
```json
{
  "stage": "hypothesis",
  "message": "Initial architecture hypothesis built",
  "data": "Frontend:\n  └─ Electron...",
  "progress": 40
}
```

Stages in order:
1. `evidence_web` — Tavily search results
2. `evidence` — LLM-augmented evidence
3. `hypothesis` — initial architecture
4. `contradictions` — challenges
5. `design` — final design + confidence
6. `mermaid` — diagram code
7. `debate_topic` → `agent_a` → `agent_b` → `verdict`
8. `saved` — stored to PostgreSQL

## Evaluation

```bash
cd backend
python evaluation.py
```

Evaluates against 5 well-known systems (Redis, Git, Docker, Nginx, Kafka) and reports:
- Component Recall
- Technology Recall
- Pattern Recall
- Confidence Calibration Error

## Project Structure

```
system-archaeologist/
├── backend/
│   ├── main.py          # FastAPI app + SSE endpoints
│   ├── pipeline.py      # DSPy ReverseEngineer module + stream generator
│   ├── debate.py        # MultiAgentDebate DSPy module
│   ├── signatures.py    # 7 DSPy Signature classes
│   ├── search.py        # Tavily evidence gatherer
│   ├── db.py            # PostgreSQL asyncpg layer
│   ├── evaluation.py    # Eval dataset + metrics + optimizer
│   └── requirements.txt
└── frontend/
    └── src/pages/index.astro
```
