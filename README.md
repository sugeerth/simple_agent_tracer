# OMNISCOPE

> A multi-agent observability platform — trace, judge, predict failures, and time-travel debug multi-agent AI systems.

**Live demo (GitHub Pages, demo mode):** https://sugeerth.github.io/simple_agent_tracer/

## Overview

OMNISCOPE is a multi-agent observability platform built for the way modern AI systems actually behave: as dynamic DAGs where agents spawn agents, share memory, contradict each other, and fail in cascading patterns invisible to traditional request-chain tracing. It captures the causal graph of every inter-agent decision, scores quality with a panel of LLM judges, predicts failures on the live execution graph, and lets you replay and fork past runs for time-travel debugging.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system design.

## Features

- **Agent DAG tracing** — causal graph of every inter-agent decision, not flat logs
- **Multi-LLM judge panel** — 6 specialized judges with calibrated scoring
- **Predictive failure detection** — real-time risk scoring on the live execution graph
- **Time-travel debugging** — reconstruct state, fork execution, and compare outcomes
- **Framework adapters** — drop-in instrumentation for LangChain, CrewAI, AutoGen, OpenAI Agents, Anthropic, and a generic adapter
- **Local-first** — judges, embeddings, and counterfactual simulation run on Ollama with no API keys required
- **Live dashboard** — React + Vite UI for exploring traces and execution graphs

## Tech Stack

- **Language:** Python
- **API / Server:** FastAPI, Uvicorn
- **Frontend:** React, Vite, TypeScript (`@xyflow/react`, Recharts)
- **LLM runtime:** Ollama (local), with Claude / GPT-4 supported as drop-in replacements
- **Storage:** SQLite

## What makes this different

Most observability tools trace linear request chains. Multi-agent systems are dynamic DAGs — agents spawn agents, share memory, contradict each other, and fail in cascading patterns invisible to traditional tracing.

OMNISCOPE combines four capabilities that don't exist together:

1. **Agent DAG Tracing** — Causal graph of every inter-agent decision, not flat logs
2. **Multi-LLM Judge Panel** — 6 specialized judges with calibrated scoring
3. **Predictive Failure Detection** — Real-time risk scoring on the live execution graph
4. **Time-Travel Debugging** — Reconstruct state, fork execution, compare outcomes

These compound via a data flywheel: traces train better failure predictors and judge calibration, which produce better traces.

## Getting Started

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull local models (no API keys required)

All LLM components (judges, embeddings, counterfactual simulation) run locally via [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b          # judge models
ollama pull nomic-embed-text     # embeddings for failure detection
```

Cloud models (Claude, GPT-4) are supported as drop-in replacements per component via environment variables.

### 3. Run the backend

```bash
uvicorn omniscope.server.app:app --reload
```

### 4. Run the dashboard

```bash
cd dashboard
npm install
npm run dev
```

### 5. Try the examples

```bash
python examples/quickstart.py
python examples/execution_graph.py
```

## Project Structure

```
multi-agent_moat/
├── omniscope/
│   ├── judges.py            # Multi-LLM judge panel
│   ├── sdk/                 # Tracing SDK (collector, decorators)
│   │   └── adapters/        # LangChain, CrewAI, AutoGen, OpenAI, Anthropic adapters
│   └── server/              # FastAPI server
├── dashboard/               # React + Vite observability UI
├── examples/                # Quickstart + execution-graph demos
├── demo*.py                 # Interactive and live demos
└── ARCHITECTURE.md          # Full system design
```
