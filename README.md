# OMNISCOPE

> Observability for multi-agent systems — capture the causal DAG of agent execution and flag risky runs with heuristic detectors. Includes a live adapter for Claude Code sessions.

**Live demo (GitHub Pages, demo mode):** https://sugeerth.github.io/simple_agent_tracer/

## Overview

OMNISCOPE traces multi-agent systems as causal DAGs rather than flat request chains: agents spawn agents, share memory, and fail in cascading ways that linear tracing misses. It captures the causal graph of every inter-agent event in a SQLite-backed trace store, exposes it over a FastAPI + WebSocket API and a React dashboard, and scores live runs with a set of heuristic risk detectors (loops, tool thrashing, context overflow, reasoning collapse). A model-backed judge panel and a time-travel / "data-flywheel" design are sketched in [`ARCHITECTURE.md`](ARCHITECTURE.md) but not yet wired up.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system design.

## Features

- **Agent DAG tracing** — causal graph of every inter-agent event, not flat logs
- **Heuristic risk detection** — rule-based detectors for loops, tool thrashing, context overflow, and reasoning-score collapse
- **Claude Code adapter** — tail live Claude Code sessions straight from their JSONL transcripts (see below)
- **Framework adapters** — instrumentation hooks for LangChain, CrewAI, AutoGen, OpenAI Agents, and Anthropic
- **Judge-panel scaffold** — 6 dimension-specific judge prompts + weighted aggregation; the model-call and calibration layers are not yet wired
- **Local-first** — tracing, risk, and the dashboard run on a single machine over SQLite; cloud LLMs optional
- **Live dashboard** — React + Vite UI for exploring traces and execution graphs

## Tech Stack

- **Language:** Python
- **API / Server:** FastAPI, Uvicorn
- **Frontend:** React, Vite, TypeScript (`@xyflow/react`, Recharts)
- **LLM runtime:** none required — the package traces model calls, it does not make them (the Anthropic adapter wraps a client you pass in). The standalone `demo_interactive.py` brings its own backends: Ollama, Anthropic API, or an offline stub
- **Storage:** SQLite

## What makes this different

Most observability tools trace linear request chains. Multi-agent systems are dynamic DAGs — agents spawn agents, share memory, contradict each other, and fail in cascading patterns invisible to traditional tracing.

Two capabilities are built today:

1. **Agent DAG tracing** — a causal graph of every inter-agent event, not flat logs
2. **Heuristic risk detection** — rule-based scoring of live runs for loops, tool thrashing, context overflow, and reasoning collapse

Two more are designed in [`ARCHITECTURE.md`](ARCHITECTURE.md) but not yet implemented: a model-backed **judge panel** (the prompts and weighted aggregation exist; the model calls and calibration do not) and **time-travel debugging** (state reconstruction and counterfactual branching).

## Getting Started

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Pull local models for the judge panel

Tracing, the heuristic risk detectors, and the dashboard need no model at all. The judge panel is designed to run locally via [Ollama](https://ollama.com) — note its model-call layer is still a scaffold (see [`ARCHITECTURE.md`](ARCHITECTURE.md)):

```bash
ollama pull llama3.1:8b          # intended judge model
```

Cloud models (Claude, GPT-4) can be set per component via environment variables.

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
# run from the repo root, as modules, so `omniscope` is importable
python -m examples.quickstart
python -m examples.execution_graph
```

## Trace live Claude Code sessions

Claude Code writes every session as append-only JSONL under `~/.claude/projects/`. The Claude Code adapter reads those transcripts directly — no hooks or changes to Claude Code — and streams them into OMNISCOPE as trace events: prompts, assistant turns with model and token usage, tool calls with per-call latency, and subagent/workflow spawns pulled into the same trace as linked child agents. It replays finished sessions and follows running ones live; content is truncated (500 chars by default) so traces stay observability-sized.

```bash
# 1. Start the server
python3 -m uvicorn omniscope.server.app:app --port 8781

# 2. Replay the most recent Claude Code session, then keep following it live
python3 -m omniscope.sdk.adapters.claude_code_adapter --latest --follow

# 3. Open the dashboard
cd dashboard && npm install && npm run dev    # http://localhost:5173
```

See `examples/claude_code_live.py` for the programmatic equivalent.

## Project Structure

```
simple_agent_tracer/
├── omniscope/
│   ├── judges.py            # Multi-LLM judge panel (scaffold — not imported by the pipeline)
│   ├── sdk/                 # Tracing SDK (collector, decorators)
│   │   └── adapters/        # LangChain, CrewAI, AutoGen, OpenAI, Anthropic adapters
│   └── server/              # FastAPI server
├── dashboard/               # React + Vite observability UI
├── examples/                # Quickstart + execution-graph demos
├── demo*.py                 # Interactive and live demos
└── ARCHITECTURE.md          # Full system design
```
