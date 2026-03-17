# OMNISCOPE: Multi-Agent Observability Platform

Trace, judge, predict failures, and time-travel debug multi-agent AI systems.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system design.

## What makes this different

Most observability tools trace linear request chains. Multi-agent systems are dynamic DAGs — agents spawn agents, share memory, contradict each other, and fail in cascading patterns invisible to traditional tracing.

OMNISCOPE combines four capabilities that don't exist together:

1. **Agent DAG Tracing** — Causal graph of every inter-agent decision, not flat logs
2. **Multi-LLM Judge Panel** — 6 specialized judges with calibrated scoring
3. **Predictive Failure Detection** — Real-time risk scoring on the live execution graph
4. **Time-Travel Debugging** — Reconstruct state, fork execution, compare outcomes

These compound via a data flywheel: traces train better failure predictors and judge calibration, which produce better traces.

## Local-first with Ollama

All LLM components (judges, embeddings, counterfactual simulation) run locally via [Ollama](https://ollama.com). No API keys required to get started:

```bash
ollama pull llama3.1:8b          # judge models
ollama pull nomic-embed-text     # embeddings for failure detection
```

Cloud models (Claude, GPT-4) are supported as drop-in replacements per component via environment variables.
