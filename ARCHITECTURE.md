# OMNISCOPE: Multi-Agent Observability Platform

## Overview

Most observability tools trace linear request chains. Multi-agent systems don't work that way — they're dynamic DAGs where agents spawn agents, share memory, contradict each other, and fail in cascading patterns invisible to traditional tracing.

OMNISCOPE is organized around four pillars. Two are built; the other two are design sketches that this document also records.

| Pillar | Status |
|---|---|
| **1. Agent DAG Tracing** — every inter-agent event captured as a node in a causal graph, not a flat log | **Built** — SQLite trace store + FastAPI/WebSocket API + React dashboard, plus a live Claude Code adapter |
| **2. Heuristic Risk Detection** — rule-based scoring of live runs (loops, tool thrashing, context overflow, reasoning collapse) | **Built** — `omniscope/server/risk.py` |
| **3. Multi-LLM Judge Panel** — dimension-specific judges with weighted aggregation | **Scaffold** — judge prompts + aggregation exist; the per-judge model call and calibration are not yet wired |
| **4. Predictive ML / Time-Travel / Data Flywheel** — learned failure prediction, state reconstruction, counterfactual branching | **Design sketch** — described below, not implemented |

Each section marks its status inline. Pseudocode in the design-sketch sections describes intended behavior, not shipped code.

**Local-first.** Tracing, risk detection, and the dashboard run on a single machine over SQLite with no API keys. The judge panel is designed to call local models via Ollama (or cloud models per component via env vars) once its model-call layer is wired.

---

## System Overview

```
+------------------------------------------------------------------+
|                        OMNISCOPE                                  |
|                                                                   |
|  +------------------+  +------------------+  +------------------+ |
|  |  AGENT DAG       |  |  JUDGE PANEL     |  |  FAILURE         | |
|  |  TRACING         |  |  (6 LLM judges)  |  |  PREDICTOR       | |
|  +--------+---------+  +--------+---------+  +--------+---------+ |
|           |                     |                     |           |
|           +---------------------+---------------------+           |
|                                 |                                 |
|                    +------------v-----------+                     |
|                    |    UNIFIED TRACE STORE  |                    |
|                    |    (Event-Sourced DAG)  |                    |
|                    +------------+------------+                    |
|                                 |                                 |
|              +------------------+------------------+              |
|              |                                     |              |
|   +----------v----------+            +-------------v-----------+ |
|   |  TIME-TRAVEL        |            |  DATA FLYWHEEL          | |
|   |  DEBUGGER           |            |  (traces -> better      | |
|   |                     |            |   models -> better      | |
|   |                     |            |   predictions)          | |
|   +---------------------+            +-------------------------+ |
+------------------------------------------------------------------+
```

---

## 1. Agent DAG Tracing

### Why it matters

In a multi-agent system, a bad output is rarely one agent's fault. Agent A retrieved low-relevance documents, Agent B built on those, Agent C approved it because its critic prompt didn't account for the domain. Traditional tracing shows you the final bad output. DAG tracing shows you the causal chain.

### Trace Event Model

Every event in the system is a node:

```
TraceEvent {
  event_id:        UUID
  trace_id:        UUID            // groups events into one execution
  parent_span_id:  UUID | null     // enables DAG construction
  agent_id:        string
  event_type:      enum {
    LLM_CALL, TOOL_CALL, AGENT_DECISION,
    MEMORY_ACCESS, JUDGE_EVALUATION
  }
  input_tokens:    int
  output_tokens:   int
  latency_ms:      float
  confidence:      float | null
  cost_usd:        float | null

  // Graph edges (materialized in graph layer)
  causal_parents:    [UUID]
  data_dependencies: [UUID]

  tags:             {string: string}
}
```

Key difference from OpenTelemetry spans: `causal_parents` and `data_dependencies` are first-class. A span can have multiple parents (Agent C's decision was caused by both Agent A's retrieval and Agent B's analysis). This makes the trace a true DAG, not a tree.

### Inter-Agent Communication

```
InterAgentMessage {
  message_id:     UUID
  trace_id:       UUID
  from_agent:     string
  to_agent:       string
  message_type:   enum {TASK, RESULT, QUERY, FEEDBACK, DELEGATION, ESCALATION}
  content:        {text, structured, embeddings}
  priority:       int
  requires_response: bool
}
```

Every message is a traced edge. You can query: "Show me all messages that led to this decision" and get a subgraph, not a list.

### Claude Code Adapter (live session tracing)

Claude Code persists every session as append-only JSONL under `~/.claude/projects/<munged-cwd>/<sessionId>.jsonl`, with subagent transcripts at `<sessionId>/subagents/agent-<agentId>.jsonl` and workflow runs under `<sessionId>/subagents/workflows/<runId>/`. The adapter (`omniscope/sdk/adapters/claude_code_adapter.py`) tails these files line-wise with incremental offsets and maps them onto the standard trace event model:

- One trace per session (`trace_id` = sessionId; name from the session's `ai-title`, falling back to the first prompt).
- Consecutive `assistant` lines sharing one `message.id` become a single `llm_call` event with model and token usage; user prompts become `system_event`s.
- Each `tool_use` block becomes a `tool_call` event, completed by the matching `tool_result` line (duration = result timestamp − use timestamp; errors from `is_error`).
- `Agent` / `Workflow` spawns are correlated via `toolUseResult` (agentId / runId), and the spawned subagent transcripts are tailed into the same trace as child agent streams linked to the spawning tool call.
- `turn_duration` system lines become turn-summary events; snapshot/mode/queue noise lines are skipped.

Deliberately KISS: stdlib file tailing plus the existing collector — no hooks into Claude Code, no daemon, no new dependencies. `trace_id` is the session id, so re-attaching maps onto the same trace, and content fields are truncated (500 chars by default): this is observability, not transcript capture.

---

## 2. Multi-LLM Judge Panel (scaffold)

### Why it matters

Single-score evaluation hides failure modes. An output can have perfect factuality but broken reasoning, or safe content that contradicts what the user asked. The design is six judges, each focused on one dimension, with calibrated scores and confidence intervals.

**Status:** the judge prompts (`JudgeConfig`/`JudgeType`) and the weighted aggregation below are implemented; the per-judge model call and calibration are **not yet wired** — `_calibrate()` is an identity pass-through and no judge currently invokes a model. Judges are designed to run locally via Ollama (`llama3.1:8b` by default, larger models for quality), with cloud models (Claude, GPT-4) as per-component drop-in replacements.

### Architecture

```
                        MODEL OUTPUT
                            |
                            v
            +---------------+----------------+
            |         JUDGE ROUTER           |
            | (selects judges by context)    |
            +--+---+---+---+---+---+---------+
               |   |   |   |   |   |
  +------------+   |   |   |   |   +------------+
  v            v   v   v   v   v                v
+--------+ +------+ +------+ +------+ +------+ +--------+
|REASON- | |FACT- | |HALLU-| |SAFETY| |CONSIS| |MM      |
|ING     | |UALITY| |CINA- | |      | |TENCY | |ALIGN   |
|        | |      | |TION  | |      | |      | |(if MM) |
+---+----+ +--+---+ +--+---+ +--+---+ +--+---+ +---+----+
    |         |         |        |        |          |
    +----+----+----+----+--------+--------+----------+
         |
    SCORE AGGREGATOR
    Weighted ensemble + isotonic calibration
         |
    COMPOSITE QUALITY SCORE
    + per-dimension breakdown
    + flagged spans with evidence

LLM BACKEND (configurable per judge):
    +------------------+     +------------------+
    |  Ollama (default)|     |  Cloud API       |
    |  llama3.1:8b     |     |  (optional)      |
    |  qwen2.5:32b     |     |  Claude / GPT-4  |
    |  llava (vision)  |     |                  |
    +------------------+     +------------------+
    All judges hit localhost:11434 by default.
    Swap per-judge via JUDGE_<NAME>_MODEL env var.
```

### Scoring

```
CompositeScore = sum(w_i * calibrate(score_i)) / sum(w_i)

where:
  w_i = judge_weight * (1 - uncertainty_i)   // down-weight uncertain judges
  calibrate(s) = s                            // identity for now; not yet calibrated

Output example:
{
    composite_score:  0.847,
    confidence:       (0.81, 0.88),
    breakdown: {
        reasoning:          0.91,
        factuality:         0.85,
        hallucination_risk: 0.12,
        safety:             0.99,
        consistency:        0.83
    },
    flags: ["mild_hallucination_span_detected"]
}
```

Once wired, judge evaluations are themselves traced (a `JUDGE_EVALUATION` event type) so their outputs can be reviewed alongside the run.

---

## 3. Risk Detection

### Why it matters

Multi-agent failures cascade. By the time the output is bad, several agents have wasted tokens and time. The risk engine watches the live event stream and raises a per-agent risk score while there's still time to intervene.

### Failure detectors

All five detectors below are heuristics over the recent event stream (`omniscope/server/risk.py`) — no embeddings or learned models are involved.

| Failure Mode | Implemented signal |
|---|---|
| **Infinite loops** | 2-/3-gram repetition in the last ~20 `(event_type, tool)` actions |
| **Hallucination** | low average model confidence over recent events |
| **Context overflow** | total tokens vs. the context window (utilization) |
| **Tool thrashing** | high tool self-transition rate + tool failure rate |
| **Reasoning collapse** | downward trend / low average in recent confidence scores |
| **Agent divergence** | *not implemented (returns 0.0)* |

### Risk Aggregation

The heuristic detectors above run on each event stream and combine into a per-agent `overall_failure_risk`. Learned detectors (anomaly models, graph models over agent topology) and a time-to-failure estimate are design goals, not implemented — `time_to_predicted_failure` is part of the payload schema below but is not currently computed.

```
RiskPayload {
    agent_id:                   string
    overall_failure_risk:       float       // 0.0 to 1.0
    time_to_predicted_failure:  float | null  // seconds
    contributing_signals: [
        {detector: string, signal: string, weight: float}
    ]
    recommended_interventions: [string]
}
```

### Example: Catching a Hallucination Spiral (illustrative / design)

```
T+2.1s  Researcher retrieves 3 docs (relevance: 0.82, 0.71, 0.45)
        hallucination_risk: 0.15

T+2.8s  Claim generated with no supporting document
        hallucination_risk: 0.38
        Signal: embedding_distance(claim, nearest_doc) = 0.73

T+3.2s  Second unsupported claim
        hallucination_risk: 0.61 -- ALERT
        System injects retrieval step, risk drops to 0.22

Without intervention (simulated): risk hits 0.91 by T+4.1s, output unusable.
```

---

## 4. Time-Travel Debugging (design sketch — not implemented)

*The operations and pseudocode in this section describe intended behavior. There is no `reconstruct_state`, `simulate_branch`, snapshot store, or execution engine in the codebase yet.*

### Why it matters

"Why did it fail?" is the hardest question in multi-agent systems. Time-travel lets you reconstruct exact state at any point, then fork execution with different parameters to test your hypothesis.

### Core Operations

```
replay(trace_id, from=T0, to=T4)       // replay entire execution
step(trace_id, direction=FORWARD)       // step one event
inspect(event_id)                       // full state at this point
branch(event_id, override={...})        // fork with different params
diff(branch_a, branch_b)               // compare outcomes
```

### State Reconstruction (Event Sourcing)

```python
def reconstruct_state(trace_id: UUID, target_time: datetime) -> SystemState:
    """
    Replays events from nearest snapshot to target_time.
    Returns all agent states, pending messages, intermediate outputs,
    retrieved documents, and prompt/response pairs at that exact moment.
    """
    nearest_snapshot = snapshot_store.get_nearest(trace_id, target_time)
    events = trace_store.get_events(
        trace_id=trace_id,
        after=nearest_snapshot.timestamp,
        before=target_time
    )
    state = nearest_snapshot.state.copy()
    for event in events:
        state = apply_event(state, event)
    return state
```

### Counterfactual Branching

Simulated branches run agent LLM calls through Ollama locally, so exploring "what if" scenarios doesn't burn API credits. The simulation uses the same judge panel (also Ollama-backed) to score the alternate outcome.

```python
def simulate_branch(branch_point: UUID, overrides: dict) -> SimulationResult:
    """
    Forks execution at branch_point with modified parameters.
    Agent calls in simulation route through Ollama by default.
    Returns alternate trace + diff against actual execution.
    """
    state = reconstruct_state(branch_point)
    state.apply_overrides(overrides)
    alternate_trace = execution_engine.run(state, mode=SIMULATION)
    return SimulationResult(
        original_trace=get_downstream_trace(branch_point),
        alternate_trace=alternate_trace,
        diff=compute_trace_diff(original, alternate),
        outcome_comparison=compare_outcomes(original, alternate)
    )
```

### Example Session

```
Developer sees composite score 0.52 on a trace.

1. Timeline view: red band at T+2.3s (Researcher, hallucination spike)
2. Clicks event -> sees retrieval returned 2 docs with relevance 0.41, 0.38
3. Clicks "Branch from here"
4. Overrides: retrieval_top_k = 10 (was 3), relevance_threshold = 0.6
5. Simulation runs in ~2 seconds
6. Comparison:
   - Original:  hallucination_risk 0.78, composite 0.52
   - Branched:  hallucination_risk 0.11, composite 0.89
7. Fix identified: increase retrieval depth for this query type
```

---

## 5. Data Flywheel (design sketch — not implemented)

*None of the training/calibration loops below exist yet; this section records the intended design.* The idea: every execution would produce data that improves the platform:

| Dataset | Source | Improves |
|---------|--------|----------|
| Failure corpus | Traces where composite < 0.5 | Failure prediction model |
| Judge calibration set | Judge scores + human labels | Isotonic calibration accuracy |
| Hallucination examples | Judge-flagged spans | Hallucination detector |
| Agent routing decisions | Planner traces + outcome quality | Task decomposition |
| Retrieval quality pairs | (query, doc, relevance) + downstream quality | Retrieval re-ranking |

*Status: none of the above is built — the heuristic detectors in section 3 are the only risk signal that ships today.*

---

## Ollama Configuration

The judge panel is designed to talk to Ollama at `localhost:11434` by default. This section documents the intended model configuration — as of today the model-call layer is not wired, so nothing actually calls Ollama yet.

### Required Models

```bash
ollama pull llama3.1:8b          # intended judge model (fast)
# nomic-embed-text is only needed if the design-sketch embedding detectors (section 3) are built
```

### Optional Models (higher quality)

```bash
ollama pull llama3.1:70b         # higher-quality judging
ollama pull qwen2.5:32b          # alternative judge model
ollama pull llava:13b            # multimodal alignment judge
```

### Per-Component Model Override

Each component reads its model from an environment variable, falling back to the default:

| Component | Env Var | Default |
|-----------|---------|---------|
| Reasoning Judge | `JUDGE_REASONING_MODEL` | `llama3.1:8b` |
| Factuality Judge | `JUDGE_FACTUALITY_MODEL` | `llama3.1:8b` |
| Hallucination Judge | `JUDGE_HALLUCINATION_MODEL` | `llama3.1:8b` |
| Safety Judge | `JUDGE_SAFETY_MODEL` | `llama3.1:8b` |
| Consistency Judge | `JUDGE_CONSISTENCY_MODEL` | `llama3.1:8b` |
| MM Alignment Judge | `JUDGE_MM_MODEL` | `llava:13b` |
| Embedding Model | `EMBEDDING_MODEL` | `nomic-embed-text` |
| Simulation LLM | `SIMULATION_MODEL` | `llama3.1:8b` |

Set any of these to a cloud model (e.g., `claude-sonnet-4-6`) to use the cloud API instead. The system auto-detects whether a model name is an Ollama model or a cloud provider model and routes accordingly.

---

## API Surface

```
POST   /api/v1/traces                         # Ingest a trace (with events)
POST   /api/v1/event                          # Append a single event
POST   /api/v1/traces/{trace_id}/complete     # Mark a trace complete
POST   /api/v1/traces/{trace_id}/fail         # Mark a trace failed
GET    /api/v1/traces                         # List traces
GET    /api/v1/traces/{trace_id}              # Get full trace
GET    /api/v1/traces/{trace_id}/graph        # Get trace as a DAG
GET    /api/v1/traces/{trace_id}/timeline     # Get trace timeline
GET    /api/v1/traces/{trace_id}/risk         # Risk scores for a trace
GET    /api/v1/risk/{trace_id}/{agent_id}     # Risk scores for one agent
GET    /api/v1/events/{event_id}              # Get one event
GET    /api/v1/events/{event_id}/causal_chain # Events that led to this one
GET    /api/v1/events/{event_id}/downstream   # Events this one affected
WS     /ws/v1/traces/{trace_id}/live          # Live trace updates
WS     /ws/v1/updates                         # Live cross-trace updates

# (plus /api/v1/auth/* for the optional dashboard login)
# Judge and time-travel endpoints are part of the design above, not yet implemented.
```
