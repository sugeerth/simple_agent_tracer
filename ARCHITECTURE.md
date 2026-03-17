# OMNISCOPE: Multi-Agent Observability Platform

## The Moat

Most observability tools trace linear request chains. Multi-agent systems don't work that way — they're dynamic DAGs where agents spawn agents, share memory, contradict each other, and fail in cascading patterns invisible to traditional tracing.

OMNISCOPE is built around **four capabilities that don't exist together anywhere else**:

1. **Agent DAG Tracing** — Every inter-agent message, decision, and tool call captured as a node in a live causal graph, not a flat log.
2. **Multi-LLM Judge Panel** — 6 specialized judges evaluate every output on orthogonal quality dimensions with calibrated scoring.
3. **Predictive Failure Detection** — Real-time risk scoring on the live execution graph, predicting agent failures before they cascade.
4. **Time-Travel Debugging** — Reconstruct system state at any point, fork execution with different parameters, compare outcomes.

These four compound: the judge panel generates training signal for the failure predictor, which improves over time via the data flywheel. No individual feature is the moat — the flywheel is.

**Local-first via Ollama.** All LLM-dependent components (judges, embedding computation, counterfactual simulation) run through Ollama by default. No API keys needed to get started, no per-evaluation cost at scale. Cloud LLMs are supported as an upgrade path for higher-quality judging, but the system is fully functional on a single machine with `ollama pull`.

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

---

## 2. Multi-LLM Judge Panel

### Why it matters

Single-score evaluation hides failure modes. An output can have perfect factuality but broken reasoning, or safe content that contradicts what the user asked. Six judges, each focused on one dimension, with calibrated scores and confidence intervals.

Judges run locally via Ollama — no API cost per evaluation. Default model: `llama3.1:8b` for speed, `llama3.1:70b` or `qwen2.5:32b` for quality. Cloud models (Claude, GPT-4) supported as drop-in replacements when higher accuracy is worth the cost.

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
  calibrate(s) = isotonic_regression(s)       // trained on human eval data

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

The judges themselves are traced — their reasoning chains become training data for improving calibration.

---

## 3. Predictive Failure Detection

### Why it matters

Multi-agent failures cascade. By the time the output is bad, 4 agents have wasted tokens and time. The failure predictor watches the live execution graph and raises alerts while there's still time to intervene.

### Six Failure Modes

| Failure Mode | Detection Signal | Method |
|---|---|---|
| **Infinite loops** | Repeated (agent, action) pairs in sliding window | DFS cycle detection + n-gram repetition |
| **Hallucination spiral** | Claims with no embedding neighbor in retrieval store | Embedding distance via Ollama (`nomic-embed-text`) |
| **Tool thrashing** | Same tool called > 3x with transition probability > 0.7 | Markov chain on tool sequences |
| **Context overflow** | Token consumption rate projected to exceed window | Linear projection |
| **Reasoning collapse** | Judge reasoning scores dropping on sliding window | Moving average threshold |
| **Agent divergence** | Agent goal embeddings becoming orthogonal | Cosine similarity via Ollama embeddings |

### Risk Aggregation

Heuristic detectors run immediately. Learned detectors (LSTM anomaly model, graph neural network on agent topology) improve as the data flywheel accumulates traces. Signals fuse via Bayesian aggregation into per-agent risk scores and a time-to-predicted-failure estimate.

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

### Example: Catching a Hallucination Spiral

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

## 4. Time-Travel Debugging

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

## 5. Data Flywheel

Every execution produces training data that improves the platform:

| Dataset | Source | Improves |
|---------|--------|----------|
| Failure corpus | Traces where composite < 0.5 | Failure prediction model |
| Judge calibration set | Judge scores + human labels | Isotonic calibration accuracy |
| Hallucination examples | Judge-flagged spans | Hallucination detector |
| Agent routing decisions | Planner traces + outcome quality | Task decomposition |
| Retrieval quality pairs | (query, doc, relevance) + downstream quality | Retrieval re-ranking |

```
Week 1:   Heuristic-only failure prediction. Uncalibrated judge scores.
Week 4:   500 traces. First learned failure predictor. Loop detection 60% -> 82%.
Week 8:   2,000 traces. Judge calibration trained. Human correlation 0.71 -> 0.84.
Week 16:  10,000 traces. GNN topology risk model. Predicts failures 15s early (vs 3s).
```

The flywheel is the moat. Competitors can copy the architecture; they can't copy 10,000 traced executions with human-validated judge scores.

---

## Ollama Configuration

All LLM-dependent components talk to Ollama at `localhost:11434` by default.

### Required Models

```bash
ollama pull llama3.1:8b          # default judge model (fast)
ollama pull nomic-embed-text     # embeddings for failure detection
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
POST   /api/v1/traces                         # Ingest trace events (batch)
GET    /api/v1/traces/{trace_id}              # Get full trace
GET    /api/v1/traces/{trace_id}/graph        # Get trace as DAG

POST   /api/v1/judge/evaluate                 # Trigger judge evaluation
GET    /api/v1/judge/{eval_id}                # Get judge results

GET    /api/v1/risk/{agent_id}                # Current risk scores
GET    /api/v1/risk/{trace_id}/heatmap        # Risk heatmap data

POST   /api/v1/timetravel/reconstruct         # Reconstruct state at timestamp
POST   /api/v1/timetravel/branch              # Create counterfactual branch
GET    /api/v1/timetravel/diff                # Compare two branches

WS     /ws/v1/traces/{trace_id}/live          # Live trace updates
WS     /ws/v1/risk/live                       # Live risk updates
```
