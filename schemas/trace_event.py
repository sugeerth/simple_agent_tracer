"""
OMNISCOPE Trace Data Model
==========================

Complete schema definitions for the unified trace store.
All execution events conform to these structures.

Every event is a node in a directed acyclic graph (the trace graph).
Edges encode causal, data-dependency, and reasoning-chain relationships.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_DECISION = "agent_decision"
    MEMORY_ACCESS = "memory_access"
    IMAGE_TRANSFORM = "image_transform"
    EMBEDDING_COMPUTE = "embedding_compute"
    RETRIEVAL = "retrieval"
    PLANNING_STEP = "planning_step"
    JUDGE_EVALUATION = "judge_evaluation"
    TRAINING_STEP = "training_step"
    INTER_AGENT_MESSAGE = "inter_agent_message"
    MODALITY_TRANSITION = "modality_transition"
    CHECKPOINT = "checkpoint"
    SYSTEM_EVENT = "system_event"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"
    STRUCTURED = "structured"


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SPECIALIST = "specialist"


class MemoryScope(str, Enum):
    LOCAL = "local"
    SHARED = "shared"
    GLOBAL = "global"


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    FEEDBACK = "feedback"
    DELEGATION = "delegation"
    ESCALATION = "escalation"


class JudgeType(str, Enum):
    REASONING = "reasoning"
    FACTUALITY = "factuality"
    HALLUCINATION = "hallucination"
    MM_ALIGNMENT = "mm_alignment"
    SAFETY = "safety"
    CONSISTENCY = "consistency"


class InfluenceMethod(str, Enum):
    ATTENTION_WEIGHT = "attention_weight"
    ABLATION = "ablation"
    GRADIENT_ATTRIBUTION = "gradient_attribution"
    HEURISTIC = "heuristic"


class EdgeType(str, Enum):
    CAUSAL = "causal"
    DATA_DEPENDENCY = "data_dependency"
    REASONING_CHAIN = "reasoning_chain"
    CONTROL_FLOW = "control_flow"
    MEMORY_INFLUENCE = "memory_influence"


# ---------------------------------------------------------------------------
# Nested Payloads
# ---------------------------------------------------------------------------

@dataclass
class PromptPayload:
    """Full prompt content and metadata."""
    system_prompt: str | None = None
    user_prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)  # artifact IDs
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)


@dataclass
class ResponsePayload:
    """Full response content and metadata."""
    text: str = ""
    finish_reason: str = ""  # "stop", "max_tokens", "tool_use"
    token_logprobs: list[dict[str, float]] | None = None  # [{token: logprob}]
    top_k_alternatives: list[list[dict[str, float]]] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolPayload:
    """Tool invocation details."""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    tool_success: bool = True
    tool_error: str | None = None
    tool_latency_ms: float = 0.0


@dataclass
class VisionPayload:
    """Vision-specific tracing data."""
    original_image_ref: str = ""          # artifact store ID
    preprocessed_ref: str | None = None
    image_resolution: tuple[int, int] = (0, 0)
    patch_grid: tuple[int, int] = (0, 0)  # e.g., (14, 14) for ViT-L
    embedding_dims: int = 0
    attention_map_ref: str | None = None   # artifact store ID for spatial attention
    top_attended_regions: list[dict[str, Any]] = field(default_factory=list)
    clip_similarity: float | None = None
    ocr_detected_text: str | None = None
    object_detections: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EmbeddingPayload:
    """Embedding computation tracing."""
    embedding_id: str = ""
    model_name: str = ""
    dimensions: int = 0
    norm: float = 0.0
    nearest_neighbors: list[dict[str, Any]] = field(default_factory=list)
    drift_from_baseline: float | None = None  # cosine distance from training mean


@dataclass
class TrainingPayload:
    """Training step tracing."""
    epoch: int = 0
    batch_index: int = 0
    global_step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    modality_balance: dict[str, float] = field(default_factory=dict)
    embedding_drift: float = 0.0
    param_update_norms: dict[str, float] = field(default_factory=dict)
    batch_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlaggedSpan:
    """A span of text flagged by a judge."""
    start: int = 0
    end: int = 0
    issue: str = ""
    severity: str = "low"  # low, medium, high, critical


@dataclass
class JudgePayload:
    """LLM judge evaluation result."""
    judge_id: str = ""
    judge_type: JudgeType = JudgeType.REASONING
    model_used: str = ""
    primary_score: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    dimension_scores: dict[str, float | None] = field(default_factory=dict)
    reasoning_trace: str = ""
    flagged_spans: list[FlaggedSpan] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    attention_regions: list[dict[str, Any]] = field(default_factory=list)
    judge_latency_ms: float = 0.0
    judge_token_cost: int = 0


@dataclass
class RiskPayload:
    """Failure prediction risk scores."""
    loop_probability: float = 0.0
    hallucination_probability: float = 0.0
    context_overflow_risk: float = 0.0
    tool_thrashing_risk: float = 0.0
    reasoning_collapse_risk: float = 0.0
    agent_divergence_risk: float = 0.0
    overall_failure_risk: float = 0.0
    time_to_predicted_failure: float | None = None  # seconds
    contributing_signals: list[dict[str, Any]] = field(default_factory=list)
    recommended_interventions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Trace Event
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """
    The fundamental unit of observability in OMNISCOPE.

    Every operation -- LLM call, tool invocation, agent decision,
    memory access, image transformation -- produces one TraceEvent.

    Events form a DAG via parent_span_id and explicit edge lists.
    """
    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trace_id: str = ""       # groups events into a single execution
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None  # enables tree/DAG construction

    # Classification
    event_type: EventType = EventType.SYSTEM_EVENT
    agent_id: str = ""
    model_name: str | None = None

    # Modality tracking
    input_modalities: list[Modality] = field(default_factory=list)
    output_modalities: list[Modality] = field(default_factory=list)

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    input_hash: str = ""     # sha256 of input
    output_hash: str = ""    # sha256 of output

    # Performance
    latency_ms: float = 0.0
    cost_usd: float | None = None

    # Quality
    confidence_score: float | None = None

    # Typed payloads (stored as JSONB, indexed)
    prompt_payload: PromptPayload | None = None
    response_payload: ResponsePayload | None = None
    tool_payload: ToolPayload | None = None
    vision_payload: VisionPayload | None = None
    embedding_payload: EmbeddingPayload | None = None
    training_payload: TrainingPayload | None = None
    judge_payload: JudgePayload | None = None
    risk_payload: RiskPayload | None = None

    # Graph edges (materialized in graph layer)
    causal_parents: list[str] = field(default_factory=list)
    data_dependencies: list[str] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)

    # Extensible metadata
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph Edge (materialized in Apache AGE)
# ---------------------------------------------------------------------------

@dataclass
class TraceEdge:
    """Explicit edge between two TraceEvents in the trace graph."""
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_event_id: str = ""
    target_event_id: str = ""
    edge_type: EdgeType = EdgeType.CAUSAL
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent Node (materialized in graph layer)
# ---------------------------------------------------------------------------

@dataclass
class AgentNode:
    """Represents an agent as a persistent node in the execution graph."""
    agent_id: str = ""
    agent_type: AgentType = AgentType.EXECUTOR
    model_config: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    memory_scope: MemoryScope = MemoryScope.LOCAL
    token_budget: int = 100_000
    tokens_consumed: int = 0
    decision_history: list[str] = field(default_factory=list)  # event IDs


# ---------------------------------------------------------------------------
# Inter-Agent Message
# ---------------------------------------------------------------------------

@dataclass
class InterAgentMessage:
    """Message passed between agents, fully traced."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.TASK
    content: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    requires_response: bool = True
    deadline_ms: int | None = None


# ---------------------------------------------------------------------------
# Memory Influence
# ---------------------------------------------------------------------------

@dataclass
class MemoryInfluence:
    """Tracks how a specific memory influenced a specific decision."""
    decision_event_id: str = ""
    memory_id: str = ""
    retrieval_score: float = 0.0
    prompt_position: int = 0
    estimated_influence: float = 0.0
    influence_method: InfluenceMethod = InfluenceMethod.HEURISTIC


# ---------------------------------------------------------------------------
# Composite Quality Score
# ---------------------------------------------------------------------------

@dataclass
class CompositeQualityScore:
    """Aggregated output from the multi-judge evaluation pipeline."""
    evaluated_event_id: str = ""
    composite_score: float = 0.0
    composite_confidence: tuple[float, float] = (0.0, 0.0)
    breakdown: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    judge_evaluations: list[JudgePayload] = field(default_factory=list)
