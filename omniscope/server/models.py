"""Pydantic models for the OMNISCOPE API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


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
    INTER_AGENT_MESSAGE = "inter_agent_message"
    MODALITY_TRANSITION = "modality_transition"
    SYSTEM_EVENT = "system_event"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    ERROR = "error"


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    trace_id: str = ""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    event_type: EventType = EventType.SYSTEM_EVENT
    agent_id: str = ""
    agent_name: str = ""
    model_name: str | None = None
    framework: str = "generic"  # langchain, crewai, openai_agents, anthropic, generic
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    confidence_score: float | None = None
    input_preview: str = ""
    output_preview: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str = ""
    tool_success: bool = True
    error_message: str | None = None
    causal_parents: list[str] = Field(default_factory=list)
    data_dependencies: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskScores(BaseModel):
    agent_id: str
    loop_probability: float = 0.0
    hallucination_probability: float = 0.0
    context_overflow_risk: float = 0.0
    tool_thrashing_risk: float = 0.0
    reasoning_collapse_risk: float = 0.0
    agent_divergence_risk: float = 0.0
    overall_failure_risk: float = 0.0
    time_to_predicted_failure: float | None = None
    contributing_signals: list[dict[str, Any]] = Field(default_factory=list)
    recommended_interventions: list[str] = Field(default_factory=list)


class JudgeScore(BaseModel):
    judge_type: str
    score: float
    confidence: tuple[float, float] = (0.0, 1.0)
    reasoning: str = ""
    flagged_spans: list[dict[str, Any]] = Field(default_factory=list)


class TraceOverview(BaseModel):
    trace_id: str
    trace_name: str = ""
    framework: str = "generic"
    started_at: str = ""
    duration_ms: float = 0.0
    event_count: int = 0
    agent_ids: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    composite_score: float | None = None
    risk_scores: dict[str, RiskScores] = Field(default_factory=dict)
    status: str = "running"  # running, completed, failed


class GraphNode(BaseModel):
    id: str
    agent_id: str
    agent_name: str = ""
    event_type: str
    label: str
    timestamp: str
    latency_ms: float = 0.0
    confidence: float | None = None
    risk_score: float = 0.0
    state: AgentState = AgentState.COMPLETED
    input_preview: str = ""
    output_preview: str = ""
    model_name: str | None = None
    tokens: int = 0
    framework: str = "generic"


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str = "causal"
    animated: bool = False


class TraceGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class TimelineEntry(BaseModel):
    event_id: str
    agent_id: str
    agent_name: str = ""
    event_type: str
    start_ms: float
    duration_ms: float
    label: str
    risk_score: float = 0.0
    confidence: float | None = None


class IngestBatch(BaseModel):
    events: list[TraceEvent]
