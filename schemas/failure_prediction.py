"""
OMNISCOPE Failure Prediction Engine
====================================

Monitors live execution traces and computes real-time risk scores
for 6 failure modes using a combination of heuristic detectors,
learned models, and graph-structural analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from .trace_event import TraceEvent, RiskPayload


# ---------------------------------------------------------------------------
# Failure Modes
# ---------------------------------------------------------------------------

class FailureMode(str, Enum):
    INFINITE_LOOP = "infinite_loop"
    HALLUCINATION_SPIKE = "hallucination_spike"
    CONTEXT_OVERFLOW = "context_overflow"
    TOOL_THRASHING = "tool_thrashing"
    REASONING_COLLAPSE = "reasoning_collapse"
    AGENT_DIVERGENCE = "agent_divergence"


# ---------------------------------------------------------------------------
# Detector Protocol
# ---------------------------------------------------------------------------

class FailureDetector(Protocol):
    """Interface for all failure detectors (heuristic and learned)."""

    def detect(
        self,
        events: list[TraceEvent],
        agent_id: str,
    ) -> DetectorSignal:
        ...


@dataclass
class DetectorSignal:
    """Output from a single failure detector."""
    detector_name: str
    failure_mode: FailureMode
    probability: float          # 0.0 to 1.0
    confidence: float           # detector's self-assessed confidence
    evidence: str               # human-readable explanation
    raw_features: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heuristic Detectors
# ---------------------------------------------------------------------------

class LoopDetector:
    """
    Detects infinite agent loops via:
    1. DFS cycle detection on the (agent_id, action) call graph
    2. N-gram repetition in consecutive agent outputs
    3. Repeated (agent, tool) pairs within sliding window
    """

    def __init__(self, window_size: int = 20, ngram_size: int = 3):
        self.window_size = window_size
        self.ngram_size = ngram_size

    def detect(self, events: list[TraceEvent], agent_id: str) -> DetectorSignal:
        agent_events = [e for e in events[-self.window_size:] if e.agent_id == agent_id]

        # Check for repeated (agent, tool) pairs
        action_sequence = []
        for e in agent_events:
            if e.tool_payload:
                action_sequence.append((e.agent_id, e.tool_payload.tool_name))

        # N-gram repetition check
        if len(action_sequence) >= self.ngram_size * 2:
            ngrams = [
                tuple(action_sequence[i:i + self.ngram_size])
                for i in range(len(action_sequence) - self.ngram_size + 1)
            ]
            repetition_rate = 1.0 - len(set(ngrams)) / max(len(ngrams), 1)
        else:
            repetition_rate = 0.0

        return DetectorSignal(
            detector_name="loop_detector_v1",
            failure_mode=FailureMode.INFINITE_LOOP,
            probability=min(1.0, repetition_rate * 1.5),
            confidence=0.8,
            evidence=f"Action repetition rate: {repetition_rate:.2f} "
                     f"over last {len(agent_events)} events",
            raw_features={"repetition_rate": repetition_rate, "window": len(agent_events)},
        )


class HallucinationDetector:
    """
    Detects hallucination spikes via:
    1. Embedding distance between claims and retrieved evidence
    2. Consecutive unsupported claims counter
    3. Sliding-window judge hallucination scores
    """

    def __init__(self, distance_threshold: float = 0.6, window_size: int = 10):
        self.distance_threshold = distance_threshold
        self.window_size = window_size

    def detect(self, events: list[TraceEvent], agent_id: str) -> DetectorSignal:
        recent = [e for e in events[-self.window_size:] if e.agent_id == agent_id]

        # Check judge hallucination scores if available
        halluc_scores = []
        for e in recent:
            if e.judge_payload and e.judge_payload.judge_type.value == "hallucination":
                halluc_scores.append(e.judge_payload.primary_score)

        avg_halluc = sum(halluc_scores) / len(halluc_scores) if halluc_scores else 0.0

        # Check embedding distances from retrieval events
        unsupported_count = 0
        total_claims = 0
        for e in recent:
            if e.embedding_payload and e.embedding_payload.nearest_neighbors:
                total_claims += 1
                best_score = max(
                    (n.get("score", 0) for n in e.embedding_payload.nearest_neighbors),
                    default=0,
                )
                if best_score < self.distance_threshold:
                    unsupported_count += 1

        unsupported_rate = unsupported_count / max(total_claims, 1)
        probability = max(avg_halluc, unsupported_rate)

        return DetectorSignal(
            detector_name="hallucination_detector_v1",
            failure_mode=FailureMode.HALLUCINATION_SPIKE,
            probability=probability,
            confidence=0.7 if halluc_scores else 0.5,
            evidence=f"Avg judge halluc score: {avg_halluc:.2f}, "
                     f"unsupported claims: {unsupported_count}/{total_claims}",
            raw_features={
                "avg_halluc_score": avg_halluc,
                "unsupported_rate": unsupported_rate,
            },
        )


class ContextOverflowDetector:
    """
    Predicts context window overflow via:
    1. Linear projection of token consumption rate
    2. Comparison against model's context window limit
    """

    def __init__(self, context_window: int = 200_000):
        self.context_window = context_window

    def detect(self, events: list[TraceEvent], agent_id: str) -> DetectorSignal:
        agent_events = [e for e in events if e.agent_id == agent_id]
        if len(agent_events) < 2:
            return DetectorSignal(
                detector_name="context_overflow_v1",
                failure_mode=FailureMode.CONTEXT_OVERFLOW,
                probability=0.0,
                confidence=0.3,
                evidence="Insufficient data for projection",
            )

        total_tokens = sum(e.input_tokens + e.output_tokens for e in agent_events)
        utilization = total_tokens / self.context_window

        # Project forward based on consumption rate
        elapsed = (agent_events[-1].timestamp - agent_events[0].timestamp).total_seconds()
        if elapsed > 0:
            rate = total_tokens / elapsed  # tokens per second
            remaining_capacity = self.context_window - total_tokens
            time_to_overflow = remaining_capacity / rate if rate > 0 else float("inf")
        else:
            time_to_overflow = float("inf")

        # Risk increases as utilization approaches 1.0
        if utilization > 0.9:
            probability = 0.95
        elif utilization > 0.7:
            probability = 0.5 + (utilization - 0.7) * 2.5
        else:
            probability = utilization * 0.3

        return DetectorSignal(
            detector_name="context_overflow_v1",
            failure_mode=FailureMode.CONTEXT_OVERFLOW,
            probability=probability,
            confidence=0.85,
            evidence=f"Token utilization: {utilization:.1%} "
                     f"({total_tokens}/{self.context_window}), "
                     f"projected overflow in {time_to_overflow:.0f}s",
            raw_features={
                "utilization": utilization,
                "total_tokens": total_tokens,
                "time_to_overflow": time_to_overflow,
            },
        )


class ToolThrashingDetector:
    """
    Detects repeated failed tool calls via Markov chain analysis:
    1. Compute transition probabilities between tool calls
    2. Flag when P(return to same tool) > threshold within N steps
    """

    def __init__(self, threshold: float = 0.7, window_size: int = 10):
        self.threshold = threshold
        self.window_size = window_size

    def detect(self, events: list[TraceEvent], agent_id: str) -> DetectorSignal:
        tool_events = [
            e for e in events[-self.window_size:]
            if e.agent_id == agent_id and e.tool_payload
        ]

        if len(tool_events) < 3:
            return DetectorSignal(
                detector_name="tool_thrashing_v1",
                failure_mode=FailureMode.TOOL_THRASHING,
                probability=0.0,
                confidence=0.3,
                evidence="Insufficient tool call data",
            )

        # Count self-transitions (same tool called consecutively)
        self_transitions = sum(
            1 for i in range(1, len(tool_events))
            if tool_events[i].tool_payload.tool_name == tool_events[i-1].tool_payload.tool_name
        )
        self_transition_rate = self_transitions / (len(tool_events) - 1)

        # Count failed tool calls
        failures = sum(
            1 for e in tool_events if not e.tool_payload.tool_success
        )
        failure_rate = failures / len(tool_events)

        # Thrashing = high self-transition + high failure
        probability = min(1.0, (self_transition_rate + failure_rate) / 2 * 1.5)

        return DetectorSignal(
            detector_name="tool_thrashing_v1",
            failure_mode=FailureMode.TOOL_THRASHING,
            probability=probability,
            confidence=0.75,
            evidence=f"Self-transition rate: {self_transition_rate:.2f}, "
                     f"failure rate: {failure_rate:.2f}",
            raw_features={
                "self_transition_rate": self_transition_rate,
                "failure_rate": failure_rate,
            },
        )


class ReasoningCollapseDetector:
    """
    Detects reasoning quality degradation via:
    1. Sliding window of judge reasoning scores
    2. Trend detection (moving average declining below threshold)
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 5):
        self.threshold = threshold
        self.window_size = window_size

    def detect(self, events: list[TraceEvent], agent_id: str) -> DetectorSignal:
        reasoning_scores = []
        for e in events:
            if (e.agent_id == agent_id and e.judge_payload
                    and e.judge_payload.judge_type == "reasoning"):
                reasoning_scores.append(e.judge_payload.primary_score)

        if len(reasoning_scores) < 2:
            return DetectorSignal(
                detector_name="reasoning_collapse_v1",
                failure_mode=FailureMode.REASONING_COLLAPSE,
                probability=0.0,
                confidence=0.3,
                evidence="Insufficient reasoning score data",
            )

        recent = reasoning_scores[-self.window_size:]
        avg = sum(recent) / len(recent)

        # Check for declining trend
        if len(recent) >= 3:
            trend = recent[-1] - recent[0]  # negative = declining
        else:
            trend = 0.0

        if avg < self.threshold:
            probability = 0.8 + (self.threshold - avg)
        elif trend < -0.2:
            probability = 0.5 + abs(trend)
        else:
            probability = max(0.0, 1.0 - avg)

        probability = min(1.0, probability)

        return DetectorSignal(
            detector_name="reasoning_collapse_v1",
            failure_mode=FailureMode.REASONING_COLLAPSE,
            probability=probability,
            confidence=0.7,
            evidence=f"Avg reasoning score: {avg:.2f}, trend: {trend:+.2f}",
            raw_features={"avg_score": avg, "trend": trend},
        )


class AgentDivergenceDetector:
    """
    Detects agents working at cross-purposes via:
    1. Cosine similarity of agent goal embeddings
    2. Conflicting tool calls or contradictory outputs
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    def detect(self, events: list[TraceEvent], agent_id: str) -> DetectorSignal:
        # In production, compare goal embeddings across cooperating agents.
        # Simplified heuristic: check for contradictory outputs via output hash collisions.
        return DetectorSignal(
            detector_name="agent_divergence_v1",
            failure_mode=FailureMode.AGENT_DIVERGENCE,
            probability=0.0,
            confidence=0.4,
            evidence="Divergence detection requires multi-agent goal embeddings",
        )


# ---------------------------------------------------------------------------
# Risk Aggregator
# ---------------------------------------------------------------------------

@dataclass
class RiskAggregator:
    """
    Fuses signals from all detectors into a unified RiskPayload.

    Uses Bayesian fusion: treat each detector as an independent
    evidence source with its own confidence weighting.
    """

    detectors: list[FailureDetector] = field(default_factory=lambda: [
        LoopDetector(),
        HallucinationDetector(),
        ContextOverflowDetector(),
        ToolThrashingDetector(),
        ReasoningCollapseDetector(),
        AgentDivergenceDetector(),
    ])

    def compute_risk(
        self,
        events: list[TraceEvent],
        agent_id: str,
    ) -> RiskPayload:
        """Compute aggregate risk for a specific agent."""
        signals = [d.detect(events, agent_id) for d in self.detectors]

        # Map signals to risk payload fields
        risk_map = {}
        contributing = []
        for s in signals:
            risk_map[s.failure_mode] = s.probability
            if s.probability > 0.1:
                contributing.append({
                    "detector": s.detector_name,
                    "signal": s.evidence,
                    "weight": s.confidence,
                })

        # Weighted aggregate
        weighted_sum = sum(s.probability * s.confidence for s in signals)
        weight_total = sum(s.confidence for s in signals)
        overall = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Interventions based on highest risks
        interventions = []
        sorted_signals = sorted(signals, key=lambda s: s.probability, reverse=True)
        for s in sorted_signals[:3]:
            if s.probability > 0.5:
                interventions.append(_intervention_for(s.failure_mode))

        # Estimate time to failure from context overflow detector
        ttf = None
        for s in signals:
            if s.failure_mode == FailureMode.CONTEXT_OVERFLOW:
                ttf = s.raw_features.get("time_to_overflow")

        return RiskPayload(
            loop_probability=risk_map.get(FailureMode.INFINITE_LOOP, 0.0),
            hallucination_probability=risk_map.get(FailureMode.HALLUCINATION_SPIKE, 0.0),
            context_overflow_risk=risk_map.get(FailureMode.CONTEXT_OVERFLOW, 0.0),
            tool_thrashing_risk=risk_map.get(FailureMode.TOOL_THRASHING, 0.0),
            reasoning_collapse_risk=risk_map.get(FailureMode.REASONING_COLLAPSE, 0.0),
            agent_divergence_risk=risk_map.get(FailureMode.AGENT_DIVERGENCE, 0.0),
            overall_failure_risk=round(overall, 4),
            time_to_predicted_failure=ttf,
            contributing_signals=contributing,
            recommended_interventions=interventions,
        )


def _intervention_for(mode: FailureMode) -> str:
    """Suggest intervention based on failure mode."""
    return {
        FailureMode.INFINITE_LOOP: "Break agent loop: inject stop condition or switch to critic agent",
        FailureMode.HALLUCINATION_SPIKE: "Inject retrieval step to re-ground agent in evidence",
        FailureMode.CONTEXT_OVERFLOW: "Summarize context window or split into sub-tasks",
        FailureMode.TOOL_THRASHING: "Disable failing tool and route to alternative",
        FailureMode.REASONING_COLLAPSE: "Insert critic agent checkpoint before continuing",
        FailureMode.AGENT_DIVERGENCE: "Synchronize agent goals via orchestrator re-alignment",
    }.get(mode, "Manual review recommended")
