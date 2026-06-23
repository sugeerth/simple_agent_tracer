"""Heuristic risk scoring over agent event streams — rule-based detectors, no learned models."""
from __future__ import annotations

from collections import defaultdict
from .models import TraceEvent, RiskScores


class RiskEngine:
    """Computes per-agent risk scores from trace event streams."""

    def compute(self, events: list[TraceEvent], agent_id: str) -> RiskScores:
        agent_events = [e for e in events if e.agent_id == agent_id]
        signals = []

        loop = self._detect_loops(agent_events)
        halluc = self._detect_hallucination(agent_events)
        overflow = self._detect_context_overflow(agent_events)
        thrashing = self._detect_tool_thrashing(agent_events)
        collapse = self._detect_reasoning_collapse(agent_events)
        divergence = 0.0

        for name, score in [
            ("loop", loop), ("hallucination", halluc), ("overflow", overflow),
            ("thrashing", thrashing), ("collapse", collapse),
        ]:
            if score > 0.1:
                signals.append({"detector": name, "signal": f"{name} risk: {score:.2f}", "weight": 0.8})

        scores = [loop, halluc, overflow, thrashing, collapse, divergence]
        weights = [0.8, 0.9, 0.85, 0.7, 0.75, 0.4]
        overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights) if weights else 0.0

        interventions = []
        if loop > 0.5:
            interventions.append("Break loop: inject stop condition")
        if halluc > 0.5:
            interventions.append("Add retrieval step to re-ground agent")
        if overflow > 0.5:
            interventions.append("Summarize context or split into sub-tasks")
        if thrashing > 0.5:
            interventions.append("Disable failing tool, route to alternative")
        if collapse > 0.5:
            interventions.append("Insert critic checkpoint")

        return RiskScores(
            agent_id=agent_id,
            loop_probability=round(loop, 3),
            hallucination_probability=round(halluc, 3),
            context_overflow_risk=round(overflow, 3),
            tool_thrashing_risk=round(thrashing, 3),
            reasoning_collapse_risk=round(collapse, 3),
            agent_divergence_risk=round(divergence, 3),
            overall_failure_risk=round(overall, 3),
            contributing_signals=signals,
            recommended_interventions=interventions,
        )

    def compute_all(self, events: list[TraceEvent]) -> dict[str, RiskScores]:
        agents = set(e.agent_id for e in events if e.agent_id)
        return {aid: self.compute(events, aid) for aid in agents}

    def _detect_loops(self, events: list[TraceEvent]) -> float:
        if len(events) < 4:
            return 0.0
        actions = []
        for e in events[-20:]:
            key = (e.event_type, e.tool_name or "")
            actions.append(key)
        if len(actions) < 4:
            return 0.0
        # Check 2-gram and 3-gram repetition
        for n in [2, 3]:
            ngrams = [tuple(actions[i:i+n]) for i in range(len(actions) - n + 1)]
            if ngrams:
                repetition = 1.0 - len(set(ngrams)) / len(ngrams)
                if repetition > 0.5:
                    return min(1.0, repetition * 1.5)
        return 0.0

    def _detect_hallucination(self, events: list[TraceEvent]) -> float:
        confidences = [e.confidence_score for e in events[-10:] if e.confidence_score is not None]
        if not confidences:
            return 0.0
        avg = sum(confidences) / len(confidences)
        if avg < 0.5:
            return 0.7
        if avg < 0.7:
            return 0.4
        return max(0.0, 1.0 - avg)

    def _detect_context_overflow(self, events: list[TraceEvent], context_window: int = 200_000) -> float:
        total = sum(e.input_tokens + e.output_tokens for e in events)
        utilization = total / context_window
        if utilization > 0.9:
            return 0.95
        if utilization > 0.7:
            return 0.5 + (utilization - 0.7) * 2.5
        return utilization * 0.3

    def _detect_tool_thrashing(self, events: list[TraceEvent]) -> float:
        tool_events = [e for e in events[-10:] if e.tool_name]
        if len(tool_events) < 3:
            return 0.0
        self_transitions = sum(
            1 for i in range(1, len(tool_events))
            if tool_events[i].tool_name == tool_events[i-1].tool_name
        )
        failures = sum(1 for e in tool_events if not e.tool_success)
        rate = (self_transitions / (len(tool_events) - 1) + failures / len(tool_events)) / 2
        return min(1.0, rate * 1.5)

    def _detect_reasoning_collapse(self, events: list[TraceEvent]) -> float:
        scores = [e.confidence_score for e in events if e.confidence_score is not None]
        if len(scores) < 3:
            return 0.0
        recent = scores[-5:]
        avg = sum(recent) / len(recent)
        trend = recent[-1] - recent[0] if len(recent) >= 2 else 0.0
        if avg < 0.5:
            return 0.8
        if trend < -0.3:
            return 0.6
        return max(0.0, 0.3 * (1.0 - avg))
