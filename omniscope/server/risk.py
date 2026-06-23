"""Heuristic risk scoring over agent event streams -- rule-based detectors, no learned models."""
from __future__ import annotations

from .models import TraceEvent, RiskScores

# Window sizes (how far back each detector looks).
_LOOP_WINDOW = 20            # recent actions scanned for repeated n-grams
_LOOP_NGRAMS = (2, 3)        # n-gram lengths checked for repetition
_LOOP_REPETITION_GATE = 0.5  # min repeated-fraction before a loop is flagged
_LOOP_AMPLIFY = 1.5          # repetition -> probability multiplier
_RECENT_CONFIDENCE = 10      # confidence samples used by the hallucination detector
_THRASH_WINDOW = 10          # recent tool events scanned for thrashing
_COLLAPSE_RECENT = 5         # confidence samples used for the collapse trend
_CONTEXT_WINDOW = 200_000    # token budget the overflow detector scores against

# A detector only contributes a "signal" / "intervention" once it clears these gates.
_SIGNAL_GATE = 0.1           # min score to record a contributing signal
_INTERVENTION_GATE = 0.5     # min score to recommend an intervention
_SIGNAL_WEIGHT = 0.8         # weight stored on every contributing-signal record


class RiskEngine:
    """Computes per-agent risk scores from trace event streams.

    Each named detector returns a 0..1 probability. ``compute`` blends them into
    one overall score and, from a single detector table, derives the contributing
    signals and recommended interventions -- so a detector's weight and remediation
    text live in exactly one place.
    """

    def compute(self, events: list[TraceEvent], agent_id: str) -> RiskScores:
        agent_events = [e for e in events if e.agent_id == agent_id]

        loop = self._detect_loops(agent_events)
        halluc = self._detect_hallucination(agent_events)
        overflow = self._detect_context_overflow(agent_events)
        thrashing = self._detect_tool_thrashing(agent_events)
        collapse = self._detect_reasoning_collapse(agent_events)
        divergence = 0.0  # not yet implemented; still weighted in the overall score

        # Single source of truth: (name, score, overall-weight, intervention text).
        # Divergence carries no signal/intervention text -- it only affects the
        # weighted average -- matching the original behavior.
        detectors = [
            ("loop", loop, 0.8, "Break loop: inject stop condition"),
            ("hallucination", halluc, 0.9, "Add retrieval step to re-ground agent"),
            ("overflow", overflow, 0.85, "Summarize context or split into sub-tasks"),
            ("thrashing", thrashing, 0.7, "Disable failing tool, route to alternative"),
            ("collapse", collapse, 0.75, "Insert critic checkpoint"),
            ("divergence", divergence, 0.4, None),
        ]

        signals = [
            {"detector": name, "signal": f"{name} risk: {score:.2f}", "weight": _SIGNAL_WEIGHT}
            for name, score, _weight, text in detectors
            if text is not None and score > _SIGNAL_GATE
        ]
        interventions = [
            text
            for _name, score, _weight, text in detectors
            if text is not None and score > _INTERVENTION_GATE
        ]

        weight_total = sum(weight for _n, _s, weight, _t in detectors)
        overall = (
            sum(score * weight for _n, score, weight, _t in detectors) / weight_total
            if weight_total else 0.0
        )

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
        actions = [(e.event_type, e.tool_name or "") for e in events[-_LOOP_WINDOW:]]
        if len(actions) < 4:
            return 0.0
        for n in _LOOP_NGRAMS:
            ngrams = [tuple(actions[i:i + n]) for i in range(len(actions) - n + 1)]
            if ngrams:
                repetition = 1.0 - len(set(ngrams)) / len(ngrams)
                if repetition > _LOOP_REPETITION_GATE:
                    return min(1.0, repetition * _LOOP_AMPLIFY)
        return 0.0

    def _detect_hallucination(self, events: list[TraceEvent]) -> float:
        confidences = [e.confidence_score for e in events[-_RECENT_CONFIDENCE:]
                       if e.confidence_score is not None]
        if not confidences:
            return 0.0
        avg = sum(confidences) / len(confidences)
        if avg < 0.5:
            return 0.7
        if avg < 0.7:
            return 0.4
        return max(0.0, 1.0 - avg)

    def _detect_context_overflow(self, events: list[TraceEvent],
                                 context_window: int = _CONTEXT_WINDOW) -> float:
        total = sum(e.input_tokens + e.output_tokens for e in events)
        utilization = total / context_window
        if utilization > 0.9:
            return 0.95
        if utilization > 0.7:
            return 0.5 + (utilization - 0.7) * 2.5
        return utilization * 0.3

    def _detect_tool_thrashing(self, events: list[TraceEvent]) -> float:
        tool_events = [e for e in events[-_THRASH_WINDOW:] if e.tool_name]
        if len(tool_events) < 3:
            return 0.0
        self_transitions = sum(
            1 for i in range(1, len(tool_events))
            if tool_events[i].tool_name == tool_events[i - 1].tool_name
        )
        failures = sum(1 for e in tool_events if not e.tool_success)
        rate = (self_transitions / (len(tool_events) - 1) + failures / len(tool_events)) / 2
        return min(1.0, rate * 1.5)

    def _detect_reasoning_collapse(self, events: list[TraceEvent]) -> float:
        scores = [e.confidence_score for e in events if e.confidence_score is not None]
        if len(scores) < 3:
            return 0.0
        recent = scores[-_COLLAPSE_RECENT:]
        avg = sum(recent) / len(recent)
        trend = recent[-1] - recent[0] if len(recent) >= 2 else 0.0
        if avg < 0.5:
            return 0.8
        if trend < -0.3:
            return 0.6
        return max(0.0, 0.3 * (1.0 - avg))
