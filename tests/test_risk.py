"""Characterization + unit tests for omniscope.server.risk.RiskEngine.

These tests pin the EXACT numeric output of the heuristic detectors and the
weighted overall score so the refactor (named constants + single detector table)
is provably behavior-preserving. Golden values were captured from the pre-refactor
implementation. Pure logic only -- no server, no network; pydantic is already a
dependency of the package.

Run: python3 -m pytest tests/test_risk.py -q
"""
from __future__ import annotations

import pytest

from omniscope.server.models import TraceEvent, EventType
from omniscope.server.risk import RiskEngine


def _event(**kw) -> TraceEvent:
    kw.setdefault("agent_id", "a")
    return TraceEvent(**kw)


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine()


# ---------------------------------------------------------------------------
# compute(): empty input is all-zeros, no signals, no interventions
# ---------------------------------------------------------------------------

def test_empty_events_are_all_zero(engine):
    scores = engine.compute([], "a")
    assert scores.agent_id == "a"
    assert scores.loop_probability == 0.0
    assert scores.hallucination_probability == 0.0
    assert scores.context_overflow_risk == 0.0
    assert scores.tool_thrashing_risk == 0.0
    assert scores.reasoning_collapse_risk == 0.0
    assert scores.agent_divergence_risk == 0.0
    assert scores.overall_failure_risk == 0.0
    assert scores.contributing_signals == []
    assert scores.recommended_interventions == []


def test_compute_only_considers_the_named_agent(engine):
    mine = [_event(agent_id="a", event_type=EventType.TOOL_CALL, tool_name="Bash")
            for _ in range(10)]
    theirs = [_event(agent_id="b", confidence_score=0.1) for _ in range(10)]
    scores = engine.compute(mine + theirs, "a")
    # Hallucination is driven only by agent "b"'s low confidence; "a" has none.
    assert scores.hallucination_probability == 0.0


# ---------------------------------------------------------------------------
# Loop detection: repeated (event_type, tool) n-grams
# ---------------------------------------------------------------------------

def test_repeated_tool_calls_flag_a_loop(engine):
    events = [_event(event_type=EventType.TOOL_CALL, tool_name="Bash") for _ in range(10)]
    scores = engine.compute(events, "a")
    assert scores.loop_probability == 1.0
    assert "Break loop: inject stop condition" in scores.recommended_interventions
    assert any(s["detector"] == "loop" for s in scores.contributing_signals)


def test_short_stream_never_loops(engine):
    events = [_event(event_type=EventType.TOOL_CALL, tool_name="Bash") for _ in range(3)]
    assert engine.compute(events, "a").loop_probability == 0.0


# ---------------------------------------------------------------------------
# Hallucination detection: average of last 10 confidence scores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence,expected", [
    (0.3, 0.7),   # avg < 0.5
    (0.6, 0.4),   # 0.5 <= avg < 0.7
    (0.9, pytest.approx(1.0 - 0.9)),  # high confidence -> 1 - avg
])
def test_hallucination_thresholds(engine, confidence, expected):
    events = [_event(event_type=EventType.LLM_CALL, confidence_score=confidence)
              for _ in range(5)]
    assert engine.compute(events, "a").hallucination_probability == expected


def test_no_confidence_means_no_hallucination_signal(engine):
    events = [_event(event_type=EventType.LLM_CALL) for _ in range(5)]
    assert engine.compute(events, "a").hallucination_probability == 0.0


# ---------------------------------------------------------------------------
# Context overflow: utilization of the 200k context window
# ---------------------------------------------------------------------------

def test_context_overflow_above_90_percent(engine):
    events = [_event(input_tokens=100_000, output_tokens=90_000)]
    assert engine.compute(events, "a").context_overflow_risk == 0.95


def test_context_overflow_low_utilization_scales_linearly(engine):
    # 20k / 200k = 0.1 utilization -> 0.1 * 0.3 = 0.03
    events = [_event(input_tokens=10_000, output_tokens=10_000)]
    assert engine.compute(events, "a").context_overflow_risk == 0.03


# ---------------------------------------------------------------------------
# Tool thrashing: self-transitions + failures over the last 10 tool events
# ---------------------------------------------------------------------------

def test_repeated_same_tool_thrashes(engine):
    events = [_event(event_type=EventType.TOOL_CALL, tool_name="Bash") for _ in range(10)]
    # All self-transitions, no failures: rate = (1.0 + 0.0)/2 = 0.5 -> 0.5*1.5 = 0.75
    assert engine.compute(events, "a").tool_thrashing_risk == 0.75


def test_failing_tools_increase_thrashing(engine):
    events = [_event(event_type=EventType.TOOL_CALL, tool_name="Bash", tool_success=False)
              for _ in range(5)]
    assert engine.compute(events, "a").tool_thrashing_risk == 1.0


# ---------------------------------------------------------------------------
# Reasoning collapse: low/declining confidence trend
# ---------------------------------------------------------------------------

def test_low_confidence_triggers_collapse(engine):
    events = [_event(event_type=EventType.LLM_CALL, confidence_score=0.3)
              for _ in range(5)]
    assert engine.compute(events, "a").reasoning_collapse_risk == 0.8


def test_declining_confidence_trend_triggers_collapse(engine):
    confs = [0.9, 0.85, 0.8, 0.75, 0.5]  # avg >= 0.5 but trend = 0.5 - 0.9 = -0.4
    events = [_event(event_type=EventType.LLM_CALL, confidence_score=c) for c in confs]
    assert engine.compute(events, "a").reasoning_collapse_risk == 0.6


# ---------------------------------------------------------------------------
# Overall score + signals: the golden composite for a loop+thrash stream
# ---------------------------------------------------------------------------

def test_overall_score_and_signals_golden(engine):
    events = [_event(event_type=EventType.TOOL_CALL, tool_name="Bash") for _ in range(10)]
    scores = engine.compute(events, "a")
    # loop=1.0 (w 0.8), thrashing=0.75 (w 0.7), rest 0; sum(weights)=4.4
    assert scores.overall_failure_risk == 0.301
    detectors = {s["detector"] for s in scores.contributing_signals}
    assert detectors == {"loop", "thrashing"}
    # Each signal records its detector weight and a formatted score string.
    loop_sig = next(s for s in scores.contributing_signals if s["detector"] == "loop")
    assert loop_sig == {"detector": "loop", "signal": "loop risk: 1.00", "weight": 0.8}
    assert scores.recommended_interventions == [
        "Break loop: inject stop condition",
        "Disable failing tool, route to alternative",
    ]


def test_divergence_is_constant_zero_but_still_weighted(engine):
    # Divergence is not yet implemented (always 0.0) but its 0.4 weight is part
    # of the overall-score denominator; this pins that contract.
    events = [_event(input_tokens=100_000, output_tokens=90_000)]  # overflow=0.95
    scores = engine.compute(events, "a")
    assert scores.agent_divergence_risk == 0.0
    # overflow 0.95 * 0.85 / sum(weights 4.4) = 0.18352... -> rounded 0.184
    assert scores.overall_failure_risk == 0.184


# ---------------------------------------------------------------------------
# compute_all(): one RiskScores per distinct agent_id
# ---------------------------------------------------------------------------

def test_compute_all_covers_every_agent(engine):
    events = [
        _event(agent_id="planner", event_type=EventType.LLM_CALL, confidence_score=0.2),
        _event(agent_id="worker", event_type=EventType.TOOL_CALL, tool_name="Bash"),
    ]
    result = engine.compute_all(events)
    assert set(result) == {"planner", "worker"}
    assert result["planner"].agent_id == "planner"


def test_compute_all_ignores_blank_agent_ids(engine):
    events = [_event(agent_id="", event_type=EventType.SYSTEM_EVENT)]
    assert engine.compute_all(events) == {}
