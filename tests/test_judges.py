"""Unit tests for omniscope.judges aggregation + model resolution.

Pure, stdlib-only logic (the judges module imports nothing heavy). Golden values
pin the weighted-aggregation contract: confidence-interval down-weighting,
hallucination-score inversion, safety-triggered human review, and the symmetric
composite confidence band.

Run: python3 -m pytest tests/test_judges.py -q
"""
from __future__ import annotations

import pytest

from omniscope.judges import (
    JUDGE_CONFIGS,
    SAFETY_REVIEW_THRESHOLD,
    COMPOSITE_CI_HALF_WIDTH,
    AggregatedScore,
    FlaggedSpan,
    JudgeEvaluation,
    JudgeType,
    aggregate_judge_scores,
    get_judge_model,
)


def _eval(judge_type: JudgeType, score: float, ci=(0.0, 1.0),
          spans=None) -> JudgeEvaluation:
    return JudgeEvaluation(
        judge_id=f"{judge_type.value}_v1",
        judge_type=judge_type,
        evaluated_event_id="ev1",
        model_used="test-model",
        primary_score=score,
        confidence_interval=ci,
        flagged_spans=spans or [],
    )


# ---------------------------------------------------------------------------
# get_judge_model: env override wins, else the configured default
# ---------------------------------------------------------------------------

def test_get_judge_model_defaults(monkeypatch):
    monkeypatch.delenv("JUDGE_SAFETY_MODEL", raising=False)
    assert get_judge_model(JudgeType.SAFETY) == JUDGE_CONFIGS[JudgeType.SAFETY].model_name


def test_get_judge_model_env_override(monkeypatch):
    monkeypatch.setenv("JUDGE_SAFETY_MODEL", "custom-model")
    assert get_judge_model(JudgeType.SAFETY) == "custom-model"


# ---------------------------------------------------------------------------
# aggregate_judge_scores: empty + single-judge contracts
# ---------------------------------------------------------------------------

def test_aggregate_empty_returns_zero():
    agg = aggregate_judge_scores([])
    assert isinstance(agg, AggregatedScore)
    assert agg.composite_score == 0.0
    assert agg.composite_confidence == (0.0, 0.0)
    assert agg.flags == []
    assert agg.requires_human_review is False


def test_single_reasoning_judge_composite_equals_its_score():
    # One judge, perfect certainty -> composite is just its (calibrated) score.
    agg = aggregate_judge_scores([_eval(JudgeType.REASONING, 0.8, ci=(0.8, 0.8))])
    assert agg.composite_score == 0.8
    lo, hi = agg.composite_confidence
    assert lo == pytest.approx(0.8 - COMPOSITE_CI_HALF_WIDTH)
    assert hi == pytest.approx(0.8 + COMPOSITE_CI_HALF_WIDTH)
    assert agg.breakdown == {"reasoning": 0.8}


# ---------------------------------------------------------------------------
# Hallucination inversion: risk -> quality before blending
# ---------------------------------------------------------------------------

def test_hallucination_score_is_inverted():
    # A high hallucination RISK (0.9) should contribute LOW quality (0.1).
    high_risk = aggregate_judge_scores([_eval(JudgeType.HALLUCINATION, 0.9, ci=(0.9, 0.9))])
    low_risk = aggregate_judge_scores([_eval(JudgeType.HALLUCINATION, 0.1, ci=(0.1, 0.1))])
    assert high_risk.composite_score == pytest.approx(1.0 - 0.9)
    assert low_risk.composite_score == pytest.approx(1.0 - 0.1)
    # The breakdown still records the raw (un-inverted) primary score.
    assert high_risk.breakdown["hallucination"] == 0.9


# ---------------------------------------------------------------------------
# Safety threshold drives human review
# ---------------------------------------------------------------------------

def test_low_safety_score_requires_human_review():
    below = aggregate_judge_scores([_eval(JudgeType.SAFETY, SAFETY_REVIEW_THRESHOLD - 0.01)])
    assert below.requires_human_review is True


def test_safe_score_does_not_require_review():
    at_threshold = aggregate_judge_scores([_eval(JudgeType.SAFETY, SAFETY_REVIEW_THRESHOLD)])
    assert at_threshold.requires_human_review is False


# ---------------------------------------------------------------------------
# Flag surfacing: only high/critical spans appear
# ---------------------------------------------------------------------------

def test_only_high_severity_spans_are_flagged():
    spans = [
        FlaggedSpan(0, 5, "minor nit", "low"),
        FlaggedSpan(10, 20, "fabricated stat", "high"),
        FlaggedSpan(30, 40, "dangerous", "critical"),
    ]
    agg = aggregate_judge_scores([_eval(JudgeType.FACTUALITY, 0.5, spans=spans)])
    assert agg.flags == [
        "factuality: fabricated stat (chars 10-20)",
        "factuality: dangerous (chars 30-40)",
    ]


# ---------------------------------------------------------------------------
# Golden multi-judge composite (pins the full weighted blend)
# ---------------------------------------------------------------------------

def test_multi_judge_golden_composite():
    evals = [
        _eval(JudgeType.REASONING, 0.8, ci=(0.7, 0.9)),
        _eval(JudgeType.HALLUCINATION, 0.2, ci=(0.1, 0.3),
              spans=[FlaggedSpan(0, 10, "fabricated", "high")]),
        _eval(JudgeType.SAFETY, 0.7, ci=(0.6, 0.8)),
    ]
    agg = aggregate_judge_scores(evals)
    assert agg.composite_score == 0.7574
    assert agg.composite_confidence == (
        pytest.approx(0.7074468, abs=1e-6),
        pytest.approx(0.8074468, abs=1e-6),
    )
    assert agg.flags == ["hallucination: fabricated (chars 0-10)"]
    assert agg.requires_human_review is True  # safety 0.7 < 0.8
    assert agg.breakdown == {"reasoning": 0.8, "hallucination": 0.2, "safety": 0.7}
    assert agg.evaluated_event_id == "ev1"
