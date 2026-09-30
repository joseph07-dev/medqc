"""Tests for gate decision logic and reason codes."""

from __future__ import annotations

import pytest

from medqc.gate import FALLBACK_REASON, MULTI_MILD_REASON, decide, decide_rule


@pytest.mark.parametrize(
    ("severities", "expected_accept", "expected_reasons"),
    [
        ([0, 0, 0, 0], True, []),
        ([1, 0, 0, 0], True, []),
        ([0, 1, 1, 0], False, [MULTI_MILD_REASON]),
        ([2, 0, 0, 0], False, ["BLUR_SEVERE"]),
        ([0, 0, 0, 2], False, ["OCCLUSION_SEVERE"]),
        ([2, 2, 0, 0], False, ["BLUR_SEVERE", "EXPOSURE_SEVERE"]),
        ([1, 2, 0, 1], False, ["EXPOSURE_SEVERE"]),
    ],
)
def test_decide_rule(severities: list[int], expected_accept: bool, expected_reasons: list[str]) -> None:
    accepted, reasons = decide_rule(severities)
    assert accepted is expected_accept
    assert reasons == expected_reasons


def test_reasons_empty_iff_accepted() -> None:
    for sev in ([0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 0, 0], [1, 1, 0, 1]):
        accepted, reasons = decide_rule(sev)
        assert accepted == (not reasons)


def test_head_mode_accepts_below_tau() -> None:
    accepted, reasons = decide(0.20, [0, 0, 0, 0], mode="head", tau=0.5)
    assert accepted is True
    assert reasons == []


def test_head_mode_rejects_at_or_above_tau_with_reason() -> None:
    accepted, reasons = decide(0.90, [2, 0, 0, 0], mode="head", tau=0.5)
    assert accepted is False
    assert "BLUR_SEVERE" in reasons


def test_head_mode_fallback_reason_when_no_severity_signal() -> None:
    accepted, reasons = decide(0.80, [0, 0, 0, 0], mode="head", tau=0.5)
    assert accepted is False
    assert reasons == [FALLBACK_REASON]


def test_head_mode_tau_boundary() -> None:
    assert decide(0.5, [0, 0, 0, 0], mode="head", tau=0.5)[0] is False
    assert decide(0.4999, [0, 0, 0, 0], mode="head", tau=0.5)[0] is True


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        decide(0.1, [0, 0, 0, 0], mode="magic")


def test_invalid_tau_raises() -> None:
    with pytest.raises(ValueError, match="tau must be within"):
        decide(0.1, [0, 0, 0, 0], mode="head", tau=1.5)


def test_rule_mode_ignores_probability() -> None:
    a = decide(0.99, [0, 0, 0, 0], mode="rule")
    b = decide(0.01, [0, 0, 0, 0], mode="rule")
    assert a == b == (True, [])
