"""Tests for calibration utilities (temperature scaling, ECE, reliability bins)."""

from __future__ import annotations

import numpy as np
import pytest

from medqc.calibrate import (
    binary_nll,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    sigmoid,
)


def test_sigmoid_known_values() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(10.0) == pytest.approx(0.9999546, abs=1e-6)
    assert sigmoid(-10.0) == pytest.approx(4.54e-5, abs=1e-6)


def test_sigmoid_vectorised_bounds() -> None:
    out = sigmoid(np.array([-50.0, 0.0, 50.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert out[0] < 0.01 < out[2]


def test_temperature_scaling_reduces_nll_on_overconfident_logits() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(0, 1, size=2000)
    logits = latent * 12.0  # overconfident: |logit| far larger than the evidence supports
    labels = (latent + rng.normal(0, 0.9, size=2000) > 0).astype(float)  # noisy labels
    nll_raw = binary_nll(logits, labels, 1.0)
    best_t = fit_temperature(logits, labels)
    nll_scaled = binary_nll(logits, labels, best_t)
    assert nll_scaled < nll_raw
    assert best_t > 1.0, "overconfident logits should require T > 1"


def test_fit_temperature_returns_reasonable_range() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(500) > 0.5).astype(float)
    logits = rng.normal(0, 1, size=labels.shape)
    t = fit_temperature(logits, labels)
    assert 0.25 <= t <= 5.0


def test_ece_perfectly_calibrated_is_zero() -> None:
    probs = np.array([0.1] * 50 + [0.9] * 50)
    labels = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_miscalibration() -> None:
    probs = np.full(100, 0.95)
    labels = np.zeros(100)
    assert expected_calibration_error(probs, labels) > 0.4


def test_ece_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        expected_calibration_error(np.array([0.5, 0.6]), np.array([0.0]))


def test_reliability_bins_counts_sum_to_sample_count() -> None:
    rng = np.random.default_rng(2)
    probs = rng.uniform(0, 1, 500)
    labels = (rng.random(500) < probs).astype(float)
    bins = reliability_bins(probs, labels, bins=10)
    assert sum(c for _, _, c in bins) == 500
    for conf, acc, count in bins:
        assert 0.0 <= conf <= 1.0 and 0.0 <= acc <= 1.0 and count > 0
