"""Tests for classical baselines used in the learned-vs-classical comparison."""

from __future__ import annotations

import numpy as np

from medqc.baseline import classical_verdict, exposure_clipped_frac, sharpness_score
from medqc.degradations import apply_blur

BASELINE = {
    "blur_severe": {"f1": 0.36, "threshold": 2.2956, "precision": 0.4154, "recall": 0.3176},
    "exposure_severe": {"f1": 0.3183, "threshold": 0.0, "precision": 0.1931, "recall": 0.906},
}


def test_sharpness_blurred_lower_than_clean(rgb_image: np.ndarray) -> None:
    assert sharpness_score(apply_blur(rgb_image, 2, seed=1)) < sharpness_score(rgb_image)


def test_exposure_frac_higher_on_blown_image(rgb_image: np.ndarray) -> None:
    blown = np.full_like(rgb_image, 254)
    assert exposure_clipped_frac(blown) > exposure_clipped_frac(rgb_image)


def test_classical_verdict_flags_severe_blur(rgb_image: np.ndarray) -> None:
    blurred = apply_blur(rgb_image, 2, seed=2)
    verdict = classical_verdict(sharpness_score(blurred), exposure_clipped_frac(blurred), BASELINE)
    assert verdict["reasons"] == ["BLUR_SEVERE"]
    assert verdict["accepted"] is False
    assert verdict["reported_f1"]["blur"] == 0.36


def test_classical_verdict_accepts_when_below_thresholds() -> None:
    verdict = classical_verdict(1000.0, 0.0, BASELINE)
    assert verdict["accepted"] is True
    assert verdict["reasons"] == []


def test_classical_verdict_tolerates_missing_thresholds() -> None:
    verdict = classical_verdict(0.1, 0.9, {})
    assert verdict["accepted"] is True
    assert verdict["blur_threshold"] is None
