"""Classical, training-free quality baselines for head-to-head comparison.

These deliberately simple heuristics are what a non-learned pipeline would use.
They exist to be *beaten*: thresholds are tuned on validation data and their F1
scores are reported next to the learned model in ``metrics.json``.
"""

from __future__ import annotations

import numpy as np


def sharpness_score(img: np.ndarray) -> float:
    """Variance of the Laplacian — lower means blurrier."""
    a = img.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(lap.var())


def exposure_clipped_frac(img: np.ndarray) -> float:
    """Fraction of near-black / near-white pixels."""
    a = img.astype(np.int16)
    if a.ndim == 3:
        a = a.max(axis=2)
    return float(np.mean((a <= 6) | (a >= 249)))


def classical_verdict(sharpness: float, clip_frac: float, baseline_metrics: dict) -> dict:
    """Threshold the classical scores using tuned baselines from ``metrics.json``.

    Returns a dict mirroring the learned gate so the UI can show both verdicts.
    """
    blur_cfg = baseline_metrics.get("blur_severe") or {}
    exp_cfg = baseline_metrics.get("exposure_severe") or {}

    reasons: list[str] = []
    blur_thr = blur_cfg.get("threshold")
    exp_thr = exp_cfg.get("threshold")
    blur_severe = blur_thr is not None and sharpness < float(blur_thr)
    exp_severe = exp_thr is not None and clip_frac > float(exp_thr)
    if blur_severe:
        reasons.append("BLUR_SEVERE")
    if exp_severe:
        reasons.append("EXPOSURE_SEVERE")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "sharpness": round(sharpness, 3),
        "clipped_frac": round(clip_frac, 4),
        "blur_threshold": blur_thr,
        "exposure_threshold": exp_thr,
        "reported_f1": {
            "blur": blur_cfg.get("f1"),
            "exposure": exp_cfg.get("f1"),
        },
    }
