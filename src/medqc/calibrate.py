"""Confidence calibration utilities: temperature scaling and ECE."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.where(arr >= 0, 1.0 / (1.0 + np.exp(-np.abs(arr))), np.exp(-np.abs(arr)) / (1.0 + np.exp(-np.abs(arr))))
    return float(out) if np.isscalar(x) or arr.ndim == 0 else out


def binary_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Binary cross-entropy of temperature-scaled logits."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    y = np.asarray(labels, dtype=np.float64)
    p = sigmoid(z)
    return float(-np.mean(y * np.log(p + EPS) + (1.0 - y) * np.log(1.0 - p + EPS)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, t_min: float = 0.25, t_max: float = 5.0, step: float = 0.05) -> float:
    """Grid-search temperature minimising validation NLL (Guo et al., 2017)."""
    best_t, best_nll = 1.0, binary_nll(logits, labels, 1.0)
    for t in np.arange(t_min, t_max + step, step):
        nll = binary_nll(logits, labels, float(t))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 3)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Standard binned ECE in [0, 1]; 0 means perfectly calibrated."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape")
    edges = np.linspace(0.0, 1.0, bins + 1)
    preds = (probs >= 0.5).astype(np.float64)
    total = len(probs)
    ece = 0.0
    for i in range(bins):
        if i == 0:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs > edges[i]) & (probs <= edges[i + 1])
        if mask.sum():
            ece += mask.sum() / total * abs(float(preds[mask].mean() - labels[mask].mean()))
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """Per-bin (mean_confidence, accuracy, count) triples for reliability plots."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    preds = (probs >= 0.5).astype(np.float64)
    out: list[tuple[float, float, int]] = []
    for i in range(bins):
        if i == 0:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs > edges[i]) & (probs <= edges[i + 1])
        if mask.sum():
            out.append((float(probs[mask].mean()), float(preds[mask].mean()), int(mask.sum())))
    return out
