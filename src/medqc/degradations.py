"""Deterministic, seed-controlled degradation synthesis for medical images.

Each function maps ``(image, severity, seed)`` to a degraded copy. Severity ``0``
is always an exact no-op, which makes the pipeline trivially testable: identical
seeds must yield byte-identical outputs, and severity ``0`` must preserve pixels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageFilter

ATTRS: tuple[str, ...] = ("BLUR", "EXPOSURE", "ROTATION", "OCCLUSION")
SEVERITY_LEVELS: tuple[str, ...] = ("OK", "MILD", "SEVERE")
GATE_RULE = "REJECT if any attribute severe or >=2 attributes mild"
_SEVERITY_PROBS: tuple[float, ...] = (0.60, 0.25, 0.15)

BLUR_SIGMA: dict[int, float] = {0: 0.0, 1: 2.5, 2: 6.0}
ROTATION_DEG: dict[int, float] = {0: 0.0, 1: 15.0, 2: 35.0}
OCCLUSION_FRAC: dict[int, float] = {0: 0.0, 1: 0.18, 2: 0.40}


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(img.astype(np.uint8)))


def _ensure_hwc3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def apply_blur(img: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Defocus blur with severity-scaled Gaussian radius."""
    if severity == 0:
        return img.copy()
    out = _to_pil(img).filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA[severity]))
    return np.asarray(out, dtype=np.uint8)


def apply_exposure(img: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Brightness/contrast shift (mild) or crushed/blown levels (severe)."""
    if severity == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    a = img.astype(np.float32)
    if severity == 1:
        a = (a - 127.5) * rng.uniform(0.7, 1.3) + 127.5
        a = a * rng.uniform(0.65, 1.35)
    else:
        if int(rng.integers(2)) == 0:
            a = np.clip(a / 255.0, 0, 1) ** 2.6 * 255.0 * 0.45
        else:
            a = np.clip(a / 255.0, 0, 1) ** 0.35 * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def apply_rotation(img: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Out-of-plane rotation with black fill corners."""
    if severity == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    deg = float(rng.uniform(ROTATION_DEG[severity] * 0.75, ROTATION_DEG[severity]))
    deg *= float(rng.choice([-1, 1]))
    out = _to_pil(img).rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    return np.asarray(out, dtype=np.uint8)


def apply_occlusion(img: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Random black rectangle covering a severity-scaled fraction of the field of view."""
    if severity == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    a = img.copy()
    h, w = a.shape[:2]
    s = math.sqrt(OCCLUSION_FRAC[severity])
    bw = max(8, int(w * s * rng.uniform(0.85, 1.15)))
    bh = max(8, int(h * s * rng.uniform(0.85, 1.15)))
    x0 = int(rng.integers(0, max(1, w - bw)))
    y0 = int(rng.integers(0, max(1, h - bh)))
    a[y0 : y0 + bh, x0 : x0 + bw] = 0
    return a


def sample_severities(seed: int) -> list[int]:
    """Draw one severity per attribute from the fixed class distribution."""
    rng = np.random.default_rng(seed)
    return [int(rng.choice(3, p=_SEVERITY_PROBS)) for _ in range(4)]


def make_sample(img: np.ndarray, seed: int) -> tuple[np.ndarray, list[int], int]:
    """Compose all four degradations; returns (image, severities, gate_label).

    Gate label follows :data:`GATE_RULE`: reject when any attribute is severe or
    at least two attributes are mildly degraded.
    """
    sev = sample_severities(seed)
    rng = np.random.default_rng(seed)
    out = img.copy()
    out = apply_blur(out, sev[0], int(rng.integers(1 << 30)))
    out = apply_exposure(out, sev[1], int(rng.integers(1 << 30)))
    out = apply_rotation(out, sev[2], int(rng.integers(1 << 30)))
    out = apply_occlusion(out, sev[3], int(rng.integers(1 << 30)))
    gate = int(gate_label_from_severities(sev))
    return out, sev, gate


def gate_label_from_severities(severities: Sequence[int]) -> bool:
    """Ground-truth rejection label for a severity vector (True = reject)."""
    sev = [int(s) for s in severities]
    return (2 in sev) or (sum(s >= 1 for s in sev) >= 2)
