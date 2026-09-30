"""Tests for the deterministic degradation synthesis pipeline."""

from __future__ import annotations

import numpy as np

from medqc.baseline import sharpness_score
from medqc.degradations import (
    ATTRS,
    apply_blur,
    apply_exposure,
    apply_occlusion,
    apply_rotation,
    gate_label_from_severities,
    make_sample,
    sample_severities,
)


def test_severity_zero_is_exact_noop(rgb_image: np.ndarray) -> None:
    for fn in (apply_blur, apply_exposure, apply_rotation, apply_occlusion):
        out = fn(rgb_image, 0, seed=7)
        np.testing.assert_array_equal(out, rgb_image)


def test_same_seed_is_byte_deterministic(rgb_image: np.ndarray) -> None:
    for fn in (apply_blur, apply_exposure, apply_rotation, apply_occlusion):
        a = fn(rgb_image, 2, seed=12345)
        b = fn(rgb_image, 2, seed=12345)
        np.testing.assert_array_equal(a, b)


def test_different_seed_can_differ_for_randomised_degradations(rgb_image: np.ndarray) -> None:
    results = {apply_exposure(rgb_image, 1, seed=s).tobytes() for s in range(8)}
    assert len(results) > 1, "randomised exposure must depend on the seed"


def test_blur_reduces_sharpness(rgb_image: np.ndarray) -> None:
    clean = sharpness_score(rgb_image)
    blurred = sharpness_score(apply_blur(rgb_image, 2, seed=1))
    assert blurred < clean


def test_severe_exposure_changes_pixels(rgb_image: np.ndarray) -> None:
    out = apply_exposure(rgb_image, 2, seed=3)
    assert out.shape == rgb_image.shape
    assert not np.array_equal(out, rgb_image)


def test_rotation_preserves_shape(rgb_image: np.ndarray) -> None:
    out = apply_rotation(rgb_image, 2, seed=5)
    assert out.shape == rgb_image.shape


def test_occlusion_marks_black_rectangle(rgb_image: np.ndarray) -> None:
    out = apply_occlusion(rgb_image, 2, seed=9)
    black_frac = float(np.mean(out.max(axis=2) == 0))
    assert black_frac > 0.30, "severe occlusion should cover >= ~35% of the frame"


def test_sample_severities_reproducible() -> None:
    assert sample_severities(42) == sample_severities(42)


def test_make_sample_reproducible_and_consistent(rgb_image: np.ndarray) -> None:
    img_a, sev_a, gate_a = make_sample(rgb_image, seed=99)
    img_b, sev_b, gate_b = make_sample(rgb_image, seed=99)
    np.testing.assert_array_equal(img_a, img_b)
    assert sev_a == sev_b
    assert gate_a == gate_b
    assert gate_a == int(gate_label_from_severities(sev_a))


def test_gate_label_rule() -> None:
    assert gate_label_from_severities([0, 0, 0, 0]) is False
    assert gate_label_from_severities([1, 0, 0, 0]) is False
    assert gate_label_from_severities([1, 1, 0, 0]) is True
    assert gate_label_from_severities([2, 0, 0, 0]) is True
    assert gate_label_from_severities([0, 0, 1, 2]) is True


def test_attributes_constant() -> None:
    assert ATTRS == ("BLUR", "EXPOSURE", "ROTATION", "OCCLUSION")
    assert len(ATTRS) == 4
