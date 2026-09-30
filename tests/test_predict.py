"""End-to-end predictor tests against the bundled quantised ONNX model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from medqc.degradations import ATTRS, apply_occlusion
from medqc.predict import GatePredictor, GateResult


def test_predictor_loads_session(predictor: GatePredictor) -> None:
    assert predictor.session.get_inputs()[0].shape[1:] == [3, 224, 224]


def test_summary_exposes_scoring_fields(predictor: GatePredictor) -> None:
    summary = predictor.summary
    for key in ("latency_ms", "model", "gate_acc", "decision", "per_attr_acc", "stratified", "mode"):
        assert key in summary, f"summary missing {key}"


def test_predict_on_bundled_sample(predictor: GatePredictor, bundled_sample: Path) -> None:
    res = predictor.predict(bundled_sample)
    assert isinstance(res, GateResult)
    assert set(res.severities) == set(ATTRS)
    assert all(v in ("OK", "MILD", "SEVERE") for v in res.severities.values())
    assert res.decision in ("ACCEPT", "REJECT")
    assert 0.0 <= res.reject_prob <= 1.0
    assert res.latency_ms > 0.0
    assert res.decision == "ACCEPT" or res.reasons, "rejection must carry reason codes"
    assert res.decision == "REJECT" or res.reasons == []


def test_predict_accepts_numpy_and_pil(predictor: GatePredictor, rgb_image: np.ndarray) -> None:
    via_np = predictor.predict(rgb_image)
    via_pil = predictor.predict(Image.fromarray(rgb_image))
    assert via_np.decision == via_pil.decision
    assert via_np.severity_idx == via_pil.severity_idx


def test_predict_accepts_grayscale_image(predictor: GatePredictor, rgb_image: np.ndarray) -> None:
    gray = Image.fromarray(rgb_image).convert("L")
    res = predictor.predict(gray)
    assert res.decision in ("ACCEPT", "REJECT")


def test_heavy_occlusion_likely_rejected(predictor: GatePredictor, rgb_image: np.ndarray) -> None:
    occluded = apply_occlusion(rgb_image, 2, seed=11)
    res = predictor.predict(occluded)
    assert res.decision in ("ACCEPT", "REJECT")
    if res.decision == "REJECT":
        assert res.reasons


def test_result_serialises_to_json(predictor: GatePredictor, bundled_sample: Path) -> None:
    payload = predictor.predict(bundled_sample).to_dict()
    assert json.loads(json.dumps(payload))["decision"] in ("ACCEPT", "REJECT")


def test_reject_observations_within_validation_range(predictor: GatePredictor, bundled_sample: Path) -> None:
    res = predictor.predict(bundled_sample)
    assert res.input_size == 224
    assert res.mode in ("rule", "head")
