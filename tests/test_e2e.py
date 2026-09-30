"""Full-pipeline integration tests: image in, decision + reason codes out."""

from __future__ import annotations

import numpy as np

from medqc.degradations import apply_blur, apply_exposure, apply_occlusion, apply_rotation


def test_pipeline_invariants_on_varied_inputs(predictor, rgb_image: np.ndarray) -> None:
    degraded = apply_occlusion(apply_rotation(apply_exposure(apply_blur(rgb_image, 2, 5), 2, 6), 2, 7), 2, 8)
    for image in (rgb_image, degraded):
        res = predictor.predict(image)
        assert res.decision in ("ACCEPT", "REJECT")
        assert res.decision == "REJECT" or res.reasons == []
        assert res.decision == "ACCEPT" or len(res.reasons) >= 1
        assert 0.0 <= res.confidence <= 1.0
        assert set(res.severities) == {"BLUR", "EXPOSURE", "ROTATION", "OCCLUSION"}


def test_pipeline_result_is_json_safe(predictor, rgb_image: np.ndarray) -> None:
    import json

    res = predictor.predict(rgb_image).to_dict()
    json.dumps(res)  # must not raise


def test_pipeline_is_fast_enough_for_real_time(predictor, rgb_image: np.ndarray) -> None:
    predictor.predict(rgb_image)  # warm-up
    res = predictor.predict(rgb_image)
    assert res.latency_ms < 200.0, "single-image gate must stay well under interactive latency"


def test_rule_and_head_modes_agree_on_structure(predictor, rgb_image: np.ndarray) -> None:
    res = predictor.predict(rgb_image)
    assert res.mode in ("rule", "head")
    assert isinstance(res.severity_idx, list) and len(res.severity_idx) == 4
    assert all(0 <= v <= 2 for v in res.severity_idx)
