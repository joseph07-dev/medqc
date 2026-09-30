"""Repository contract tests: metrics/config integrity and latency budget."""

from __future__ import annotations

import json
from pathlib import Path

from medqc.degradations import ATTRS, GATE_RULE
from medqc.gate import DECISION_MODES


def test_metrics_structure(metrics: dict) -> None:
    assert metrics["input_size"] == 224
    assert metrics["gate_rule"] == GATE_RULE
    assert metrics["model"]["arch"]
    assert metrics["model"]["params"] > 0


def test_latency_budget_met(metrics: dict) -> None:
    p95 = metrics["latency_ms"]["p95"]
    assert p95 < 50.0, f"p95 latency {p95} ms exceeds the 50 ms budget promised in the problem statement"
    assert metrics["latency_ms"]["provider"] == "CPUExecutionProvider"


def test_per_attribute_accuracies_present_and_sane(metrics: dict) -> None:
    per_attr = metrics["val"]["per_attr_acc"]
    assert len(per_attr) == len(ATTRS)
    assert all(0.0 <= v <= 1.0 for v in per_attr)
    assert min(per_attr) > 0.5, "every attribute head must beat chance"


def test_decision_config_valid(metrics: dict, decision_cfg: dict) -> None:
    assert decision_cfg.get("mode") in DECISION_MODES
    if decision_cfg["mode"] == "head":
        tau = decision_cfg.get("tau")
        assert tau is not None and 0.0 < float(tau) < 1.0
    assert "clean_accept_rate" in metrics["val"]


def test_stratified_metrics_cover_both_modalities(metrics: dict) -> None:
    strat = metrics["val"]["stratified_by_modality"]
    assert {"chest_xray", "retina"} <= set(strat)
    for name, block in strat.items():
        assert 0.0 <= block["gate_auroc"] <= 1.0, f"{name} AUROC out of range"


def test_calibration_fields_reported(metrics: dict) -> None:
    val = metrics["val"]
    assert "ece_pre_calibration" in val
    assert "ece_post_calibration" in val
    assert "temperature" in val


def test_classical_baseline_documented(metrics: dict) -> None:
    baseline = metrics["classical_baseline"]
    assert "blur_severe" in baseline and "exposure_severe" in baseline
    assert baseline["method"]
    assert metrics["val"]["gate_auroc"] > baseline["blur_severe"]["f1"], (
        "learned gate AUROC should exceed the classical blur baseline F1"
    )


def test_downstream_pairs_schema(repo_root: Path) -> None:
    path = repo_root / "downstream_results.json"
    assert path.is_file()
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    assert isinstance(pairs, list) and pairs, "at least one demo pair must be bundled"
    for pair in pairs:
        for key in ("modality", "clean_png", "deg_png", "gate_clean", "gate_deg", "reasons"):
            assert key in pair
        for img_key in ("clean_png", "deg_png"):
            assert (repo_root / pair[img_key]).is_file(), f"missing {pair[img_key]}"
        assert pair["gate_deg"] in ("ACCEPT", "REJECT")


def test_sample_assets_exist(repo_root: Path) -> None:
    samples = list((repo_root / "assets" / "samples").glob("*.png"))
    assert len(samples) >= 2, "bundled demo images required for sample-first UX"


def test_no_plaintext_secrets_in_tree(repo_root: Path) -> None:
    """Guard used by the audit: no obvious credential patterns anywhere."""
    # Assembled at runtime so this test file does not trip its own scanner.
    forbidden = ("AK" + "IA", "BEGIN RSA PRIVATE " + "KEY", "gh" + "p_", "h" + "f_", "s" + "k-")
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
    for path in repo_root.rglob("*"):
        if not path.is_file() or any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix in {".png", ".onnx", ".jpg", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in forbidden:
            assert token not in text, f"possible secret {token!r} found in {path}"
