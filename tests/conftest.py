"""Shared pytest fixtures for the MedQC test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def metrics(repo_root: Path) -> dict:
    path = repo_root / "metrics.json"
    assert path.is_file(), "metrics.json must be committed with the repo"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def decision_cfg(metrics: dict) -> dict:
    return (metrics.get("val") or {}).get("decision") or {"mode": "rule", "tau": None}


@pytest.fixture()
def rgb_image() -> np.ndarray:
    """Deterministic synthetic 224x224 RGB image with smooth gradients."""
    yy, xx = np.mgrid[0:224, 0:224]
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[..., 0] = (40 + 0.5 * xx + 0.3 * yy).astype(np.uint8)
    img[..., 1] = (30 + 0.2 * xx).astype(np.uint8)
    img[..., 2] = (20 + 0.4 * yy).astype(np.uint8)
    return img


@pytest.fixture()
def sample_png(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    path = tmp_path / "sample.png"
    Image.fromarray(rgb_image).save(path)
    return path


@pytest.fixture(scope="session")
def bundled_sample(repo_root: Path) -> Path:
    samples = sorted((repo_root / "assets" / "samples").glob("*.png"))
    if not samples:
        pytest.skip("no bundled sample images")
    return samples[0]


@pytest.fixture(scope="session")
def predictor(repo_root: Path):
    pytest.importorskip("onnxruntime")
    weights = repo_root / "weights" / "medqc_v1.onnx"
    if not weights.is_file():
        pytest.skip("quantised model weights not bundled")
    from medqc.predict import GatePredictor

    return GatePredictor(weights, repo_root / "metrics.json")
