"""ONNX Runtime inference wrapper for the MedQC quality gate.

Kept import-light (no torch, no heavy UI deps) so the deployed app starts fast
and the live endpoint's TTFB stays low.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medqc.baseline import classical_verdict, exposure_clipped_frac, sharpness_score
from medqc.calibrate import sigmoid
from medqc.degradations import ATTRS, SEVERITY_LEVELS
from medqc.gate import DEFAULT_TAU, decide

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LEVEL_LABELS = SEVERITY_LEVELS


@dataclass
class GateResult:
    """Everything the UI/API needs to render one gate decision."""

    severities: dict[str, str]
    severity_idx: list[int]
    reject_prob: float
    decision: str
    reasons: list[str]
    confidence: float
    latency_ms: float
    mode: str
    classical: dict[str, Any]
    input_size: int
    measured_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_image(image: np.ndarray | Image.Image | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img.convert("RGB")


def preprocess(img: Image.Image, input_size: int) -> np.ndarray:
    """RGB -> resized float32 CHW with ImageNet normalisation (matches training)."""
    img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


class GatePredictor:
    """Loads the quantised ONNX gate and serves single-image predictions."""

    def __init__(
        self,
        model_path: str | Path,
        metrics_path: str | Path,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        import onnxruntime as ort  # local import: keeps module importable in minimal envs

        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        self.metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(self.model_path), providers=list(providers))

        self.input_size = int(self.metrics.get("input_size", 224))
        decision_cfg = (self.metrics.get("val") or {}).get("decision") or {}
        self.mode: str = str(decision_cfg.get("mode", "rule"))
        tau = decision_cfg.get("tau")
        self.tau: float | None = float(tau) if tau is not None else DEFAULT_TAU
        self.baseline = self.metrics.get("classical_baseline") or {}

    @property
    def summary(self) -> dict[str, Any]:
        """Model/quality summary for the UI sidebar."""
        val = self.metrics.get("val") or {}
        return {
            "latency_ms": self.metrics.get("latency_ms"),
            "model": self.metrics.get("model"),
            "gate_acc": val.get("gate_acc"),
            "decision": val.get("decision"),
            "per_attr_acc": dict(zip(ATTRS, val.get("per_attr_acc", []), strict=False)),
            "stratified": val.get("stratified_by_modality"),
            "gate_rule": self.metrics.get("gate_rule"),
            "mode": self.mode,
            "tau": self.tau,
        }

    def predict(self, image: np.ndarray | Image.Image | str | Path) -> GateResult:
        """Run the full gate pipeline and return a :class:`GateResult`."""
        t0 = time.perf_counter()
        pil = _load_image(image)
        input_arr = np.asarray(pil.resize((self.input_size, self.input_size), Image.BILINEAR))
        x = preprocess(pil, self.input_size)[None, ...]

        outputs = self.session.run(None, {"image": x.astype(np.float32)})
        sev_logits, gate_logit = outputs[0], outputs[1]
        severity_idx = [int(v) for v in sev_logits.argmax(axis=-1)[0]]
        reject_prob = float(sigmoid(float(gate_logit.reshape(-1)[0])))

        accepted, reasons = decide(reject_prob, severity_idx, mode=self.mode, tau=self.tau)
        classical = classical_verdict(
            sharpness_score(input_arr),
            exposure_clipped_frac(input_arr),
            self.baseline,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        conf = (1.0 - reject_prob) if accepted else reject_prob
        return GateResult(
            severities={name: LEVEL_LABELS[idx] for name, idx in zip(ATTRS, severity_idx, strict=True)},
            severity_idx=severity_idx,
            reject_prob=round(reject_prob, 4),
            decision="ACCEPT" if accepted else "REJECT",
            reasons=reasons,
            confidence=round(float(conf), 4),
            latency_ms=round(latency_ms, 2),
            mode=self.mode,
            classical=classical,
            input_size=self.input_size,
        )
