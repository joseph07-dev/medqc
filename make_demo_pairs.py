"""Generate honest silent-failure demo pairs (single pass, deterministic).

For each modality keep the first image where:
  gate(clean) == ACCEPT  and  gate(degraded) == REJECT with reason codes.
Prefers pairs where the classical baseline still accepts the degraded image
(literal silent failure), otherwise keeps the first clean->reject transition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medmnist import PneumoniaMNIST, RetinaMNIST  # noqa: E402

from medqc.degradations import make_sample  # noqa: E402
from medqc.predict import GatePredictor  # noqa: E402

pred = GatePredictor(ROOT / "weights" / "medqc_v1.onnx", ROOT / "metrics.json")

DATASETS = [
    ("chest_xray", PneumoniaMNIST, 400_000),
    ("retinal_fundus", RetinaMNIST, 500_000),
]

pairs: list[dict] = []

for name, cls, offset in DATASETS:
    ds = cls(split="val", download=True, root=str(ROOT / "medmnist_data"), size=224)
    first_reject: dict | None = None
    chosen: dict | None = None
    for i in range(min(len(ds), 80)):
        clean = np.asarray(ds[i][0])
        if clean.ndim == 2:
            clean = np.repeat(clean[..., None], 3, axis=2)
        clean = np.ascontiguousarray(clean.astype(np.uint8))
        degraded, sev, _ = make_sample(clean, offset + i)
        rc = pred.predict(Image.fromarray(clean))
        rd = pred.predict(Image.fromarray(degraded))
        if rc.decision != "ACCEPT" or rd.decision != "REJECT":
            continue
        cand = {
            "entry": {
                "modality": name,
                "clean_png": f"assets/samples/pair_{name}_clean.png",
                "deg_png": f"assets/samples/pair_{name}_degraded.png",
                "probe": False,
                "y_true": None,
                "clean_pred": None,
                "clean_conf": None,
                "deg_pred": None,
                "deg_conf": None,
                "gate_clean": "ACCEPT",
                "gate_deg": "REJECT",
                "reasons": rd.reasons,
            },
            "clean": clean,
            "degraded": degraded,
            "sev": sev,
            "idx": i,
            "classical_ok": bool(rd.classical.get("accepted")),
        }
        if cand["classical_ok"]:
            chosen = cand
            break
        if first_reject is None:
            first_reject = cand
    chosen = chosen or first_reject
    if chosen is None:
        print(f"{name}: NO qualifying pair found")
        continue
    Image.fromarray(chosen["clean"]).save(ROOT / chosen["entry"]["clean_png"])
    Image.fromarray(chosen["degraded"]).save(ROOT / chosen["entry"]["deg_png"])
    pairs.append(chosen["entry"])
    print(
        f"{name}: idx={chosen['idx']} sev={chosen['sev']} reasons={chosen['entry']['reasons']} "
        f"classical_accepts_degraded={chosen['classical_ok']}"
    )

assert len(pairs) >= 1, "need at least one demo pair"
(ROOT / "downstream_results.json").write_text(json.dumps({"pairs": pairs}, indent=2), encoding="utf-8")
print(f"wrote {len(pairs)} pairs")
