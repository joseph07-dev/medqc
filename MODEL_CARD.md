# Model Card — MedQC quality gate v1.0

## Model details

| | |
|---|---|
| Name | `medqc_v1` (int8 dynamic quantisation of `medqc_fp32`) |
| Architecture | MobileNetV3-Small backbone (ImageNet-pretrained) + 4 severity heads (3 classes each) + 1 gate head |
| Parameters | 1 078 061 |
| Input | 224 × 224 RGB, ImageNet normalization |
| Outputs | `sev_logits` [batch, 4, 3], `gate_logit` [batch] |
| Format | ONNX (`weights/medqc_v1.onnx`, `weights/medqc_fp32.onnx`) |
| Runtime | ONNX Runtime, CPUExecutionProvider, p50 31.3 ms / p95 33.1 ms |
| Training | 5 epochs, Kaggle T4×2, 5 788 train / 644 validation samples |
| License | MIT (model weights included in this repository) |

## Intended use

- **Primary:** runtime accept/reject gate placed *before* a downstream medical-imaging
  classifier or reader workflow, rejecting non-diagnostic images with reason codes.
- **Secondary:** research baseline for quality-control / silent-failure studies.

## Out-of-scope use

- Not a diagnostic model: it never predicts disease.
- Not a medical device; no regulatory clearance. Must not be used as the sole basis for
  any clinical decision.
- Not validated on modalities beyond chest X-ray and retinal fundus (CT, MRI, ultrasound,
  pathology, ultrasound, mammography are out of scope until re-trained and re-validated).

## Training data

- **PneumoniaMNIST** (chest X-ray, pediatric) and **RetinaMNIST** (retinal fundus) from
  MedMNIST v2 — public, benchmark-standard, license-compatible.
- Labels for quality attributes are **synthetic**: each training image is degraded with
  deterministic, seed-controlled transforms (defocus blur, exposure shift, rotation,
  occlusion) at severity 0/1/2 using the exact functions in `src/medqc/degradations.py`.
- Gate label = rule: reject if any attribute is severe or ≥ 2 are mild.

## Metrics

Reported verbatim from `metrics.json` (validation, n = 644):

- Gate accuracy **0.9053**, gate AUROC **0.9634**
- Per-attribute accuracy: BLUR 0.992, EXPOSURE 0.713, ROTATION 0.933, OCCLUSION 0.983
- ECE pre/post calibration 0.095 (T = 1.0)
- Stratified: chest X-ray acc 0.924 / AUROC 0.974; retina acc 0.825 / AUROC 0.881
- Classical baselines for the same targets: F1 0.360 (blur), 0.318 (exposure)

Decision configuration (mode / tau) is committed under `metrics.json → val.decision`.

## Ethical considerations & fairness

- Synthetic degradations do not capture every real acquisition artifact; performance on a
  new site must be re-measured before deployment.
- Retina subset is small (n = 120) — its AUROC 0.881 carries wider uncertainty.
- The gate can reject a diagnostically usable image (false reject). The UI therefore
  always shows reason codes + confidence so a human can override; never auto-discard.

## Carets / maintenance

- Re-run `notebooks/train.ipynb` to reproduce weights and regenerate `metrics.json`.
- `tests/test_metrics_config.py` enforces the reported latency budget and metric ranges in CI.
