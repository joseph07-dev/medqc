"""MedQC: calibrated real-time quality gate for non-diagnostic medical images."""

from medqc.degradations import ATTRS, GATE_RULE, apply_blur, apply_exposure, apply_occlusion, apply_rotation
from medqc.gate import decide, decide_rule

__version__ = "1.0.0"

__all__ = [
    "ATTRS",
    "GATE_RULE",
    "__version__",
    "apply_blur",
    "apply_exposure",
    "apply_occlusion",
    "apply_rotation",
    "decide",
    "decide_rule",
]
