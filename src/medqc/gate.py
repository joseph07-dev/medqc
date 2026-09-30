"""Accept/reject decision logic for the MedQC quality gate.

Two decision modes:

* ``rule``  – deterministic rule over predicted per-attribute severities
  (``GATE_RULE``): reject if any attribute is severe or >= 2 are mild.
* ``head``  – calibrated gate-head probability thresholded at ``tau``;
  reason codes are still derived from the predicted severities.

Both modes always return machine-readable reason codes on rejection, which the
API/UI surface verbatim (e.g. ``BLUR_SEVERE``).
"""

from __future__ import annotations

from collections.abc import Sequence

from medqc.degradations import ATTRS

DECISION_MODES = ("rule", "head")
DEFAULT_TAU = 0.5
FALLBACK_REASON = "GATE_THRESHOLD"
MULTI_MILD_REASON = "MULTI_MILD"


def decide_rule(severities: Sequence[int]) -> tuple[bool, list[str]]:
    """Apply :data:`GATE_RULE` to predicted severities.

    Returns ``(accepted, reasons)``. ``reasons`` is empty iff accepted.
    """
    reasons: list[str] = []
    n_mild = 0
    for name, level in zip(ATTRS, severities, strict=False):
        level = int(level)
        if level == 2:
            reasons.append(f"{name}_SEVERE")
        elif level == 1:
            n_mild += 1
    if not reasons and n_mild >= 2:
        reasons.append(MULTI_MILD_REASON)
    return (not reasons), reasons


def decide(
    reject_prob: float,
    severities: Sequence[int],
    mode: str = "rule",
    tau: float | None = DEFAULT_TAU,
) -> tuple[bool, list[str]]:
    """Final gate decision with reason codes.

    Parameters
    ----------
    reject_prob:
        Calibrated probability that the image is non-diagnostic.
    severities:
        Predicted severity per attribute (0/1/2).
    mode:
        ``"rule"`` or ``"head"`` (see module docstring).
    tau:
        Rejection threshold for ``head`` mode.

    Returns ``(accepted, reasons)`` with the invariant ``accepted == (not reasons)``.
    """
    if mode not in DECISION_MODES:
        raise ValueError(f"mode must be one of {DECISION_MODES}, got {mode!r}")

    if mode == "rule":
        return decide_rule(severities)

    threshold = DEFAULT_TAU if tau is None else float(tau)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"tau must be within [0, 1], got {threshold!r}")

    accepted = float(reject_prob) < threshold
    if accepted:
        return True, []
    _, reasons = decide_rule(severities)
    return False, reasons or [FALLBACK_REASON]
