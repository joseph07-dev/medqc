"""MedQC — Clinical Dashboard (Streamlit).

Three tabs:

1. **Upload & Gate Check** — upload first, or pick a bundled sample, then see the
   big ACCEPT/REJECT badge, four severity bars, reason chips, and live latency.
2. **Live Degradation Test** — apply severity-0/1/2 degradations in real time
   and watch the gate react.
3. **Silent Failure Demo** — side-by-side clean/degraded pairs with downstream
   classifier verdicts, showing what a passing-but-wrong image costs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medqc.degradations import (  # noqa: E402
    ATTRS,
    SEVERITY_LEVELS,
    apply_blur,
    apply_exposure,
    apply_occlusion,
    apply_rotation,
)
from medqc.predict import GatePredictor, GateResult  # noqa: E402

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = ("jpg", "jpeg", "png", "bmp", "webp")
SEEDS = dict(zip(ATTRS, (101, 102, 103, 104), strict=True))
APPLIERS = dict(zip(ATTRS, (apply_blur, apply_exposure, apply_rotation, apply_occlusion), strict=True))

st.set_page_config(
    page_title="MedQC · Clinical Dashboard",
    page_icon="🛟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; }
      .badge { border-radius: 14px; padding: 18px 26px; text-align: center;
               font-size: 30px; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px; }
      .badge-accept { background: #06371f; color: #4ade80; border: 2px solid #16a34a; }
      .badge-reject { background: #3f1215; color: #f87171; border: 2px solid #dc2626; }
      .badge-sub { text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 14px; }
      .chip { display: inline-block; background: #7f1d1d; color: #fecaca; border: 1px solid #ef4444;
              border-radius: 999px; padding: 3px 12px; margin: 3px 4px 0 0; font-size: 13px;
              font-family: ui-monospace, monospace; }
      .chip-ok { background: #052e16; color: #bbf7d0; border-color: #22c55e; }
      .kv { color: #cbd5e1; font-size: 14px; margin: 2px 0; }
      .kv b { color: #f8fafc; }
      h1, h2, h3 { letter-spacing: 0.2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MedQC gate (int8 ONNX)…")
def get_predictor() -> GatePredictor:
    return GatePredictor(ROOT / "weights" / "medqc_v1.onnx", ROOT / "metrics.json")


@st.cache_data(show_spinner=False)
def load_pairs() -> list[dict[str, Any]]:
    path = ROOT / "downstream_results.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("pairs", [])


@st.cache_data(show_spinner=False)
def bundled_samples() -> list[Path]:
    return sorted((ROOT / "assets" / "samples").glob("*.png"))


@st.cache_data(show_spinner=False)
def load_image(source: str | Path | bytes) -> Image.Image:
    if isinstance(source, bytes):
        img = Image.open(source)
    else:
        img = Image.open(source)
    return img.convert("RGB")


def badge(result: GateResult) -> None:
    if result.decision == "ACCEPT":
        st.markdown('<div class="badge badge-accept">✔ ACCEPT — DIAGNOSTIC-GRADE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge badge-reject">✖ REJECT — NON-DIAGNOSTIC</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="badge-sub">confidence {result.confidence:.1%} · p(reject) {result.reject_prob:.3f}'
        f' · mode <code>{result.mode}</code> · <b>{result.latency_ms:.1f} ms</b></div>',
        unsafe_allow_html=True,
    )
    chips = "".join(f'<span class="chip">{r}</span>' for r in result.reasons)
    if not chips:
        chips = '<span class="chip chip-ok">NO_DEFECTS_FLAGGED</span>'
    st.markdown(chips, unsafe_allow_html=True)


def severity_bars(result: GateResult) -> None:
    st.caption("Predicted severity per quality attribute")
    for name in ATTRS:
        level = result.severities[name]
        frac = {"OK": 1.0, "MILD": 0.66, "SEVERE": 0.33}[level]
        st.progress(frac, text=f"{name}: {level}")


def classical_row(result: GateResult) -> None:
    c = result.classical
    f1 = c.get("reported_f1") or {}
    st.caption("Classical baseline head-to-head (Laplacian sharpness + clipped-pixel exposure)")
    st.markdown(
        f'<div class="kv">classical verdict: <b>{"ACCEPT" if c.get("accepted") else "REJECT"}</b>'
        f' · Laplacian var {c.get("sharpness")}'
        f' · clipped {c.get("clipped_frac")}'
        f' · reported F1 blur {f1.get("blur")} / exposure {f1.get("exposure")}</div>',
        unsafe_allow_html=True,
    )


def result_panel(image: Image.Image, result: GateResult, title: str) -> None:
    st.subheader(title)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(image, width='stretch', caption="input image")
    with right:
        badge(result)
        severity_bars(result)
        st.divider()
        st.markdown(
            f'<div class="kv">latency: <b>{result.latency_ms:.1f} ms</b> · input: {result.input_size}px'
            f' · measured {result.measured_at}</div>',
            unsafe_allow_html=True,
        )
        classical_row(result)


def run_gate(image: Image.Image) -> GateResult:
    return get_predictor().predict(image)


def tab_upload() -> None:
    st.markdown("Upload a chest X-ray or retinal fundus image — **upload first**, quality gate runs instantly.")
    uploaded = st.file_uploader("Medical image", type=list(ALLOWED_TYPES), label_visibility="collapsed")
    image: Image.Image | None = None
    if uploaded is not None:
        raw = uploaded.getvalue()
        if len(raw) > MAX_UPLOAD_BYTES:
            st.error("File exceeds 5 MB limit.")
            return
        try:
            image = load_image(raw)
        except Exception:
            st.error("Could not decode that file as an image.")
            return

    samples = bundled_samples()
    if image is None and samples:
        st.caption("…or check a bundled sample:")
        cols = st.columns(min(4, len(samples)))
        for i, path in enumerate(samples):
            with cols[i % len(cols)]:
                if st.button(path.stem, width='stretch', key=f"sample_{path.stem}"):
                    st.session_state["sample_choice"] = str(path)
                    st.rerun()
        choice = st.session_state.get("sample_choice")
        if choice:
            image = load_image(choice)

    if image is None:
        st.info("Awaiting an image. Every decision below includes machine-readable reason codes.")
        return
    result_panel(image, run_gate(image), "Gate decision")


def tab_degrade() -> None:
    st.markdown("Set per-attribute severity and watch the gate react in real time (deterministic seeds).")
    samples = bundled_samples()
    if not samples:
        st.warning("No bundled sample images available.")
        return
    choice = st.selectbox("Base image", [p.stem for p in samples], format_func=lambda s: s.replace("pair_", ""))
    base = load_image(samples[[p.stem for p in samples].index(choice)])
    st.image(base, width=260, caption=f"{choice} (clean)")

    cols = st.columns(4)
    sev: dict[str, str] = {}
    for col, name in zip(cols, ATTRS, strict=True):
        with col:
            sev[name] = st.select_slider(f"{name}", options=list(SEVERITY_LEVELS), value="OK", key=f"sev_{name}")

    arr = np.asarray(base)
    applied = arr
    for name in ATTRS:
        level_idx = SEVERITY_LEVELS.index(sev[name])
        applied = APPLIERS[name](applied, level_idx, SEEDS[name])

    summary = " · ".join(f"{n}={s}" for n, s in sev.items())
    st.caption(f"Applied: {summary}")
    result_panel(Image.fromarray(applied), run_gate(Image.fromarray(applied)), "Degraded gate decision")


def tab_silent() -> None:
    st.markdown(
        "A degraded image can stay **above every classical sharpness/exposure threshold** and still be "
        "non-diagnostic — downstream classifiers then fail silently. MedQC rejects it with reason codes."
    )
    pairs = load_pairs()
    if not pairs:
        st.info("No demo pairs bundled with this build.")
        return
    for i, pair in enumerate(pairs):
        st.divider()
        st.markdown(f"### Pair {i + 1} · {pair.get('modality', '?')}")
        left, right = st.columns(2, gap="large")
        with left:
            st.image(load_image(ROOT / pair["clean_png"]), width='stretch', caption="clean")
            st.image(load_image(ROOT / pair["deg_png"]), width='stretch', caption="degraded")
        with right:
            probe = pair.get("probe")
            if probe and pair.get("clean_pred") is not None:
                lab = {0: "NORMAL", 1: "ABNORMAL"}
                st.markdown(
                    f'<div class="kv">classifier (clean): <b>{lab.get(pair["clean_pred"], pair["clean_pred"])}'
                    f' · {pair["clean_conf"]:.0%}</b></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="kv">classifier (degraded): <b>{lab.get(pair["deg_pred"], pair["deg_pred"])}'
                    f' · {pair["deg_conf"]:.0%}</b></div>',
                    unsafe_allow_html=True,
                )
                if pair.get("y_true") is not None:
                    st.markdown(
                        f'<div class="kv">ground truth: <b>{lab.get(pair["y_true"], pair["y_true"])}</b></div>',
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f'<div class="kv">MedQC clean: <b style="color:#4ade80">{pair["gate_clean"]}</b></div>'
                f'<div class="kv">MedQC degraded: <b style="color:#f87171">{pair["gate_deg"]}</b>'
                f' &nbsp; {" ".join(f"<code>{r}</code>" for r in pair.get("reasons", []))}</div>',
                unsafe_allow_html=True,
            )
            live = run_gate(load_image(ROOT / pair["deg_png"]))
            st.caption(f"live re-run: {live.decision} in {live.latency_ms:.1f} ms ({live.mode} mode)")


def sidebar() -> None:
    with st.sidebar:
        st.title("🛟 MedQC")
        st.caption("Calibrated real-time quality gate for non-diagnostic medical images.")
        try:
            summary = get_predictor().summary
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Model failed to load: {exc}")
            return
        st.divider()
        st.markdown("**Model**")
        model = summary.get("model") or {}
        st.markdown(
            f'<div class="kv">arch: <b>{model.get("arch", "—")}</b></div>'
            f'<div class="kv">params: <b>{model.get("params", 0):,}</b></div>',
            unsafe_allow_html=True,
        )
        lat = summary.get("latency_ms") or {}
        st.markdown(
            f'<div class="kv">latency p50/p95: <b>{lat.get("p50", "—")}/{lat.get("p95", "—")} ms</b>'
            f' ({lat.get("provider", "")})</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Validation**")
        val_lat = summary.get("gate_acc")
        strat = summary.get("stratified") or {}
        rows = "| metric | value |\n|---|---|"
        rows += f"\n| gate accuracy | {val_lat} |"
        for name, block in strat.items():
            rows += f"\n| {name} AUROC | {block.get('gate_auroc')} |"
        st.markdown(rows)
        st.markdown(f'<div class="kv">decision mode: <b>{summary.get("mode")}</b></div>', unsafe_allow_html=True)
        st.caption(summary.get("gate_rule") or "")
        st.divider()
        st.markdown(
            "**Research prototype — not a medical device.** Decisions are advisory and must be "
            "reviewed by a qualified clinician."
        )


def main() -> None:
    sidebar()
    st.title("MedQC · Clinical Dashboard")
    st.caption(
        "Deep quality gate (MobileNetV3-Small, int8 ONNX) that rejects non-diagnostic medical images "
        "with machine-readable reason codes — before they reach a downstream classifier or radiologist workflow."
    )
    tabs = st.tabs(["1 · Upload & Gate Check", "2 · Live Degradation Test", "3 · Silent Failure Demo"])
    with tabs[0]:
        tab_upload()
    with tabs[1]:
        tab_degrade()
    with tabs[2]:
        tab_silent()


main()
