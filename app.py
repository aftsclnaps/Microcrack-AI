"""MicroCrack AI demo.   Run:  streamlit run app.py

Loads any checkpoint found in artifacts/runs/*/best.pt. Binary crack / no-crack screening aid only:
it does NOT estimate severity and does NOT determine structural safety.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.explainability.gradcam import grad_cam, overlay
from src.models.model_factory import count_parameters, load_checkpoint, model_size_mb

RUNS = Path("artifacts/runs")
CLASS_NAMES = ["No Crack", "Crack"]

st.set_page_config(page_title="MicroCrack AI", page_icon="🧱", layout="wide")


@st.cache_resource
def get_model(ckpt_path: str):
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    return model, ckpt


def predict(model, ckpt, image: Image.Image):
    t0 = time.perf_counter()
    x = build_transforms("test", ckpt["input_size"])(image.convert("RGB")).unsqueeze(0)
    t1 = time.perf_counter()
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0, 1].item()
    t2 = time.perf_counter()
    return prob, x, (t1 - t0) * 1000, (t2 - t1) * 1000


st.title("🧱 MicroCrack AI")
st.caption("Lightweight concrete-surface crack screening · binary output: Crack / No Crack")
st.warning("Research demo. A prediction here is **not** a structural assessment, says nothing about "
           "crack severity, and must not be used to declare a structure safe. Professional inspection is required.")

checkpoints = sorted(RUNS.glob("*/best.pt")) if RUNS.exists() else []
if not checkpoints:
    st.error("No trained model found. Train one first, e.g.\n\n"
             "`python -m src.training.train --data-root /path/to/dataset`\n\n"
             "This app only displays real model outputs; it has no demo/fake mode.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model (run)", checkpoints, format_func=lambda p: p.parent.name)
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    st.markdown("**Uncertainty review band**")
    low, high = st.slider("Flag if P(crack) is between", 0.0, 1.0, (0.30, 0.70), 0.01)
    st.caption("Default band is an uncalibrated placeholder. Choose it from validation data "
               "(Stage F) before relying on it.")

model, ckpt = get_model(str(choice))
name = ckpt["model_name"]

tab_single, tab_batch, tab_model = st.tabs(["Single image", "Batch report", "Model details"])

with tab_single:
    up = st.file_uploader("Upload a concrete surface image", type=["jpg", "jpeg", "png"], key="single")
    if up:
        image = Image.open(up).convert("RGB")
        prob, x, t_pre, t_fwd = predict(model, ckpt, image)
        pred = int(prob >= 0.5)
        conf = prob if pred else 1 - prob
        flagged = low <= prob <= high
        c1, c2 = st.columns(2)
        c1.image(image, caption="Uploaded image", use_container_width=True)
        if show_cam:
            cam = grad_cam(model, name, x)
            c2.image(overlay(image.resize((ckpt["input_size"],) * 2), cam),
                     caption="Grad-CAM (interpretability aid, not proof of correctness)", use_container_width=True)
        (st.error if pred else st.success)(f"**{CLASS_NAMES[pred]}**  ·  confidence {conf:.1%}")
        m1, m2, m3 = st.columns(3)
        m1.metric("P(crack)", f"{prob:.3f}")
        m2.metric("Forward pass", f"{t_fwd:.1f} ms")
        m3.metric("Preprocessing", f"{t_pre:.1f} ms")
        st.caption("Timings are single uncontrolled measurements on this machine (includes warm-up effects); "
                   "use the benchmark script for proper latency numbers.")
        if flagged:
            st.warning("⚠️ Uncertain: flagged for human review.")

with tab_batch:
    files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="batch")
    if files:
        rows = []
        for f in files:
            prob, _, t_pre, t_fwd = predict(model, ckpt, Image.open(f))
            rows.append({"file": f.name, "prediction": CLASS_NAMES[int(prob >= 0.5)],
                         "p_crack": round(prob, 4), "needs_review": low <= prob <= high,
                         "forward_ms": round(t_fwd, 2), "model_run": choice.parent.name})
        report = pd.DataFrame(rows)
        st.dataframe(report, use_container_width=True)
        st.download_button("Download CSV report", report.to_csv(index=False).encode(),
                           "inspection_report.csv", "text/csv")

with tab_model:
    st.write({"architecture": name, "input_size": ckpt["input_size"],
              "trainable_parameters": count_parameters(model), "serialized_size_MiB": round(model_size_mb(model), 3),
              "best_epoch": ckpt.get("epoch"), "seed": ckpt.get("seed")})
    for fname in ("metrics_val.json", "metrics_test.json"):
        p = choice.parent / fname
        if p.exists():
            st.subheader(fname)
            st.json(json.loads(p.read_text()), expanded=False)
    st.caption("Benchmark and difficulty-category results will appear here once those stages are run.")
