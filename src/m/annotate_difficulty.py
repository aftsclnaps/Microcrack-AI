"""Difficulty annotation workflow for a representative subset of TEST images.

1) Make a CSV template (stratified sample, ground truth is pre-filled from the dataset label, category left empty):
     python -m src.data.annotate_difficulty template --n 400 --out data/annotations/test_annotations.csv
2) Annotate in the browser (saves to the same CSV after every image):
     streamlit run src/data/annotate_difficulty.py -- --csv data/annotations/test_annotations.csv
3) Validate (and compare two independent annotators for ambiguous cases):
     python -m src.data.annotate_difficulty validate --csv data/annotations/test_annotations.csv
     python -m src.data.annotate_difficulty agree --a annotatorA.csv --b annotatorB.csv

Definitions (judge by the IMAGE and its TRUE label, never by a contrast score):
  easy_positive : crack clearly visible at normal viewing size
  hard_positive : crack faint / thin / low-contrast, easy to miss at a glance
  hard_negative : NO crack but shadows, stains, joints, pores or textures that could be mistaken for one
  easy_negative : plain crack-free concrete
  uncertain     : annotator cannot decide (excluded from metrics; send to second reviewer)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

CATS = {1: ["easy_positive", "hard_positive", "uncertain"], 0: ["easy_negative", "hard_negative", "uncertain"]}
COLS = ["image_path", "ground_truth", "visibility_category", "difficulty", "annotation_notes"]
DIFF = {"easy_positive": "easy", "easy_negative": "easy", "hard_positive": "hard", "hard_negative": "hard", "uncertain": "uncertain"}


def make_template(split_csv, n, out, seed=42):
    df = pd.read_csv(split_csv)
    per = n // 2
    s = pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in df.groupby("label")]).sample(frac=1, random_state=seed)
    t = pd.DataFrame({"image_path": s.rel_path, "ground_truth": s.label.map({1: "Crack", 0: "No Crack"}),
                      "visibility_category": "", "difficulty": "", "annotation_notes": ""})
    Path(out).parent.mkdir(parents=True, exist_ok=True); t[COLS].to_csv(out, index=False)
    print(f"Wrote {len(t)} rows to {out}")


def validate(csv):
    d = pd.read_csv(csv).fillna("")
    done = d[d.visibility_category != ""]
    bad = done[[(r.ground_truth == "Crack") != r.visibility_category.endswith("positive") and r.visibility_category != "uncertain"
                for r in done.itertuples()]]
    print(f"{len(done)}/{len(d)} annotated | counts: {done.visibility_category.value_counts().to_dict()}")
    print("Category/label mismatches:", len(bad)); [print("  ", r.image_path) for r in bad.itertuples()]


def agree(a, b):
    from sklearn.metrics import cohen_kappa_score
    da, db = pd.read_csv(a).fillna(""), pd.read_csv(b).fillna("")
    m = da.merge(db, on="image_path", suffixes=("_a", "_b")); m = m[(m.visibility_category_a != "") & (m.visibility_category_b != "")]
    print(f"{len(m)} jointly annotated | Cohen's kappa = {cohen_kappa_score(m.visibility_category_a, m.visibility_category_b):.3f}")
    dis = m[m.visibility_category_a != m.visibility_category_b]; print(f"{len(dis)} disagreements -> adjudicate and set a final category")
    dis[["image_path", "visibility_category_a", "visibility_category_b"]].to_csv("annotation_disagreements.csv", index=False)


def run_app(csv, root):
    import streamlit as st
    from PIL import Image
    st.set_page_config(page_title="MicroCrack annotator", layout="wide")
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(csv).fillna("")
    d = st.session_state.df
    todo = d.index[d.visibility_category == ""].tolist()
    st.progress((len(d) - len(todo)) / len(d), text=f"{len(d) - len(todo)} / {len(d)} annotated")
    if not todo:
        st.success("All images annotated."); return
    i = todo[0]; row = d.loc[i]
    st.image(Image.open(Path(root) / row.image_path), width=454, caption=f"Ground truth (from dataset): {row.ground_truth}")
    opts = CATS[1 if row.ground_truth == "Crack" else 0]
    cat = st.radio("Visibility category", opts, horizontal=True); note = st.text_input("Notes (optional)")
    if st.button("Save and next"):
        d.loc[i, ["visibility_category", "difficulty", "annotation_notes"]] = [cat, DIFF[cat], note]
        d.to_csv(csv, index=False); st.rerun()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"template", "validate", "agree"}:
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--n", type=int, default=400)
        ap.add_argument("--out", default="data/annotations/test_annotations.csv"); ap.add_argument("--split-csv", default="data/splits/test.csv")
        ap.add_argument("--csv", default="data/annotations/test_annotations.csv"); ap.add_argument("--a"); ap.add_argument("--b")
        a = ap.parse_args()
        {"template": lambda: make_template(a.split_csv, a.n, a.out), "validate": lambda: validate(a.csv), "agree": lambda: agree(a.a, a.b)}[a.cmd]()
    else:  # launched by streamlit
        ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="data/annotations/test_annotations.csv")
        ap.add_argument("--root", default="data/raw"); a, _ = ap.parse_known_args(); run_app(a.csv, a.root)
