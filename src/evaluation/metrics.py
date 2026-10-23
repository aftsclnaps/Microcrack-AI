"""Metrics (numpy/sklearn only, no torch)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    out = {
        "n": int(len(y_true)), "accuracy": div(tp + tn, len(y_true)), "precision": prec, "recall": rec,
        "f1": div(2 * prec * rec, prec + rec) if tp else (0.0 if len(y_true) else float("nan")),
        "specificity": div(tn, tn + fp), "fnr": div(fn, tp + fn), "fpr": div(fp, tn + fp),
        "confusion_matrix": [[tn, fp], [fn, tp]], "threshold": float(threshold),
    }
    out["auc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) == 2 else float("nan")
    return out


def category_metrics(df, threshold: float = 0.5) -> dict:
    """df needs columns: label, prob_crack and optionally visibility_category in
    {easy_positive, hard_positive, hard_negative, easy_negative} (annotated subset only)."""
    res = {"all": compute_metrics(df.label, df.prob_crack, threshold)}
    if "visibility_category" not in df.columns:
        return res
    ann = df[df.visibility_category.notna()]
    res["annotated_subset_n"] = int(len(ann))
    for cat in ["easy_positive", "hard_positive", "hard_negative", "easy_negative"]:
        sub = ann[ann.visibility_category == cat]
        if not len(sub):
            res[cat] = {"n": 0}
            continue
        pred = (sub.prob_crack.values >= threshold).astype(int)
        if cat.endswith("positive"):
            res[cat] = {"n": int(len(sub)), "recall": float(pred.mean()), "fnr": float(1 - pred.mean())}
        else:
            res[cat] = {"n": int(len(sub)), "false_positive_rate": float(pred.mean()), "specificity": float(1 - pred.mean())}
    return res


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    conf, pred = np.maximum(y_prob, 1 - y_prob), (y_prob >= 0.5).astype(int)
    correct, bins, ece = (pred == y_true).astype(float), np.linspace(0.5, 1.0, n_bins + 1), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def pick_threshold_for_recall(y_true, y_prob, recall_target: float = 0.99) -> float:
    """Highest threshold whose validation recall still meets the target (keeps false positives as low as possible)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    pos = np.sort(y_prob[y_true == 1])
    if not len(pos):
        return 0.5
    k = int(np.floor((1 - recall_target) * len(pos)))
    return float(min(pos[k], 0.5) if k < len(pos) else 0.5)


def choose_review_confidence(y_true, y_prob, threshold: float = 0.5, accuracy_target: float = 0.99) -> dict:
    """Smallest confidence cut-off tau such that predictions with confidence >= tau reach `accuracy_target`.
    Predictions below tau are flagged for human review."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    conf = np.where(pred == 1, y_prob, 1 - y_prob)
    ok = (pred == y_true)
    best = None
    for tau in np.unique(np.round(conf, 4)):
        keep = conf >= tau
        if keep.sum() and ok[keep].mean() >= accuracy_target:
            best = float(tau)
            break
    if best is None:
        best = 1.01  # target unreachable on validation data: flag everything
    keep = conf >= best
    return {"tau": best, "flagged_fraction": float(1 - keep.mean()),
            "accuracy_on_unflagged": float(ok[keep].mean()) if keep.any() else float("nan"),
            "accuracy_target": accuracy_target}
