"""Inspect the dataset, validate files, find duplicates, and write leakage-aware split manifests.

Usage:  python -m src.data.prepare_data --config configs/config.yaml

IMPORTANT LIMITATION: the public dataset has no source-image identifiers (458 large images were cut into
~40k 227x227 patches). Patches from one source image can therefore land in different splits. This script
removes exact/near duplicates across splits and (optionally) groups visually similar patches via
`data.pseudo_source_clusters`, which only APPROXIMATES source grouping. Do not claim source-grouped evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.data.common import CLASS_NAMES, discover_images
from src.utils import load_config, log, save_json, set_seed, setup_logging


def analyse_image(path: Path):
    """Return (ok, w, h, md5, dhash, thumb_or_error) for one file."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
            bits = (g[:, 1:] > g[:, :-1]).flatten()
            dhash = int("".join("1" if b else "0" for b in bits), 2)
            thumb = np.asarray(im.resize((4, 4), Image.BILINEAR), dtype=np.float32).flatten()
        return True, w, h, hashlib.md5(path.read_bytes()).hexdigest(), dhash, thumb
    except Exception as e:  # noqa: BLE001 - catalogue every failure
        return False, 0, 0, "", 0, str(e)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def near_duplicate_pairs(hashes, max_dist: int, bucket_cap: int = 3000):
    """Multi-index hashing: split 64 bits into (max_dist+1) chunks; hashes within max_dist share one chunk exactly."""
    n_chunks = max_dist + 1
    bounds = np.linspace(0, 64, n_chunks + 1).astype(int)
    pairs = set()
    for c in range(n_chunks):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        buckets: dict[int, list[int]] = {}
        for i, h in enumerate(hashes):
            buckets.setdefault((int(h) >> lo) & ((1 << (hi - lo)) - 1), []).append(i)
        for idxs in buckets.values():
            if len(idxs) < 2 or len(idxs) > bucket_cap:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= max_dist:
                        pairs.add((i, j))
    return pairs


def make_sample_figure(df, root, out_path, n=8, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for r, label in enumerate([1, 0]):
        sub = df[df.label == label]
        for c, idx in enumerate(rng.choice(len(sub), size=min(n, len(sub)), replace=False)):
            axes[r, c].imshow(Image.open(Path(root) / sub.iloc[idx].rel_path))
            axes[r, c].axis("off")
        axes[r, 0].set_title(CLASS_NAMES[label], loc="left", fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.max_per_class=500")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    seed, root, d = cfg["seed"], cfg["paths"]["dataset_root"], cfg["data"]
    set_seed(seed)

    df = discover_images(root)
    log.info("Found %d image files: %s", len(df), dict(Counter(df.label)))
    if d["max_per_class"]:
        df = pd.concat([g.sample(min(len(g), d["max_per_class"]), random_state=seed) for _, g in df.groupby("label")])
        df = df.reset_index(drop=True)
        log.info("Subsampled to %d images for a quick run", len(df))

    recs, corrupt, thumbs = [], [], []
    for rp in tqdm(df.rel_path, desc="validating"):
        ok, w, h, md5, dh, extra = analyse_image(Path(root) / rp)
        if ok:
            recs.append((rp, w, h, md5, dh))
            thumbs.append(extra)
        else:
            corrupt.append({"rel_path": rp, "error": extra})
    info = pd.DataFrame(recs, columns=["rel_path", "width", "height", "md5", "dhash"])
    df = df.merge(info, on="rel_path", how="inner").reset_index(drop=True)
    thumbs = np.stack(thumbs)
    log.info("Readable: %d | corrupt: %d | sizes: %s", len(df), len(corrupt), dict(Counter(zip(df.width, df.height))))

    uf = UnionFind(len(df))
    first_seen: dict[str, int] = {}
    n_exact = 0
    for i, m in enumerate(df.md5):
        if m in first_seen:
            uf.union(i, first_seen[m])
            n_exact += 1
        else:
            first_seen[m] = i
    pairs = near_duplicate_pairs(df.dhash.values, d["near_dup_hamming"])
    for i, j in pairs:
        uf.union(i, j)
    cross_label = sum(1 for i, j in pairs if df.label[i] != df.label[j])
    log.info("Exact duplicates: %d | near-duplicate pairs (Hamming<=%d): %d (%d cross-label)",
             n_exact, d["near_dup_hamming"], len(pairs), cross_label)

    grouping = "exact + near-duplicate components"
    if d["pseudo_source_clusters"] > 0:
        from sklearn.cluster import MiniBatchKMeans
        k = d["pseudo_source_clusters"]
        z = (thumbs - thumbs.mean(0)) / (thumbs.std(0) + 1e-6)
        cl = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(z)
        first: dict[int, int] = {}
        for i, c in enumerate(cl):
            if c in first:
                uf.union(i, first[c])
            else:
                first[c] = i
        grouping += f" + {k} appearance clusters (APPROXIMATE source proxy)"
    df["group"] = [uf.find(i) for i in range(len(df))]
    log.info("Groups: %d for %d images (%s)", df.group.nunique(), len(df), grouping)

    n_folds = d["n_folds"]
    folds = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, idx) in enumerate(sgkf.split(df, df.label, df.group)):
        folds[idx] = f
    n_test, n_val = round(n_folds * d["test_frac"]), round(n_folds * d["val_frac"])
    df["split"] = np.where(folds < n_test, "test", np.where(folds < n_test + n_val, "val", "train"))
    out = Path(cfg["paths"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        df[df.split == s][["rel_path", "label", "group", "md5"]].to_csv(out / f"{s}.csv", index=False)
    g = {s: set(df[df.split == s].group) for s in ["train", "val", "test"]}
    assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]), "group leakage across splits!"

    summary = {
        "seed": seed, "grouping": grouping, "n_images": len(df), "n_corrupt": len(corrupt), "corrupt": corrupt,
        "exact_duplicates": n_exact, "near_duplicate_pairs": len(pairs), "near_dup_hamming": d["near_dup_hamming"],
        "sizes": {f"{k[0]}x{k[1]}": v for k, v in Counter(zip(df.width, df.height)).items()},
        "splits": {s: {"n": int((df.split == s).sum()), "crack": int(((df.split == s) & (df.label == 1)).sum()),
                       "no_crack": int(((df.split == s) & (df.label == 0)).sum())} for s in ["train", "val", "test"]},
        "limitations": [
            "No source-image identifiers or difficulty/visibility labels exist in the public dataset.",
            "Patches from the same source image may appear in different splits; evaluation is NOT source-grouped.",
            "Duplicate/near-duplicate grouping reduces, but does not eliminate, leakage.",
        ],
    }
    save_json(summary, out / "split_meta.json")
    make_sample_figure(df, root, Path(cfg["paths"]["reports_dir"]) / "figures" / "samples.png", seed=seed)
    log.info("Split sizes: %s", {s: v["n"] for s, v in summary["splits"].items()})
    log.info("Wrote manifests + split_meta.json to %s", out)


if __name__ == "__main__":
    main()
