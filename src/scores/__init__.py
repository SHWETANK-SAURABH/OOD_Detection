"""Compute every training-free score from the saved .npz features (CPU only)."""
import os

import numpy as np

from src.scores import clip_score, knn, logit, mahalanobis, react

SCORERS = ["msp", "maxlogit", "energy", "react", "knn", "mahalanobis"]


def compute_scores(feat_dir, sets, cfg):
    """Returns {set_name: {scorer: (N,) array}}. Higher = more in-distribution.
    ReAct threshold, KNN index and Mahalanobis statistics are fitted on the ID training features only."""
    load = lambda n: np.load(os.path.join(feat_dir, f"{n}.npz"))
    tr, fc = load("train"), load("fc")
    thr = react.fit(tr["features"], cfg["react_percentile"])
    index = knn.fit(tr["features"])
    means, prec = mahalanobis.fit(tr["features"], tr["labels"])
    out = {}
    for name in sets:
        d = load(name)
        lg, ft = d["logits"], d["features"]
        out[name] = {
            "msp": logit.msp(lg),
            "maxlogit": logit.max_logit(lg),
            "energy": logit.energy(lg),
            "react": react.score(ft, fc["weight"], fc["bias"], thr),
            "knn": knn.score(ft, index, cfg["knn_k"]),
            "mahalanobis": mahalanobis.score(ft, means, prec),
        }
    if cfg.get("use_clip"):  # optional MCM column; CLIP is frozen, so it is the same for every seed
        cdir = cfg["clip_feature_dir"]
        text = np.load(os.path.join(cdir, "text.npz"))["descriptions" if cfg["clip_descriptions"] else "names"]
        for name in sets:
            out[name]["clip"] = clip_score.score(np.load(os.path.join(cdir, f"{name}.npz"))["image_emb"], text)
    return out
