"""Score, fuse and evaluate one trained run; or aggregate several runs (seeds) into mean +- std.

Usage:
  python -m src.evaluate --features results/features/resnet18_seed0 [--fusion configs/fusion.yaml]
  python -m src.evaluate --aggregate results/tables/runs/resnet18_seed*__fusion.csv --out results/tables/summary.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src import fusion
from src.metrics import auroc, fpr_at_95_tpr
from src.scores import compute_scores
from src.utils import load_config


def method_scores(scores, cfg):
    """{method_name: {set_name: fused/single score array}} for every set in `scores`."""
    names = list(cfg["scorers"]) + (["clip"] if cfg.get("use_clip") else [])
    val = scores["val"]
    params = fusion.zscore_params(val, names)
    z = {s: fusion.zmatrix(scores[s], params, names) for s in scores}
    out = {n: {s: scores[s][n] for s in scores} for n in names}  # each scorer alone

    w_eq = fusion.equal_weights(len(names))
    out["fused_equal"] = {s: z[s] @ w_eq for s in scores}
    w_tuned = fusion.tune_weights(z["val"], z[cfg["tune_set"]])
    out["fused_tuned"] = {s: z[s] @ w_tuned for s in scores}
    out["fused_rank"] = {s: fusion.rank_matrix(scores[s], val, names) @ w_eq for s in scores}
    for i, n in enumerate(names):  # leave-one-scorer-out, equal weights
        keep = [j for j in range(len(names)) if j != i]
        out[f"loo_equal_-{n}"] = {s: z[s][:, keep] @ fusion.equal_weights(len(keep)) for s in scores}
    weights = {k: {n: round(float(x), 4) for n, x in zip(names, w)} for k, w in [("equal", w_eq), ("tuned", w_tuned)]}
    return out, weights


def evaluate_run(feat_dir, cfg, dcfg):
    ood = dcfg["near_ood"] + dcfg["far_ood"] + [s for s in dcfg["optional_ood"]
                                                 if os.path.exists(os.path.join(feat_dir, f"{s}.npz"))]
    sets = ["val", "test", cfg["tune_set"]] + ood
    scores = compute_scores(feat_dir, sets, cfg)
    methods, weights = method_scores(scores, cfg)
    seed = int(np.load(os.path.join(feat_dir, "fc.npz"))["seed"])

    rows = []
    for m, sc in methods.items():
        for s in ood:
            rows.append({"method": m, "ood_set": s, "group": "near" if s in dcfg["near_ood"] else
                         "far" if s in dcfg["far_ood"] else "optional",
                         "auroc": auroc(sc["test"], sc[s]), "fpr95": fpr_at_95_tpr(sc["test"], sc[s])})
    df = pd.DataFrame(rows)
    # near / far averages (optional sets such as MNIST are reported but not averaged)
    avg = df[df.group.isin(["near", "far"])].groupby(["method", "group"], as_index=False)[["auroc", "fpr95"]].mean()
    avg["ood_set"] = avg["group"] + "_avg"
    df = pd.concat([df, avg], ignore_index=True)
    df["seed"], df["run"] = seed, os.path.basename(feat_dir.rstrip("/\\"))
    val_means = {m: float(np.mean(sc["val"])) for m, sc in methods.items() if m.startswith("fused")}
    return df, weights, val_means


def aggregate(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = df.groupby(["method", "ood_set"])[["auroc", "fpr95"]]
    out = g.agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features")
    p.add_argument("--fusion", default="configs/fusion.yaml")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--aggregate", nargs="+")
    p.add_argument("--out")
    a = p.parse_args()

    if a.aggregate:
        files = sorted({f for pat in a.aggregate for f in glob.glob(pat)})
        out = aggregate(files)
        out.to_csv(a.out, index=False)
        print(f"aggregated {len(files)} runs -> {a.out}")
        return

    cfg, dcfg = load_config(a.fusion), load_config(a.data)
    df, weights, val_means = evaluate_run(a.features, cfg, dcfg)
    os.makedirs("results/tables/runs", exist_ok=True)
    name = os.path.basename(a.features.rstrip("/\\"))
    tag = os.path.splitext(os.path.basename(a.fusion))[0]  # one result file per (run, fusion config)
    out = a.out or f"results/tables/runs/{name}__{tag}.csv"
    df.to_csv(out, index=False)
    with open(out.replace(".csv", "_meta.json"), "w") as f:
        json.dump({"seed": int(df.seed.iloc[0]), "fusion_config_file": a.fusion, "fusion_config": cfg, "data_config": dcfg,
                   "weights": weights, "fused_mean_on_id_val": val_means}, f, indent=2)
    pd.set_option("display.width", 200)
    for metric in ("auroc", "fpr95"):
        print(f"--- {metric}")
        print(df.pivot_table(index="method", columns="ood_set", values=metric).round(4).to_string())
    print("fused score mean on ID val:", {k: round(v, 4) for k, v in val_means.items()})
    print("weights:", weights)
    print("wrote", out)


if __name__ == "__main__":
    main()
