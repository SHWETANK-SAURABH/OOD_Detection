"""Stage 10 report: summary tables (CSV + Excel), ROC plots and a markdown results table.

Usage: python -m src.report
Reads results/tables/runs/*.csv written by src.evaluate; writes results/tables/*.csv, results.xlsx,
results/plots/roc_*.png and results/tables/results.md (pasted into README).
"""
import glob
import os

import numpy as np
import pandas as pd

# keep matplotlib's font cache inside the project instead of the user profile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   ".cache", "mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.evaluate import aggregate, method_scores
from src.scores import compute_scores
from src.utils import load_config

TABLES, PLOTS = "results/tables", "results/plots"
SET_ORDER = ["cifar100", "tin", "near_avg", "svhn", "texture", "places365", "far_avg", "mnist"]


def wide(summary, metric):
    """method x set table with 'mean +- std' strings (std omitted when only one seed)."""
    s = summary.copy()
    fmt = lambda r: (f"{100 * r[metric + '_mean']:.2f} ± {100 * r[metric + '_std']:.2f}"
                     if r["n_seeds"] > 1 else f"{100 * r[metric + '_mean']:.2f}")
    s["cell"] = s.apply(fmt, axis=1)
    t = s.pivot(index="method", columns="ood_set", values="cell")
    return t[[c for c in SET_ORDER if c in t.columns]]


def roc_plots(feat_dir, fusion_cfg, dcfg, methods, tag):
    cfg = load_config(fusion_cfg)
    ood = dcfg["near_ood"] + dcfg["far_ood"]
    scores = compute_scores(feat_dir, ["val", "test", cfg["tune_set"]] + ood, cfg)
    ms, _ = method_scores(scores, cfg)
    os.makedirs(PLOTS, exist_ok=True)
    for s in ood:
        plt.figure(figsize=(4.5, 4.5))
        for m in methods:
            y = np.r_[np.ones(len(ms[m]["test"])), np.zeros(len(ms[m][s]))]
            fpr, tpr, _ = roc_curve(y, np.r_[ms[m]["test"], ms[m][s]])
            plt.plot(fpr, tpr, label=m, lw=1.4)
        plt.plot([0, 1], [0, 1], "k:", lw=0.8)
        plt.xlabel("FPR (OOD accepted)")
        plt.ylabel("TPR (ID accepted)")
        plt.title(f"CIFAR-10 vs {s} ({os.path.basename(feat_dir)}, {tag})", fontsize=9)
        plt.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, f"roc_{tag}_{s}.png"), dpi=150)
        plt.close()


def main():
    dcfg = load_config("configs/data.yaml")
    runs = os.path.join(TABLES, "runs")
    groups = {  # summary name -> run files (one per seed)
        "summary": glob.glob(f"{runs}/resnet18_seed*__fusion.csv"),
        "summary_clip": glob.glob(f"{runs}/resnet18_seed*__fusion_clip.csv"),
        "summary_clip_desc": glob.glob(f"{runs}/resnet18_seed*__fusion_clip_desc.csv"),
        "holdout_40k": glob.glob(f"{runs}/resnet18_holdout_seed*__fusion.csv"),
    }
    sheets = {}
    for name, files in groups.items():
        if not files:
            continue
        summ = aggregate(sorted(files))
        summ.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
        sheets[f"{name}_auroc"] = wide(summ, "auroc")
        sheets[f"{name}_fpr95"] = wide(summ, "fpr95")
        print(f"{name}: {len(files)} run(s)")

    # ID accuracy per training run
    acc = pd.DataFrame([{"run": os.path.basename(os.path.dirname(f)), **pd.read_json(f, typ="series")}
                        for f in sorted(glob.glob("results/checkpoints/resnet18*_seed*/result.json"))
                        if "smoke" not in f])
    acc["test_set"] = acc.run.map(lambda r: "Kaggle hold-out 5k" if "holdout" in r else "torchvision test 10k")
    acc.to_csv(os.path.join(TABLES, "id_accuracy.csv"), index=False)
    sheets["id_accuracy"] = acc.set_index("run")

    with pd.ExcelWriter(os.path.join(TABLES, "results.xlsx")) as xw:
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31])

    # markdown table for the README: main methods, AUROC and FPR95
    main_methods = ["msp", "maxlogit", "energy", "react", "knn", "mahalanobis",
                    "fused_equal", "fused_tuned", "fused_rank"]
    with open(os.path.join(TABLES, "results.md"), "w", encoding="utf-8") as f:
        for key in [k for k in sheets if k.endswith(("_auroc", "_fpr95"))]:
            t = sheets[key]
            t = t.loc[[m for m in t.index if m in main_methods or m == "clip" or m.startswith("loo")]]
            f.write(f"\n#### {key} (%)\n\n{t.to_markdown()}\n")
    print("wrote", os.path.join(TABLES, "results.md"))

    feat = "results/features/resnet18_seed0"
    roc_plots(feat, "configs/fusion.yaml", dcfg,
              ["msp", "energy", "knn", "mahalanobis", "fused_equal", "fused_tuned"], "resnet")
    if os.path.exists("results/features/clip_vitb16/test.npz"):
        roc_plots(feat, "configs/fusion_clip.yaml", dcfg,
                  ["knn", "clip", "fused_equal", "fused_tuned", "fused_rank"], "with_clip")
    print("wrote ROC plots to", PLOTS)


if __name__ == "__main__":
    main()
