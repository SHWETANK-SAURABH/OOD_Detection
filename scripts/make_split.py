"""Stratified 45,000 / 5,000 train/val split of the Kaggle CIFAR-10 train set (fixed seed)."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import CLASSES
from src.utils import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/data.yaml")
df = pd.read_csv(os.path.join(cfg["kaggle_dir"], "trainLabels.csv")).sort_values("id")
labels = df["label"].map(CLASSES.index).to_numpy()
idx = np.arange(len(df))  # position i <-> image (i+1).png
train_idx, val_idx = train_test_split(idx, test_size=cfg["val_size"], stratify=labels,
                                      random_state=cfg["seed"])
train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

os.makedirs(cfg["splits_dir"], exist_ok=True)
np.save(os.path.join(cfg["splits_dir"], "train_idx.npy"), train_idx)
np.save(os.path.join(cfg["splits_dir"], "val_idx.npy"), val_idx)
assert len(np.intersect1d(train_idx, val_idx)) == 0
print(f"seed={cfg['seed']} train={len(train_idx)} val={len(val_idx)}")
print("train per class:", np.bincount(labels[train_idx], minlength=10).tolist())
print("val per class:  ", np.bincount(labels[val_idx], minlength=10).tolist())

# Kaggle-only hold-out (stage 9 ablation): split a stratified 5,000-image test set off the 45k train
# split -> 40,000 train / 5,000 val (unchanged) / 5,000 test, all from the Kaggle images.
tr40, te5 = train_test_split(train_idx, test_size=cfg["val_size"], stratify=labels[train_idx],
                             random_state=cfg["seed"])
tr40, te5 = np.sort(tr40), np.sort(te5)
np.save(os.path.join(cfg["splits_dir"], "train40k_idx.npy"), tr40)
np.save(os.path.join(cfg["splits_dir"], "test_kaggle5k_idx.npy"), te5)
print(f"hold-out: train40k={len(tr40)} val={len(val_idx)} test_kaggle5k={len(te5)}")
print("test_kaggle5k per class:", np.bincount(labels[te5], minlength=10).tolist())
