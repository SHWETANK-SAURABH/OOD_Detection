"""Hash every Kaggle train PNG (decoded pixels) and every torchvision CIFAR-10 train image; compare sets.
Also checks that matched images carry the same label, which validates the alphabetical class mapping."""
import hashlib
import os
import sys

import numpy as np
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import load_kaggle_arrays
from src.utils import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/data.yaml")
h = lambda a: hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()

k_imgs, k_labels = load_kaggle_arrays(cfg)
tv = torchvision.datasets.CIFAR10(cfg["torchvision_root"], train=True, download=False)
k_map = {h(x): y for x, y in zip(k_imgs, k_labels)}
t_map = {h(x): y for x, y in zip(tv.data, tv.targets)}

k_set, t_set = set(k_map), set(t_map)
common = k_set & t_set
print(f"kaggle images: {len(k_imgs)}  unique hashes: {len(k_set)}")
print(f"torchvision train images: {len(tv.data)}  unique hashes: {len(t_set)}")
print(f"common: {len(common)}  only-kaggle: {len(k_set - t_set)}  only-torchvision: {len(t_set - k_set)}")
print("hash sets match:", k_set == t_set)
mism = sum(k_map[c] != t_map[c] for c in common)
print(f"label mismatches among common images: {mism}")
