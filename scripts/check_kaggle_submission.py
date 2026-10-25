"""Score a Kaggle submission on the Kaggle test images whose pixels match a labeled torchvision
CIFAR-10 test image (the remaining Kaggle test images are unlabeled filler)."""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import CLASSES
from src.utils import load_config


def _hash(path):
    return hashlib.sha1(np.ascontiguousarray(np.array(Image.open(path).convert("RGB"))).tobytes()).hexdigest()


def main():
    cfg = load_config("configs/data.yaml")
    sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_submission.csv")
    img_dir = os.path.join(cfg["kaggle_dir"], "test")
    if os.path.isdir(os.path.join(img_dir, "test")):
        img_dir = os.path.join(img_dir, "test")

    tv = torchvision.datasets.CIFAR10(cfg["torchvision_root"], train=False)
    tv_label = {hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest(): y for x, y in zip(tv.data, tv.targets)}

    paths = [os.path.join(img_dir, f"{i}.png") for i in sub["id"]]
    with ProcessPoolExecutor(4) as ex:
        hashes = list(ex.map(_hash, paths, chunksize=2000))

    truth = np.array([tv_label.get(h, -1) for h in hashes])
    found = truth >= 0
    pred = sub["label"].map(CLASSES.index).to_numpy()
    print(f"Kaggle test images: {len(sub)}  matched to torchvision test: {found.sum()}")
    print(f"accuracy on matched images: {np.mean(pred[found] == truth[found]):.4f}")
    # some torchvision test images occur several times in the Kaggle test set; also score each once
    matched = np.array(hashes)[found]
    _, first = np.unique(matched, return_index=True)
    print(f"distinct torchvision test images among matches: {len(first)}")
    print(f"accuracy counting each distinct image once: {np.mean(pred[found][first] == truth[found][first]):.4f}")


if __name__ == "__main__":
    main()
