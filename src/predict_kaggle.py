"""Stage 11 (optional): predict labels for the Kaggle test images and write a submission file.

Checks classification only; the Kaggle test set is unlabeled and mostly filler, so it is never used for OOD.
Needs data/kaggle/test.7z and data/kaggle/sampleSubmission.csv.

Usage: python -m src.predict_kaggle --ckpt results/checkpoints/resnet18_seed0/best.pt
"""
import argparse
import os

import numpy as np
import pandas as pd
import py7zr
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets import CLASSES, make_loader, transforms
from src.models import ResNet18
from src.utils import get_device, load_config


class KaggleTest(Dataset):
    def __init__(self, img_dir, ids, transform):
        self.img_dir, self.ids, self.transform = img_dir, ids, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, f"{self.ids[i]}.png")).convert("RGB")
        return self.transform(img), self.ids[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/checkpoints/resnet18_seed0/best.pt")
    p.add_argument("--out", default="results/kaggle_submission.csv")
    p.add_argument("--num_workers", type=int, default=2)
    a = p.parse_args()

    dev = get_device()
    ck = torch.load(a.ckpt, map_location=dev)
    dcfg = load_config(ck["config"]["data_config"])
    kdir = dcfg["kaggle_dir"]
    sample = pd.read_csv(os.path.join(kdir, "sampleSubmission.csv"))  # defines the id order

    img_dir = os.path.join(kdir, "test")
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) == 0:
        print("extracting test.7z (300,000 PNGs) ...", flush=True)
        with py7zr.SevenZipFile(os.path.join(kdir, "test.7z")) as z:
            z.extractall(kdir)
    if os.path.isdir(os.path.join(img_dir, "test")):  # archive may contain a nested test/ folder
        img_dir = os.path.join(img_dir, "test")

    model = ResNet18().to(dev).eval()
    model.load_state_dict(ck["model"])
    loader = make_loader(KaggleTest(img_dir, sample["id"].tolist(), transforms(dcfg)), 512, False, a.num_workers)
    preds, ids = [], []
    with torch.no_grad():
        for x, i in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                preds.append(model(x.to(dev, non_blocking=True)).argmax(1).cpu())
            ids.append(i)
    preds, ids = torch.cat(preds).numpy(), torch.cat(ids).numpy()
    assert (ids == sample["id"].to_numpy()).all()

    sub = pd.DataFrame({"id": ids, "label": np.array(CLASSES)[preds]})
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sub)} rows, columns {list(sub.columns)} (sample: {list(sample.columns)})")
    print("predicted class counts:", sub["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
