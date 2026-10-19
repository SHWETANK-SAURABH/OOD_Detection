"""Run the frozen model once over every split and save logits + 512-d features as .npz.

Usage: python -m src.extract --ckpt results/checkpoints/resnet18_seed0/best.pt [--sets test svhn ...]
Writes results/features/<run>/<set>.npz (logits, features, labels) and fc.npz (final layer weights).
All scoring reads these files and needs no GPU.
"""
import argparse
import os

import numpy as np
import torch

from src.datasets import get_dataset, make_loader, transforms
from src.models import ResNet18
from src.utils import get_device, load_config


@torch.no_grad()
def run(model, loader, dev):
    logits, feats, labels = [], [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            lg, ft = model(x.to(dev, non_blocking=True), return_features=True)
        logits.append(lg.float().cpu())
        feats.append(ft.float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(feats).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sets", nargs="+", help="default: train val test + all OOD sets in data.yaml + tin_val")
    p.add_argument("--out_root", default="results/features")
    p.add_argument("--num_workers", type=int, help="override data.yaml (0 = lowest RAM)")
    a = p.parse_args()

    dev = get_device()
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = ck["config"]
    dcfg = load_config(cfg["data_config"])
    model = ResNet18().to(dev).eval()
    model.load_state_dict(ck["model"])

    run_name = os.path.basename(os.path.dirname(a.ckpt))
    out = os.path.join(a.out_root, run_name)
    os.makedirs(out, exist_ok=True)
    # final linear layer, needed by ReAct to recompute logits from clipped features
    np.savez(os.path.join(out, "fc.npz"), weight=model.fc.weight.detach().cpu().numpy(),
             bias=model.fc.bias.detach().cpu().numpy(), seed=cfg["seed"], config=repr(cfg))

    sets = a.sets or (["train", "val", "test"] + dcfg["near_ood"] + dcfg["far_ood"]
                      + dcfg["optional_ood"] + ["tin_val"])
    # a run may train/test on other Kaggle splits (e.g. the 40k hold-out); save them as train/val/test
    source = {s: cfg.get(f"{s}_split", s) for s in ("train", "val", "test")}
    for name in sets:
        ds = get_dataset(source.get(name, name), dcfg, transforms(dcfg))  # no augmentation, even for train
        nw = dcfg["num_workers"] if a.num_workers is None else a.num_workers
        loader = make_loader(ds, 256, False, nw)
        logits, feats, labels = run(model, loader, dev)
        np.savez(os.path.join(out, f"{name}.npz"), logits=logits, features=feats, labels=labels,
                 seed=cfg["seed"], ckpt=a.ckpt)
        acc = f" acc={np.mean(logits.argmax(1) == labels):.4f}" if labels.min() >= 0 else ""
        print(f"{name}: n={len(labels)} logits{logits.shape} features{feats.shape}{acc}", flush=True)


if __name__ == "__main__":
    main()
