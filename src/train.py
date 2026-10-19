"""Train ResNet-18 on the Kaggle CIFAR-10 split; keep the best checkpoint by val accuracy.

Usage: python -m src.train --config configs/train.yaml [--seed 1] [--epochs 2]
"""
import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.datasets import get_dataset, make_loader, transforms
from src.models import ResNet18
from src.utils import get_device, load_config, set_seed


@torch.no_grad()
def evaluate(model, loader, dev, amp):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def train(cfg, batch_size, out):
    dev = get_device()
    dcfg = load_config(cfg["data_config"])
    set_seed(cfg["seed"])
    nw = cfg["num_workers"]
    train_dl = make_loader(get_dataset(cfg.get("train_split", "train"), dcfg, transforms(dcfg, True)),
                           batch_size, True, nw)
    val_dl = make_loader(get_dataset(cfg.get("val_split", "val"), dcfg, transforms(dcfg)), 256, False, nw)

    model = ResNet18().to(dev)
    opt = torch.optim.SGD(model.parameters(), cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
    total_steps = cfg["epochs"] * len(train_dl)
    lo = cfg["lr_min"] / cfg["lr"]  # cosine from 1 to lo, per iteration
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    writer = SummaryWriter(os.path.join("runs", os.path.basename(out)))
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    logw = csv.writer(log)
    logw.writerow(["epoch", "lr", "train_loss", "train_acc", "val_acc", "sec"])

    best = -1.0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        torch.cuda.synchronize()
        sec = time.time() - t0
        val_acc = evaluate(model, val_dl, dev, cfg["amp"])
        lr = opt.param_groups[0]["lr"]
        row = [epoch, f"{lr:.6f}", f"{loss_sum / n:.4f}", f"{correct / n:.4f}", f"{val_acc:.4f}", f"{sec:.1f}"]
        logw.writerow(row)
        log.flush()
        for k, v in zip(["lr", "train_loss", "train_acc", "val_acc"], [lr, loss_sum / n, correct / n, val_acc]):
            writer.add_scalar(k, v, epoch)
        print(f"epoch {epoch}/{cfg['epochs']} loss {loss_sum / n:.4f} train_acc {correct / n:.4f} "
              f"val_acc {val_acc:.4f} ({sec:.1f}s)", flush=True)
        if epoch == 1:
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f"[speed] epoch 1: {sec:.1f} s/epoch (train loop only), peak GPU memory "
                  f"{peak:.0f} MiB allocated, batch size {batch_size}", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "config": cfg},
                       os.path.join(out, "best.pt"))
    log.close()
    writer.close()

    # ID test accuracy of the best checkpoint
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev)["model"])
    test_dl = make_loader(get_dataset(cfg.get("test_split", "test"), dcfg, transforms(dcfg)), 256, False, nw)
    test_acc = evaluate(model, test_dl, dev, cfg["amp"])
    return {"seed": cfg["seed"], "best_val_acc": best, "test_acc": test_acc, "batch_size": batch_size,
            "epochs": cfg["epochs"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--tag", default="", help="suffix for the run directory, e.g. smoke")
    a = p.parse_args()
    cfg = load_config(a.config)
    for k in ("seed", "epochs"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    out = os.path.join(cfg["out_dir"], f"{cfg.get('name', 'resnet18')}_seed{cfg['seed']}{a.tag}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    bs = cfg["batch_size"]
    while True:
        try:
            res = train(cfg, bs, out)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 16:
                raise
            bs //= 2
            print(f"[OOM] out of GPU memory; retrying with batch size {bs}", flush=True)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
