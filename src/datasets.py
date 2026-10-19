"""Datasets: Kaggle CIFAR-10 (ID train/val), torchvision CIFAR-10 test (ID test), and the OOD sets."""
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Alphabetical, identical to torchvision's CIFAR10.classes, so Kaggle labels line up.
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def transforms(cfg, train=False):
    norm = [T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])]
    if train:
        return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] + norm)
    # OOD images come in various sizes; resize to 32x32 (no-op for 32x32 inputs).
    return T.Compose([T.Resize((32, 32))] + norm)


def load_kaggle_arrays(cfg):
    """All 50,000 Kaggle PNGs as a uint8 array (N,32,32,3), ordered by id, plus int labels.
    Decoding 50k PNGs is slow, so the result is cached next to the images as .npz."""
    cache = os.path.join(cfg["kaggle_dir"], "kaggle_train_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["images"], d["labels"]
    df = pd.read_csv(os.path.join(cfg["kaggle_dir"], "trainLabels.csv")).sort_values("id")
    images = np.stack([np.array(Image.open(os.path.join(cfg["kaggle_img_dir"], f"{i}.png")).convert("RGB"))
                       for i in df["id"]])
    labels = df["label"].map(CLASSES.index).to_numpy(np.int64)
    np.savez(cache, images=images, labels=labels)
    return images, labels


class KaggleCIFAR10(Dataset):
    """Kaggle train images selected by split indices (position i <-> (i+1).png)."""

    def __init__(self, cfg, split="train", transform=None):
        images, labels = load_kaggle_arrays(cfg)
        idx = np.load(os.path.join(cfg["splits_dir"], f"{split}_idx.npy"))
        self.images, self.labels, self.transform = images[idx], labels[idx], transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return (self.transform(img) if self.transform else img), int(self.labels[i])


class ImglistDataset(Dataset):
    """OpenOOD-style image list: each line is '<path relative to images_classic/> <label>'."""

    def __init__(self, root, imglist, transform=None):
        with open(imglist) as f:
            self.paths = [line.split()[0] for line in f if line.strip()]
        self.root, self.transform = root, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")
        return (self.transform(img) if self.transform else img), -1


def _ood_label(_):
    return -1  # module-level (not a lambda) so Windows DataLoader workers can pickle it


def get_dataset(name, cfg, transform=None):
    """ID splits: 'train', 'val' (Kaggle), 'test' (torchvision). OOD: names below. OOD labels are -1."""
    tv = cfg["torchvision_root"]
    imgs = os.path.join(cfg["ood_root"], "images_classic")
    lists = os.path.join(cfg["ood_root"], "benchmark_imglist", "cifar10")
    if name in ("train", "val"):
        return KaggleCIFAR10(cfg, name, transform)
    if name == "test":
        return torchvision.datasets.CIFAR10(tv, train=False, transform=transform)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(tv, train=False, transform=transform,
                                             target_transform=_ood_label)
    if name == "svhn":
        return torchvision.datasets.SVHN(os.path.join(tv, "svhn"), split="test", transform=transform,
                                         target_transform=_ood_label)
    if name == "mnist":  # grayscale -> 3 channels before the shared transform
        t = T.Compose([T.Grayscale(3), transform]) if transform else T.Grayscale(3)
        return torchvision.datasets.MNIST(tv, train=False, transform=t, target_transform=_ood_label)
    # OpenOOD cleaned lists. tin_val is the held-out OOD slice for fusion tuning (disjoint from tin test).
    openood = {"tin": "test_tin.txt", "places365": "test_places365.txt",
               "texture": "test_texture.txt", "tin_val": "val_tin.txt"}
    if name in openood:
        return ImglistDataset(imgs, os.path.join(lists, openood[name]), transform)
    raise ValueError(f"unknown dataset {name}")


def make_loader(ds, batch_size, shuffle, num_workers):
    """DataLoader with workers; falls back to num_workers=0 if worker startup fails."""
    try:
        loader = DataLoader(ds, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
        if num_workers > 0:
            next(iter(loader))  # surface worker errors now rather than mid-epoch
        return loader
    except Exception as e:  # noqa: BLE001 - any worker failure -> single-process loading
        print(f"DataLoader with {num_workers} workers failed ({e!r}); using num_workers=0")
        return DataLoader(ds, batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
