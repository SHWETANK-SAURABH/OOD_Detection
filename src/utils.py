"""Shared helpers: seeding, device, config loading."""
import os
import random

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; a CPU-only torch build is not accepted.")
    return torch.device("cuda")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
