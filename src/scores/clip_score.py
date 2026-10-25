"""Optional CLIP score: MCM (maximum concept matching) with ViT-B/16.

score = max_c softmax_c(cos(image, text_c) / T), T = 1, over the 10 CIFAR-10 classes.
Text embedding per class: one prompt "a photo of a {class}", or (descriptions mode) the mean of
several LLM-written descriptions from configs/clip_descriptions.yaml.

Extract once (GPU):  python -m src.scores.clip_score --out results/features/clip_vitb16
Scoring (CPU) reads the saved image embeddings and text embeddings.
"""
import argparse
import os

import numpy as np
from scipy.special import softmax


def score(image_emb, text_emb, T=1.0):
    """image_emb (N, D), text_emb (C, D), both L2-normalised."""
    return softmax(image_emb @ text_emb.T / T, axis=1).max(1)


def main():
    # keep model downloads inside the project
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("HF_HOME", os.path.join(root, ".cache", "hf"))
    os.environ.setdefault("TORCH_HOME", os.path.join(root, ".cache", "torch"))
    import open_clip
    import torch
    import yaml

    from src.datasets import CLASSES, get_dataset, make_loader
    from src.utils import get_device, load_config

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/features/clip_vitb16")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--descriptions", default="configs/clip_descriptions.yaml")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sets", nargs="+")
    p.add_argument("--num_workers", type=int, help="override data.yaml (0 = lowest RAM)")
    a = p.parse_args()

    dev, dcfg = get_device(), load_config(a.data)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-16")
    model.eval()
    os.makedirs(a.out, exist_ok=True)

    @torch.no_grad()
    def encode_text(prompts):
        e = model.encode_text(tok(prompts).to(dev)).float()
        return e / e.norm(dim=-1, keepdim=True)

    names = encode_text([f"a photo of a {c}" for c in CLASSES]).cpu().numpy()
    with open(a.descriptions) as f:
        desc = yaml.safe_load(f)
    # average the normalised description embeddings per class, then re-normalise
    d = torch.stack([encode_text(desc[c]).mean(0) for c in CLASSES])
    d = (d / d.norm(dim=-1, keepdim=True)).cpu().numpy()
    np.savez(os.path.join(a.out, "text.npz"), names=names, descriptions=d)

    sets = a.sets or (["val", "test"] + dcfg["near_ood"] + dcfg["far_ood"] + dcfg["optional_ood"] + ["tin_val"])
    for name in sets:
        # raw images through CLIP's own preprocessing (224px, CLIP normalisation), not the CIFAR transform
        nw = dcfg["num_workers"] if a.num_workers is None else a.num_workers
        loader = make_loader(get_dataset(name, dcfg, preprocess), a.batch_size, False, nw)
        embs = []
        with torch.no_grad():
            for x, _ in loader:
                with torch.autocast("cuda", dtype=torch.float16):
                    e = model.encode_image(x.to(dev, non_blocking=True)).float()
                embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        emb = torch.cat(embs).numpy()
        np.savez(os.path.join(a.out, f"{name}.npz"), image_emb=emb)
        print(f"{name}: n={len(emb)} emb{emb.shape}", flush=True)


if __name__ == "__main__":
    main()
