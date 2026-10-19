# ood-cifar10

Out-of-distribution detection on CIFAR-10: one ResNet-18, several training-free OOD scores, fused into one score.
Reports AUROC and FPR95 per OOD set.

## Setup (Windows, RTX 4060)

```bash
python -m venv .venv            # Python 3.10 or 3.11
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py     # stage 1 check: CUDA + GPU + one forward pass
```

## Data

1. Kaggle CIFAR-10 competition: put `train.7z` and `trainLabels.csv` in `data/kaggle/` and extract
   `train.7z` there (images end up in `data/kaggle/train/train/1.png ... 50000.png`; path set in `configs/data.yaml`).
   `test.7z` is not used.
2. `bash scripts/download_data.sh` downloads:
   - Tiny-ImageNet, Places365, Textures and OpenOOD's cleaned image lists, using OpenOOD's own
     `scripts/download/download.py` (pinned commit `3c35632`, `--dataset_mode dataset`) into `data/ood/`.
   - CIFAR-10 (train + test), CIFAR-100 test, SVHN test, MNIST test via torchvision into `data/torchvision/`.
3. `python scripts/make_split.py` creates the stratified 45,000 / 5,000 split (`splits/*.npy`, seed 0).
4. `python scripts/check_hashes.py` checks the Kaggle images equal torchvision's CIFAR-10 train set.

| Role | Set | Source | Images |
|---|---|---|---|
| ID train / val | Kaggle CIFAR-10 train | Kaggle | 45,000 / 5,000 |
| ID test | CIFAR-10 test | torchvision | 10,000 |
| near-OOD | CIFAR-100 test | torchvision | 10,000 |
| near-OOD | Tiny-ImageNet | OpenOOD `test_tin.txt` | 7,793 |
| far-OOD | SVHN test | torchvision | 26,032 |
| far-OOD | Textures (DTD) | OpenOOD `test_texture.txt` | 5,640 |
| far-OOD | Places365 | OpenOOD `test_places365.txt` | 35,195 |
| optional far | MNIST test | torchvision | 10,000 |
| fusion tuning only | Tiny-ImageNet val | OpenOOD `val_tin.txt` (disjoint from `test_tin.txt`) | 1,000 |

Work in progress; more sections are added as stages are completed.
