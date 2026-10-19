#!/usr/bin/env bash
# Downloads every dataset except the Kaggle files (put train.7z/trainLabels.csv in data/kaggle/ yourself).
# Run from the repo root: bash scripts/download_data.sh
set -e
PY=${PY:-.venv/Scripts/python.exe}   # Windows venv; on Linux use PY=.venv/bin/python
OPENOOD_SHA=3c35632ee91b54b09d1f085d04f94744cece7d0b

# 1) OpenOOD image lists + images for Tiny-ImageNet, Places365, Textures.
#    Uses OpenOOD's own script (scripts/download/download.py) with --dataset_mode dataset
#    so only the listed sets are fetched. Writes data/ood/benchmark_imglist and data/ood/images_classic/<set>.
mkdir -p .cache/openood data/ood
curl -sL "https://raw.githubusercontent.com/Jingkang50/OpenOOD/${OPENOOD_SHA}/scripts/download/download.py" \
  -o .cache/openood/download.py
"$PY" .cache/openood/download.py --contents datasets --datasets tin places365 texture \
  --dataset_mode dataset --save_dir ./data/ood ./results

# 2) torchvision sets: CIFAR-10 test (ID test) + train (hash check), CIFAR-100 test, SVHN test, MNIST test.
"$PY" - <<'PYEOF'
import torchvision.datasets as D
root = "data/torchvision"
D.CIFAR10(root, train=True, download=True)
D.CIFAR10(root, train=False, download=True)
D.CIFAR100(root, train=False, download=True)
D.SVHN(root + "/svhn", split="test", download=True)
D.MNIST(root, train=False, download=True)
print("torchvision datasets ready")
PYEOF
