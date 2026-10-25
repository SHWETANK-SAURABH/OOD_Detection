# ood-cifar10

Out-of-distribution (OOD) detection on CIFAR-10. One ResNet-18 is trained on CIFAR-10, several training-free
OOD scores are computed from that single model, and the scores are fused into one. AUROC and FPR95 are
reported per OOD set. An optional CLIP score (MCM) can be added as one more signal.

**Claim tested:** the fused score beats the best single score on both near-OOD and far-OOD sets.
See [Results](#results) for what we found.

## Setup

Tested on Windows 11, Python 3.11.15, an RTX 4060 Laptop GPU (8 GB) and NVIDIA driver 591.66.

```bash
python -m venv .venv            # Python 3.10 or 3.11
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -r requirements.txt # CUDA 13.0 build of torch; CPU-only builds are rejected
pip install open_clip_torch==3.3.0   # optional, only for the CLIP score
python scripts/check_env.py     # CUDA available, GPU name, one forward pass under autocast
```

## Data

1. Kaggle CIFAR-10 competition: put `train.7z` and `trainLabels.csv` in `data/kaggle/` and extract `train.7z`
   there. The images should end up at `data/kaggle/train/train/1.png ... 50000.png`, which is the path set in
   `configs/data.yaml`. `test.7z` is not used.
2. `bash scripts/download_data.sh` downloads:
   - Tiny-ImageNet, Places365, Textures and OpenOOD's cleaned image lists into `data/ood/`. It uses OpenOOD's own
     `scripts/download/download.py` (pinned commit `3c35632`) with `--dataset_mode dataset`, so only these sets
     are fetched.
   - CIFAR-10 (train + test), CIFAR-100 test, SVHN test and MNIST test via torchvision into `data/torchvision/`.
3. `python scripts/make_split.py` creates the stratified 45,000 / 5,000 train / val split, plus the 40k / 5k / 5k
   Kaggle-only hold-out. The split files are in `splits/` (seed 0) and are committed to the repo.
4. `python scripts/check_hashes.py` hashes the decoded pixels of all 50,000 Kaggle images and torchvision's CIFAR-10
   train set. Result: **the hash sets match** (50,000 common images, 0 label mismatches), which also confirms that
   the alphabetical class order lines up.

| Role | Set | Source | Images |
|---|---|---|---|
| ID train / val | Kaggle CIFAR-10 train | Kaggle | 45,000 / 5,000 |
| ID test | CIFAR-10 test | torchvision | 10,000 |
| near-OOD | CIFAR-100 test | torchvision | 10,000 |
| near-OOD | Tiny-ImageNet (TIN) | OpenOOD `test_tin.txt` | 7,793 |
| far-OOD | SVHN test | torchvision | 26,032 |
| far-OOD | Textures (DTD) | OpenOOD `test_texture.txt` | 5,640 |
| far-OOD | Places365 | OpenOOD `test_places365.txt` | 35,195 |
| optional far | MNIST test | torchvision | 10,000 |
| fusion tuning only | TIN val | OpenOOD `val_tin.txt` (no overlap with `test_tin.txt`) | 1,000 |

All OOD images are resized to 32x32 and use the CIFAR-10 normalisation.

## Pipeline

```bash
bash scripts/run_all.sh          # train -> extract -> evaluate for seeds 0 1 2, then aggregate
# or step by step:
python -m src.train   --config configs/train.yaml --seed 0
python -m src.extract --ckpt results/checkpoints/resnet18_seed0/best.pt
python -m src.evaluate --features results/features/resnet18_seed0 --fusion configs/fusion.yaml
python -m src.scores.clip_score                                # optional CLIP image embeddings
python -m src.evaluate --features results/features/resnet18_seed0 --fusion configs/fusion_clip.yaml
python -m src.report                                           # tables, Excel, ROC plots
pytest -q tests
```

- **Model:** ResNet-18, CIFAR variant: 3x3 stride-1 first conv, no max pool, 512-d penultimate features.
- **Training** (`configs/train.yaml`): SGD with momentum 0.9 and weight decay 5e-4, lr 0.1 with cosine decay,
  batch 128, 100 epochs, random crop (padding 4) + horizontal flip, float16 autocast. The best checkpoint is
  chosen by val accuracy. Compared with OpenOOD v1.5's CIFAR-10 recipe (`openood/trainers/base_trainer.py`,
  `configs/pipelines/train/baseline.yml` at commit `3c35632`), two settings were taken from OpenOOD: Nesterov
  momentum, and cosine decay stepped per iteration down to 1e-6.
- **Measured speed:** about 14-15 s per epoch (training loop), with 475 MiB peak GPU memory allocated at batch
  128, so the full run takes about 25 minutes.
- **Extraction:** `src/extract.py` runs the frozen model once per set and stores logits and features in
  `results/features/<run>/*.npz`. All scoring after that runs on the CPU.
- **Scores** (higher = more in-distribution):
  - MSP, max logit and energy.
  - ReAct: clip features at the 90th percentile of ID training activations, then take the energy.
  - KNN: L2-normalised features, k = 50.
  - Mahalanobis: class means with one shared covariance.
  - CLIP MCM (optional): ViT-B/16 `openai` weights, T = 1, either the prompt "a photo of a {class}" or the
    mean of 5 LLM-written descriptions per class (`configs/clip_descriptions.yaml`).
- **Fusion:** each score is z-scored with the mean and std of its ID val scores, then combined in one of three
  modes:
  - Equal weights.
  - Tuned weights: logistic regression of ID val against TIN val, rescaled so that sum(|w|) = 1.
  - Rank normalisation: each score becomes its empirical CDF on ID val.
- **Metrics:** ID is the positive class. AUROC comes from scikit-learn. FPR95 is the share of OOD samples
  accepted at the threshold that keeps 95% of ID test samples. Results are the mean ± std over 3 training
  seeds (0, 1, 2). The near/far averages exclude MNIST.

## Results

All numbers below were computed by this code (`python -m src.report`; the full tables are in
`results/tables/*.csv` and `results/tables/results.xlsx`). No published values are quoted.

**ID accuracy (CIFAR-10 test, 10,000 images):** 94.61% / 94.51% / 94.64% for seeds 0 / 1 / 2. The Kaggle-only
hold-out model reaches 94.74% on its 5,000-image Kaggle test split (`results/tables/id_accuracy.csv`).

**Verdict on the claim:**

- **ResNet scores only (6 scorers): not supported for AUROC.** KNN alone is the best single score. On near-OOD,
  only tuned fusion beats it (90.72 ± 0.05 vs 90.53 ± 0.09). On far-OOD, no fusion mode beats KNN (best is
  equal-weight fusion at 92.98 ± 0.52 vs 93.15 ± 0.51).
- **ResNet scores only, FPR95: supported, but by small margins.** Lower is better here, and the best single
  score on both averages is energy. Tuned fusion gets 48.09 on near vs 48.66 for energy, and rank fusion gets
  40.14 on far vs 41.79. Both gaps are within about one standard deviation.
- **With CLIP as a 7th score: supported, by a wide margin.** Tuned fusion reaches 94.87 near / 96.86 far AUROC,
  against 90.60 / 93.15 for the best single score. It beats every single score, CLIP included, on every OOD set.
  FPR95 falls to 25.07 near / 14.52 far. Equal-weight and rank fusion also beat the best single score on both
  averages once CLIP is included.

**Other findings:**

- **Leave-one-scorer-out:** dropping Mahalanobis or KNN hurts equal-weight fusion the most. Dropping MSP
  improves it (far AUROC 93.48, far FPR95 35.89), so equal weighting is held back by redundant, weaker scores.
- **CLIP prompts:** plain class names and the averaged LLM descriptions perform within about 0.9 AUROC of each
  other. Neither is consistently better.
- **Training data:** the Kaggle-only 40k / 5k / 5k hold-out (one seed) gives near/far averages within 1.1
  AUROC points of the 45k model with the same seed, and the method ranking is unchanged.

**Caveats:**

- **Tuned weights are fitted on TIN val** (1,000 images, no overlap with TIN test). TIN test is a reported
  near-OOD set from the same distribution, which may favour the tuned mode on near-OOD. With CLIP, the tuned
  mode also wins on all three far-OOD sets, which TIN val does not cover.
- **CLIP is a frozen model pretrained on large web data.** It is the same for every seed (std 0.00 in the
  CLIP row), and "+CLIP" results no longer come from a single CIFAR-trained model.
- **SVHN results vary a lot across seeds** (std up to 5 AUROC points and 21 FPR95 points), so differences on
  SVHN alone are not reliable with 3 seeds.

ROC curves for seed 0 are in `results/plots/`: `roc_resnet_*.png` (ResNet scores only) and `roc_with_clip_*.png`.

### ResNet-18 scores and fusion (no CLIP), 3 seeds

AUROC % (higher is better), mean ± std:

| method      | cifar100     | tin          | **near avg**   | svhn         | texture      | places365    | **far avg**   | mnist        |
|:------------|:-------------|:-------------|:---------------|:-------------|:-------------|:-------------|:--------------|:-------------|
| msp         | 87.26 ± 0.13 | 88.99 ± 0.16 | 88.13 ± 0.10   | 92.07 ± 3.22 | 90.37 ± 0.56 | 89.27 ± 0.46 | 90.57 ± 0.82  | 91.11 ± 1.57 |
| maxlogit    | 86.59 ± 0.11 | 88.94 ± 0.28 | 87.76 ± 0.12   | 91.79 ± 5.11 | 89.58 ± 1.18 | 89.94 ± 0.55 | 90.44 ± 1.26  | 91.95 ± 2.31 |
| energy      | 86.64 ± 0.12 | 89.02 ± 0.31 | 87.83 ± 0.14   | 91.86 ± 5.16 | 89.60 ± 1.24 | 90.06 ± 0.55 | 90.51 ± 1.28  | 92.08 ± 2.35 |
| react       | 86.18 ± 0.13 | 88.55 ± 0.17 | 87.36 ± 0.15   | 91.86 ± 2.95 | 90.35 ± 0.34 | 90.91 ± 1.39 | 91.04 ± 0.74  | 89.90 ± 2.53 |
| knn         | 89.59 ± 0.07 | 91.48 ± 0.11 | 90.53 ± 0.09   | 94.45 ± 1.45 | 93.25 ± 0.34 | 91.75 ± 0.21 | 93.15 ± 0.51  | 93.88 ± 0.74 |
| mahalanobis | 83.47 ± 0.96 | 85.00 ± 0.73 | 84.23 ± 0.82   | 92.93 ± 2.98 | 92.86 ± 0.98 | 85.45 ± 1.48 | 90.41 ± 1.03  | 93.66 ± 0.47 |
| fused_equal | 88.31 ± 0.18 | 90.28 ± 0.14 | 89.29 ± 0.15   | 94.18 ± 1.63 | 93.54 ± 0.37 | 91.22 ± 0.50 | 92.98 ± 0.52  | 93.45 ± 1.16 |
| fused_tuned | 89.74 ± 0.08 | 91.70 ± 0.04 | 90.72 ± 0.05   | 93.83 ± 1.70 | 92.22 ± 0.83 | 92.49 ± 0.53 | 92.85 ± 0.62  | 93.66 ± 0.58 |
| fused_rank  | 88.40 ± 0.18 | 90.36 ± 0.14 | 89.38 ± 0.15   | 94.03 ± 2.49 | 92.63 ± 0.31 | 91.26 ± 0.47 | 92.64 ± 0.71  | 93.70 ± 1.37 |

FPR95 % (lower is better), mean ± std:

| method      | cifar100     | tin          | **near avg**   | svhn          | texture      | places365    | **far avg**   | mnist         |
|:------------|:-------------|:-------------|:---------------|:--------------|:-------------|:-------------|:--------------|:--------------|
| msp         | 61.82 ± 1.64 | 57.78 ± 1.19 | 59.80 ± 1.39   | 50.08 ± 10.70 | 56.29 ± 1.58 | 57.50 ± 1.47 | 54.62 ± 3.59  | 56.67 ± 5.28  |
| maxlogit    | 53.07 ± 0.19 | 46.87 ± 1.07 | 49.97 ± 0.54   | 38.02 ± 14.90 | 47.06 ± 2.27 | 44.42 ± 1.27 | 43.17 ± 4.22  | 42.82 ± 9.30  |
| energy      | 51.80 ± 0.19 | 45.51 ± 1.40 | 48.66 ± 0.74   | 36.82 ± 15.63 | 45.88 ± 2.87 | 42.67 ± 1.03 | 41.79 ± 4.52  | 40.88 ± 9.73  |
| react       | 54.16 ± 0.46 | 47.76 ± 0.64 | 50.96 ± 0.55   | 42.59 ± 11.88 | 46.39 ± 0.75 | 43.15 ± 3.65 | 44.05 ± 3.40  | 48.08 ± 10.06 |
| knn         | 53.37 ± 0.65 | 47.79 ± 0.76 | 50.58 ± 0.69   | 38.44 ± 8.74  | 42.84 ± 1.83 | 45.41 ± 0.76 | 42.23 ± 2.75  | 43.22 ± 5.06  |
| mahalanobis | 72.02 ± 3.00 | 70.79 ± 2.00 | 71.40 ± 2.34   | 48.24 ± 21.37 | 40.61 ± 6.65 | 68.99 ± 3.88 | 52.62 ± 5.49  | 46.50 ± 4.37  |
| fused_equal | 55.21 ± 0.93 | 49.89 ± 0.67 | 52.55 ± 0.78   | 38.75 ± 9.01  | 39.04 ± 2.58 | 47.98 ± 1.06 | 41.92 ± 2.55  | 44.25 ± 6.17  |
| fused_tuned | 51.55 ± 1.35 | 44.63 ± 1.71 | 48.09 ± 1.53   | 40.68 ± 8.14  | 47.43 ± 3.12 | 39.84 ± 1.05 | 42.65 ± 1.89  | 42.25 ± 2.97  |
| fused_rank  | 53.63 ± 1.06 | 48.05 ± 0.84 | 50.84 ± 0.95   | 34.68 ± 10.04 | 40.33 ± 1.62 | 45.40 ± 1.45 | 40.14 ± 3.06  | 38.24 ± 6.03  |

### With the CLIP MCM score as a 7th column, 3 seeds

AUROC %:

| method      | cifar100     | tin          | **near avg**   | svhn         | texture      | places365    | **far avg**   | mnist        |
|:------------|:-------------|:-------------|:---------------|:-------------|:-------------|:-------------|:--------------|:-------------|
| knn         | 89.59 ± 0.07 | 91.48 ± 0.11 | 90.53 ± 0.09   | 94.45 ± 1.45 | 93.25 ± 0.34 | 91.75 ± 0.21 | 93.15 ± 0.51  | 93.88 ± 0.74 |
| clip        | 90.85 ± 0.00 | 90.36 ± 0.00 | 90.60 ± 0.00   | 97.62 ± 0.00 | 93.53 ± 0.00 | 87.34 ± 0.00 | 92.83 ± 0.00  | 98.22 ± 0.00 |
| fused_equal | 90.67 ± 0.16 | 92.13 ± 0.14 | 91.40 ± 0.14   | 95.70 ± 1.20 | 94.92 ± 0.34 | 92.72 ± 0.41 | 94.45 ± 0.41  | 95.13 ± 0.82 |
| fused_tuned | 94.50 ± 0.00 | 95.24 ± 0.02 | 94.87 ± 0.01   | 98.77 ± 0.32 | 97.15 ± 0.31 | 94.67 ± 0.34 | 96.86 ± 0.23  | 98.74 ± 0.12 |
| fused_rank  | 91.15 ± 0.16 | 92.73 ± 0.13 | 91.94 ± 0.14   | 96.44 ± 1.95 | 94.92 ± 0.24 | 93.14 ± 0.42 | 94.83 ± 0.55  | 96.28 ± 1.02 |

FPR95 %:

| method      | cifar100     | tin          | **near avg**   | svhn         | texture      | places365    | **far avg**   | mnist        |
|:------------|:-------------|:-------------|:---------------|:-------------|:-------------|:-------------|:--------------|:-------------|
| knn         | 53.37 ± 0.65 | 47.79 ± 0.76 | 50.58 ± 0.69   | 38.44 ± 8.74 | 42.84 ± 1.83 | 45.41 ± 0.76 | 42.23 ± 2.75  | 43.22 ± 5.06 |
| clip        | 47.23 ± 0.00 | 50.33 ± 0.00 | 48.78 ± 0.00   | 12.16 ± 0.00 | 37.45 ± 0.00 | 58.90 ± 0.00 | 36.17 ± 0.00  | 2.08 ± 0.00  |
| fused_equal | 50.15 ± 0.80 | 44.85 ± 0.39 | 47.50 ± 0.59   | 30.72 ± 8.91 | 33.01 ± 2.32 | 43.65 ± 1.16 | 35.79 ± 2.85  | 35.96 ± 6.16 |
| fused_tuned | 27.15 ± 0.54 | 23.00 ± 0.26 | 25.07 ± 0.39   | 3.91 ± 2.73  | 14.77 ± 1.79 | 24.88 ± 1.39 | 14.52 ± 1.13  | 4.60 ± 1.07  |
| fused_rank  | 39.31 ± 0.33 | 34.42 ± 0.13 | 36.87 ± 0.21   | 17.90 ± 9.64 | 26.26 ± 0.49 | 35.04 ± 0.70 | 26.40 ± 3.06  | 19.94 ± 5.00 |

### Ablations (near/far averages, %)

Leave-one-scorer-out, equal weights, no CLIP, 3 seeds:

| method                 |   near AUROC |   far AUROC |   near FPR95 |   far FPR95 |
|:-----------------------|-------------:|------------:|-------------:|------------:|
| fused_equal            |        89.29 |       92.98 |        52.55 |       41.92 |
| loo_equal_-msp         |        89.59 |       93.48 |        48.66 |       35.89 |
| loo_equal_-maxlogit    |        89.43 |       93.15 |        53.53 |       42.49 |
| loo_equal_-energy      |        89.41 |       93.13 |        53.69 |       42.71 |
| loo_equal_-react       |        89.45 |       93.06 |        53.32 |       42.29 |
| loo_equal_-knn         |        88.78 |       92.77 |        52.95 |       41.95 |
| loo_equal_-mahalanobis |        88.67 |       91.58 |        53.23 |       46.72 |

CLIP prompt: class names vs LLM descriptions, 3 seeds:

| setting              | method      |   near AUROC |   far AUROC |   near FPR95 |   far FPR95 |
|:---------------------|:------------|-------------:|------------:|-------------:|------------:|
| a photo of a {class} | clip        |        90.6  |       92.83 |        48.78 |       36.17 |
| a photo of a {class} | fused_equal |        91.4  |       94.45 |        47.5  |       35.79 |
| a photo of a {class} | fused_tuned |        94.87 |       96.86 |        25.07 |       14.52 |
| a photo of a {class} | fused_rank  |        91.94 |       94.83 |        36.87 |       26.4  |
| LLM descriptions     | clip        |        89.73 |       92.9  |        48.25 |       34.46 |
| LLM descriptions     | fused_equal |        91.28 |       94.47 |        47.35 |       35.38 |
| LLM descriptions     | fused_tuned |        94.58 |       96.78 |        25.74 |       15.03 |
| LLM descriptions     | fused_rank  |        91.93 |       94.92 |        36.14 |       25.21 |

Training data: 45k Kaggle train + torchvision test (seed 0) vs Kaggle-only 40k / 5k / 5k hold-out (seed 0, one run):

| method      |   near AUROC 45k |   near AUROC 40k |   far AUROC 45k |   far AUROC 40k |
|:------------|-----------------:|-----------------:|----------------:|----------------:|
| msp         |            88.12 |            87.91 |           89.63 |           90.02 |
| energy      |            87.95 |            87.69 |           89.1  |           90.21 |
| knn         |            90.45 |            90.33 |           92.56 |           92.7  |
| mahalanobis |            84.48 |            85.59 |           91.08 |           91.85 |
| fused_equal |            89.18 |            89.29 |           92.38 |           92.57 |
| fused_tuned |            90.66 |            90.32 |           92.28 |           92.72 |
| fused_rank  |            89.29 |            89.35 |           91.82 |           92.39 |

Stage 11 (Kaggle test-set submission) was not run: it needs `test.7z`, which the data rules exclude.


## Layout

```
configs/   data.yaml train.yaml train_holdout.yaml fusion.yaml fusion_clip.yaml fusion_clip_desc.yaml clip_descriptions.yaml
splits/    train_idx.npy val_idx.npy train40k_idx.npy test_kaggle5k_idx.npy
src/       datasets.py models.py train.py extract.py fusion.py metrics.py evaluate.py report.py utils.py
src/scores/ logit.py react.py knn.py mahalanobis.py clip_score.py
scripts/   download_data.sh run_all.sh make_split.py check_hashes.py check_env.py
results/   tables/ plots/   (checkpoints/ and features/ are git-ignored)
tests/     test_metrics.py test_scores.py
```

Every per-run result file (`results/tables/runs/*.csv`) carries its seed, and the matching `*_meta.json` stores
the full fusion and data config. Each checkpoint stores its training config.
