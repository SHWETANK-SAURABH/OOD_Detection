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

Work in progress; more sections are added as stages are completed.
