#!/usr/bin/env bash
# Full pipeline: train -> extract -> evaluate for each seed, then aggregate.
# Run from the repo root after scripts/download_data.sh and scripts/make_split.py.
set -e
PY=${PY:-.venv/Scripts/python.exe}   # Linux: PY=.venv/bin/python
SEEDS=${SEEDS:-"0 1 2"}
for s in $SEEDS; do
  "$PY" -m src.train --config configs/train.yaml --seed "$s"
  "$PY" -m src.extract --ckpt "results/checkpoints/resnet18_seed$s/best.pt"
  "$PY" -m src.evaluate --features "results/features/resnet18_seed$s"
done
"$PY" -m src.evaluate --aggregate "results/tables/runs/resnet18_seed*__fusion.csv" --out results/tables/summary.csv
