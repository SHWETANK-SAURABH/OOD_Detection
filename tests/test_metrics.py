import numpy as np

from src.metrics import auroc, fpr_at_95_tpr


def test_fpr95_hand_example():
    # 20 ID scores 1..20: keeping 95% (19 of 20) gives threshold 2.
    id_s = np.arange(1, 21, dtype=float)
    # OOD scores: 2 and 3 are accepted (>= 2), 0, 1, 1.5 are rejected -> FPR = 2/5
    ood_s = np.array([0.0, 1.0, 1.5, 2.0, 3.0])
    assert fpr_at_95_tpr(id_s, ood_s) == 0.4


def test_fpr95_perfect_and_worst():
    id_s = np.linspace(10, 20, 100)
    assert fpr_at_95_tpr(id_s, np.zeros(50)) == 0.0
    assert fpr_at_95_tpr(id_s, np.full(50, 30.0)) == 1.0


def test_auroc():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    # one of four ID/OOD pairs mis-ordered -> 0.75
    assert auroc(np.array([1.0, 3.0]), np.array([0.0, 2.0])) == 0.75
