"""OOD metrics. ID is the positive class; scores are higher for in-distribution."""
import math

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(id_scores, ood_scores):
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    return roc_auc_score(y, np.r_[id_scores, ood_scores])


def fpr_at_95_tpr(id_scores, ood_scores, tpr=0.95):
    """Share of OOD samples accepted (score >= t) at the largest threshold t that keeps >= 95% of ID."""
    s = np.sort(id_scores)[::-1]
    t = s[math.ceil(tpr * len(s)) - 1]
    return float(np.mean(ood_scores >= t))
