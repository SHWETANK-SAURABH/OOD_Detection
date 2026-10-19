"""Score fusion. Each scorer is z-scored with the mean/std of its ID validation scores, then combined."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def zscore_params(val_scores, names):
    return {n: (val_scores[n].mean(), val_scores[n].std()) for n in names}


def zmatrix(scores, params, names):
    """(N, k) matrix of z-scored scorers."""
    return np.stack([(scores[n] - params[n][0]) / params[n][1] for n in names], axis=1)


def rank_matrix(scores, val_scores, names):
    """Rank normalisation: each score -> fraction of ID val scores below it, centred to [-0.5, 0.5]."""
    cols = []
    for n in names:
        ref = np.sort(val_scores[n])
        cols.append(np.searchsorted(ref, scores[n], side="right") / len(ref) - 0.5)
    return np.stack(cols, axis=1)


def equal_weights(k):
    return np.full(k, 1.0 / k)


def tune_weights(z_id, z_ood):
    """Logistic regression of ID (1) vs held-out OOD (0) on z-scores; weights rescaled to sum(|w|) = 1."""
    x = np.r_[z_id, z_ood]
    y = np.r_[np.ones(len(z_id)), np.zeros(len(z_ood))]
    w = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x, y).coef_[0]
    return w / np.abs(w).sum()
