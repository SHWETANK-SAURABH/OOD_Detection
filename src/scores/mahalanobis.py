"""Mahalanobis score: class means + one shared covariance on training features; -min distance."""
import numpy as np


def fit(train_features, train_labels, num_classes=10):
    means = np.stack([train_features[train_labels == c].mean(0) for c in range(num_classes)])
    centered = train_features - means[train_labels]
    cov = centered.T @ centered / len(train_features)
    return means, np.linalg.pinv(cov)


def score(features, means, precision):
    # squared Mahalanobis distance to each class mean (one class at a time to bound memory)
    d = np.stack([np.sum((features - m) @ precision * (features - m), axis=1) for m in means], axis=1)
    return -d.min(1)
