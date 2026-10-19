"""KNN score (Sun et al. 2022): negative L2 distance to the k-th nearest normalised training feature."""
import faiss
import numpy as np


def _normalize(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)).astype(np.float32)


def fit(train_features):
    index = faiss.IndexFlatL2(train_features.shape[1])
    index.add(_normalize(train_features))
    return index


def score(features, index, k=50):
    d2, _ = index.search(_normalize(features), k)  # squared L2, sorted ascending
    return -np.sqrt(np.maximum(d2[:, -1], 0))
