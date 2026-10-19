"""ReAct: clip penultimate activations, recompute logits with the final linear layer, take energy."""
import numpy as np

from src.scores.logit import energy


def fit(train_features, percentile=90):
    """Clip threshold = given percentile of all ID training activations."""
    return float(np.percentile(train_features, percentile))


def score(features, weight, bias, threshold):
    logits = np.minimum(features, threshold) @ weight.T + bias
    return energy(logits)
