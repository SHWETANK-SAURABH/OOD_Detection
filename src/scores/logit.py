"""Logit-based scores. Higher = more in-distribution."""
import numpy as np
from scipy.special import logsumexp, softmax


def msp(logits):
    return softmax(logits, axis=1).max(1)


def max_logit(logits):
    return logits.max(1)


def energy(logits, T=1.0):
    # negative free energy: T * logsumexp(logits / T)
    return T * logsumexp(logits / T, axis=1)
