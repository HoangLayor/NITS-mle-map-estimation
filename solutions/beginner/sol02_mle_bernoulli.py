"""
SOLUTION — ex02_mle_bernoulli.py
"""

import numpy as np


def compute_p_mle(data):
    x = np.array(data, dtype=float)
    return np.sum(x) / len(x)


def compute_bernoulli_log_likelihood(data, p):
    x = np.array(data, dtype=float)
    k = np.sum(x)
    n = len(x)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return k * np.log(p) + (n - k) * np.log(1 - p)


def find_best_p(data, p_candidates):
    best_p = None
    best_ll = -np.inf
    for p in p_candidates:
        ll = compute_bernoulli_log_likelihood(data, p)
        if ll > best_ll:
            best_ll = ll
            best_p = p
    return best_p
