"""
SOLUTION — ex03_map_bernoulli.py
"""

import numpy as np


def compute_p_map(data, alpha, beta):
    x = np.array(data, dtype=float)
    n = len(x)
    k = int(np.sum(x))
    return (alpha + k - 1) / (alpha + beta + n - 2)


def compare_mle_map(data, alpha, beta):
    x = np.array(data, dtype=float)
    p_mle = float(np.mean(x))
    p_map = float(compute_p_map(data, alpha, beta))
    return {
        "p_mle": p_mle,
        "p_map": p_map,
        "difference": abs(p_mle - p_map),
    }
