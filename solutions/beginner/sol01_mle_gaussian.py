"""
SOLUTION — ex01_mle_gaussian.py
Chỉ dành cho team lead. Không chia sẻ cho thành viên.
"""

import numpy as np


def compute_mu_mle(data):
    x = np.array(data, dtype=float)
    return np.sum(x) / len(x)


def compute_sigma_mle(data):
    x = np.array(data, dtype=float)
    mu = np.sum(x) / len(x)
    variance = np.sum((x - mu) ** 2) / len(x)
    return np.sqrt(variance)


def compute_log_likelihood(data, mu, sigma):
    x = np.array(data, dtype=float)
    n = len(x)
    return (
        -n / 2 * np.log(2 * np.pi * sigma**2)
        - np.sum((x - mu) ** 2) / (2 * sigma**2)
    )
