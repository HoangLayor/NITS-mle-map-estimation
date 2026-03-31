"""
Bài tập 1: MLE cho Gaussian
═══════════════════════════════════════════════════════════

Mục tiêu:
    - Tự tính μ_MLE và σ_MLE bằng công thức
    - Tính log-likelihood
    - So sánh với GaussianMLE trong src/

Lý thuyết nhắc lại:
    μ_MLE = (1/n) * Σ xi
    σ²_MLE = (1/n) * Σ (xi - μ)²

Hướng dẫn:
    1. Đọc kỹ docstring của từng hàm
    2. Chỉ dùng numpy (không dùng scipy.stats)
    3. Chạy: pytest tests/test_mle.py::test_ex01 -v
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


# ───────────────────────────────────────────────
# Thử nghiệm nhanh (chạy file này trực tiếp)
# ───────────────────────────────────────────────

if __name__ == "__main__":
    data = [2.1, 1.9, 2.3, 2.0, 2.2, 1.8, 2.4, 1.7, 2.1, 1.9]

    mu = compute_mu_mle(data)
    sigma = compute_sigma_mle(data)
    ll = compute_log_likelihood(data, mu, sigma)

    print(f"μ_MLE  = {mu}")
    print(f"σ_MLE  = {sigma}")
    print(f"log L  = {ll}")

    # Kiểm tra với numpy
    print("\n--- Kiểm tra với numpy ---")
    print(f"np.mean = {np.mean(data):.6f}")
    print(f"np.std  = {np.std(data, ddof=0):.6f}")
