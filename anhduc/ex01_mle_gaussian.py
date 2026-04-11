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


# ───────────────────────────────────────────────
# PHẦN 1: Tính μ_MLE
# ───────────────────────────────────────────────

def compute_mu_mle(data):
    """
    Tính μ_MLE (trung bình mẫu).

    Args:
        data (list hoặc np.ndarray): danh sách số thực

    Returns:
        float: μ_MLE

    Gợi ý: μ_MLE = tổng / số phần tử
    """
    x = np.array(data, dtype=float)

    # TODO: viết công thức tính μ_MLE
    mu = np.sum(x) / len(x)

    return mu


# ───────────────────────────────────────────────
# PHẦN 2: Tính σ_MLE
# ───────────────────────────────────────────────

def compute_sigma_mle(data):
    """
    Tính σ_MLE (độ lệch chuẩn MLE — KHÔNG phải unbiased).

    Args:
        data (list hoặc np.ndarray): danh sách số thực

    Returns:
        float: σ_MLE

    Gợi ý:
        σ²_MLE = (1/n) * Σ (xi - μ_MLE)²
        σ_MLE = sqrt(σ²_MLE)

    Lưu ý: np.std(x, ddof=0) cũng cho kết quả này,
            nhưng hãy tự tính bằng công thức!
    """
    x = np.array(data, dtype=float)

    # TODO: tính mu trước
    mu = np.sum(x) / len(x)

    # TODO: tính variance
    variance = np.sum((x - mu) ** 2) / len(x)

    # TODO: trả về sigma
    sigma = np.sqrt(variance)

    return sigma


# ───────────────────────────────────────────────
# PHẦN 3: Tính log-likelihood
# ───────────────────────────────────────────────

def compute_log_likelihood(data, mu, sigma):
    """
    Tính log-likelihood của Gaussian tại (μ, σ) cho data.

    log L(μ, σ | X) = -n/2 * log(2πσ²) - 1/(2σ²) * Σ(xi - μ)²

    Args:
        data: list hoặc np.ndarray
        mu (float): giá trị μ
        sigma (float): giá trị σ (> 0)

    Returns:
        float: log-likelihood

    Gợi ý: dùng np.log(), np.sum(), np.pi
    """
    x = np.array(data, dtype=float)
    n = len(x)

    # TODO: tính log-likelihood theo công thức
    log_lik = -n/2 * np.log(2 * np.pi * sigma**2) \
          - (1 / (2 * sigma**2)) * np.sum((x - mu) ** 2)

    return log_lik


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
