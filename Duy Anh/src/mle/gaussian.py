"""
MLE cho phân phối Gaussian (Normal).

Lý thuyết:
    Cho data X = {x1, x2, ..., xn} ~ N(μ, σ²)

    MLE tìm μ, σ để maximize log-likelihood:
        log L(μ, σ | X) = -n/2 * log(2πσ²) - 1/(2σ²) * Σ(xi - μ)²

    Nghiệm dạng đóng:
        μ_MLE = (1/n) * Σ xi          (trung bình mẫu)
        σ²_MLE = (1/n) * Σ(xi - μ)²  (phương sai mẫu — MLE bị bias)
"""

import numpy as np
from .base import BaseMLEEstimator


class GaussianMLE(BaseMLEEstimator):
    """
    MLE cho phân phối Gaussian.

    Attributes:
        mu (float): Ước lượng MLE của trung bình μ
        sigma (float): Ước lượng MLE của độ lệch chuẩn σ

    Example:
        >>> est = GaussianMLE()
        >>> est.fit([1.2, 0.8, 1.1, 0.9, 1.0])
        >>> print(f"mu={est.mu:.3f}, sigma={est.sigma:.3f}")
        mu=1.000, sigma=0.141
    """

    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None

    def fit(self, data):
        """
        Ước lượng μ và σ từ data.

        Args:
            data: list hoặc array các số thực

        Returns:
            self (để có thể chain: est.fit(data).log_likelihood(data))
        """
        x = np.array(data, dtype=float)
        n = len(x)

        # MLE: nghiệm dạng đóng
        self.mu = np.mean(x)           # μ_MLE = trung bình mẫu
        self.sigma = np.std(x, ddof=0) # σ_MLE = std với ddof=0 (MLE, không unbiased)

        self._fitted = True
        return self

    def log_likelihood(self, data):
        """
        Tính log-likelihood tại (μ, σ) hiện tại.

        Args:
            data: list hoặc array các số thực

        Returns:
            float: giá trị log-likelihood
        """
        self._check_fitted()
        x = np.array(data, dtype=float)
        n = len(x)

        ll = (
            -n / 2 * np.log(2 * np.pi * self.sigma**2)
            - 1 / (2 * self.sigma**2) * np.sum((x - self.mu) ** 2)
        )
        return ll

    def summary(self):
        """In tóm tắt kết quả ước lượng."""
        self._check_fitted()
        print("=" * 35)
        print("  Gaussian MLE — Kết quả")
        print("=" * 35)
        print(f"  μ (mean)  = {self.mu:.6f}")
        print(f"  σ (std)   = {self.sigma:.6f}")
        print(f"  σ² (var)  = {self.sigma**2:.6f}")
        print("=" * 35)
