"""
MAP cho phân phối Gaussian với Gaussian prior trên μ.

Lý thuyết:
    Prior:       μ ~ N(μ₀, τ²)
    Likelihood:  X | μ ~ N(μ, σ²)  với σ² đã biết

    MAP tìm μ để maximize log-posterior:
        log P(μ | X) ∝ log P(X | μ) + log P(μ)

    Nghiệm dạng đóng:
                    n/σ²  * x̄  +  1/τ²  * μ₀
        μ_MAP = ─────────────────────────────
                         n/σ²  +  1/τ²

    Trực giác:
        - Khi n lớn: μ_MAP → μ_MLE = x̄  (data thắng)
        - Khi n nhỏ: μ_MAP → μ₀          (prior thắng)
        - τ² → ∞   : μ_MAP → μ_MLE       (prior yếu, không bias)
"""

import numpy as np
from .base import BaseMAPEstimator


class GaussianMAPEstimator(BaseMAPEstimator):
    """
    MAP estimation cho μ của Gaussian với Gaussian prior.

    Args:
        mu0 (float): Trung bình của prior (niềm tin ban đầu về μ)
        tau (float): Độ lệch chuẩn của prior (càng lớn → prior càng yếu)
        sigma (float): Độ lệch chuẩn của likelihood (giả sử đã biết)

    Example:
        >>> est = GaussianMAPEstimator(mu0=0.0, tau=1.0, sigma=2.0)
        >>> est.fit([0.5, 1.2, 0.8])
        >>> print(f"MAP mu = {est.mu:.3f}")
    """

    def __init__(self, mu0: float = 0.0, tau: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.mu0 = mu0    # prior mean
        self.tau = tau    # prior std
        self.sigma = sigma  # likelihood std (known)
        self.mu = None    # MAP estimate (sau khi fit)

    def fit(self, data):
        """
        Tính μ_MAP từ data.

        Args:
            data: list hoặc array số thực

        Returns:
            self
        """
        x = np.array(data, dtype=float)
        n = len(x)
        x_bar = np.mean(x)

        # Precision (độ chính xác) = 1/variance
        precision_likelihood = n / self.sigma**2
        precision_prior = 1 / self.tau**2

        # Công thức MAP (trung bình có trọng số theo precision)
        self.mu = (
            precision_likelihood * x_bar + precision_prior * self.mu0
        ) / (precision_likelihood + precision_prior)

        self._fitted = True
        return self

    def log_prior(self):
        """
        Tính log P(μ) — log prior Gaussian.

        Returns:
            float: giá trị log-prior
        """
        self._check_fitted()
        return (
            -0.5 * np.log(2 * np.pi * self.tau**2)
            - (self.mu - self.mu0)**2 / (2 * self.tau**2)
        )

    def log_posterior(self, data):
        """
        Tính log-posterior = log-likelihood + log-prior.

        Args:
            data: list hoặc array số thực

        Returns:
            float: log-posterior (chưa chuẩn hóa)
        """
        self._check_fitted()
        x = np.array(data, dtype=float)
        n = len(x)

        log_lik = (
            -n / 2 * np.log(2 * np.pi * self.sigma**2)
            - np.sum((x - self.mu)**2) / (2 * self.sigma**2)
        )
        return log_lik + self.log_prior()

    def summary(self, data=None):
        """In so sánh MAP vs MLE."""
        self._check_fitted()
        mle_mu = np.mean(data) if data is not None else "N/A"

        print("=" * 45)
        print("  Gaussian MAP — Kết quả")
        print("=" * 45)
        print(f"  Prior: μ₀ = {self.mu0},  τ = {self.tau}")
        print(f"  Likelihood σ = {self.sigma}")
        print("-" * 45)
        if data is not None:
            print(f"  μ_MLE     = {float(mle_mu):.6f}  (chỉ nhìn data)")
        print(f"  μ_MAP     = {self.mu:.6f}  (kết hợp prior)")
        if data is not None:
            print(f"  Shift về prior: {abs(self.mu - float(mle_mu)):.6f}")
        print("=" * 45)
