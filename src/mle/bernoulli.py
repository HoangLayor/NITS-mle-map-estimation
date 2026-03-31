"""
MLE cho phân phối Bernoulli.

Lý thuyết:
    Cho data X = {x1, ..., xn}, xi ∈ {0, 1} ~ Bernoulli(p)

    Log-likelihood:
        log L(p | X) = k*log(p) + (n-k)*log(1-p)
        với k = số lần xi = 1

    Nghiệm dạng đóng:
        p_MLE = k / n   (tỉ lệ thành công trong mẫu)
"""

import numpy as np
from .base import BaseMLEEstimator


class BernoulliMLE(BaseMLEEstimator):
    """
    MLE cho phân phối Bernoulli.

    Attributes:
        p (float): Ước lượng MLE của xác suất thành công

    Example:
        >>> est = BernoulliMLE()
        >>> est.fit([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        >>> print(f"p={est.p:.2f}")
        p=0.60
    """

    def __init__(self):
        super().__init__()
        self.p = None

    def fit(self, data):
        """
        Ước lượng p từ data nhị phân (0/1).

        Args:
            data: list hoặc array các giá trị 0 hoặc 1

        Returns:
            self
        """
        x = np.array(data, dtype=float)

        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Data phải là nhị phân (chỉ gồm 0 và 1).")

        # p_MLE = tỉ lệ số 1 trong mẫu
        self.p = np.mean(x)
        self._fitted = True
        return self

    def log_likelihood(self, data):
        """
        Tính log-likelihood tại p hiện tại.

        Args:
            data: list hoặc array các giá trị 0/1

        Returns:
            float: giá trị log-likelihood
        """
        self._check_fitted()
        x = np.array(data, dtype=float)
        k = np.sum(x)
        n = len(x)

        # Tránh log(0)
        eps = 1e-12
        p = np.clip(self.p, eps, 1 - eps)

        return k * np.log(p) + (n - k) * np.log(1 - p)

    def summary(self):
        """In tóm tắt kết quả."""
        self._check_fitted()
        print("=" * 35)
        print("  Bernoulli MLE — Kết quả")
        print("=" * 35)
        print(f"  p (prob)  = {self.p:.6f}")
        print(f"  1 - p     = {1 - self.p:.6f}")
        print("=" * 35)
