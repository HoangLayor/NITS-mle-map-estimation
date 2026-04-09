"""
MAP cho Bernoulli với Beta prior.

Lý thuyết:
    Prior:       p ~ Beta(α, β)       ← "đã thấy" α-1 thành công, β-1 thất bại
    Likelihood:  X | p ~ Bernoulli(p)

    Posterior:   p | X ~ Beta(α + k, β + n - k)

    Nghiệm MAP = mode của posterior:
        p_MAP = (α + k - 1) / (α + β + n - 2)

    So sánh:
        p_MLE = k / n                           (chỉ nhìn data)
        p_MAP = (α + k - 1) / (α + β + n - 2)  (kết hợp prior)

    Trực giác về α, β:
        Beta(1, 1) → prior phẳng (uniform), không bias
        Beta(2, 2) → tin rằng p ≈ 0.5
        Beta(5, 2) → tin rằng p hơi cao
        Khi n >> α+β: MAP → MLE
"""

import numpy as np
from .base import BaseMAPEstimator


class BernoulliMAPEstimator(BaseMAPEstimator):
    """
    MAP estimation cho p của Bernoulli với Beta prior.

    Args:
        alpha (float): Tham số α của Beta prior (> 0)
        beta  (float): Tham số β của Beta prior (> 0)

    Example:
        >>> est = BernoulliMAPEstimator(alpha=2, beta=2)
        >>> est.fit([1, 1, 0])  # 2 thành công / 3 lần
        >>> print(f"MAP p = {est.p:.3f}")  # kéo về 0.5 do prior
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha và beta phải > 0")
        self.alpha = alpha
        self.beta = beta
        self.p = None

    def fit(self, data):
        """
        Tính p_MAP từ data nhị phân.

        Args:
            data: list hoặc array gồm 0 và 1

        Returns:
            self
        """
        x = np.array(data, dtype=float)
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Data phải là nhị phân (0 hoặc 1).")

        n = len(x)
        k = int(np.sum(x))  # số thành công

        # MAP = mode của Beta posterior
        # Chỉ tính được khi α + k > 1 và β + (n-k) > 1
        numerator = self.alpha + k - 1
        denominator = self.alpha + self.beta + n - 2

        if denominator <= 0:
            raise ValueError("α + β + n - 2 phải > 0. Thêm data hoặc tăng α, β.")

        self.p = numerator / denominator
        self.p = float(np.clip(self.p, 1e-10, 1 - 1e-10))

        self._n = n
        self._k = k
        self._fitted = True
        return self

    def log_prior(self):
        """
        Tính log P(p) — log Beta prior.

        Returns:
            float
        """
        self._check_fitted()
        from scipy.special import betaln
        return (
            (self.alpha - 1) * np.log(self.p)
            + (self.beta - 1) * np.log(1 - self.p)
            - betaln(self.alpha, self.beta)
        )

    def log_posterior(self, data):
        """
        Tính log-posterior = log-likelihood + log-prior.

        Returns:
            float
        """
        self._check_fitted()
        x = np.array(data, dtype=float)
        k = np.sum(x)
        n = len(x)
        eps = 1e-12
        p = np.clip(self.p, eps, 1 - eps)

        log_lik = k * np.log(p) + (n - k) * np.log(1 - p)
        return log_lik + self.log_prior()

    def summary(self, data=None):
        """In so sánh MAP vs MLE."""
        self._check_fitted()
        print("=" * 45)
        print("  Bernoulli MAP — Kết quả")
        print("=" * 45)
        print(f"  Prior: Beta(α={self.alpha}, β={self.beta})")
        print(f"  Data:  n={self._n},  k={self._k} thành công")
        print("-" * 45)
        mle_p = self._k / self._n if self._n > 0 else float("nan")
        print(f"  p_MLE = {mle_p:.6f}  (chỉ nhìn data)")
        print(f"  p_MAP = {self.p:.6f}  (kết hợp prior)")
        print(f"  Posterior: Beta({self.alpha + self._k:.0f}, {self.beta + self._n - self._k:.0f})")
        print("=" * 45)
