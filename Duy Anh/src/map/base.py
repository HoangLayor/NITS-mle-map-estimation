"""
Base class cho tất cả MAP estimators.
"""

from abc import ABC, abstractmethod


class BaseMAPEstimator(ABC):
    """
    Abstract base class cho Maximum A Posteriori Estimation.

    MAP = argmax_θ [ log P(X|θ) + log P(θ) ]
              = MLE + log prior
    """

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, data):
        """Ước lượng tham số từ data bằng MAP."""
        pass

    @abstractmethod
    def log_posterior(self, data):
        """Tính log-posterior (chưa chuẩn hóa) = log-likelihood + log-prior."""
        pass

    @abstractmethod
    def log_prior(self):
        """Tính log-prior tại tham số hiện tại."""
        pass

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Chưa gọi fit(). Hãy gọi fit(data) trước.")
