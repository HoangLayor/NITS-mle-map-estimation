"""
Base class cho tất cả MLE estimators.
Mọi estimator đều kế thừa từ class này.
"""

from abc import ABC, abstractmethod


class BaseMLEEstimator(ABC):
    """
    Abstract base class cho Maximum Likelihood Estimation.

    Cách dùng:
        class MyEstimator(BaseMLEEstimator):
            def fit(self, data): ...
            def log_likelihood(self, data): ...
    """

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, data):
        """Ước lượng tham số từ data bằng MLE."""
        pass

    @abstractmethod
    def log_likelihood(self, data):
        """Tính log-likelihood tại tham số hiện tại."""
        pass

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Chưa gọi fit(). Hãy gọi fit(data) trước.")
