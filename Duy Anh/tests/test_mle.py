"""
Tests tự động cho MLE exercises.
Chạy: pytest tests/test_mle.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exercises.beginner.ex01_mle_gaussian import (
    compute_mu_mle,
    compute_sigma_mle,
    compute_log_likelihood,
)
from exercises.beginner.ex02_mle_bernoulli import (
    compute_p_mle,
    compute_bernoulli_log_likelihood,
    find_best_p,
)
from src.mle import GaussianMLE, BernoulliMLE


# ─── Fixtures ─────────────────────────────────

@pytest.fixture
def gaussian_data():
    """Data cố định để test ổn định."""
    rng = np.random.default_rng(42)
    return rng.normal(2.0, 1.5, size=100).tolist()

@pytest.fixture
def bernoulli_data():
    rng = np.random.default_rng(7)
    return rng.binomial(1, 0.65, size=50).tolist()


# ─── Bài tập 01: MLE Gaussian ─────────────────

class TestEx01:
    def test_mu_mle_correct(self, gaussian_data):
        result = compute_mu_mle(gaussian_data)
        assert result is not None, "compute_mu_mle chưa được implement (đang trả None)"
        expected = np.mean(gaussian_data)
        assert abs(result - expected) < 1e-8, (
            f"μ_MLE sai: kết quả={result:.6f}, đúng={expected:.6f}"
        )

    def test_sigma_mle_correct(self, gaussian_data):
        result = compute_sigma_mle(gaussian_data)
        assert result is not None, "compute_sigma_mle chưa được implement"
        expected = np.std(gaussian_data, ddof=0)
        assert abs(result - expected) < 1e-8, (
            f"σ_MLE sai: kết quả={result:.6f}, đúng={expected:.6f}"
        )

    def test_sigma_positive(self, gaussian_data):
        result = compute_sigma_mle(gaussian_data)
        if result is not None:
            assert result > 0, "σ_MLE phải > 0"

    def test_log_likelihood_at_mle_is_maximum(self, gaussian_data):
        """Log-likelihood tại MLE phải lớn hơn tại điểm bất kỳ khác."""
        mu_mle = compute_mu_mle(gaussian_data)
        sigma_mle = compute_sigma_mle(gaussian_data)

        if mu_mle is None or sigma_mle is None:
            pytest.skip("Chưa implement mu/sigma")

        ll_mle = compute_log_likelihood(gaussian_data, mu_mle, sigma_mle)
        ll_other = compute_log_likelihood(gaussian_data, mu_mle + 1.0, sigma_mle)

        assert ll_mle is not None, "compute_log_likelihood chưa implement"
        assert ll_mle > ll_other, (
            "log-likelihood tại MLE phải lớn hơn tại điểm khác"
        )

    def test_matches_library(self, gaussian_data):
        """Kết quả phải khớp với GaussianMLE trong src/."""
        est = GaussianMLE().fit(gaussian_data)
        mu = compute_mu_mle(gaussian_data)
        sigma = compute_sigma_mle(gaussian_data)

        if mu is None or sigma is None:
            pytest.skip("Chưa implement")

        assert abs(mu - est.mu) < 1e-8
        assert abs(sigma - est.sigma) < 1e-8


# ─── Bài tập 02: MLE Bernoulli ─────────────────

class TestEx02:
    def test_p_mle_correct(self, bernoulli_data):
        result = compute_p_mle(bernoulli_data)
        assert result is not None, "compute_p_mle chưa implement"
        expected = np.mean(bernoulli_data)
        assert abs(result - expected) < 1e-8, (
            f"p_MLE sai: kết quả={result:.4f}, đúng={expected:.4f}"
        )

    def test_p_in_range(self, bernoulli_data):
        result = compute_p_mle(bernoulli_data)
        if result is not None:
            assert 0 <= result <= 1, "p_MLE phải nằm trong [0, 1]"

    def test_log_likelihood_finite(self, bernoulli_data):
        p = compute_p_mle(bernoulli_data)
        if p is None:
            pytest.skip("Chưa implement p_mle")
        ll = compute_bernoulli_log_likelihood(bernoulli_data, p)
        assert ll is not None, "log_likelihood chưa implement"
        assert np.isfinite(ll), "log-likelihood phải là số hữu hạn"

    def test_find_best_p_close_to_mle(self, bernoulli_data):
        p_mle = np.mean(bernoulli_data)
        candidates = np.arange(0.05, 1.0, 0.05)
        best = find_best_p(bernoulli_data, candidates)
        assert best is not None, "find_best_p chưa implement"
        # Best p trong grid phải gần p_mle (sai số tối đa 0.1)
        assert abs(best - p_mle) < 0.1, (
            f"find_best_p={best:.2f} cách xa MLE={p_mle:.2f}"
        )

    def test_all_ones(self):
        """Khi tất cả là 1 thì p_MLE = 1.0."""
        data = [1] * 10
        result = compute_p_mle(data)
        if result is not None:
            assert abs(result - 1.0) < 1e-8

    def test_all_zeros(self):
        """Khi tất cả là 0 thì p_MLE = 0.0."""
        data = [0] * 10
        result = compute_p_mle(data)
        if result is not None:
            assert abs(result - 0.0) < 1e-8

    def test_matches_library(self, bernoulli_data):
        est = BernoulliMLE().fit(bernoulli_data)
        p = compute_p_mle(bernoulli_data)
        if p is not None:
            assert abs(p - est.p) < 1e-8
