"""
Tests tự động cho MAP exercises.
Chạy: pytest tests/test_map.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exercises.beginner.ex03_map_bernoulli import compute_p_map, compare_mle_map
from src.map import BernoulliMAPEstimator


@pytest.fixture
def small_data():
    return [1, 1, 1]  # 3 lần ngửa — case cực đoan để thấy MAP vs MLE rõ

@pytest.fixture
def medium_data():
    rng = np.random.default_rng(10)
    return rng.binomial(1, 0.7, size=30).tolist()


# ─── Bài tập 03: MAP Bernoulli ─────────────────

class TestEx03:
    def test_not_none(self, small_data):
        result = compute_p_map(small_data, alpha=2, beta=2)
        assert result is not None, "compute_p_map chưa implement (đang trả None)"

    def test_p_map_less_than_mle_for_extreme_data(self, small_data):
        """Khi data cực đoan (3/3 thành công), MAP phải < MLE = 1.0."""
        p_mle = np.mean(small_data)  # = 1.0
        p_map = compute_p_map(small_data, alpha=2, beta=2)
        if p_map is None:
            pytest.skip("Chưa implement")
        assert p_map < p_mle, (
            f"MAP ({p_map:.3f}) phải nhỏ hơn MLE ({p_mle:.3f}) "
            "khi prior kéo về 0.5"
        )

    def test_p_map_in_range(self, medium_data):
        p_map = compute_p_map(medium_data, alpha=2, beta=2)
        if p_map is not None:
            assert 0 < p_map < 1, "p_MAP phải nằm trong (0, 1)"

    def test_correct_formula(self, medium_data):
        """Kiểm tra công thức đúng."""
        alpha, beta = 3, 3
        x = np.array(medium_data)
        n = len(x)
        k = int(np.sum(x))
        expected = (alpha + k - 1) / (alpha + beta + n - 2)

        result = compute_p_map(medium_data, alpha=alpha, beta=beta)
        if result is not None:
            assert abs(result - expected) < 1e-8, (
                f"Công thức sai: kết quả={result:.6f}, đúng={expected:.6f}"
            )

    def test_flat_prior_gives_mle(self):
        """Beta(1,1) là uniform prior → MAP = MLE."""
        data = [1, 0, 1, 1, 0]
        p_mle = np.mean(data)
        p_map = compute_p_map(data, alpha=1, beta=1)
        if p_map is not None:
            # Với prior phẳng: (1+k-1)/(1+1+n-2) = k/n = MLE
            assert abs(p_map - p_mle) < 1e-8, (
                "Prior Beta(1,1) phải cho kết quả bằng MLE"
            )

    def test_map_converges_to_mle_for_large_n(self):
        """Khi n lớn, MAP phải gần MLE."""
        rng = np.random.default_rng(99)
        data = rng.binomial(1, 0.65, size=500).tolist()
        p_mle = np.mean(data)
        p_map = compute_p_map(data, alpha=2, beta=2)
        if p_map is not None:
            assert abs(p_map - p_mle) < 0.01, (
                "Với n=500, MAP và MLE phải rất gần nhau (<0.01)"
            )

    def test_compare_mle_map_returns_dict(self, medium_data):
        result = compare_mle_map(medium_data, alpha=2, beta=2)
        assert result is not None
        assert "p_mle" in result
        assert "p_map" in result
        assert "difference" in result

    def test_compare_difference_correct(self, medium_data):
        result = compare_mle_map(medium_data, alpha=2, beta=2)
        if result["p_mle"] and result["p_map"]:
            expected_diff = abs(result["p_mle"] - result["p_map"])
            assert abs(result["difference"] - expected_diff) < 1e-8

    def test_matches_library(self, medium_data):
        """Kết quả phải khớp với BernoulliMAPEstimator trong src/."""
        alpha, beta = 2, 5
        est = BernoulliMAPEstimator(alpha=alpha, beta=beta).fit(medium_data)
        result = compute_p_map(medium_data, alpha=alpha, beta=beta)
        if result is not None:
            assert abs(result - est.p) < 1e-6
