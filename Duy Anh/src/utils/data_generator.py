"""
Sinh data giả lập để thực hành MLE và MAP.
"""

import numpy as np


def make_gaussian(n=50, mu=2.0, sigma=1.0, seed=None):
    """
    Sinh n mẫu từ N(mu, sigma²).

    Args:
        n: số mẫu
        mu: trung bình thật
        sigma: độ lệch chuẩn thật
        seed: random seed để tái tạo kết quả

    Returns:
        numpy array shape (n,)

    Example:
        >>> data = make_gaussian(n=100, mu=5.0, sigma=2.0, seed=42)
        >>> print(data[:5])
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


def make_bernoulli(n=30, p=0.7, seed=None):
    """
    Sinh n mẫu từ Bernoulli(p).

    Args:
        n: số mẫu
        p: xác suất thành công thật
        seed: random seed

    Returns:
        numpy array shape (n,) gồm 0 và 1

    Example:
        >>> data = make_bernoulli(n=20, p=0.3, seed=0)
        >>> print(f"Tỉ lệ 1 trong data: {data.mean():.2f}")
    """
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, size=n).astype(float)


def make_small_sample(distribution="gaussian", seed=42):
    """
    Sinh sample nhỏ để quan sát sự khác biệt MLE vs MAP rõ hơn.
    (n nhỏ → prior ảnh hưởng mạnh hơn)

    Args:
        distribution: 'gaussian' hoặc 'bernoulli'
        seed: random seed

    Returns:
        numpy array
    """
    if distribution == "gaussian":
        return make_gaussian(n=5, mu=3.0, sigma=1.0, seed=seed)
    elif distribution == "bernoulli":
        return make_bernoulli(n=5, p=0.8, seed=seed)
    else:
        raise ValueError("distribution phải là 'gaussian' hoặc 'bernoulli'")
