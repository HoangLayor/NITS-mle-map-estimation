"""
Hàm vẽ đồ thị tái sử dụng cho cả nhóm.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_mle_gaussian(data, ax=None, title="Gaussian MLE"):
    """
    Vẽ histogram của data và đường Gaussian MLE fit.

    Args:
        data: list hoặc array
        ax: matplotlib Axes (nếu None sẽ tạo mới)
        title: tiêu đề biểu đồ
    """
    x = np.array(data)
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(x, bins="auto", density=True, alpha=0.5, color="steelblue", label="Data")

    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
            "r-", lw=2, label=f"MLE: μ={mu:.2f}, σ={sigma:.2f}")

    ax.axvline(mu, color="red", linestyle="--", alpha=0.7, label=f"μ_MLE = {mu:.2f}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_mle_vs_map_bernoulli(data, alpha=2, beta=2, ax=None):
    """
    Vẽ Beta posterior và so sánh MLE vs MAP cho Bernoulli.

    Args:
        data: list nhị phân (0/1)
        alpha, beta: tham số Beta prior
        ax: matplotlib Axes
    """
    x = np.array(data)
    n = len(x)
    k = int(np.sum(x))

    p_mle = k / n
    alpha_post = alpha + k
    beta_post = beta + (n - k)
    p_map = (alpha + k - 1) / (alpha + beta + n - 2) if (alpha + beta + n - 2) > 0 else p_mle

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    p_range = np.linspace(0.001, 0.999, 500)

    # Prior
    ax.plot(p_range, stats.beta.pdf(p_range, alpha, beta),
            "gray", lw=1.5, linestyle="--", label=f"Prior: Beta({alpha}, {beta})", alpha=0.7)

    # Posterior
    ax.plot(p_range, stats.beta.pdf(p_range, alpha_post, beta_post),
            "steelblue", lw=2, label=f"Posterior: Beta({alpha_post}, {beta_post})")

    # MLE và MAP
    ax.axvline(p_mle, color="red", lw=2, linestyle="-",
               label=f"MLE: p = {p_mle:.3f}")
    ax.axvline(p_map, color="darkorange", lw=2, linestyle="-",
               label=f"MAP: p = {p_map:.3f}")

    ax.set_title(f"Bernoulli: MLE vs MAP  (n={n}, k={k} thành công)")
    ax.set_xlabel("p")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_mle_vs_map_gaussian(data, mu0=0.0, tau=1.0, sigma=1.0, ax=None):
    """
    Vẽ so sánh MLE vs MAP cho Gaussian với Gaussian prior.

    Args:
        data: list số thực
        mu0, tau: tham số prior
        sigma: độ lệch chuẩn likelihood (known)
        ax: matplotlib Axes
    """
    x = np.array(data)
    n = len(x)
    x_bar = np.mean(x)

    precision_lik = n / sigma**2
    precision_prior = 1 / tau**2
    mu_map = (precision_lik * x_bar + precision_prior * mu0) / (precision_lik + precision_prior)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Range vẽ
    center = (x_bar + mu0) / 2
    width = max(abs(x_bar - mu0) * 2, 3)
    mu_range = np.linspace(center - width, center + width, 500)

    # Prior
    ax.plot(mu_range, stats.norm.pdf(mu_range, mu0, tau),
            "gray", lw=1.5, linestyle="--", label=f"Prior: N({mu0}, {tau}²)", alpha=0.7)

    # Posterior precision và std
    post_precision = precision_lik + precision_prior
    post_std = 1 / np.sqrt(post_precision)
    ax.plot(mu_range, stats.norm.pdf(mu_range, mu_map, post_std),
            "steelblue", lw=2, label=f"Posterior (n={n})")

    ax.axvline(x_bar, color="red", lw=2, label=f"MLE: μ = {x_bar:.3f}")
    ax.axvline(mu_map, color="darkorange", lw=2, label=f"MAP: μ = {mu_map:.3f}")
    ax.axvline(mu0, color="gray", lw=1.5, linestyle=":",
               label=f"Prior mean: μ₀ = {mu0}")

    ax.set_title(f"Gaussian: MLE vs MAP  (n={n}, σ={sigma})")
    ax.set_xlabel("μ")
    ax.set_ylabel("Density")
    ax.legend()
    return ax
