"""
Bài tập 3: MAP cho Bernoulli (Beta prior)
═══════════════════════════════════════════════════════════

Tình huống:
    Bạn chỉ tung đồng xu 3 lần và được 3 lần ngửa (1, 1, 1).
    MLE sẽ nói p = 1.0 — nhưng điều đó có hợp lý không?
    MAP với prior giúp "kéo" ước lượng về thực tế hơn.

Lý thuyết:
    Prior:     p ~ Beta(α, β)
    Posterior: p | X ~ Beta(α + k, β + n - k)
    MAP:       p_MAP = (α + k - 1) / (α + β + n - 2)

Hướng dẫn:
    Chạy: pytest tests/test_map.py::test_ex03 -v
"""

import numpy as np


def compute_p_map(data, alpha, beta):
    """
    Tính p_MAP cho Bernoulli với Beta(alpha, beta) prior.

    Args:
        data: list nhị phân (0/1)
        alpha (float): tham số α của Beta prior (> 0)
        beta  (float): tham số β của Beta prior (> 0)

    Returns:
        float: p_MAP

    Công thức:
        p_MAP = (α + k - 1) / (α + β + n - 2)
        với k = số lần xuất hiện 1

    Gợi ý: tính n và k trước, sau đó áp dụng công thức
    """
    x = np.array(data, dtype=float)
    n = len(x)
    k = int(np.sum(x))

    # TODO
    p_map = (alpha + k - 1) / (alpha + beta + n - 2)

    return p_map


def compare_mle_map(data, alpha, beta):
    """
    So sánh p_MLE và p_MAP, trả về dict kết quả.

    Args:
        data: list nhị phân
        alpha, beta: tham số prior

    Returns:
        dict với keys: 'p_mle', 'p_map', 'difference'

    Gợi ý:
        p_mle = sum(data) / len(data)
        p_map = compute_p_map(data, alpha, beta)
        difference = abs(p_mle - p_map)
    """
    # TODO
    p_mle = np.sum(data) / len(data)
    p_map = compute_p_map(data, alpha, beta)
    difference = abs(p_mle - p_map)

    result = {
    "p_mle": p_mle,
    "p_map": p_map,
    "difference": difference,
}
    return result
def effect_of_sample_size(alpha=2, beta=2, true_p=0.7):
    """
    (Câu hỏi khám phá — không có test tự động)

    Quan sát: khi n tăng, MAP → MLE hay không?

    Returns:
        dict: {n: (p_mle, p_map)} cho n trong [3, 10, 50, 200]
    """
    rng = np.random.default_rng(0)
    results = {}

    for n in [3, 10, 50, 200]:
        data = rng.binomial(1, true_p, size=n).tolist()
        p_mle = float(np.mean(data))
        p_map = compute_p_map(data, alpha, beta)
        results[n] = (round(p_mle, 4), round(float(p_map), 4) if p_map is not None else None)

    return results


# ───────────────────────────────────────────────
# Thử nghiệm nhanh
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # Tình huống: 3 lần tung, 3 lần ngửa
    data = [1, 1, 1]

    p_mle = np.mean(data)
    p_map = compute_p_map(data, alpha=2, beta=2)

    print("Tung đồng xu 3 lần, cả 3 lần ngửa:")
    print(f"  p_MLE = {p_mle:.3f}  ← quá tự tin!")
    print(f"  p_MAP = {p_map}  ← hợp lý hơn với prior Beta(2,2)")

    print("\nHiệu ứng của cỡ mẫu:")
    sizes = effect_of_sample_size()
    for n, (mle, map_) in sizes.items():
        print(f"  n={n:4d}: MLE={mle:.3f}, MAP={map_}")