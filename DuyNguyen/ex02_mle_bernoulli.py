"""
Bài tập 2: MLE cho Bernoulli
═══════════════════════════════════════════════════════════

Tình huống thực tế:
    Bạn tung đồng xu 20 lần, được 13 lần mặt ngửa.
    Hỏi: xác suất ngửa p là bao nhiêu theo MLE?

Lý thuyết:
    p_MLE = k / n   (k = số thành công, n = số lần thử)

Hướng dẫn:
    Điền vào các hàm bên dưới.
    Chạy: pytest tests/test_mle.py::test_ex02 -v
"""

import numpy as np


def compute_p_mle(data):
    """
    Tính p_MLE cho phân phối Bernoulli.

    Args:
        data: list nhị phân (0 hoặc 1)

    Returns:
        float: p_MLE ∈ [0, 1]

    Gợi ý: p_MLE = số lần xuất hiện 1 / tổng số mẫu
    """
    x = np.array(data, dtype=float)

    # TODO
    p = np.sum(x) / len(x)

    return p


def compute_bernoulli_log_likelihood(data, p):
    """
    Tính log-likelihood của Bernoulli tại p.

    log L(p | X) = k * log(p) + (n-k) * log(1-p)

    Args:
        data: list nhị phân (0/1)
        p (float): xác suất thành công (0 < p < 1)

    Returns:
        float: log-likelihood

    Gợi ý: dùng np.log(), đếm k = np.sum(data)
    """
    x = np.array(data, dtype=float)
    n = len(x)
    k = int(np.sum(x))

    # TODO: tính log-likelihood
    log_lik = k * np.log(p) + (n - k) * np.log(1 - p)

    return log_lik


"""
Tìm p tốt nhất trong danh sách candidates bằng cách
so sánh log-likelihood.

(Đây là cách brute-force để "thấy" MLE hoạt động)

Args:
    data: list nhị phân
    p_candidates: list các giá trị p cần thử

Returns:
    float: p có log-likelihood cao nhất

Gợi ý: dùng vòng lặp for hoặc max() với key
"""
# TODO
def find_best_p(data, p_candidates):
    best_p = p_candidates[0]
    best_ll = compute_bernoulli_log_likelihood(data, p_candidates[0])

    for p in p_candidates:
        ll = compute_bernoulli_log_likelihood(data, p)
        if ll > best_ll:
            best_ll = ll
            best_p = p

    return best_p


# ───────────────────────────────────────────────
# Thử nghiệm nhanh
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # Tung đồng xu 20 lần, 13 lần ngửa
    data = [1]*13 + [0]*7
    np.random.shuffle(data)

    p = compute_p_mle(data)
    ll = compute_bernoulli_log_likelihood(data, p)

    print(f"Dữ liệu: {sum(data)} ngửa / {len(data)} lần")
    print(f"p_MLE   = {p:.4f}   (kỳ vọng: 0.65)")
    print(f"log L   = {ll:.4f}")

    # Tìm p tốt nhất trong grid
    candidates = np.arange(0.1, 1.0, 0.05)
    best = find_best_p(data, candidates)
    print(f"Best p (grid) = {best:.2f}")