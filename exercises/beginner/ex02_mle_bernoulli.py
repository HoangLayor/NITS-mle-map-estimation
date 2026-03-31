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
    x = np.array(data, dtype=float)
    return np.sum(x) / len(x)


def compute_bernoulli_log_likelihood(data, p):
    x = np.array(data, dtype=float)
    k = np.sum(x)
    n = len(x)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return k * np.log(p) + (n - k) * np.log(1 - p)


def find_best_p(data, p_candidates):
    best_p = None
    best_ll = -np.inf
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
