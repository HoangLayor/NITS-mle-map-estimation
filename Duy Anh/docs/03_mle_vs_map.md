# 03 — MLE vs MAP: Bảng so sánh

## Tổng quan

| | MLE | MAP |
|---|---|---|
| **Tối ưu** | log P(X\|θ) | log P(X\|θ) + log P(θ) |
| **Cần prior?** | Không | Có |
| **n nhỏ** | Dễ overfit | Ổn định hơn |
| **n lớn** | Tốt | ≈ MLE |
| **Kết quả** | Điểm ước lượng | Điểm ước lượng |
| **Bayesian?** | Không | Một phần |

---

## Khi nào chúng bằng nhau?

MAP = MLE khi:
1. **Prior phẳng (uniform)**: log P(θ) = const → không ảnh hưởng
2. **n → ∞**: Data áp đảo prior

Ví dụ: Beta(1,1) prior cho Bernoulli:
```
p_MAP = (1 + k - 1) / (1 + 1 + n - 2) = k / n = p_MLE ✓
```

---

## Ví dụ minh họa: 3 lần tung đồng xu

```python
data = [1, 1, 1]  # 3 lần ngửa liên tiếp

# MLE
p_mle = 3/3 = 1.0  # "đồng xu chắc chắn ngửa" — quá tự tin!

# MAP với Beta(2, 2)
p_map = (2+3-1)/(2+2+3-2) = 4/5 = 0.8  # hợp lý hơn

# MAP với Beta(5, 5)  — prior mạnh hơn
p_map = (5+3-1)/(5+5+3-2) = 7/11 ≈ 0.636  # kéo về 0.5 nhiều hơn
```

---

## Liên hệ với Machine Learning

| Khái niệm ML | Tương đương Bayesian |
|---|---|
| L2 regularization (Ridge) | MAP với Gaussian prior |
| L1 regularization (Lasso) | MAP với Laplace prior |
| Dropout | Approximate Bayesian inference |
| Early stopping | Implicit regularization |

---

## Bước tiếp theo sau MLE/MAP

```
MLE → MAP → Full Bayesian Inference
                    ↓
             Tính toàn bộ posterior P(θ|X)
             (không chỉ mode)
                    ↓
              MCMC, Variational Inference
```
