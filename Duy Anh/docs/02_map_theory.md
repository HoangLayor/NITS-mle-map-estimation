# 02 — Maximum A Posteriori (MAP)

## Ý tưởng cốt lõi

> **MAP = MLE + kiến thức trước (prior)**

Thay vì chỉ nhìn vào data, MAP kết hợp thêm niềm tin ban đầu của ta về tham số.

---

## Công thức (Bayes' theorem)

```
P(θ | X) ∝ P(X | θ) · P(θ)
 posterior   likelihood  prior
```

MAP tìm:
```
θ_MAP = argmax_θ  [log P(X | θ) + log P(θ)]
                     log-likelihood  log-prior
```

So với MLE chỉ có: `argmax_θ  log P(X | θ)`

---

## Ví dụ 1: Bernoulli với Beta prior

**Bài toán**: Tung đồng xu 3 lần, cả 3 đều ngửa.

MLE sẽ nói: `p = 3/3 = 1.0` — quá tự tin!

**Với Beta(2, 2) prior** (tin rằng p ≈ 0.5):
```
p_MAP = (α + k - 1) / (α + β + n - 2)
      = (2 + 3 - 1) / (2 + 2 + 3 - 2)
      = 4 / 5
      = 0.8
```

Hợp lý hơn nhiều — MAP "kéo" ước lượng về phía prior.

Code:
```python
alpha, beta = 2, 2
k, n = 3, 3
p_map = (alpha + k - 1) / (alpha + beta + n - 2)  # 0.8
```

---

## Ví dụ 2: Gaussian với Gaussian prior

Prior: μ ~ N(μ₀, τ²)  
Likelihood: x | μ ~ N(μ, σ²)

Nghiệm MAP:
```
μ_MAP = (n/σ² · x̄ + 1/τ² · μ₀) / (n/σ² + 1/τ²)
```

Đây là **trung bình có trọng số** giữa x̄ (data) và μ₀ (prior), trọng số là precision (1/variance).

---

## Trực giác về prior

| Prior | Ý nghĩa |
|-------|---------|
| Beta(1, 1) | Không bias — uniform, bằng MLE |
| Beta(2, 2) | Tin p ≈ 0.5 nhẹ nhàng |
| Beta(10, 10) | Rất tin p ≈ 0.5 |
| Beta(5, 2) | Tin p hơi cao |

**Quy tắc:** n >> (α + β) → MAP ≈ MLE (data thắng prior)

---

## MAP = MLE + Regularization

MAP với Gaussian prior ↔ **Ridge Regression** (L2 regularization):
```
MAP objective = log-likelihood - λ · ||θ||²
              = MLE            + log-prior (Gaussian)
```

Đây là lý do regularization có nền tảng Bayesian!

---

## Khi nào dùng MAP?

- Data ít (n nhỏ), MLE không đáng tin
- Có kiến thức trước về tham số
- Muốn tránh overfitting
- Làm bước đệm trước khi học full Bayesian inference
