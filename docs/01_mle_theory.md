# 01 — Maximum Likelihood Estimation (MLE)

## Ý tưởng cốt lõi

> **MLE tìm tham số θ sao cho data quan sát được có khả năng xảy ra cao nhất.**

Nói đơn giản: nếu bạn quan sát được data X, MLE hỏi *"tham số nào của mô hình khiến data này có xác suất xuất hiện cao nhất?"*

---

## Công thức

Likelihood:
```
L(θ | X) = P(X | θ) = ∏ P(xi | θ)
```

Vì tích nhiều số nhỏ dễ bị tràn số, ta thường dùng **log-likelihood**:
```
log L(θ | X) = Σ log P(xi | θ)
```

MLE tìm:
```
θ_MLE = argmax_θ  log L(θ | X)
```

---

## Ví dụ 1: Gaussian

Data: X = {1.2, 0.8, 1.1, 0.9, 1.0} ~ N(μ, σ²)

**Nghiệm dạng đóng** (đạo hàm và cho bằng 0):
```
μ_MLE  = (1/n) Σ xi            = trung bình mẫu
σ²_MLE = (1/n) Σ (xi - μ)²    = phương sai mẫu
```

Code:
```python
import numpy as np
data = [1.2, 0.8, 1.1, 0.9, 1.0]
mu    = np.mean(data)        # 1.0
sigma = np.std(data, ddof=0) # 0.1414...
```

---

## Ví dụ 2: Bernoulli

Data: X = {1, 0, 1, 1, 0} — tung đồng xu 5 lần, 3 lần ngửa

```
p_MLE = k / n = 3 / 5 = 0.6
```

Code:
```python
data = [1, 0, 1, 1, 0]
p_mle = sum(data) / len(data)  # 0.6
```

---

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Đơn giản, công thức rõ ràng | Dễ overfit khi n nhỏ |
| Hiệu quả khi n lớn | Không dùng được kiến thức trước |
| Không cần giả định về prior | p_MLE = 1.0 nếu 3/3 lần ngửa |

---

## Khi nào dùng MLE?

- Có nhiều data (n lớn)
- Không có thông tin trước về tham số
- Muốn kết quả đơn giản, dễ giải thích
