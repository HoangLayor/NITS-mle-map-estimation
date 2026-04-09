# 📊 MLE & MAP Learning Repository

Repo học tập cho nhóm — tập trung vào **Maximum Likelihood Estimation (MLE)** và **Maximum A Posteriori (MAP)**.

---

## 🗂️ Cấu trúc repo

```
mle-map-learning/
├── docs/                        # Tài liệu lý thuyết
│   ├── 01_mle_theory.md
│   ├── 02_map_theory.md
│   └── 03_mle_vs_map.md
│
├── src/                         # Code nền tảng (đã viết sẵn, mọi người dùng chung)
│   ├── mle/
│   │   ├── gaussian.py          # MLE cho Gaussian distribution
│   │   ├── bernoulli.py         # MLE cho Bernoulli/Binomial
│   │   └── base.py              # Abstract base class
│   ├── map/
│   │   ├── gaussian_prior.py    # MAP với Gaussian prior
│   │   ├── beta_prior.py        # MAP với Beta prior
│   │   └── base.py
│   └── utils/
│       ├── plotting.py          # Hàm vẽ đồ thị tái sử dụng
│       └── data_generator.py    # Sinh data giả lập
│
├── notebooks/                   # Jupyter notebooks minh họa
│   ├── 01_mle_intro.ipynb
│   ├── 02_map_intro.ipynb
│   └── 03_comparison.ipynb
│
├── exercises/                   # Bài tập
│   ├── beginner/                # Bài tập cơ bản (điền vào chỗ trống)
│   └── intermediate/            # Bài tập nâng cao
│
├── tests/                       # Test tự động chấm điểm
│   ├── test_mle.py
│   └── test_map.py
│
├── solutions/                   # Đáp án (chỉ team lead xem)
│
├── requirements.txt
├── .github/workflows/           # GitHub Actions tự chạy test
│   └── check_exercises.yml
└── CONTRIBUTING.md
```

---

## 🚀 Bắt đầu nhanh

### 1. Clone repo
```bash
git clone https://github.com/<your-org>/mle-map-learning.git
cd mle-map-learning
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

### 3. Chạy thử
```python
from src.mle.gaussian import GaussianMLE

estimator = GaussianMLE()
estimator.fit([2.1, 1.9, 2.3, 2.0, 2.2])
print(estimator.mu, estimator.sigma)
```

### 4. Làm bài tập
Vào thư mục `exercises/beginner/` và đọc hướng dẫn trong từng file.

---

## 📋 Quy trình làm bài

```
1. Đọc lý thuyết trong docs/
2. Chạy notebook tương ứng để xem ví dụ
3. Làm bài tập trong exercises/
4. Chạy pytest để kiểm tra kết quả
5. Tạo Pull Request để team lead review
```

---

## ✅ Chạy tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cho MLE
pytest tests/test_mle.py -v

# Chạy test cho MAP
pytest tests/test_map.py -v
```

---

## 👥 Thành viên nhóm

| Tên | GitHub | Tiến độ |
|-----|--------|---------|
| (Team lead) | @... | ✅ |
| ... | @... | 🔄 |

---

## 📚 Tài liệu thêm

- [Pattern Recognition and Machine Learning – Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [Machine Learnin cơ bản](https://machinelearningcoban.com)
