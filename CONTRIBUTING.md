# Hướng dẫn đóng góp cho nhóm

## Quy trình làm bài tập

```
main branch (team lead)
    └── feature/ten-ban/ex01   ← mỗi người tạo branch riêng
    └── feature/ten-ban/ex02
    └── feature/ten-ban/ex03
```

### Bước 1: Clone repo và tạo branch

```bash
git clone https://github.com/<your-org>/mle-map-learning.git
cd mle-map-learning
git checkout -b feature/<ten-ban>/ex01
```

### Bước 2: Làm bài tập

Mở file trong `exercises/beginner/`, điền vào chỗ có `# TODO`.

### Bước 3: Chạy test cục bộ

```bash
pip install -r requirements.txt
pytest tests/test_mle.py::TestEx01 -v   # bài 01
pytest tests/test_mle.py::TestEx02 -v   # bài 02
pytest tests/test_map.py::TestEx03 -v   # bài 03
```

Kết quả xanh = đúng ✅ | Đỏ = sai, đọc thông báo lỗi để sửa ❌

### Bước 4: Commit và push

```bash
git add exercises/beginner/ex01_mle_gaussian.py
git commit -m "ex01: implement compute_mu_mle and compute_sigma_mle"
git push origin feature/<ten-ban>/ex01
```

### Bước 5: Tạo Pull Request

Vào GitHub → New Pull Request → base: `main` ← compare: `feature/<ten-ban>/ex01`

GitHub Actions sẽ tự động chạy tests. Team lead sẽ review và merge.

---

## Quy tắc commit message

```
ex01: mô tả ngắn gọn việc đã làm
ex02: implement compute_p_mle
ex03: fix công thức MAP, thêm test case
```

## Không được phép

- Copy đáp án từ `solutions/`
- Hardcode kết quả vào hàm (ví dụ `return 1.0` để qua test)
- Sửa file trong `tests/` hoặc `src/`
