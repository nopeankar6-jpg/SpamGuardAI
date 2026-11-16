# SpamGuard AI – Phần SV1 

## 1. Cài đặt

Cần Python 3.10+ (khuyến nghị 3.11/3.12). Mở thư mục này bằng VS Code, chạy:

```bash
python -m pip install -r requirements.txt
```

## 2. Tải tập dữ liệu

Nguồn: https://spamassassin.apache.org/old/publiccorpus/

- `20021010_easy_ham.tar.bz2`
- `20030228_spam.tar.bz2`

Đặt hai file tại `data/raw/`. Nếu bản tải lên tên là `20021010_easy_ham_tar.bz2` và `20030228_spam_tar.bz2` thì **cứ giữ nguyên**: `preprocess.py` nhận cả hai kiểu tên nhờ tìm `*.bz2`. Không cần giải nén.

## 3. Chạy phần SV1

```bash
python preprocess.py
python data_analysis.py
```

Sau đó có các file:

| File | Nội dung |
|---|---|
| `data/emails_clean.csv` | Dữ liệu sạch, 4 cột `subject`, `body`, `text`, `label` |
| `data/preprocess_stats.json` | Số email đọc được, email trùng và mâu thuẫn bị loại, thống kê nhãn |
| `data/label_distribution.png` | Biểu đồ số email mỗi nhãn |
| `data/length_distribution.png` | Histogram log10 độ dài văn bản |
| `data/data_report.md` | Báo cáo thống kê thực tế, có số email bị loại |

Mở `data_analysis.ipynb` bằng VS Code/Jupyter, chọn kernel Python có cài các thư viện. Notebook có thể chạy nếu thư mục làm việc hiện tại là thư mục dự án hoặc một thư mục con của dự án (notebook tự dò về thư mục gốc).

## 4. SV2 gọi lại hàm <bàn giao>

```python
from preprocess import clean_text, extract_email, prepare_input, load_clean_data

print(clean_text("  hello  \n world  "))

raw = b"Subject: Class meeting\nContent-Type: text/plain; charset=utf-8\n\nMeeting at 9 AM."
subject, body, text = extract_email(raw)
print(subject, body, text)

# Dùng khi người dùng nhập tiêu đề và nội dung vào giao diện
text = prepare_input("Class meeting", "Meeting at 9 AM.")

df = load_clean_data("data/emails_clean.csv")
print(df[["text", "label"]].head())
```

**Chia train/test (SV2 thực hiện):**

```python
from sklearn.model_selection import train_test_split
from preprocess import load_clean_data

df = load_clean_data()
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2,
    stratify=df["label"], random_state=42
)
```

SV2 phải fit TF-IDF trên train *sau khi chia*, và báo cáo recall/F1 của lớp spam. Không có phần huấn luyện mô hình trong gói SV1 này.

## 5. Chạy giao diện Web (Streamlit - SV3)

Sau khi SV2 huấn luyện và tạo ra các file mô hình trong thư mục `models/` (`naive_bayes.pkl` và `svm.pkl`), khởi chạy ứng dụng web bằng lệnh:

```bash
python -m streamlit run app.py
```

Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.

Các tính năng trên giao diện:
- **Phân loại đơn lẻ:** Nhập Tiêu đề (Subject) và Nội dung (Body) email, chọn mô hình (Naive Bayes hoặc SVM), bấm **Phân loại** để nhận diện `Spam` hoặc `Ham`.
- **Phân loại hàng loạt:** Tải lên tệp CSV (yêu cầu cột `text` hoặc cặp cột `subject` và `body`), xem trước dữ liệu, phân loại toàn bộ và tải về tệp kết quả CSV kèm biểu đồ thống kê.

## 6. Kiểm thử riêng mà không cần email thật

```bash
python -m unittest discover -s tests -v
```

Bài test tự tạo corpus nhỏ trong thư mục tạm (không dùng tài liệu thực) để kiểm tra đọc `_tar.bz2`, loại trùng, nhãn mâu thuẫn, `extract_email()` và CSV.

## 7. Giới hạn

Hai file email gốc **không nằm trong gói mã nguồn**. Xem `DATA_SOURCE.md` để biết nguồn, thông tin bản quyền, hạn chế lệch nguồn/nhãn và thống kê chạy thử mà nhóm cung cấp.
