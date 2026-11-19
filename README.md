# SpamGuard AI

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

## 5. Hướng dẫn demo giao diện Web (Streamlit - SV3)

### Chuẩn bị và khởi chạy

- Cài các thư viện trong `requirements.txt` theo mục 1.
- Đảm bảo đã có hai mô hình `models/naive_bayes.pkl` và `models/svm.pkl`.
  - Nếu chưa có mô hình nhưng đã tạo `data/emails_clean.csv`, huấn luyện bằng lệnh `python train.py`.
- Từ thư mục gốc của dự án, khởi chạy ứng dụng:

  ```bash
  python -m streamlit run app.py
  ```

- Mở địa chỉ `http://localhost:8501` nếu trình duyệt không tự bật.
- Thanh bên trái có ba mục điều hướng: **Dashboard**, **Phân loại đơn lẻ** và **Phân loại hàng loạt**. Nút bên dưới dùng để đổi giữa chế độ sáng và tối.

### Demo phân loại một email

- Chọn **Phân loại đơn lẻ** ở thanh điều hướng.
- Nhập tiêu đề vào ô **Tiêu đề email** và nội dung vào ô **Nội dung email**. Có thể mở **Xem ví dụ mẫu** để dùng thử:
  - Spam: `Congratulations! You WON a FREE iPhone 15` / `Click here NOW to claim your prize: http://bit.ly/xyz`
  - Ham: `Meeting moved to 3pm tomorrow` / `Hi team, please confirm your availability. Thanks.`
- Bấm **Phân loại email**. Ứng dụng chạy cả hai mô hình **Naive Bayes** và **SVM**, rồi hiển thị nhãn dự đoán của từng mô hình.
- Nhãn `SPAM` cho biết email bị nhận diện là thư rác; `HAM` là thư hợp lệ. Nếu nội dung quá ngắn hoặc không hợp lệ, kết quả có thể là `UNKNOWN`.
- Nếu cả tiêu đề và nội dung đều để trống, giao diện sẽ yêu cầu nhập email trước khi phân loại.

### Demo phân loại nhiều email từ CSV

- Chọn **Phân loại hàng loạt**.
- Tải lên tệp `.csv` mã hóa UTF-8. Tệp cần có một trong hai định dạng cột:
  - Cột `text` chứa nội dung email đã ghép; hoặc
  - Cả hai cột `subject` và `body`.
- Sau khi tải tệp lên, kiểm tra số dòng đã đọc và phần xem trước dữ liệu.
- Bấm **Phân loại tất cả** để phân loại từng email bằng cả Naive Bayes và SVM. Giao diện hiển thị tiến độ trong khi xử lý.
- Xem bảng kết quả, sau đó bấm **Tải kết quả CSV** để lưu kết quả phân loại. Nếu có dòng gặp lỗi, giao diện sẽ thông báo số email bị lỗi.
- Bấm **Tạo phiên mới** nếu muốn tải một tệp khác và thực hiện lượt phân loại mới.

### Xem dashboard và lịch sử

- Sau khi phân loại hàng loạt, chọn **Dashboard** để xem các phiên đã chạy trong phiên làm việc hiện tại.
- Chọn một phiên để xem phân bố Spam/Ham, thống kê và so sánh kết quả hai mô hình, cùng bảng chi tiết email.
- Biểu đồ phân bố có thể chuyển giữa Naive Bayes và SVM; bảng chi tiết hiển thị tối đa 100 email đầu tiên.
- Bấm **Tải toàn bộ kết quả CSV** để tải toàn bộ dữ liệu của phiên đang xem.

## 6. Kiểm thử riêng mà không cần email thật

```bash
python -m unittest discover -s tests -v
```

Bài test tự tạo corpus nhỏ trong thư mục tạm (không dùng tài liệu thực) để kiểm tra đọc `_tar.bz2`, loại trùng, nhãn mâu thuẫn, `extract_email()` và CSV.

## 7. Giới hạn

Hai file email gốc **không nằm trong gói mã nguồn**. Xem `DATA_SOURCE.md` để biết nguồn, thông tin bản quyền, hạn chế lệch nguồn/nhãn và thống kê chạy thử mà nhóm cung cấp.
