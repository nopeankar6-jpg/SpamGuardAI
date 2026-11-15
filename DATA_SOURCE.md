# DATA SOURCE – SpamGuard AI (SV1)

## 1. Nguồn dữ liệu

- **Tên:** Apache SpamAssassin Public Corpus.
- **Trang chủ tập dữ liệu:** https://spamassassin.apache.org/old/publiccorpus/
- **Thông tin mô tả gốc (README):** https://spamassassin.apache.org/old/publiccorpus/readme.html
- **Ngày kiểm tra / tổng hợp thông tin trong dự án:** 15/11/2026.
- **Mục đích sử dụng:** nghiên cứu, huấn luyện và kiểm thử chương trình phân loại email Spam/Ham trong bài tập nhóm học phần Trí tuệ nhân tạo.

| Vai trò | Tên tệp gốc trên máy chủ | Tên tệp đã đổi có thể gặp | Dữ liệu ghi nhận trước xử lý* |
|---|---|---|---:|
| Email hợp lệ (ham) | `20021010_easy_ham.tar.bz2` | `20021010_easy_ham_tar.bz2` | 2.551 |
| Thư rác (spam) | `20030228_spam.tar.bz2` | `20030228_spam_tar.bz2` | 501 |

* Các số đếm này được trích từ biên bản đánh giá chạy thử ngày 15/11/2026 do nhóm cung cấp, không phải là thông số chính thức của toàn bộ bộ sưu tập. Kết quả có thể thay đổi tùy tệp, bộ lọc và quy tắc tiền xử lý. Khi chạy lại, chương trình lưu số đếm thực tế vào `data/preprocess_stats.json`.

Các dãy số `20021010`, `20030228` là một phần tên bộ dữ liệu, liên quan đến thời điểm tập hợp corpus; không phải ngày mà nhóm tải dữ liệu. Bộ dữ liệu SpamAssassin có nhiều tập con khác ngoài hai tập đang dùng.

## 2. Thông tin quyền sử dụng và lưu ý an toàn

Trang README của bộ dữ liệu cho biết **bản quyền văn bản trong email vẫn thuộc về những người gửi gốc**. Trong tài liệu nguồn được tham chiếu, không có một giấy phép riêng rõ ràng cấp quyền sử dụng lại toàn bộ nội dung từng email. Vì vậy nhóm **không tự khẳng định dữ liệu thuộc giấy phép Apache, MIT hay CC0**.

README còn cảnh báo **không được đưa các email trong corpus vào hệ thống thư điện tử thực đang hoạt động để thử nghiệm**. Nhóm chỉ phân tích dữ liệu cục bộ; không gửi hay phát tán email gốc đến địa chỉ thật. Không đưa corpus nguyên bản lên GitHub công khai nếu chưa đánh giá quyền sử dụng và thông tin riêng tư.

## 3. Quy trình chuẩn bị dữ liệu

1. Đọc trực tiếp các file tar+bzip2 (`.tar.bz2` hoặc `_tar.bz2`).
2. Parse định dạng email RFC822/MIME để lấy `subject`, `body` và nhãn dựa vào tệp nguồn.
3. Bỏ nội dung đính kèm, ưu tiên `text/plain`; nếu không có, trích chữ từ `text/html`.
4. Xóa tiền tố `***SPAM***`/`[SPAM]` ở **đầu tiêu đề** để giảm nguy cơ lộ nhãn; không xóa tự động từ “spam” ở mọi nơi trong thân thư.
5. Chuẩn hóa khoảng trắng/ký tự không hợp lệ, bỏ văn bản quá ngắn.
6. Loại tất cả email có cùng `text` nhưng nhãn mâu thuẫn, sau đó loại bản ghi trùng theo `text`.
7. Xuất `data/emails_clean.csv` (UTF-8 BOM) với 4 cột: `subject`, `body`, `text`, `label`.
8. Ghi `data/preprocess_stats.json` để lưu các bước loại bỏ có thể kiểm tra lại.

## 4. Số liệu đã ghi nhận tại lần đánh giá 15/11/2026

| Chỉ số | Giá trị do nhóm cung cấp |
|---|---:|
| Email hợp lệ đọc được | 2.551 |
| Email spam đọc được | 501 |
| Tổng trước loại trùng | 3.052 |
| Email bị loại sau bước đọc | 162 |
| Nhãn mâu thuẫn | 0 |
| Tổng sau loại trùng | 2.890 |
| Ham sau xử lý | 2.408 |
| Spam sau xử lý | 482 |
| Độ dài text trung vị ham | khoảng 865 ký tự |
| Độ dài text trung vị spam | khoảng 1.078 ký tự |

**Lưu ý:** bảng trên là kết quả kiểm tra đã được nhóm báo cáo, **không phải kết quả mà bản mã sửa này đã chạy trên hai tệp gốc**. Để cập nhật số liệu đo trực tiếp, chạy `python preprocess.py` và `python data_analysis.py` trên chính hai file dữ liệu trong `data/raw`.

## 5. Hạn chế và rủi ro thiên lệch dữ liệu

- **Lệch tỷ lệ nhãn:** tập đã xử lý có khoảng 5 ham/1 spam. Nếu chỉ đo accuracy, mô hình có thể được đánh giá quá lạc quan. Cần theo dõi recall lớp spam, precision lớp spam và macro F1.
- **Lệch nguồn:** email ham chứa nhiều nội dung mailing list (ví dụ footer SourceForge), email spam thường có kiểu viết khác. Mô hình có nguy cơ học dấu hiệu nguồn thay vì thật sự hiểu spam.
- **Dấu hiệu nhận dạng đặc thù:** theo kiểm tra của nhóm, khoảng 437 ham và 50 spam chứa từ “spam”; 101 ham có “Razor”. Các từ này xuất hiện trong ngữ cảnh hợp lệ, không nhất thiết là nhãn. Trong bản mã sửa, JSON sẽ thống kê lại các từ đứng độc lập (ranh giới từ), nên số có thể khác số cũ.
- **Lệch thời điểm:** nhóm dùng ham mang tiền tố `20021010` và spam mang tiền tố `20030228`. Dữ liệu ở các giai đoạn thu thập khác nhau; thói quen gửi email có thể thay đổi.
- **Tính lịch sử:** không đại diện cho email hiện đại và không đủ để khẳng định khả năng phân loại tiếng Việt.
- **Thiếu trường con:** một số email có subject hoặc body rỗng nhưng `text` vẫn khác rỗng; chấp nhận được nếu ghi rõ khi phân tích.

## 6. Hướng dẫn chuyển giao cho SV2

SV2 đọc dữ liệu bằng `load_clean_data()` từ `preprocess.py`. Trước khi huấn luyện, tách dữ liệu có **stratify** và chỉ fit TF-IDF trên tập train. Ví dụ:

```python
from sklearn.model_selection import train_test_split
from preprocess import load_clean_data

df = load_clean_data()
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=42
)
```

**Không fit TF-IDF trước khi chia dữ liệu**, không dùng tập test để thử nhiều tham số rồi chọn mô hình. Nên dùng `Pipeline` và đánh giá macro F1, recall spam, confusion matrix bên cạnh accuracy.
