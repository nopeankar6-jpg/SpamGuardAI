"""Tiền xử lý Apache SpamAssassin Public Corpus cho SpamGuard AI (SV1).

Đọc trực tiếp tập tin .bz2 dạng tar, chấp nhận cả tên .tar.bz2 và _tar.bz2.
Không tách train/test ở bước này: SV2 thực hiện sau khi loại trùng.
"""
from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import tarfile

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "emails_clean.csv"
STATS_FILE = BASE_DIR / "data" / "preprocess_stats.json"
COLUMNS = ["subject", "body", "text", "label"]
SPAM_PREFIX = re.compile(r"^(?:(?:\*+\s*SPAM\s*\*+|\[SPAM\])\s*)+", re.I)


class HTMLToText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        elif self.ignored_depth == 0 and tag in {"p", "div", "br", "li", "tr", "td"}:
            self.parts.append(" ")

    def handle_startendtag(self, tag, attrs):
        if self.ignored_depth == 0 and tag in {"br", "hr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif self.ignored_depth == 0 and tag in {"p", "div", "li", "tr", "td"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if self.ignored_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = HTMLToText()
    parser.feed(html)
    return " ".join(parser.parts)


def clean_text(value: object) -> str:
    """Chuẩn hóa khoảng trắng/ký tự NUL, giữ lại từ, URL và dấu câu.

    Không dùng hàm này để tính TF-IDF hoặc tách tập train/test.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def prepare_input(subject: str = "", body: str = "") -> str:
    """Dùng cho email nhập thủ công, nhất quán với cột text ở CSV."""
    normalized_subject = SPAM_PREFIX.sub("", clean_text(subject)).strip()
    return clean_text(f"{normalized_subject} {clean_text(body)}")


def decode_part(part) -> str:
    try:
        value = part.get_content()
        if isinstance(value, str):
            return value
    except (LookupError, UnicodeError, ValueError, AssertionError, TypeError):
        pass
    try:
        raw = part.get_payload(decode=True)
    except Exception:
        return ""
    if not isinstance(raw, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def extract_email(raw_email: bytes | str | Path) -> tuple[str, str, str]:
    """Trích xuất (subject, body, text) từ email định dạng RFC822/.eml.

    raw_email: bytes của một email, văn bản RFC822 hoặc đường dẫn Path tới .eml.
    Chỉ dùng text/plain, nếu không có thì chuyển text/html thành văn bản;
    bỏ phần đính kèm để tránh lấy nội dung file ngoài email.
    """
    if isinstance(raw_email, Path):
        raw_email = raw_email.read_bytes()
    elif isinstance(raw_email, str):
        raw_email = raw_email.encode("utf-8", errors="replace")
    if not isinstance(raw_email, bytes):
        raise TypeError("raw_email phải là bytes, str RFC822 hoặc pathlib.Path")

    message = BytesParser(policy=policy.default).parsebytes(raw_email)
    subject = SPAM_PREFIX.sub("", clean_text(message.get("Subject", ""))).strip()

    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        content = decode_part(part)
        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(html_to_text(content))

    body = clean_text(" ".join(plain_parts or html_parts))
    return subject, body, prepare_input(subject, body)


def label_from_name(filename: str) -> str | None:
    name = filename.lower()
    if "easy_ham" in name or "hard_ham" in name or re.search(r"(?:^|_)ham(?:_|\.|$)", name):
        return "ham"
    if re.search(r"(?:^|_)spam(?:_|\.|$)", name):
        return "spam"
    return None


def load_clean_data(path: str | Path = OUTPUT_FILE) -> pd.DataFrame:
    """Đọc dữ liệu sạch để SV2 sử dụng. Cột text và label luôn hợp lệ."""
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")
    df = df[COLUMNS].copy()
    for column in ["subject", "body", "text", "label"]:
        df[column] = df[column].fillna("").astype(str)
    if df["text"].str.strip().eq("").any():
        raise ValueError("CSV có cột text rỗng")
    if not df["label"].isin(["ham", "spam"]).all():
        raise ValueError("Nhãn chỉ được là 'ham' hoặc 'spam'")
    return df


def process_archives(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_FILE,
                     stats_path: Path = STATS_FILE) -> tuple[pd.DataFrame, dict]:
    """Xử lý tất cả .bz2 có nhãn trong thư mục raw, xuất CSV và JSON thống kê."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    stats_path = Path(stats_path)
    archives = sorted(raw_dir.glob("*.bz2"))  # Nhận cả *_tar.bz2 lẫn *.tar.bz2
    if not archives:
        raise FileNotFoundError(
            f"Không tìm thấy file .bz2 trong {raw_dir}. "
            "Cần 20021010_easy_ham.tar.bz2 (hoặc _tar.bz2) "
            "và 20030228_spam.tar.bz2 (hoặc _tar.bz2)."
        )

    records: list[dict] = []
    file_stats: list[dict] = []
    for archive in archives:
        label = label_from_name(archive.name)
        if label is None:
            print(f"Bỏ qua file không nhận diện được nhãn: {archive.name}")
            continue
        status = {"filename": archive.name, "label": label, "members_seen": 0,
                  "valid_emails": 0, "skipped_empty": 0, "skipped_error": 0}
        try:
            with tarfile.open(archive, "r:bz2") as tar:
                for member in tar:
                    if not member.isfile() or Path(member.name).name.startswith("."):
                        continue
                    status["members_seen"] += 1
                    try:
                        stream = tar.extractfile(member)  # KHÔNG giải nén xuống ổ đĩa
                        if stream is None:
                            status["skipped_error"] += 1
                            continue
                        subject, body, text = extract_email(stream.read())
                    except Exception as exc:
                        status["skipped_error"] += 1
                        print(f"Lỗi đọc {archive.name}:{member.name}: {exc}")
                        continue
                    if len(text) < 3:
                        status["skipped_empty"] += 1
                        continue
                    records.append({"subject": subject, "body": body, "text": text,
                                    "label": label})
                    status["valid_emails"] += 1
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise RuntimeError(f"Không thể mở file {archive.name}: {exc}") from exc
        file_stats.append(status)
        print(f"Đã đọc {archive.name}: {status['valid_emails']} email ({label})")

    if not records:
        raise ValueError("Không có dữ liệu email hợp lệ. Kiểm tra lại tên/tập tin .bz2")

    df = pd.DataFrame(records, columns=COLUMNS)
    before = len(df)
    # Xóa cả hai nhãn nếu trùng nguyên văn text nhưng khác nhãn.
    unique_labels = df.groupby("text")["label"].nunique()
    conflicts = set(unique_labels[unique_labels > 1].index)
    conflicting_rows = int(df["text"].isin(conflicts).sum())
    df = df[~df["text"].isin(conflicts)].copy()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"])
    duplicates_removed = before_dedup - len(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    stats = {
        "source": "Apache SpamAssassin Public Corpus",
        "archives": file_stats,
        "valid_before_dedup": before,
        "contradictory_unique_texts": len(conflicts),
        "contradictory_rows_removed": conflicting_rows,
        "duplicates_removed": int(duplicates_removed),
        "total_removed_after_parsing": before - len(df),
        "total_after_cleaning": len(df),
        "labels_after_cleaning": {k: int(v) for k, v in df["label"].value_counts().items()},
        "empty_subject": int(df["subject"].eq("").sum()),
        "empty_body": int(df["body"].eq("").sum()),
        "empty_text": int(df["text"].eq("").sum()),
        "source_term_diagnostics": {
            "spam_term_by_label": {
                label: int(df.loc[df["label"].eq(label), "text"].str.contains(
                    r"\bspam\b", case=False, regex=True).sum())
                for label in ("ham", "spam")
            },
            "razor_term_in_ham": int(df.loc[df["label"].eq("ham"), "text"].str.contains(
                r"\brazor\b", case=False, regex=True).sum())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Trước loại trùng: {before}; loại nhãn mâu thuẫn: {conflicting_rows}; "
          f"loại trùng: {duplicates_removed}")
    print(f"Sau làm sạch: {len(df)}; nhãn: {stats['labels_after_cleaning']}")
    print(f"Đã tạo: {output_path} và {stats_path}")
    return df, stats


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu Spam/Ham (SV1)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--stats", type=Path, default=STATS_FILE)
    args = parser.parse_args()
    process_archives(args.raw_dir, args.output, args.stats)


if __name__ == "__main__":
    main()
