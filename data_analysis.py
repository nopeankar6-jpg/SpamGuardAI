"""Thống kê dữ liệu Spam/Ham và xuất biểu đồ, báo cáo của SV1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import BASE_DIR, OUTPUT_FILE, STATS_FILE, load_clean_data


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    label_stats = pd.DataFrame({"count": counts, "percent": counts / len(df) * 100})
    lengths = df.assign(text_length=df["text"].str.len())
    length_stats = (lengths.groupby("label")["text_length"]
                    .agg(count="count", mean="mean", median="median", max="max"))
    length_stats["p95"] = lengths.groupby("label")["text_length"].quantile(.95)
    return label_stats, length_stats


def plot_labels(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(int(bar.get_height())), ha="center", va="bottom")
    ax.set_title("Email count by label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of emails")
    ax.set_ylim(0, max(counts.max() * 1.15, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_lengths(df: pd.DataFrame, output_path: Path) -> None:
    """Biểu đồ log10(độ dài + 1) để các email cực dài không che mất phân bố."""
    lengths = df.assign(text_length=df["text"].str.len())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in ("ham", "spam"):
        values = lengths.loc[lengths["label"].eq(label), "text_length"]
        if len(values):
            ax.hist(np.log10(values + 1), bins=35, alpha=0.55, label=f"{label} (n={len(values)})")
    ax.set_title("Email text length distribution")
    ax.set_xlabel("log10(number of characters + 1)")
    ax.set_ylabel("Number of emails")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def create_report(df: pd.DataFrame, label_stats: pd.DataFrame,
                  length_stats: pd.DataFrame, stats: dict | None,
                  out_path: Path) -> None:
    def n(v):
        return f"{v:,}" if isinstance(v, int) else str(v)

    lines = [
        "# Báo cáo thống kê dữ liệu email Spam/Ham", "",
        f"- Tổng số email trong CSV: **{len(df):,}**",
        f"- Thiếu `text`: **{int(df['text'].isna().sum())}**",
        f"- Thiếu/để trống `subject`: **{int(df['subject'].eq('').sum())}**",
        f"- Thiếu/để trống `body`: **{int(df['body'].eq('').sum())}**", "",
        "## Phân bố nhãn", "",
        "| Nhãn | Số lượng | Tỷ lệ |", "|---|---:|---:|",
    ]
    for label, row in label_stats.iterrows():
        lines.append(f"| {label} | {int(row['count']):,} | {row['percent']:.2f}% |")
    lines += ["", "## Độ dài văn bản", "", "| Nhãn | Trung bình | Trung vị | P95 | Lớn nhất |",
              "|---|---:|---:|---:|---:|"]
    for label, row in length_stats.iterrows():
        lines.append(f"| {label} | {row['mean']:.0f} | {row['median']:.0f} | {row['p95']:.0f} | {row['max']:.0f} |")

    lines += ["", "## Lọc trùng và nhãn mâu thuẫn", ""]
    if stats:
        fields = [
            ("Đọc hợp lệ trước loại trùng", "valid_before_dedup"),
            ("Số nội dung mâu thuẫn nhãn", "contradictory_unique_texts"),
            ("Số bản ghi bị loại vì nhãn mâu thuẫn", "contradictory_rows_removed"),
            ("Số bản ghi trùng bị loại", "duplicates_removed"),
            ("Tổng bản ghi bị loại sau bước đọc", "total_removed_after_parsing"),
            ("Còn lại sau làm sạch", "total_after_cleaning"),
        ]
        for title, key in fields:
            lines.append(f"- {title}: **{n(stats.get(key, 'N/A'))}**")
        diag = stats.get("source_term_diagnostics", {})
        lines += ["", "## Dấu hiệu lệch nguồn cần cảnh báo", "",
                  f"- Email chứa từ 'spam' (theo nhãn): {diag.get('spam_term_by_label', {})}",
                  f"- Ham chứa từ 'razor': {diag.get('razor_term_in_ham', 'N/A')}"]
    else:
        lines.append("- Chưa có `preprocess_stats.json`; chạy `python preprocess.py` để ghi thống kê loại trùng.")
    lines += [
        "", "## Hạn chế dữ liệu", "",
        "- Nhãn mất cân bằng: ham thường nhiều hơn spam; nên sử dụng stratified split và Macro F1/Recall spam.",
        "- Email ham có thể đến từ mailing list, trong khi spam đến từ nguồn khác, dễ gây lệch nguồn.",
        "- Từ 'spam' hoặc 'Razor' xuất hiện trong cả ham; không tự động loại các từ này vì có thể là ngữ cảnh hợp lệ.",
        "- Nếu dùng 20021010_easy_ham và 20030228_spam thì hai phần dữ liệu khác thời điểm tập hợp.",
        "- Không dùng tập này để gửi thư vào hệ thống email thật.",
        "", "## Các biểu đồ", "",
        "- `label_distribution.png`: số email theo nhãn.",
        "- `length_distribution.png`: histogram log10 độ dài email, giảm ảnh hưởng outlier.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_data(csv_path: Path = OUTPUT_FILE,
                 stats_path: Path = STATS_FILE,
                 out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean_data(csv_path)
    label_stats, length_stats = summarize(df)
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else None

    print("Tổng email:", len(df))
    print("\nPhân bố nhãn:\n", label_stats.to_string())
    print("\nThiếu dữ liệu:\n", df.isna().sum().to_string())
    print("\nRỗng subject/body:\n", df[["subject", "body"]].eq("").sum().to_string())
    print("\nĐộ dài văn bản (ký tự):\n", length_stats.to_string())
    if stats:
        print("\nLoại trùng:", stats["duplicates_removed"],
              "; loại do nhãn mâu thuẫn:", stats["contradictory_rows_removed"])

    plot_labels(df, out_dir / "label_distribution.png")
    plot_lengths(df, out_dir / "length_distribution.png")
    create_report(df, label_stats, length_stats, stats, out_dir / "data_report.md")
    print("\nĐã lưu: label_distribution.png, length_distribution.png, data_report.md")
    return label_stats, length_stats


def main():
    parser = argparse.ArgumentParser(description="Thống kê dữ liệu Spam/Ham")
    parser.add_argument("--csv", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--stats", type=Path, default=STATS_FILE)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    analyze_data(args.csv, args.stats, args.out_dir)


if __name__ == "__main__":
    main()
