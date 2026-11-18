"""SV4 - Đánh giá hai mô hình Naive Bayes và SVM trên tập test cố định.

Chạy:  python evaluate.py
Cần có sẵn: data/test_set.csv (do train.py của SV2 tạo ra)
            models/naive_bayes.pkl, models/svm.pkl
Kết quả được lưu trong thư mục reports/.
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # vẽ hình ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

TEST_PATH = "data/test_set.csv"
REPORT_DIR = "reports"
MODELS = {
    "Naive Bayes": "models/naive_bayes.pkl",
    "SVM": "models/svm.pkl",
}
POS_LABEL = "spam"          # nhãn dương = spam
LABELS = ["ham", "spam"]    # thứ tự hàng/cột trong confusion matrix


def load_test_set():
    if not os.path.exists(TEST_PATH):
        sys.exit(f"[SV4] Không tìm thấy {TEST_PATH}. Hãy chạy `python train.py` trước.")
    df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_test_set()
    X_test, y_test = df["text"], df["label"]
    print(f"[SV4] Số email trong tập test: {len(df)}")
    print(y_test.value_counts().to_string(), "\n")

    rows = []
    errors = []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.5))

    for ax, (name, path) in zip(axes, MODELS.items()):
        if not os.path.exists(path):
            sys.exit(f"[SV4] Không tìm thấy mô hình {path}. Hãy chạy `python train.py` trước.")
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision_spam": round(precision_score(y_test, y_pred, pos_label=POS_LABEL), 4),
            "recall_spam": round(recall_score(y_test, y_pred, pos_label=POS_LABEL), 4),
            "f1_spam": round(f1_score(y_test, y_pred, pos_label=POS_LABEL), 4),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        })

        print(f"===== {name} =====")
        print(classification_report(y_test, y_pred, labels=LABELS, digits=4))

        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")

        # Ghi lại các email bị đoán sai để phân tích lỗi
        wrong = df[y_test.values != y_pred].copy()
        wrong["model"] = name
        wrong["predicted"] = y_pred[y_test.values != y_pred]
        wrong["error_type"] = wrong.apply(
            lambda r: "False Positive (ham bị coi là spam)" if r["label"] == "ham"
            else "False Negative (spam bị bỏ lọt)", axis=1
        )
        wrong["text"] = wrong["text"].str.slice(0, 500)
        errors.append(wrong[["model", "error_type", "label", "predicted", "text"]])

    fig.suptitle("Confusion matrix trên tập test (nhãn dương = spam)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(REPORT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
    pd.concat(errors).to_csv(os.path.join(REPORT_DIR, "misclassified.csv"),
                             index=False, encoding="utf-8-sig")

    print("===== BẢNG TỔNG HỢP =====")
    print(metrics.to_string(index=False))
    print(f"\n[SV4] Đã lưu: {REPORT_DIR}/metrics.csv, {REPORT_DIR}/confusion_matrix.png, "
          f"{REPORT_DIR}/misclassified.csv")


if __name__ == "__main__":
    main()