"""SV4 - Đánh giá hai mô hình Naive Bayes và SVM trên tập test cố định.

Chạy:  python evaluate.py
Cần có sẵn: data/test_set.csv (do train.py của SV2 tạo ra)
            models/naive_bayes.pkl, models/svm.pkl
Kết quả được lưu trong thư mục reports/.

Đánh giá ở 2 mức:
  1. Mức mô hình: gọi trực tiếp Pipeline (TF-IDF + bộ phân loại) đã lưu.
     -> reports/metrics.csv, reports/confusion_matrix.png, reports/misclassified.csv
  2. Mức hệ thống: gọi predict_email() giống như website, có bộ lọc đầu vào
     (email không hợp lệ trả về "unknown").
     -> reports/system_metrics.csv
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
    "Naive Bayes": ("nb", "models/naive_bayes.pkl"),
    "SVM": ("svm", "models/svm.pkl"),
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
    return df.reset_index(drop=True)


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = cm.ravel()
    return cm, {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_spam": round(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0), 4),
        "recall_spam": round(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0), 4),
        "f1_spam": round(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def evaluate_models(df):
    """Mức 1: đánh giá trực tiếp hai Pipeline đã huấn luyện."""
    X_test, y_test = df["text"], df["label"]
    rows, errors = [], []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.5))

    for ax, (name, (_, path)) in zip(axes, MODELS.items()):
        if not os.path.exists(path):
            sys.exit(f"[SV4] Không tìm thấy mô hình {path}. Hãy chạy `python train.py` trước.")
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        cm, m = compute_metrics(y_test, y_pred)
        rows.append({"model": name, **m})

        print(f"===== {name} (mức mô hình) =====")
        print(classification_report(y_test, y_pred, labels=LABELS, digits=4))

        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")

        # Ghi lại các email bị đoán sai để phân tích lỗi
        wrong_mask = y_test.values != y_pred
        wrong = df[wrong_mask].copy()
        wrong["model"] = name
        wrong["predicted"] = y_pred[wrong_mask]
        wrong["error_type"] = wrong["label"].map({
            "ham": "False Positive (ham bị coi là spam)",
            "spam": "False Negative (spam bị bỏ lọt)",
        })
        wrong["text"] = wrong["text"].str.slice(0, 500)
        errors.append(wrong[["model", "error_type", "label", "predicted", "text"]])

    fig.suptitle("Confusion matrix trên tập test (nhãn dương = spam)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(REPORT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
    pd.concat(errors).to_csv(os.path.join(REPORT_DIR, "misclassified.csv"),
                             index=False, encoding="utf-8-sig")
    return metrics


def evaluate_system(df):
    """Mức 2: đánh giá qua predict_email() như website (có bộ lọc 'unknown')."""
    try:
        from predict import predict_email
    except ImportError as e:
        print(f"[SV4] Bỏ qua đánh giá mức hệ thống: không import được predict.py ({e})")
        return None

    rows = []
    n_total = len(df)
    for name, (key, _) in MODELS.items():
        preds = pd.Series([predict_email(t, key) for t in df["text"]], index=df.index)
        unknown = preds.eq("unknown")
        classified = ~unknown
        y_true, y_pred = df.loc[classified, "label"], preds[classified]
        _, m = compute_metrics(y_true, y_pred)
        rows.append({
            "model": name,
            "n_test": n_total,
            "n_classified": int(classified.sum()),
            "n_unknown": int(unknown.sum()),
            "unknown_ham": int((unknown & df["label"].eq("ham")).sum()),
            "unknown_spam": int((unknown & df["label"].eq("spam")).sum()),
            "coverage": round(classified.mean(), 4),
            **m,  # metric tính trên các email được phân loại (không tính unknown)
        })

    system = pd.DataFrame(rows)
    system.to_csv(os.path.join(REPORT_DIR, "system_metrics.csv"), index=False, encoding="utf-8-sig")
    return system


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_test_set()
    print(f"[SV4] Số email trong tập test: {len(df)}")
    print(df["label"].value_counts().to_string(), "\n")

    metrics = evaluate_models(df)
    print("===== BẢNG TỔNG HỢP – MỨC MÔ HÌNH =====")
    print(metrics.to_string(index=False))

    system = evaluate_system(df)
    if system is not None:
        print("\n===== MỨC HỆ THỐNG (qua predict_email, có bộ lọc unknown) =====")
        print(system[["model", "n_classified", "n_unknown", "coverage",
                      "accuracy", "precision_spam", "recall_spam", "f1_spam"]].to_string(index=False))
        print("(Metric mức hệ thống chỉ tính trên các email được phân loại, không tính 'unknown'.)")

    print(f"\n[SV4] Đã lưu các file trong thư mục {REPORT_DIR}/")


if __name__ == "__main__":
    main()
