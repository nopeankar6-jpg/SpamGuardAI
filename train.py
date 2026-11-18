import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from preprocess import load_clean_data

RANDOM_STATE = 42
DATA_PATH    = "data/emails_clean.csv"
MODEL_DIR    = "models"
TEST_PATH    = "data/test_set.csv"
STATS_PATH   = "data/train_stats.json"

def build_pipelines():
    """Tạo 2 pipeline: vector + classifier."""
    vectorizer_args = dict(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        stop_words=None,
    )
    nb = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_args)),
        ("clf",   MultinomialNB(alpha=0.5)),
    ])
    svm = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_args)),
        ("clf",   LinearSVC(C=1.0)),
    ])
    return {"nb": nb, "svm": svm}
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[SV2] Đọc dữ liệu từ {DATA_PATH} ...")
    df = load_clean_data(DATA_PATH)
    print(f"[SV2] Tổng số email: {len(df)}")
    print("[SV2] Phân bố nhãn:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"[SV2] Train: {len(X_train)} | Test: {len(X_test)}")

    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(TEST_PATH, index=False)
    print(f"[SV2] Đã lưu test set cố định -> {TEST_PATH}")

    pipelines = build_pipelines()
    file_names = {"nb": "naive_bayes.pkl", "svm": "svm.pkl"}
    stats = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_distribution_train": y_train.value_counts().to_dict(),
        "label_distribution_test":  y_test.value_counts().to_dict(),
        "random_state": RANDOM_STATE,
    }
    for name, pipe in pipelines.items():
        print(f"\n[SV2] Đang huấn luyện {name} ...")
        pipe.fit(X_train, y_train)          # TF-IDF chỉ fit trên train
        path = os.path.join(MODEL_DIR, file_names[name])
        joblib.dump(pipe, path)
        print(f"[SV2] Đã lưu {name} -> {path}")
        stats[name] = {
            "model_path": path,
            "tfidf_max_features": pipe.named_steps["tfidf"].max_features,
            "n_features_actual": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[SV2] Đã lưu thống kê -> {STATS_PATH}")
    print("[SV2] HOÀN TẤT HUẤN LUYỆN")
if __name__ == "__main__":
    main()