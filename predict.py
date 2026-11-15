import os
import joblib
from preprocess import prepare_input

_MODEL_PATHS = {
    "nb":  "models/naive_bayes.pkl",
    "svm": "models/svm.pkl",
}
_cache = {}

def _load(model_name: str):
    if model_name not in _MODEL_PATHS:
        raise ValueError(f"model_name phải là 'nb' hoặc 'svm', nhận được: {model_name!r}")
    if model_name not in _cache:
        path = _MODEL_PATHS[model_name]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy mô hình: {path}. Hãy chạy `python train.py` trước."
            )
        _cache[model_name] = joblib.load(path)
    return _cache[model_name]

def predict_email(text: str, model_name: str = "nb") -> str:
    """Dự đoán nhãn từ text đã được xử lý (dùng cho CSV upload)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nội dung email rỗng.")
    model = _load(model_name)
    return str(model.predict([text])[0])

def predict_email_from_parts(subject: str, body: str, model_name: str = "nb") -> str:
    """
    Dự đoán từ subject + body (dùng cho app Streamlit).
    Dùng prepare_input của SV1 để ghép chuẩn — KHÔNG tự nối chuỗi thủ công.
    """
    if not (subject or "").strip() and not (body or "").strip():
        raise ValueError("Vui lòng nhập tiêu đề hoặc nội dung email.")
    text = prepare_input(subject, body)
    return predict_email(text, model_name)

if __name__ == "__main__":
    spam_subject = "Congratulations! You WON a FREE iPhone 15"
    spam_body    = "Click here NOW to claim your prize: http://bit.ly/xyz"
    ham_subject  = "Meeting moved to 3pm tomorrow"
    ham_body     = "Hi team, please confirm your availability. Thanks."

    for name in ("nb", "svm"):
        print(f"\n--- {name.upper()} ---")
        print("Spam ->", predict_email_from_parts(spam_subject, spam_body, name))
        print("Ham  ->", predict_email_from_parts(ham_subject, ham_body, name))