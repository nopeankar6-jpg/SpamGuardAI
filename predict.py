import os
import re
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

def _is_likely_english_word(word: str) -> bool:
    """
    Heuristic kiểm tra một từ có giống từ tiếng Anh không.
    Dựa trên tỷ lệ nguyên âm hợp lý và không có chuỗi phụ âm quá dài.
    """
    w = re.sub(r'[^a-zA-Z]', '', word).lower()
    if not w:
        return False
    # Từ 1-2 ký tự: cho qua (a, I, an, to, ...)
    if len(w) <= 2:
        return True
    # Kiểm tra tỷ lệ nguyên âm (tiếng Anh thường 30-50%)
    vowels = sum(1 for c in w if c in 'aeiouy')
    vowel_ratio = vowels / len(w)
    if vowel_ratio < 0.15 or vowel_ratio > 0.85:
        return False
    # Kiểm tra chuỗi phụ âm liên tiếp quá dài (>4 rất hiếm trong tiếng Anh)
    if re.search(r'[^aeiouy]{5,}', w):
        return False
    # Kiểm tra chuỗi nguyên âm liên tiếp quá dài (>3 rất hiếm)
    if re.search(r'[aeiouy]{4,}', w):
        return False
    return True


def is_valid_input(text: str) -> bool:
    """
    Kiểm tra xem văn bản có đủ ý nghĩa để phân loại không.
    Chặn dữ liệu rác (spam ký tự) và chặn tiếng Việt (nhưng cho phép dấu câu Unicode).
    """
    text = text.strip()
    
    words = text.split()
    if len(words) < 3:
        return False
        
    if not re.search(r'[aeiouyAEIOUY]', text):
        return False
        
    if re.search(r'(.)\1{3,}', text):
        return False

    allowed_punctuation = r'\u2014\u2013\u2018\u2019\u201C\u201D\u2026' 
    
    invalid_chars = re.findall(fr'[^\x00-\x7F{allowed_punctuation}]', text)
    if len(invalid_chars) > 0:
        return False

    # Kiểm tra tỷ lệ từ giống tiếng Anh — chặn text rác kiểu "sdasdfasdfasd"
    alpha_words = [w for w in words if re.search(r'[a-zA-Z]', w)]
    if alpha_words:
        english_count = sum(1 for w in alpha_words if _is_likely_english_word(w))
        if english_count / len(alpha_words) < 0.5:
            return False
        
    return True

def predict_email(text: str, model_name: str = "nb") -> str:
    """Dự đoán nhãn từ text đã được xử lý (dùng cho CSV upload)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nội dung email rỗng.")
        
    if not is_valid_input(text):
        return "unknown"        
        
    model = _load(model_name)
    return str(model.predict([text])[0])
def predict_email_from_parts(subject: str, body: str, model_name: str = "nb") -> str:
    if not (subject or "").strip() and not (body or "").strip():
        raise ValueError("Vui lòng nhập tiêu đề hoặc nội dung email.")
        
    text = prepare_input(subject, body)
    
    if not is_valid_input(text):
        return "unknown"        
        
    return predict_email(text, model_name)
if __name__ == "__main__":
    spam_subject = "Congratulations! You WON a FREE iPhone 15"
    spam_body    = "Click here NOW to claim your prize: http://bit.ly/xyz"
    
    ham_subject  = "Meeting moved to 3pm tomorrow"
    ham_body     = "Hi team, please confirm your availability. Thanks."
    
    junk_text    = "asdfghjkl"
    
    vn_subject   = "Trúng thưởng"
    vn_body      = "Chúc mừng bạn đã trúng thưởng xe SH, bấm vào link để nhận"
    
    short_text   = "Hello there" 
    
    en_special_subject = "Meeting update"
    en_special_body    = "Hi team — let’s meet at 3pm tomorrow… Thanks!"

    print("\n--- TEST MÔ HÌNH ---")
    for name in ("nb", "svm"):
        print(f"\n--- {name.upper()} ---")
        print("Spam ->", predict_email_from_parts(spam_subject, spam_body, name))
        print("Ham  ->", predict_email_from_parts(ham_subject, ham_body, name))    
    
    print("\n--- TEST DỮ LIỆU RÁC (Kỳ vọng: unknown) ---")
    print("Rác  ->", predict_email_from_parts("", junk_text, "nb"))
    
    print("\n--- TEST TIẾNG VIỆT (Kỳ vọng: unknown) ---")
    print("Tiếng Việt ->", predict_email_from_parts(vn_subject, vn_body, "nb"))
    
    print("\n--- TEST QUÁ NGẮN (Kỳ vọng: unknown) ---")
    print("Ngắn ->", predict_email_from_parts("", short_text, "nb"))
    
    print("\n--- TEST TIẾNG ANH CÓ DẤU CÂU UNICODE (Kỳ vọng: ham/spam) ---")
    print("Anh đặc biệt ->", predict_email_from_parts(en_special_subject, en_special_body, "nb"))