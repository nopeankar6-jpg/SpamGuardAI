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

def is_valid_input(text: str) -> bool:

    text = text.strip()
    
    words = text.split()
    if len(words) < 3:
        return False
        
    if not re.search(r'[aeiouyAEIOUY]', text):
        return False
        
    unique_chars = len(set(text.lower()))
    if len(text) > 5 and (unique_chars / len(text)) < 0.2:
        return False

    vietnamese_chars = re.findall(r'[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵĂÂĐÊÔƠƯÀẢÃÁẠẰẲẴẮẶẦẨẪẤẬÈẺẼÉẸỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌỒỔỖỐỘỜỞỠỚỢÙỦŨÚỤỪỬỮỨỰỲỶỸÝỴ]', text)
    if len(vietnamese_chars) > 0:
        return False
               
    return True

def predict_email(text: str, model_name: str = "nb") -> str:
   
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
    # Câu tiếng Anh có chứa ký tự đặc biệt (dấu gạch dài, dấu nháy lạ)
    en_special_subject = "Meeting update"
    en_special_body    = "Hi team — let’s meet at 3pm tomorrow."

    print("\n--- TEST MÔ HÌNH ---")
    for name in ("nb", "svm"):
        print(f"\n--- {name.upper()} ---")
        print("Spam ->", predict_email_from_parts(spam_subject, spam_body, name))
        print("Ham  ->", predict_email_from_parts(ham_subject, ham_body, name))    
    
    print("\n--- TEST DỮ LIỆU RÁC ---")
    print("Rác  ->", predict_email_from_parts("", junk_text, "nb"))
    
    print("\n--- TEST TIẾNG VIỆT ---")
    print("Tiếng Việt ->", predict_email_from_parts(vn_subject, vn_body, "nb"))
    
    print("\n--- TEST TIẾNG ANH CÓ KÝ TỰ ĐẶC BIỆT ---")
    print("Anh đặc biệt ->", predict_email_from_parts(en_special_subject, en_special_body, "nb"))