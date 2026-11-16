"""SpamGuard AI – Giao diện phân loại email Spam/Ham bằng Streamlit (SV3).

Sử dụng hàm predict_email_from_parts(subject, body, model_name)
từ predict.py do SV2 cung cấp.
"""

import streamlit as st
import pandas as pd
import os
import io

# ──────────────────────────── Cấu hình trang ────────────────────────────
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── CSS tuỳ chỉnh ─────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #1f77b4;
        font-size: 2.4rem;
    }
    .main-header p {
        color: #666;
        font-size: 1.05rem;
    }
    .result-spam {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border-left: 5px solid #ff4b4b;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-ham {
        background: linear-gradient(135deg, #21c35422, #21c35411);
        border-left: 5px solid #21c354;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .result-desc {
        font-size: 0.95rem;
        color: #555;
        margin: 0.3rem 0 0 0;
    }
    div.stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ───────────────────── Kiểm tra mô hình tồn tại ────────────────────────
def check_model_exists(model_name: str) -> bool:
    """Kiểm tra file mô hình có tồn tại không."""
    paths = {"nb": "models/naive_bayes.pkl", "svm": "models/svm.pkl"}
    return os.path.exists(paths.get(model_name, ""))


# ─────────────────── Import hàm predict an toàn ─────────────────────────
def get_predict_function():
    """Import predict_email_from_parts từ predict.py của SV2."""
    try:
        from predict import predict_email_from_parts
        return predict_email_from_parts, None
    except ImportError as e:
        return None, f"Không thể import predict.py: {e}"
    except Exception as e:
        return None, f"Lỗi khi nạp module predict: {e}"


def get_predict_email_function():
    """Import predict_email từ predict.py (dùng cho CSV upload)."""
    try:
        from predict import predict_email
        return predict_email, None
    except ImportError as e:
        return None, f"Không thể import predict.py: {e}"
    except Exception as e:
        return None, f"Lỗi khi nạp module predict: {e}"


# ──────────────────── Hàm hiển thị kết quả ──────────────────────────────
def render_result(label: str, model_display: str):
    """Hiển thị kết quả phân loại với giao diện đẹp."""
    is_spam = label.lower() == "spam"
    css_class = "result-spam" if is_spam else "result-ham"
    icon = "🚫" if is_spam else "✅"
    desc = "Email này được nhận diện là **thư rác (spam)**." if is_spam \
        else "Email này được nhận diện là **thư hợp lệ (ham)**."
    color = "#ff4b4b" if is_spam else "#21c354"

    st.markdown(f"""
    <div class="{css_class}">
        <p class="result-label">{icon} <span style="color:{color}">{label.upper()}</span></p>
        <p class="result-desc">Mô hình: {model_display}</p>
    </div>
    """, unsafe_allow_html=True)
    if is_spam:
        st.warning(desc)
    else:
        st.success(desc)


# ═══════════════════════════ GIAO DIỆN CHÍNH ═════════════════════════════
# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ SpamGuard AI</h1>
    <p>Phân loại email Spam / Ham bằng học máy<br>
    <em>Naive Bayes &nbsp;•&nbsp; SVM &nbsp;•&nbsp; TF-IDF</em></p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ────────────────────────── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    MODEL_OPTIONS = {"Naive Bayes": "nb", "SVM (Support Vector Machine)": "svm"}
    selected_model_display = st.selectbox(
        "🤖 Chọn mô hình",
        list(MODEL_OPTIONS.keys()),
        help="Naive Bayes: nhanh, phù hợp dữ liệu văn bản.\n"
             "SVM: chính xác hơn trên nhiều tập dữ liệu.",
    )
    selected_model = MODEL_OPTIONS[selected_model_display]

    st.divider()

    # Trạng thái mô hình
    st.subheader("📦 Trạng thái mô hình")
    for display_name, key in MODEL_OPTIONS.items():
        exists = check_model_exists(key)
        status = "✅ Sẵn sàng" if exists else "❌ Chưa huấn luyện"
        st.markdown(f"**{display_name}:** {status}")

    st.divider()

    st.subheader("ℹ️ Hướng dẫn")
    st.markdown("""
    1. Nhập **tiêu đề** và **nội dung** email
    2. Chọn mô hình phân loại
    3. Nhấn **🔍 Phân loại**
    4. Xem kết quả Spam / Ham

    **Hoặc:** Upload file CSV để phân loại hàng loạt.
    """)

    st.divider()
    st.caption("SpamGuard AI © 2026 — Bài tập nhóm TTNTTM")


# ────────────────────────── Tabs chính ───────────────────────────────────
tab_single, tab_batch = st.tabs(["📧 Phân loại đơn lẻ", "📁 Phân loại hàng loạt (CSV)"])

# ========================= TAB 1: PHÂN LOẠI ĐƠN LẺ =======================
with tab_single:
    st.subheader("Nhập nội dung email cần kiểm tra")

    col_subject, _ = st.columns([1, 0.01])
    with col_subject:
        subject = st.text_input(
            "📌 Tiêu đề email (Subject)",
            placeholder="Ví dụ: Congratulations! You won a FREE gift",
        )

    body = st.text_area(
        "📝 Nội dung email (Body)",
        height=180,
        placeholder="Ví dụ: Click here to claim your prize now! Limited time offer...",
    )

    # Nút phân loại
    classify_btn = st.button(
        "🔍 Phân loại",
        type="primary",
        use_container_width=True,
    )

    if classify_btn:
        # Kiểm tra đầu vào rỗng
        if not subject.strip() and not body.strip():
            st.error("⚠️ Vui lòng nhập tiêu đề hoặc nội dung email.")
        # Kiểm tra mô hình
        elif not check_model_exists(selected_model):
            st.error(
                f"❌ Mô hình **{selected_model_display}** chưa được huấn luyện.\n\n"
                "Hãy chạy `python train.py` trước để tạo file mô hình."
            )
        else:
            predict_fn, err = get_predict_function()
            if predict_fn is None:
                st.error(f"❌ {err}")
            else:
                try:
                    with st.spinner("Đang phân loại..."):
                        result = predict_fn(subject, body, selected_model)
                    render_result(result, selected_model_display)
                except ValueError as ve:
                    st.error(f"⚠️ Lỗi đầu vào: {ve}")
                except FileNotFoundError as fe:
                    st.error(f"❌ {fe}")
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi không mong muốn: {e}")

    # Ví dụ nhanh
    st.divider()
    with st.expander("💡 Xem ví dụ mẫu"):
        st.markdown("**Ví dụ Spam:**")
        st.code(
            "Subject: Congratulations! You WON a FREE iPhone 15\n"
            "Body: Click here NOW to claim your prize: http://bit.ly/xyz",
            language=None,
        )
        st.markdown("**Ví dụ Ham:**")
        st.code(
            "Subject: Meeting moved to 3pm tomorrow\n"
            "Body: Hi team, please confirm your availability. Thanks.",
            language=None,
        )


# ======================== TAB 2: PHÂN LOẠI HÀNG LOẠT =====================
with tab_batch:
    st.subheader("Upload file CSV để phân loại hàng loạt")

    st.info(
        "📋 **Yêu cầu định dạng CSV:**\n"
        "- Phải có cột `text` (nội dung email đã xử lý), "
        "**hoặc** hai cột `subject` và `body`.\n"
        "- Encoding: UTF-8.\n"
        "- Nếu có cả `text`, `subject`, `body` — ưu tiên dùng `subject` + `body`."
    )

    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type=["csv"],
        help="File CSV chứa các email cần phân loại.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Không thể đọc file CSV: {e}")
            df = None

        if df is not None:
            st.write(f"📊 Đã đọc **{len(df)}** dòng dữ liệu.")
            st.dataframe(df.head(5), use_container_width=True)

            has_subject_body = "subject" in df.columns and "body" in df.columns
            has_text = "text" in df.columns

            if not has_subject_body and not has_text:
                st.error(
                    "❌ File CSV cần có cột `text`, hoặc cả `subject` và `body`."
                )
            elif not check_model_exists(selected_model):
                st.error(
                    f"❌ Mô hình **{selected_model_display}** chưa được huấn luyện.\n\n"
                    "Hãy chạy `python train.py` trước."
                )
            else:
                batch_btn = st.button(
                    "🚀 Phân loại tất cả",
                    type="primary",
                    use_container_width=True,
                )

                if batch_btn:
                    # Chọn hàm phù hợp
                    if has_subject_body:
                        predict_fn, err = get_predict_function()
                        mode = "subject_body"
                    else:
                        predict_fn, err = get_predict_email_function()
                        mode = "text"

                    if predict_fn is None:
                        st.error(f"❌ {err}")
                    else:
                        results = []
                        errors = 0
                        progress = st.progress(0, text="Đang phân loại...")

                        for idx, row in df.iterrows():
                            try:
                                if mode == "subject_body":
                                    subj = str(row.get("subject", ""))
                                    bd = str(row.get("body", ""))
                                    label = predict_fn(subj, bd, selected_model)
                                else:
                                    txt = str(row.get("text", ""))
                                    label = predict_fn(txt, selected_model)
                            except Exception:
                                label = "error"
                                errors += 1
                            results.append(label)
                            progress.progress(
                                (idx + 1) / len(df),
                                text=f"Đang phân loại... {idx + 1}/{len(df)}",
                            )

                        progress.empty()
                        df["prediction"] = results

                        # Thống kê
                        st.success(f"✅ Hoàn tất phân loại {len(df)} email!")

                        if errors > 0:
                            st.warning(f"⚠️ Có {errors} email gặp lỗi khi phân loại.")

                        col1, col2, col3 = st.columns(3)
                        spam_count = (df["prediction"].str.lower() == "spam").sum()
                        ham_count = (df["prediction"].str.lower() == "ham").sum()
                        error_count = (df["prediction"] == "error").sum()

                        col1.metric("🚫 Spam", spam_count)
                        col2.metric("✅ Ham", ham_count)
                        col3.metric("❌ Lỗi", error_count)

                        # Biểu đồ
                        if spam_count + ham_count > 0:
                            chart_data = pd.DataFrame(
                                {"Nhãn": ["Spam", "Ham"], "Số lượng": [spam_count, ham_count]}
                            )
                            st.bar_chart(
                                chart_data.set_index("Nhãn"),
                                use_container_width=True,
                            )

                        # Bảng kết quả
                        st.subheader("📋 Kết quả chi tiết")
                        st.dataframe(df, use_container_width=True)

                        # Tải xuống
                        csv_output = df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="📥 Tải kết quả (CSV)",
                            data=csv_output,
                            file_name="spamguard_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
