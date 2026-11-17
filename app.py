"""SpamGuard AI – Giao diện phân loại email Spam/Ham bằng Streamlit (SV3).

Sử dụng hàm predict_email_from_parts(subject, body, model_name)
từ predict.py do SV2 cung cấp.
"""

import streamlit as st
import pandas as pd
import altair as alt
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

# ──────────────────────────── Cấu hình trang ────────────────────────────
st.set_page_config(
    page_title="SpamGuard AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── CSS tuỳ chỉnh ─────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: #0b1018;
        color: #e7edf5;
    }
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stMainBlockContainer"] {
        animation: fade-up 420ms ease-out both;
    }
    [data-testid="stSidebar"] {
        background: #111927;
        border-right: 1px solid #263244;
        width: 210px !important;
        min-width: 210px !important;
        max-width: 210px !important;
        flex: 0 0 210px !important;
    }
    [data-testid="stSidebarResizeHandle"] {
        display: none;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 1.5rem;
    }
    .sidebar-brand {
        padding: 0.25rem 0 0.8rem;
    }
    .sidebar-brand-title {
        color: #f3f6fb;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .sidebar-brand-description {
        color: #9aa8bb;
        font-size: 0.78rem;
        line-height: 1.4;
        margin: 0.35rem 0 0;
    }
    .session-heading {
        color: #f3f6fb;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 1.1rem 0 1rem;
    }
    .result-spam {
        background: #241719;
        border: 1px solid #69353b;
        border-left: 4px solid #ed6a70;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .result-ham {
        background: #14231f;
        border: 1px solid #31584c;
        border-left: 4px solid #59bd91;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .result-label {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
    }
    .result-desc {
        font-size: 0.85rem;
        color: #aab6c6;
        margin: 0.2rem 0 0 0;
    }
    [data-testid="stMetric"] {
        background: #151e2c;
        border: 1px solid #263244;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        transition: transform 180ms ease, border-color 180ms ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #47627f;
    }
    [data-testid="stSidebar"] button {
        transition: transform 160ms ease, border-color 160ms ease;
    }
    [data-testid="stSidebar"] button:hover {
        transform: translateX(2px);
    }
    button[data-testid="stBaseButton-secondary"] {
        background: #1d1b19;
        border-color: #49443f;
        color: #e7edf5;
        transition: border-color 160ms ease, transform 160ms ease;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: #1d1b19;
        border-color: #81776e;
        transform: translateY(-1px);
    }
    button[data-testid="stBaseButton-primary"] {
        background: #285d50;
        border-color: #4a9d83;
        color: #f3fff9;
        box-shadow: inset 3px 0 #91d7bd;
        transition: border-color 160ms ease, transform 160ms ease;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background: #285d50;
        border-color: #73c5a9;
        color: #ffffff;
        transform: translateY(-1px);
    }
    [data-testid="stMetricLabel"] p {
        color: #9aa8bb;
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


MODEL_OPTIONS = {"Naive Bayes": "nb", "SVM": "svm"}


def start_new_batch():
    st.session_state.batch_results = None
    st.session_state.batch_upload_version += 1
    st.rerun()


def select_batch_session(session_id):
    st.session_state.selected_batch_session_id = session_id


def prediction_style(label):
    normalized_label = str(label).strip().lower()
    if normalized_label == "ham":
        return "background-color: #284936; color: #d8f2df"
    if normalized_label == "spam":
        return "background-color: #512e33; color: #f7dadd"
    return ""


def get_prediction_columns(df):
    return [
        column for column in ("Naive Bayes", "SVM", "prediction")
        if column in df.columns
    ]


def render_results_table(df, *, height=520):
    prediction_columns = get_prediction_columns(df)
    styled_df = df.style.map(prediction_style, subset=prediction_columns)
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config={
            column: st.column_config.Column(width="small")
            for column in df.columns
        },
    )


def calculate_model_metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, pos_label="spam", zero_division=0),
        "Recall": recall_score(actual, predicted, pos_label="spam", zero_division=0),
        "F1-score": f1_score(actual, predicted, pos_label="spam", zero_division=0),
    }


def render_metrics_comparison(df):
    prediction_columns = list(MODEL_OPTIONS)
    if "label" not in df.columns or any(column not in df.columns for column in prediction_columns):
        metrics = None
        evaluated_count = 0
    else:
        actual = df["label"].astype(str).str.strip().str.lower()
        predictions = {
            model_name: df[model_name].astype(str).str.strip().str.lower()
            for model_name in MODEL_OPTIONS
        }
        valid_rows = actual.isin(["spam", "ham"])
        for predicted in predictions.values():
            valid_rows &= predicted.isin(["spam", "ham"])
        evaluated_count = int(valid_rows.sum())
        if evaluated_count:
            actual = actual[valid_rows]
            metrics = {
                model_name: calculate_model_metrics(actual, predicted[valid_rows])
                for model_name, predicted in predictions.items()
            }
        else:
            metrics = None

    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_table = pd.DataFrame(
        [
            {
                "Chỉ số": metric_name,
                "NB": f"{metrics['Naive Bayes'][metric_name]:.1%}" if metrics else "—",
                "SVM": f"{metrics['SVM'][metric_name]:.1%}" if metrics else "—",
            }
            for metric_name in metric_names
        ]
    )
    st.dataframe(
        metric_table,
        use_container_width=True,
        hide_index=True,
        height=225,
        column_config={
            "Chỉ số": st.column_config.TextColumn(width="medium"),
            "NB": st.column_config.TextColumn(width="small"),
            "SVM": st.column_config.TextColumn(width="small"),
        },
    )
    if metrics:
        st.caption(
            f"Đánh giá trên cùng {evaluated_count} email có nhãn hợp lệ. "
            "Spam là lớp dương."
        )
    else:
        st.caption("Thêm cột `label` với nhãn `spam` hoặc `ham` để tính điểm.")


def render_prediction_distribution(df):
    prediction_values = pd.concat(
        [df[model_name].astype(str).str.strip().str.lower() for model_name in MODEL_OPTIONS],
        ignore_index=True,
    )
    prediction_counts = prediction_values[prediction_values.isin(["ham", "spam"])].value_counts()
    total_predictions = int(prediction_counts.sum())
    distribution = pd.DataFrame(
        [
            {
                "Nhãn": label.title(),
                "Số lượng": int(prediction_counts.get(label, 0)),
                "Tỷ lệ": prediction_counts.get(label, 0) / total_predictions,
            }
            for label in ("ham", "spam")
        ]
    )
    if not total_predictions:
        st.info("Phiên này chưa có dự đoán hợp lệ để hiển thị.", icon=None)
        return

    base = alt.Chart(distribution).encode(
        theta=alt.Theta("Số lượng:Q", stack=True),
        color=alt.Color(
            "Nhãn:N",
            scale=alt.Scale(
                domain=["Ham", "Spam"],
                range=["#5797F5", "#6CC47D"],
            ),
            legend=alt.Legend(title=None, orient="bottom", labelFontSize=15),
        ),
        tooltip=[
            "Nhãn:N",
            "Số lượng:Q",
            alt.Tooltip("Tỷ lệ:Q", format=".1%"),
        ],
    )
    arcs = base.mark_arc(innerRadius=68, outerRadius=116)
    percentage_labels = base.mark_text(
        radius=138,
        color="#e7edf5",
        fontSize=14,
        fontWeight="bold",
    ).encode(text=alt.Text("Tỷ lệ:Q", format=".0%"))
    chart = (
        (arcs + percentage_labels)
        .properties(height=330, background="transparent")
        .configure_view(stroke=None)
        .configure_legend(symbolSize=140)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Tổng hợp dự đoán từ Naive Bayes và SVM.")


def render_batch_results():
    result = st.session_state.batch_results
    heading, action = st.columns([4, 1])
    with heading:
        st.subheader("Kết quả chi tiết")
    with action:
        if st.button("Tạo phiên mới", use_container_width=True):
            start_new_batch()

    st.success(f"Hoàn tất phân loại {len(result['data'])} email.", icon=None)
    if result["errors"]:
        st.warning(f"Có {result['errors']} email gặp lỗi khi phân loại.", icon=None)

    df = result["data"]
    render_results_table(df)
    csv_output = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Tải kết quả CSV",
        data=csv_output,
        file_name="spamguard_results.csv",
        mime="text/csv",
    )


# ──────────────────── Hàm hiển thị kết quả ──────────────────────────────
def render_result(label: str, model_display: str):
    """Hiển thị kết quả phân loại với giao diện đẹp."""
    is_spam = label.lower() == "spam"
    css_class = "result-spam" if is_spam else "result-ham"
    desc = "Email này được nhận diện là **thư rác (spam)**." if is_spam \
        else "Email này được nhận diện là **thư hợp lệ (Ham)**."
    color = "#ed6a70" if is_spam else "#59bd91"

    st.markdown(f"""
    <div class="{css_class}">
        <p class="result-label"><span style="color:{color}">{label.upper()}</span></p>
        <p class="result-desc">Mô hình: {model_display}</p>
    </div>
    """, unsafe_allow_html=True)
    if is_spam:
        st.warning(desc, icon=None)
    else:
        st.success(desc, icon=None)


# ═══════════════════════════ GIAO DIỆN CHÍNH ═════════════════════════════
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "batch_upload_version" not in st.session_state:
    st.session_state.batch_upload_version = 0
if "batch_sessions" not in st.session_state:
    st.session_state.batch_sessions = []
    previous_result = st.session_state.batch_results
    if previous_result is not None:
        st.session_state.batch_sessions.append({
            "id": 1,
            "created_at": "Phiên gần nhất",
            "model": "Không xác định",
            **previous_result,
        })
if "selected_batch_session_id" not in st.session_state:
    sessions = st.session_state.batch_sessions
    st.session_state.selected_batch_session_id = sessions[-1]["id"] if sessions else None

# ────────────────────────── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">SpamGuard AI</div>
            <p class="sidebar-brand-description">
                Phân loại email Spam / Ham bằng học máy
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.radio(
        "Điều hướng",
        ["Dashboard", "Phân loại đơn lẻ", "Phân loại hàng loạt"],
        label_visibility="collapsed",
        key="active_page",
    )
    st.divider()
    st.caption("SpamGuard AI © 2026 — Bài tập nhóm TTNTTM")


# =========================== DASHBOARD PHIÊN ============================
if st.session_state.active_page == "Dashboard":
    st.subheader("Lịch sử phân loại hàng loạt")
    batch_sessions = st.session_state.batch_sessions
    if not batch_sessions:
        st.info("Chưa có phiên phân loại hàng loạt nào.", icon=None)
    else:
        for batch_session in reversed(batch_sessions):
            session_id = batch_session["id"]
            row_count = len(batch_session["data"])
            is_selected = session_id == st.session_state.selected_batch_session_id
            button_label = (
                f"Phiên {session_id}  |  {batch_session['created_at']}  |  "
                f"{row_count} email  |  {batch_session['model']}"
                f"{'  |  Đang xem' if is_selected else ''}"
            )
            st.button(
                button_label,
                key=f"select_batch_session_{session_id}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
                on_click=select_batch_session,
                args=(session_id,),
            )

        selected_session = next(
            (
                session
                for session in batch_sessions
                if session["id"] == st.session_state.selected_batch_session_id
            ),
            batch_sessions[-1],
        )
        session_df = selected_session["data"]
        st.markdown(
            f'<div class="session-heading">Kết quả của phiên {selected_session["id"]}</div>',
            unsafe_allow_html=True,
        )
        distribution_column, metrics_column = st.columns([1, 1])
        with distribution_column:
            st.markdown("#### Phân bố dự đoán Spam/Ham")
            render_prediction_distribution(session_df)
        with metrics_column:
            st.markdown("#### So sánh NB và SVM")
            render_metrics_comparison(session_df)

        st.subheader("Kết quả chi tiết")
        render_results_table(session_df.head(100), height=420)
        if len(session_df) > 100:
            st.caption(f"Đang hiển thị 100/{len(session_df)} email đầu tiên.")
        session_csv = session_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Tải toàn bộ kết quả CSV",
            data=session_csv,
            file_name=f"spamguard_session_{selected_session['id']}.csv",
            mime="text/csv",
        )

# ========================= TAB 1: PHÂN LOẠI ĐƠN LẺ =======================
elif st.session_state.active_page == "Phân loại đơn lẻ":
    st.subheader("Nhập nội dung email cần kiểm tra")

    subject_col, body_col = st.columns([1, 1])
    with subject_col:
        subject = st.text_input(
            "Tiêu đề email",
            placeholder="Ví dụ: Congratulations! You won a FREE gift",
        )
    with body_col:
        body = st.text_area(
            "Nội dung email",
            height=110,
            placeholder="Ví dụ: Click here to claim your prize now! Limited time offer...",
        )

    # Nút phân loại
    classify_btn = st.button(
        "Phân loại email",
        type="primary",
        use_container_width=True,
    )

    if classify_btn:
        # Kiểm tra đầu vào rỗng
        if not subject.strip() and not body.strip():
            st.error("Vui lòng nhập tiêu đề hoặc nội dung email.", icon=None)
        elif any(not check_model_exists(name) for name in MODEL_OPTIONS.values()):
            st.error("Cần huấn luyện cả Naive Bayes và SVM trước khi phân loại.", icon=None)
        else:
            predict_fn, err = get_predict_function()
            if predict_fn is None:
                st.error(err, icon=None)
            else:
                try:
                    with st.spinner("Đang phân loại..."):
                        predictions = {
                            model_name: predict_fn(subject, body, model_key)
                            for model_name, model_key in MODEL_OPTIONS.items()
                        }
                    result_columns = st.columns(2)
                    for column, (model_name, prediction) in zip(
                        result_columns, predictions.items()
                    ):
                        with column:
                            render_result(prediction, model_name)
                except ValueError as ve:
                    st.error(f"Lỗi đầu vào: {ve}", icon=None)
                except FileNotFoundError as fe:
                    st.error(str(fe), icon=None)
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi không mong muốn: {e}", icon=None)

    with st.expander("Xem ví dụ mẫu"):
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
elif st.session_state.active_page == "Phân loại hàng loạt":
    if st.session_state.batch_results is not None:
        render_batch_results()
    else:
        st.subheader("Phân loại email từ file CSV")
        st.info(
            "File cần có cột `text` hoặc cả hai cột `subject` và `body`. "
            "Định dạng mã hóa UTF-8.",
            icon=None,
        )

        uploaded_file = st.file_uploader(
            "Chọn file CSV",
            type=["csv"],
            help="Chọn file CSV chứa các email cần phân loại.",
            key=f"batch_upload_{st.session_state.batch_upload_version}",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Không thể đọc file CSV: {e}", icon=None)
                df = None

            if df is not None:
                st.write(f"Đã đọc **{len(df)}** dòng dữ liệu.")
                st.dataframe(
                    df.head(5),
                    use_container_width=True,
                    height=180,
                    hide_index=True,
                )

                has_subject_body = "subject" in df.columns and "body" in df.columns
                has_text = "text" in df.columns

                if not has_subject_body and not has_text:
                    st.error(
                        "File CSV cần có cột `text`, hoặc cả `subject` và `body`.",
                        icon=None,
                    )
                else:
                    action_slot = st.empty()
                    batch_btn = action_slot.button(
                        "Phân loại tất cả",
                        type="primary",
                        use_container_width=True,
                    )

                    if batch_btn:
                        progress = action_slot.progress(0, text="Đang chuẩn bị...")
                        missing_models = [
                            name for name, key in MODEL_OPTIONS.items()
                            if not check_model_exists(key)
                        ]
                        if missing_models:
                            action_slot.empty()
                            st.error(
                                f"Chưa có mô hình: {', '.join(missing_models)}. "
                                "Hãy chạy `python train.py` trước.",
                                icon=None,
                            )
                        else:
                            if has_subject_body:
                                predict_fn, err = get_predict_function()
                                mode = "subject_body"
                            else:
                                predict_fn, err = get_predict_email_function()
                                mode = "text"

                            if predict_fn is None:
                                action_slot.empty()
                                st.error(err, icon=None)
                            else:
                                results = {model_name: [] for model_name in MODEL_OPTIONS}
                                errors = 0
                                row_count = len(df)

                                for row_number, (_, row) in enumerate(
                                    df.iterrows(), start=1
                                ):
                                    for model_name, model_key in MODEL_OPTIONS.items():
                                        try:
                                            if mode == "subject_body":
                                                subject_text = str(row.get("subject", ""))
                                                body_text = str(row.get("body", ""))
                                                label = predict_fn(
                                                    subject_text, body_text, model_key
                                                )
                                            else:
                                                text = str(row.get("text", ""))
                                                label = predict_fn(text, model_key)
                                        except Exception:
                                            label = "error"
                                            errors += 1
                                        results[model_name].append(label)
                                    progress.progress(
                                        row_number / row_count if row_count else 1,
                                        text=f"Đang phân loại... {row_number}/{row_count}",
                                    )

                                progress.empty()
                                for model_name, labels in results.items():
                                    df[model_name] = labels
                                st.session_state.batch_results = {
                                    "data": df,
                                    "errors": errors,
                                }
                                session_id = max(
                                    (session["id"] for session in st.session_state.batch_sessions),
                                    default=0,
                                ) + 1
                                batch_session = {
                                    "id": session_id,
                                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "model": "Naive Bayes + SVM",
                                    "data": df,
                                    "errors": errors,
                                }
                                st.session_state.batch_sessions.append(batch_session)
                                st.session_state.selected_batch_session_id = session_id
                                st.rerun()
