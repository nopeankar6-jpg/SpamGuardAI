"""SpamGuard AI – Giao dien phan loai email Spam/Ham bang Streamlit (SV3).

Su dung ham predict_email_from_parts(subject, body, model_name)
tu predict.py do SV2 cung cap.
"""

import streamlit as st
import pandas as pd
import os

# ─────────────────────── Cau hinh trang ──────────────────────────────────
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────── CSS tuy chinh ───────────────────────────────────
st.markdown("""
<style>
    /* Thu gon padding mac dinh cua Streamlit */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    header[data-testid="stHeader"] {
        height: 2.5rem;
    }

    /* Header chinh */
    .app-header {
        margin-bottom: 0.6rem;
    }
    .app-header h1 {
        font-size: 1.7rem;
        margin: 0 0 0.15rem 0;
        font-weight: 700;
    }
    .app-header p {
        color: #888;
        font-size: 0.85rem;
        margin: 0;
    }

    /* Metric card */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.15rem 0 0 0;
    }
    .metric-card.spam .metric-value { color: #e74c3c; }
    .metric-card.ham  .metric-value { color: #27ae60; }
    .metric-card.rate .metric-value { color: #2980b9; }
    .metric-card.total .metric-value { color: #2c3e50; }

    /* Ket qua phan loai */
    .result-box {
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .result-box.spam {
        background: #fdecea;
        border-left: 4px solid #e74c3c;
    }
    .result-box.ham {
        background: #eafaf1;
        border-left: 4px solid #27ae60;
    }
    .result-box .label {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
    }
    .result-box.spam .label { color: #c0392b; }
    .result-box.ham  .label { color: #1e8449; }
    .result-box .desc {
        font-size: 0.82rem;
        color: #555;
        margin: 0.15rem 0 0 0;
    }

    /* Bo khoang trong thua cua tabs */
    div.stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    div.stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.8rem;
    }

    /* Chart container */
    .chart-section {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────── Session state ──────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"subject","body","model","result"}


# ──────────────────── Ham tro giup ───────────────────────────────────────
def check_model_exists(model_name: str) -> bool:
    paths = {"nb": "models/naive_bayes.pkl", "svm": "models/svm.pkl"}
    return os.path.exists(paths.get(model_name, ""))


def get_predict_function():
    try:
        from predict import predict_email_from_parts
        return predict_email_from_parts, None
    except ImportError as e:
        return None, f"Khong the import predict.py: {e}"
    except Exception as e:
        return None, f"Loi khi nap module predict: {e}"


def get_predict_email_function():
    try:
        from predict import predict_email
        return predict_email, None
    except ImportError as e:
        return None, f"Khong the import predict.py: {e}"
    except Exception as e:
        return None, f"Loi khi nap module predict: {e}"


def render_result(label: str, model_display: str):
    is_spam = label.lower() == "spam"
    css = "spam" if is_spam else "ham"
    tag = "SPAM" if is_spam else "HAM"
    desc = "Thu rac (spam)" if is_spam else "Thu hop le (ham)"
    st.markdown(f"""
    <div class="result-box {css}">
        <p class="label">{tag}</p>
        <p class="desc">Nhan dien: {desc} — Mo hinh: {model_display}</p>
    </div>
    """, unsafe_allow_html=True)


MODEL_OPTIONS = {"Naive Bayes": "nb", "SVM": "svm"}


# ═══════════════════════ HEADER ══════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1>SpamGuard AI</h1>
    <p>Phan loai email Spam / Ham — Naive Bayes &bull; SVM &bull; TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════ TABS CHINH ══════════════════════════════════════
tab_single, tab_batch, tab_dashboard = st.tabs([
    "Phan loai don le",
    "Phan loai hang loat",
    "Dashboard",
])

# ======================== TAB 1: DON LE ==================================
with tab_single:
    # Layout 2 cot: form ben trai, ket qua ben phai
    col_form, col_result = st.columns([3, 2], gap="medium")

    with col_form:
        selected_model_display = st.selectbox(
            "Mo hinh",
            list(MODEL_OPTIONS.keys()),
        )
        selected_model = MODEL_OPTIONS[selected_model_display]

        subject = st.text_input(
            "Tieu de (Subject)",
            placeholder="VD: Congratulations! You won a FREE gift",
        )
        body = st.text_area(
            "Noi dung (Body)",
            height=150,
            placeholder="VD: Click here to claim your prize now...",
        )
        classify_btn = st.button("Phan loai", type="primary", use_container_width=True)

    with col_result:
        st.markdown("**Ket qua**")

        if classify_btn:
            if not subject.strip() and not body.strip():
                st.error("Vui long nhap tieu de hoac noi dung email.")
            elif not check_model_exists(selected_model):
                st.error(
                    f"Mo hinh {selected_model_display} chua duoc huan luyen. "
                    "Hay chay `python train.py` truoc."
                )
            else:
                predict_fn, err = get_predict_function()
                if predict_fn is None:
                    st.error(err)
                else:
                    try:
                        result = predict_fn(subject, body, selected_model)
                        render_result(result, selected_model_display)
                        # Luu vao session
                        st.session_state.history.append({
                            "subject": subject,
                            "body": body[:80],
                            "model": selected_model_display,
                            "result": result,
                        })
                    except ValueError as ve:
                        st.error(f"Loi dau vao: {ve}")
                    except FileNotFoundError as fe:
                        st.error(str(fe))
                    except Exception as e:
                        st.error(f"Loi khong mong muon: {e}")
        else:
            st.caption("Nhap email va nhan Phan loai de xem ket qua.")

    # Vi du mau (gon)
    with st.expander("Vi du mau"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Spam**")
            st.code(
                "Subject: You WON a FREE iPhone 15\n"
                "Body: Click here NOW: http://bit.ly/xyz",
                language=None,
            )
        with c2:
            st.markdown("**Ham**")
            st.code(
                "Subject: Meeting moved to 3pm\n"
                "Body: Hi team, please confirm. Thanks.",
                language=None,
            )


# ======================== TAB 2: HANG LOAT ===============================
with tab_batch:
    col_upload, col_cfg = st.columns([3, 1], gap="medium")

    with col_cfg:
        batch_model_display = st.selectbox(
            "Mo hinh",
            list(MODEL_OPTIONS.keys()),
            key="batch_model",
        )
        batch_model = MODEL_OPTIONS[batch_model_display]

        st.markdown(
            "**Dinh dang CSV:**\n"
            "- Cot `text`, hoac\n"
            "- Hai cot `subject` va `body`"
        )

    with col_upload:
        uploaded_file = st.file_uploader("Chon file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Khong the doc file CSV: {e}")
            df = None

        if df is not None:
            st.caption(f"Da doc {len(df)} dong du lieu.")
            st.dataframe(df.head(5), use_container_width=True, height=180)

            has_sb = "subject" in df.columns and "body" in df.columns
            has_text = "text" in df.columns

            if not has_sb and not has_text:
                st.error("File CSV can co cot `text`, hoac ca `subject` va `body`.")
            elif not check_model_exists(batch_model):
                st.error(
                    f"Mo hinh {batch_model_display} chua duoc huan luyen. "
                    "Hay chay `python train.py` truoc."
                )
            else:
                batch_btn = st.button("Phan loai tat ca", type="primary", use_container_width=True)
                if batch_btn:
                    if has_sb:
                        predict_fn, err = get_predict_function()
                        mode = "sb"
                    else:
                        predict_fn, err = get_predict_email_function()
                        mode = "text"

                    if predict_fn is None:
                        st.error(err)
                    else:
                        results = []
                        errors = 0
                        progress = st.progress(0, text="Dang phan loai...")
                        for idx, row in df.iterrows():
                            try:
                                if mode == "sb":
                                    lbl = predict_fn(str(row.get("subject", "")),
                                                     str(row.get("body", "")),
                                                     batch_model)
                                else:
                                    lbl = predict_fn(str(row.get("text", "")),
                                                     batch_model)
                            except Exception:
                                lbl = "error"
                                errors += 1
                            results.append(lbl)
                            progress.progress((idx + 1) / len(df),
                                              text=f"Dang phan loai... {idx + 1}/{len(df)}")
                        progress.empty()

                        df["prediction"] = results
                        spam_n = (df["prediction"].str.lower() == "spam").sum()
                        ham_n = (df["prediction"].str.lower() == "ham").sum()
                        err_n = (df["prediction"] == "error").sum()

                        # Luu vao session
                        for _, r in df.iterrows():
                            st.session_state.history.append({
                                "subject": str(r.get("subject", ""))[:40],
                                "body": str(r.get("body", str(r.get("text", ""))))[:40],
                                "model": batch_model_display,
                                "result": r["prediction"],
                            })

                        # Thong ke nhanh
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Tong", len(df))
                        m2.metric("Spam", int(spam_n))
                        m3.metric("Ham", int(ham_n))
                        m4.metric("Loi", int(err_n))

                        if errors > 0:
                            st.warning(f"Co {errors} email gap loi khi phan loai.")

                        # Bang + download
                        st.dataframe(df, use_container_width=True)
                        csv_out = df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="Tai ket qua (CSV)",
                            data=csv_out,
                            file_name="spamguard_results.csv",
                            mime="text/csv",
                        )


# ======================== TAB 3: DASHBOARD ===============================
with tab_dashboard:
    history = st.session_state.history

    if not history:
        st.info("Chua co du lieu. Hay phan loai it nhat mot email de xem thong ke.")
    else:
        df_hist = pd.DataFrame(history)
        total = len(df_hist)
        spam_count = (df_hist["result"].str.lower() == "spam").sum()
        ham_count = (df_hist["result"].str.lower() == "ham").sum()
        error_count = total - spam_count - ham_count
        spam_pct = (spam_count / total * 100) if total > 0 else 0
        ham_pct = (ham_count / total * 100) if total > 0 else 0

        # ---- Metric cards ----
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.markdown(f"""
            <div class="metric-card total">
                <p class="metric-value">{total:,}</p>
                <p class="metric-label">Tong email</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card spam">
                <p class="metric-value">{int(spam_count):,}</p>
                <p class="metric-label">Spam</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card ham">
                <p class="metric-value">{int(ham_count):,}</p>
                <p class="metric-label">Ham</p>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card rate">
                <p class="metric-value">{spam_pct:.1f}%</p>
                <p class="metric-label">Ty le Spam</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")  # spacer nho

        # ---- Bieu do ----
        chart_left, chart_right = st.columns(2, gap="medium")

        with chart_left:
            st.markdown("**Phan bo ket qua**")
            dist_data = pd.DataFrame({
                "Nhan": ["Spam", "Ham"],
                "So luong": [int(spam_count), int(ham_count)],
            }).set_index("Nhan")
            st.bar_chart(dist_data, use_container_width=True, height=280)

        with chart_right:
            st.markdown("**Ty le phan tram**")
            pct_data = pd.DataFrame({
                "Nhan": ["Spam", "Ham"],
                "Phan tram": [round(spam_pct, 1), round(ham_pct, 1)],
            }).set_index("Nhan")
            st.bar_chart(pct_data, use_container_width=True, height=280)

        # ---- Thong ke theo mo hinh ----
        st.markdown("**Thong ke theo mo hinh**")
        model_stats = df_hist.groupby("model")["result"].value_counts().unstack(fill_value=0)
        st.dataframe(model_stats, use_container_width=True)

        # ---- Lich su phan loai ----
        with st.expander("Lich su phan loai chi tiet"):
            display_df = df_hist.copy()
            display_df.index = range(1, len(display_df) + 1)
            display_df.index.name = "STT"
            st.dataframe(display_df, use_container_width=True)

        # Nut xoa lich su
        if st.button("Xoa lich su phien"):
            st.session_state.history = []
            st.rerun()
