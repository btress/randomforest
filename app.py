import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ==============================================================================
# CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Phân tích & Dự đoán Hiến tạng (Random Forest)",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
TARGET = "oppose"

NUMERIC_COLS = [
    "age",                  # tuổi (10-17)
    "knowledge_score",      # điểm kiến thức về hiến tạng (0-10)
    "trust_healthcare",     # tin tưởng hệ thống y tế (1-5)
    "fear_body_integrity",  # lo ngại mất sự toàn vẹn cơ thể (1-5)
    "belief_brain_death",   # nghi ngờ chẩn đoán chết não (1-5)
    "religiosity",          # mức độ tôn giáo/tín ngưỡng (1-5)
    "media_exposure",       # mức độ tiếp cận thông tin về hiến tạng (1-5)
    "parent_support",       # phụ huynh ủng hộ hiến tạng (1-5)
]

CATEGORICAL_COLS = [
    "gender",               # Nam / Nữ
    "residence",            # Thành thị / Nông thôn
    "religion",             # Không / Phật giáo / Công giáo / Khác
    "family_income",        # Thấp / Trung bình / Cao
    "knows_donor",          # Có / Không
    "school_education",     # Có / Không
]

FEATURE_NAMES_VN = {
    "age": "Độ tuổi",
    "knowledge_score": "Điểm kiến thức hiến tạng",
    "trust_healthcare": "Niềm tin vào y tế",
    "fear_body_integrity": "Lo sợ mất toàn vẹn cơ thể",
    "belief_brain_death": "Nghi ngờ chẩn đoán chết não",
    "religiosity": "Mức độ tín ngưỡng/tôn giáo",
    "media_exposure": "Tiếp cận thông tin truyền thông",
    "parent_support": "Phụ huynh ủng hộ",
    "gender": "Giới tính",
    "residence": "Nơi cư trú",
    "religion": "Tôn giáo",
    "family_income": "Thu nhập gia đình",
    "knows_donor": "Quen biết người hiến/nhận",
    "school_education": "Đã học ở trường",
}


# ==============================================================================
# HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN
# ==============================================================================
@st.cache_data
def load_csv_data(filepath_or_buffer):
    if isinstance(filepath_or_buffer, str):
        for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
            try:
                df = pd.read_csv(filepath_or_buffer, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        df = pd.read_csv(filepath_or_buffer)

    missing = [c for c in NUMERIC_COLS + CATEGORICAL_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"File thiếu các cột bắt buộc: {missing}")
    
    df = df[df["age"] < 18].copy()
    df = df.dropna(subset=[TARGET])
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)
    return df


def build_pipeline(n_estimators=300, max_depth=8, min_samples_leaf=3):
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", num, NUMERIC_COLS), ("cat", cat, CATEGORICAL_COLS)])
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("rf", rf)])


@st.cache_resource
def train_model(df, test_size=0.2, n_estimators=300, max_depth=8, do_tune=False):
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    base_pipe = build_pipeline(n_estimators=n_estimators, max_depth=max_depth)

    if do_tune:
        param_dist = {
            "rf__n_estimators": [150, 300, 500],
            "rf__max_depth": [5, 8, 12, None],
            "rf__min_samples_leaf": [1, 3, 5],
            "rf__max_features": ["sqrt", "log2", 0.5],
        }
        search = RandomizedSearchCV(
            base_pipe,
            param_dist,
            n_iter=15,
            scoring="roc_auc",
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
    else:
        base_pipe.fit(X_train, y_train)
        model = base_pipe
        best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
        cv_score = None

    # Đánh giá
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred_test),
        "roc_auc": roc_auc_score(y_test, proba_test),
        "f1": f1_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "report": classification_report(y_test, pred_test, target_names=["Ủng hộ (0)", "Phản đối (1)"], output_dict=True),
    }

    # Gini importance
    names = model.named_steps["pre"].get_feature_names_out()
    imp = model.named_steps["rf"].feature_importances_
    gini_dict = {}
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        mask = [(n == f"num__{col}") or n.startswith(f"cat__{col}_") for n in names]
        gini_dict[col] = imp[np.array(mask)].sum()
    imp_gini = pd.Series(gini_dict).sort_values(ascending=False)

    # Permutation importance
    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp_perm = pd.DataFrame(
        {"mean": perm.importances_mean, "std": perm.importances_std}, index=X_test.columns
    ).sort_values("mean", ascending=False)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "proba_test": proba_test,
        "pred_test": pred_test,
        "metrics": metrics,
        "imp_gini": imp_gini,
        "imp_perm": imp_perm,
        "best_params": best_params,
        "cv_score": cv_score,
    }


def compute_direction_table(df):
    rows = []
    for col in NUMERIC_COLS:
        r = df[[col, TARGET]].corr().iloc[0, 1]
        tinh_chat = "Tăng nguy cơ phản đối (+)" if r > 0.05 else ("Giảm nguy cơ phản đối (-)" if r < -0.05 else "Ảnh hưởng ít")
        rows.append({
            "Đặc trưng": FEATURE_NAMES_VN.get(col, col),
            "Mã biến": col,
            "Loại": "Biến số (Thang đo)",
            "Chỉ số": f"Tương quan r = {r:+.2f}",
            "Xu hướng ảnh hưởng": tinh_chat,
        })
    for col in CATEGORICAL_COLS:
        rates = df.groupby(col)[TARGET].mean().round(3).to_dict()
        desc = ", ".join([f"{k}: {v*100:.1f}%" for k, v in rates.items()])
        rows.append({
            "Đặc trưng": FEATURE_NAMES_VN.get(col, col),
            "Mã biến": col,
            "Loại": "Biến phân loại",
            "Chỉ số": f"Tỷ lệ phản đối: {desc}",
            "Xu hướng ảnh hưởng": "Thay đổi theo nhóm",
        })
    return pd.DataFrame(rows)


# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
def main():
    st.title("🫀 Dự đoán & Phân tích Quan điểm Phản đối Hiến tạng")
    st.caption("Ứng dụng học máy Random Forest phân tích các nhân tố tâm lý - xã hội ở lứa tuổi thanh thiếu niên (< 18 tuổi)")

    # ---- SIDEBAR ----
    st.sidebar.header("⚙️ Cấu hình Dữ liệu & Mô hình")

    # 1. Chọn file dữ liệu
    available_files = [f for f in os.listdir(".") if f.endswith(".csv") and "survey" in f]
    default_index = 0
    if "organ_donation_survey_1500-v4.csv" in available_files:
        default_index = available_files.index("organ_donation_survey_1500-v4.csv")

    source_type = st.sidebar.radio("Nguồn dữ liệu huấn luyện:", ["File có sẵn", "Tải file mới lên (.csv)"])

    df = None
    if source_type == "File có sẵn":
        if available_files:
            selected_file = st.sidebar.selectbox("Chọn file khảo sát:", available_files, index=default_index)
            try:
                df = load_csv_data(selected_file)
                st.sidebar.success(f"Đã tải {len(df)} dòng dữ liệu hợp lệ.")
            except Exception as e:
                st.sidebar.error(f"Lỗi đọc file: {e}")
        else:
            st.sidebar.warning("Không tìm thấy file khảo sát có sẵn.")
    else:
        uploaded_file = st.sidebar.file_uploader("Tải file CSV khảo sát lên", type=["csv"])
        if uploaded_file is not None:
            try:
                df = load_csv_data(uploaded_file)
                st.sidebar.success(f"Đã tải {len(df)} dòng từ file tải lên.")
            except Exception as e:
                st.sidebar.error(f"Lỗi đọc file: {e}")

    if df is None:
        st.info("Vui lòng chọn hoặc tải lên file dữ liệu hợp lệ để bắt đầu.")
        return

    # 2. Siêu tham số
    st.sidebar.subheader("Tham số mô hình")
    test_size = st.sidebar.slider("Tỷ lệ tập kiểm tra (Test size):", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.sidebar.slider("Số lượng cây (n_estimators):", 50, 600, 300, 50)
    max_depth = st.sidebar.slider("Độ sâu tối đa (max_depth):", 3, 20, 8, 1)
    do_tune = st.sidebar.checkbox("Tự động tinh chỉnh siêu tham số (Tuning CV)", value=False)

    train_btn = st.sidebar.button("🔄 Huấn luyện lại mô hình", use_container_width=True)
    if train_btn:
        st.cache_resource.clear()

    # Huấn luyện mô hình
    with st.spinner("Đang huấn luyện mô hình Random Forest..."):
        try:
            train_results = train_model(
                df,
                test_size=test_size,
                n_estimators=n_estimators,
                max_depth=max_depth,
                do_tune=do_tune,
            )
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {e}")
            return

    # TABS CHÍNH
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tổng quan Dữ liệu",
        "🤖 Đánh giá Mô hình",
        "🔍 Phân tích Tầm quan trọng",
        "🎯 Dự đoán Cá nhân",
        "📁 Dự đoán Hàng loạt (File)",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: TỔNG QUAN DỮ LIỆU
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("1. Tổng quan Dữ liệu Khảo sát")
        c1, c2, c3, c4 = st.columns(4)
        n_total = len(df)
        n_oppose = int(df[TARGET].sum())
        n_agree = n_total - n_oppose
        pct_oppose = n_oppose / n_total * 100

        c1.metric("Tổng số người tham gia", f"{n_total:,}")
        c2.metric("Số phản đối (Nhãn 1)", f"{n_oppose:,}", f"{pct_oppose:.1f}%")
        c3.metric("Số ủng hộ/không phản đối (Nhãn 0)", f"{n_agree:,}", f"{100-pct_oppose:.1f}%")
        c4.metric("Độ tuổi trung bình", f"{df['age'].mean():.1f} tuổi")

        st.markdown("---")
        st.write("##### Bảng dữ liệu khảo sát (10 dòng đầu):")
        st.dataframe(df.head(10), use_container_width=True)

        st.write("##### Phân bố các đặc trưng theo quan điểm phản đối:")
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            fig_p1, ax_p1 = plt.subplots(figsize=(6, 4))
            df.groupby("residence")[TARGET].mean().plot(kind="bar", color=["#4C78A8", "#F58518"], ax=ax_p1)
            ax_p1.set_title("Tỷ lệ phản đối theo Nơi cư trú")
            ax_p1.set_ylabel("Tỷ lệ phản đối")
            ax_p1.set_ylim(0, 1)
            plt.xticks(rotation=0)
            st.pyplot(fig_p1)
            plt.close(fig_p1)

        with col_plot2:
            fig_p2, ax_p2 = plt.subplots(figsize=(6, 4))
            df.groupby("religion")[TARGET].mean().plot(kind="bar", color="#54A24B", ax=ax_p2)
            ax_p2.set_title("Tỷ lệ phản đối theo Tôn giáo")
            ax_p2.set_ylabel("Tỷ lệ phản đối")
            ax_p2.set_ylim(0, 1)
            plt.xticks(rotation=0)
            st.pyplot(fig_p2)
            plt.close(fig_p2)

    # --------------------------------------------------------------------------
    # TAB 2: ĐÁNH GIÁ MÔ HÌNH
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("2. Kết quả Đánh giá Mô hình trên Tập kiểm tra")
        metrics = train_results["metrics"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.3f}")
        m2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        m3.metric("F1-Score", f"{metrics['f1']:.3f}")
        m4.metric("Precision (Phản đối)", f"{metrics['precision']:.3f}")
        m5.metric("Recall (Phản đối)", f"{metrics['recall']:.3f}")

        if train_results["cv_score"] is not None:
            st.info(f"Kết quả Tinh chỉnh CV (5-fold ROC-AUC): **{train_results['cv_score']:.3f}** | Siêu tham số tối ưu: `{train_results['best_params']}`")

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.write("##### Ma trận nhầm lẫn (Confusion Matrix)")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                train_results["y_test"],
                train_results["pred_test"],
                display_labels=["Không PĐ", "Phản đối"],
                cmap="Blues",
                ax=ax_cm,
                colorbar=False,
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_m2:
            st.write("##### Đường cong ROC (ROC Curve)")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(
                train_results["y_test"],
                train_results["proba_test"],
                ax=ax_roc,
                plot_chance_level=True,
                curve_kwargs={"color": "#E45756"},
            )
            ax_roc.set_title(f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        st.write("##### Báo cáo phân loại chi tiết (Classification Report):")
        report_df = pd.DataFrame(metrics["report"]).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: PHÂN TÍCH TẦM QUAN TRỌNG ĐẶC TRƯNG
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("3. Phân tích Các yếu tố Ảnh hưởng đến Quyết định")

        col_imp1, col_imp2 = st.columns(2)
        imp_gini = train_results["imp_gini"]
        imp_perm = train_results["imp_perm"]

        with col_imp1:
            st.write("##### Gini Importance (Độ suy giảm tạp chất)")
            fig_g, ax_g = plt.subplots(figsize=(7, 5))
            plot_gini = imp_gini.sort_values(ascending=True)
            plot_gini.index = [FEATURE_NAMES_VN.get(c, c) for c in plot_gini.index]
            plot_gini.plot.barh(ax=ax_g, color="#4C78A8")
            ax_g.set_xlabel("Độ quan trọng Gini")
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)

        with col_imp2:
            st.write("##### Permutation Importance (Độ sụt giảm ROC-AUC)")
            fig_p, ax_p = plt.subplots(figsize=(7, 5))
            plot_perm = imp_perm["mean"].sort_values(ascending=True)
            plot_perm.index = [FEATURE_NAMES_VN.get(c, c) for c in plot_perm.index]
            plot_perm.plot.barh(ax=ax_p, color="#F58518")
            ax_p.set_xlabel("Mức giảm ROC-AUC")
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

        st.markdown("---")
        st.write("##### Bảng tổng hợp Chiều ảnh hưởng và Mức độ tương quan:")
        direction_df = compute_direction_table(df)
        st.dataframe(direction_df, use_container_width=True)

        st.markdown("---")
        st.write("##### Biểu đồ Partial Dependence (Mối quan hệ phi tuyến của các yếu tố hàng đầu):")
        top_num = [c for c in imp_perm.index if c in NUMERIC_COLS][:4]
        if top_num:
            fig_pdp, ax_pdp = plt.subplots(1, len(top_num), figsize=(4 * len(top_num), 3.5))
            if len(top_num) == 1:
                ax_pdp = [ax_pdp]
            PartialDependenceDisplay.from_estimator(
                train_results["model"],
                train_results["X_train"],
                top_num,
                ax=ax_pdp,
                kind="average",
            )
            plt.tight_layout()
            st.pyplot(fig_pdp)
            plt.close(fig_pdp)

    # --------------------------------------------------------------------------
    # TAB 4: DỰ ĐOÁN CHO CÁ NHÂN
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("4. Dự đoán Quan điểm cho Từng Học sinh / Thanh thiếu niên")
        st.write("Nhập thông tin cá nhân dưới đây để mô hình tính toán xác suất phản đối hiến tạng:")

        with st.form("single_predict_form"):
            col_u1, col_u2 = st.columns(2)

            with col_u1:
                st.markdown("##### 👤 Thông tin Nhân khẩu học")
                age_val = st.slider("Tuổi (10 - 17)", 10, 17, 15)
                gender_val = st.selectbox("Giới tính", ["Nam", "Nữ"])
                residence_val = st.selectbox("Nơi cư trú", ["Thành thị", "Nông thôn"])
                religion_val = st.selectbox("Tôn giáo", ["Không", "Phật giáo", "Công giáo", "Khác"])
                family_income_val = st.selectbox("Thu nhập gia đình", ["Thấp", "Trung bình", "Cao"])
                knows_donor_val = st.selectbox("Có quen biết người từng hiến/nhận tạng?", ["Có", "Không"])
                school_education_val = st.selectbox("Đã từng được học về hiến tạng ở trường học?", ["Có", "Không"])

            with col_u2:
                st.markdown("##### 🧠 Nhận thức, Tâm lý & Xã hội")
                knowledge_score_val = st.slider("Điểm kiến thức về hiến tạng (0 - 10)", 0.0, 10.0, 5.0, 0.5)
                trust_healthcare_val = st.slider("Mức độ tin tưởng hệ thống y tế (1: Rất thấp -> 5: Rất cao)", 1, 5, 3)
                fear_body_integrity_val = st.slider("Lo ngại mất sự toàn vẹn cơ thể sau chết (1: Không lo -> 5: Rất lo)", 1, 5, 3)
                belief_brain_death_val = st.slider("Nghi ngờ tiêu chuẩn chẩn đoán chết não (1: Không nghi ngờ -> 5: Rất nghi ngờ)", 1, 5, 3)
                religiosity_val = st.slider("Mức độ sùng đạo / gắn kết tín ngưỡng (1: Thấp -> 5: Rất cao)", 1, 5, 3)
                media_exposure_val = st.slider("Mức độ tiếp cận truyền thông về hiến tạng (1: Hiếm khi -> 5: Rất thường xuyên)", 1, 5, 3)
                parent_support_val = st.slider("Mức độ phụ huynh ủng hộ hiến tạng (1: Hoàn toàn phản đối -> 5: Rất ủng hộ)", 1, 5, 3)

            submit_btn = st.form_submit_button("🚀 Dự đoán Quan điểm", use_container_width=True)

        if submit_btn:
            input_dict = {
                "age": [age_val],
                "gender": [gender_val],
                "residence": [residence_val],
                "religion": [religion_val],
                "family_income": [family_income_val],
                "knows_donor": [knows_donor_val],
                "school_education": [school_education_val],
                "knowledge_score": [knowledge_score_val],
                "trust_healthcare": [trust_healthcare_val],
                "fear_body_integrity": [fear_body_integrity_val],
                "belief_brain_death": [belief_brain_death_val],
                "religiosity": [religiosity_val],
                "media_exposure": [media_exposure_val],
                "parent_support": [parent_support_val],
            }
            input_df = pd.DataFrame(input_dict)
            model = train_results["model"]
            proba_oppose = model.predict_proba(input_df)[0, 1]
            pred_oppose = int(proba_oppose >= 0.5)

            st.markdown("---")
            st.write("### 📌 Kết quả Dự đoán:")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred_oppose == 1:
                    st.error(f"### Dự đoán: PHẢN ĐỐI\n**Xác suất phản đối: {proba_oppose*100:.1f}%**")
                else:
                    st.success(f"### Dự đoán: ỦNG HỘ / KHÔNG PHẢN ĐỐI\n**Xác suất phản đối: {proba_oppose*100:.1f}%**")

                st.progress(float(proba_oppose))

            with res_col2:
                st.write("##### 💡 Phân tích & Gợi ý định hướng can thiệp:")
                recommendations = []
                if knowledge_score_val < 5.0:
                    recommendations.append("- **Nâng cao hiểu biết:** Học sinh có điểm kiến thức thấp (< 5.0). Cần tăng cường giáo dục thông tin khoa học chính xác về quy trình hiến tạng.")
                if fear_body_integrity_val >= 4:
                    recommendations.append("- **Giải tỏa nỗi sợ toàn vẹn cơ thể:** Tâm lý sợ biến dạng thi thể rất cao. Cần tư vấn y khoa giải thích về kỹ thuật phẫu thuật vô trùng, trang nghiêm và tôn trọng cơ thể người hiến.")
                if belief_brain_death_val >= 4:
                    recommendations.append("- **Giải thích khái niệm chết não:** Có sự hoài nghi cao về chẩn đoán chết não. Cần giải thích quy trình chẩn đoán nghiêm ngặt của hội đồng chuyên môn y khoa độc lập.")
                if parent_support_val <= 2:
                    recommendations.append("- **Gia đình và phụ huynh:** Phụ huynh có xu hướng không ủng hộ. Các chương trình truyền thông nên hướng đến cả gia đình thay vì chỉ cá nhân học sinh.")
                if school_education_val == "Không":
                    recommendations.append("- **Tích hợp học đường:** Chưa từng được tiếp cận chủ đề này ở trường học. Nên đưa vào các buổi sinh hoạt ngoại khóa hoặc môn giáo dục công dân / sinh học.")

                if recommendations:
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.markdown("✅ Các chỉ số nhận thức và tâm lý của học sinh này đều rất tích cực đối với việc ủng hộ hiến tạng.")

    # --------------------------------------------------------------------------
    # TAB 5: DỰ ĐOÁN HÀNG LOẠT (FILE)
    # --------------------------------------------------------------------------
    with tab5:
        st.subheader("5. Dự đoán theo File Dữ liệu Hàng loạt")
        st.write("Dự đoán nhãn `oppose` và `xac_suat_phan_doi` cho danh sách học sinh từ file CSV.")

        batch_source = st.radio("Chọn nguồn file cần dự đoán:", ["Dùng file test.csv có sẵn trong thư mục", "Tải file CSV mới lên"], horizontal=True)

        test_df = None
        if batch_source == "Dùng file test.csv có sẵn trong thư mục":
            if os.path.exists("test.csv"):
                try:
                    for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
                        try:
                            test_df = pd.read_csv("test.csv", encoding=enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    st.write(f"Đã đọc file `test.csv` ({len(test_df)} dòng):")
                    st.dataframe(test_df.head(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi đọc test.csv: {e}")
            else:
                st.warning("Không tìm thấy file `test.csv` trong thư mục.")
        else:
            up_test = st.file_uploader("Tải file CSV kiểm tra (cần có các cột đặc trưng, cột oppose có thể để trống)", type=["csv"], key="batch_up")
            if up_test is not None:
                try:
                    test_df = pd.read_csv(up_test)
                    st.write(f"Đã tải file ({len(test_df)} dòng):")
                    st.dataframe(test_df.head(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi đọc file tải lên: {e}")

        if test_df is not None:
            need_cols = NUMERIC_COLS + CATEGORICAL_COLS
            missing_cols = [c for c in need_cols if c not in test_df.columns]
            if missing_cols:
                st.error(f"File còn thiếu các cột bắt buộc: {missing_cols}")
            else:
                if st.button("⚡ Thực hiện Dự đoán Hàng loạt", use_container_width=True):
                    with st.spinner("Đang tính toán dự đoán..."):
                        X_pred = test_df[need_cols].copy()
                        X_pred[NUMERIC_COLS] = X_pred[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
                        
                        probas = train_results["model"].predict_proba(X_pred)[:, 1]
                        preds = (probas >= 0.5).astype(int)

                        result_df = test_df.copy()
                        result_df[TARGET] = preds
                        result_df["xac_suat_phan_doi"] = probas.round(3)
                        result_df["phan_loai"] = np.where(preds == 1, "Phản đối", "Không phản đối / Ủng hộ")

                        n_opp = int(preds.sum())
                        n_tot = len(preds)
                        st.success(f"Dự đoán hoàn tất cho {n_tot} dòng!")

                        b_c1, b_c2, b_c3 = st.columns(3)
                        b_c1.metric("Tổng số dòng dự đoán", n_tot)
                        b_c2.metric("Số lượng phản đối (1)", f"{n_opp}", f"{n_opp/n_tot*100:.1f}%")
                        b_c3.metric("Số lượng không phản đối (0)", f"{n_tot - n_opp}", f"{(n_tot - n_opp)/n_tot*100:.1f}%")

                        st.write("##### Kết quả chi tiết:")
                        st.dataframe(result_df, use_container_width=True)

                        # Nút download CSV
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                        st.download_button(
                            label="📥 Tải file kết quả CSV",
                            data=csv_buffer.getvalue().encode("utf-8-sig"),
                            file_name="ket_qua_du_doan_test.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )


if __name__ == "__main__":
    main()
