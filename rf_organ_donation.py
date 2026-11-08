"""
Random Forest dự đoán & phân tích các yếu tố ảnh hưởng đến quan điểm PHẢN ĐỐI hiến tạng
ở người dưới 18 tuổi.

Cách dùng:
    python rf_organ_donation.py                                                                 # chạy với dữ liệu mô phỏng (demo)
    python rf_organ_donation.py --data organ_donation_survey_1500-v4.csv                        # chạy với dữ liệu khảo sát thật
    python rf_organ_donation.py --data organ_donation_survey_1500-v4.csv --predict test.csv     # chạy với dữ liệu khảo sát thật và dự đoán cho file test.csv (cột oppose để trống)
Dữ liệu thật cần có cột mục tiêu (mặc định 'oppose': 1 = phản đối, 0 = không phản đối)
và các cột đặc trưng. Sửa NUMERIC_COLS / CATEGORICAL_COLS bên dưới cho khớp bảng hỏi.

Lưu ý: dữ liệu mô phỏng chỉ để minh họa quy trình; KẾT QUẢ KHÔNG có giá trị nghiên cứu.
Với dữ liệu khảo sát trẻ vị thành niên thật, cần có sự đồng ý của phụ huynh/người giám hộ
và thông qua hội đồng đạo đức theo quy định.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    "knows_donor",          # Có / Không (quen biết người hiến/nhận tạng)
    "school_education",     # Có / Không (đã được học về hiến tạng ở trường)
]


# --------------------------------------------------------------------------- #
# 1. Dữ liệu
# --------------------------------------------------------------------------- #
def make_synthetic_data(n=1500, seed=RANDOM_STATE):
    """Sinh dữ liệu khảo sát mô phỏng (chỉ để demo quy trình)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(10, 18, n),
        "gender": rng.choice(["Nam", "Nữ"], n),
        "residence": rng.choice(["Thành thị", "Nông thôn"], n, p=[0.45, 0.55]),
        "religion": rng.choice(["Không", "Phật giáo", "Công giáo", "Khác"], n,
                               p=[0.45, 0.35, 0.1, 0.1]),
        "family_income": rng.choice(["Thấp", "Trung bình", "Cao"], n, p=[0.3, 0.5, 0.2]),
        "knows_donor": rng.choice(["Có", "Không"], n, p=[0.1, 0.9]),
        "school_education": rng.choice(["Có", "Không"], n, p=[0.25, 0.75]),
        "knowledge_score": np.clip(rng.normal(5, 2.2, n), 0, 10).round(1),
        "trust_healthcare": rng.integers(1, 6, n),
        "fear_body_integrity": rng.integers(1, 6, n),
        "belief_brain_death": rng.integers(1, 6, n),
        "religiosity": rng.integers(1, 6, n),
        "media_exposure": rng.integers(1, 6, n),
        "parent_support": rng.integers(1, 6, n),
    })
    z = (-1.2
         - 0.35 * (df["knowledge_score"] - 5)
         - 0.40 * (df["trust_healthcare"] - 3)
         + 0.55 * (df["fear_body_integrity"] - 3)
         + 0.45 * (df["belief_brain_death"] - 3)
         + 0.20 * (df["religiosity"] - 3)
         - 0.30 * (df["parent_support"] - 3)
         - 0.15 * (df["media_exposure"] - 3)
         - 0.50 * (df["school_education"] == "Có")
         - 0.40 * (df["knows_donor"] == "Có")
         + 0.25 * (df["residence"] == "Nông thôn")
         + rng.normal(0, 0.8, n))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    # thêm ít giá trị thiếu để minh họa xử lý missing
    for col in ["knowledge_score", "family_income"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df


def load_data(path):
    if path is None:
        print("[i] Không có --data, dùng dữ liệu mô phỏng.")
        return make_synthetic_data()
    df = pd.read_csv(path)
    missing = [c for c in NUMERIC_COLS + CATEGORICAL_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong dữ liệu: {missing}")
    df = df[df["age"] < 18]  # chỉ giữ đối tượng dưới 18 tuổi
    return df.dropna(subset=[TARGET])


# --------------------------------------------------------------------------- #
# 2. Mô hình
# --------------------------------------------------------------------------- #
def build_pipeline():
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", num, NUMERIC_COLS), ("cat", cat, CATEGORICAL_COLS)])
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("rf", rf)])


def tune(pipe, X_train, y_train):
    param_dist = {
        "rf__n_estimators": [200, 400, 600],
        "rf__max_depth": [None, 5, 8, 12],
        "rf__min_samples_leaf": [1, 3, 5, 10],
        "rf__max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"[i] Tham số tốt nhất: {search.best_params_}")
    print(f"[i] ROC-AUC CV (5-fold): {search.best_score_:.3f}")
    return search.best_estimator_


# --------------------------------------------------------------------------- #
# 3. Phân tích yếu tố ảnh hưởng
# --------------------------------------------------------------------------- #
def grouped_impurity_importance(model):
    """Gộp độ quan trọng (Gini) của các cột one-hot về biến gốc."""
    names = model.named_steps["pre"].get_feature_names_out()
    imp = model.named_steps["rf"].feature_importances_
    result = {}
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        mask = [(n == f"num__{col}") or n.startswith(f"cat__{col}_") for n in names]
        result[col] = imp[np.array(mask)].sum()
    return pd.Series(result).sort_values(ascending=False)


def direction_table(df):
    """Chiều ảnh hưởng gần đúng: tỷ lệ phản đối theo từng nhóm / hệ số tương quan."""
    rows = []
    for col in NUMERIC_COLS:
        r = df[[col, TARGET]].corr().iloc[0, 1]
        rows.append((col, f"corr với 'phản đối' = {r:+.2f}"))
    for col in CATEGORICAL_COLS:
        rate = df.groupby(col)[TARGET].mean().round(3).to_dict()
        rows.append((col, f"tỷ lệ phản đối theo nhóm: {rate}"))
    return pd.DataFrame(rows, columns=["feature", "chiều ảnh hưởng (mô tả)"])


def read_csv_vn(path):
    """Đọc CSV, tự thử các bảng mã hay gặp khi lưu từ Excel."""
    for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không đọc được file {path} (lỗi bảng mã)")


def predict_file(model, in_path, out_path):
    """Dự đoán cho từng dòng trong file CSV, điền 0/1 vào cột 'oppose'."""
    new = read_csv_vn(in_path)
    need = NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in need if c not in new.columns]
    if missing:
        raise ValueError(f"File {in_path} thiếu cột: {missing}")

    X_new = new[need].copy()
    X_new[NUMERIC_COLS] = X_new[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
    proba = model.predict_proba(X_new)[:, 1]

    new[TARGET] = (proba >= 0.5).astype(int)
    new["xac_suat_phan_doi"] = proba.round(3)
    new.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Đã dự đoán {len(new)} học sinh -> '{out_path}'")
    print(f"    Phản đối (1): {int(new[TARGET].sum())} | Không phản đối (0): {int((1 - new[TARGET]).sum())}")


def main(args):
    os.makedirs(args.out, exist_ok=True)
    df = load_data(args.data)
    print(f"[i] Số mẫu: {len(df)} | Tỷ lệ phản đối: {df[TARGET].mean():.1%}")
    counts = df[TARGET].value_counts()
    if len(counts) < 2 or counts.min() < 30:
        raise SystemExit(
            f"[LỖI] Cột '{TARGET}' trong dữ liệu huấn luyện chưa đủ để học: {counts.to_dict()}.\n"
            "      Cần có cả hai giá trị 0 và 1 (mỗi giá trị nên từ 30 dòng trở lên) và không để trống.\n"
            "      Lưu ý: file huấn luyện (--data) phải có sẵn cột oppose đã điền, file cần dự đoán (--predict) thì để trống.")

    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)  # PDP yêu cầu kiểu float
    X, y = df[NUMERIC_COLS + CATEGORICAL_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = tune(build_pipeline(), X_train, y_train)

    # ---- Đánh giá trên tập kiểm tra ----
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n=== ĐÁNH GIÁ TRÊN TẬP KIỂM TRA ===")
    print(classification_report(y_test, pred, target_names=["Không phản đối", "Phản đối"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f} | F1: {f1_score(y_test, pred):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[0], colorbar=False,
                                            display_labels=["Không PĐ", "Phản đối"])
    ax[0].set_title("Ma trận nhầm lẫn")
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[1])
    ax[1].set_title("Đường ROC")
    plt.tight_layout()
    plt.savefig(f"{args.out}/danh_gia_mo_hinh.png", dpi=150)
    plt.close()

    # ---- Độ quan trọng đặc trưng ----
    imp_gini = grouped_impurity_importance(model)
    perm = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                  n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    imp_perm = pd.DataFrame({"mean": perm.importances_mean, "std": perm.importances_std},
                            index=X_test.columns).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    imp_gini.sort_values().plot.barh(ax=ax[0], color="#4C78A8")
    ax[0].set_title("Độ quan trọng Gini (gộp biến gốc)")
    imp_perm["mean"].sort_values().plot.barh(
        ax=ax[1], xerr=imp_perm["std"].reindex(imp_perm["mean"].sort_values().index),
        color="#F58518")
    ax[1].set_title("Permutation importance (giảm ROC-AUC)")
    plt.tight_layout()
    plt.savefig(f"{args.out}/do_quan_trong_dac_trung.png", dpi=150)
    plt.close()

    summary = pd.concat([imp_gini.rename("gini"), imp_perm["mean"].rename("permutation")],
                        axis=1).sort_values("permutation", ascending=False)
    summary.to_csv(f"{args.out}/do_quan_trong.csv", encoding="utf-8-sig")
    print("\n=== ĐỘ QUAN TRỌNG ĐẶC TRƯNG ===")
    print(summary.round(4).to_string())

    # ---- Partial dependence cho các biến số quan trọng nhất ----
    top_num = [c for c in imp_perm.index if c in NUMERIC_COLS][:4]
    fig, ax = plt.subplots(figsize=(11, 7))
    PartialDependenceDisplay.from_estimator(model, X_train, top_num, ax=ax, kind="average")
    plt.suptitle("Partial dependence: xác suất phản đối theo từng biến")
    plt.tight_layout()
    plt.savefig(f"{args.out}/partial_dependence.png", dpi=150)
    plt.close()

    # ---- Chiều ảnh hưởng (mô tả) ----
    dt = direction_table(df)
    dt.to_csv(f"{args.out}/chieu_anh_huong.csv", index=False, encoding="utf-8-sig")
    print("\n=== CHIỀU ẢNH HƯỞNG (MÔ TẢ) ===")
    print(dt.to_string(index=False))

    # ---- Dự đoán cho file test.csv (cột oppose để trống) ----
    if args.predict:
        predict_file(model, args.predict, args.pred_out)

    print(f"\n[✓] Kết quả đã lưu vào thư mục '{args.out}/'")
    print("[!] Random Forest cho thấy mức độ ảnh hưởng dự đoán, KHÔNG chứng minh quan hệ nhân quả.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Đường dẫn file CSV khảo sát")
    ap.add_argument("--out", default="ket_qua", help="Thư mục lưu kết quả")
    ap.add_argument("--predict", default=None, help="File CSV cần dự đoán (cột oppose để trống), ví dụ test.csv")
    ap.add_argument("--pred-out", default="test_du_doan.csv", help="File CSV kết quả dự đoán")
    main(ap.parse_args())
