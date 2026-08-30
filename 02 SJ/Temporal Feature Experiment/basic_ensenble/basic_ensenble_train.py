# train.py
# CatBoost + XGBoost Ensemble
# CatBoost 0.5655685 / XGBoost 0.4344315

import os
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import VotingClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# =======================
# 기본 설정
# =======================

DATA_DIR = "./data"

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
]

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# =======================
# 데이터 불러오기
# =======================

print("Load data...")


test_cols = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="utf-8-sig",
    nrows=0
).columns.tolist()


FEATURES = [
    col for col in test_cols
    if col != ID_COL
]


NUM_COLS = [
    col for col in FEATURES
    if col not in CAT_COLS
]


train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="utf-8-sig",
    usecols=FEATURES + [TARGET_COL]
)


print(
    f"train={train.shape} | "
    f"features={len(FEATURES)} "
    f"(cat={len(CAT_COLS)}, num={len(NUM_COLS)})"
)

print(
    f"season: "
    f"{train['season'].min()} ~ "
    f"{train['season'].max()}"
)

print(
    f"control_success rate: "
    f"{train[TARGET_COL].mean():.4f}"
)


# =======================
# 전처리
# =======================

print("\nBuild preprocessing...")


cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    ),
])


num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
])


preprocessor = ColumnTransformer([
    (
        "cat",
        cat_pipeline,
        CAT_COLS
    ),
    (
        "num",
        num_pipeline,
        NUM_COLS
    ),
])


# =======================
# CatBoost
# 현재 658.51 나온 설정
# =======================

print("Build CatBoost...")


catboost = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    thread_count=-1,
)


# =======================
# XGBoost
# 앞서 사용한 설정
# =======================

print("Build XGBoost...")


xgboost = XGBClassifier(
    objective="binary:logistic",

    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=1.0,

    tree_method="hist",
    eval_metric="logloss",

    n_jobs=-1,
    random_state=42,
)


# =======================
# Soft Voting Ensemble
# =======================

print("Build Ensemble...")


ensemble = VotingClassifier(
    estimators=[
        ("catboost", catboost),
        ("xgboost", xgboost),
    ],

    voting="soft",

    weights=[
        CAT_WEIGHT,
        XGB_WEIGHT,
    ]
)


# 전처리 + Ensemble
model = Pipeline([
    ("pre", preprocessor),
    ("clf", ensemble),
])


# =======================
# Train / Validation 분리
# =======================

print("\nSplit train / validation...")


is_val = train["season"] == 2024


X_train = train.loc[
    ~is_val,
    FEATURES
]

y_train = train.loc[
    ~is_val,
    TARGET_COL
]


X_val = train.loc[
    is_val,
    FEATURES
]

y_val = train.loc[
    is_val,
    TARGET_COL
]


print(
    f"train={len(X_train)} | "
    f"val={len(X_val)}"
)


# =======================
# 학습
# =======================

print("\nTrain Ensemble...")


start = time.time()


model.fit(
    X_train,
    y_train
)


print(
    f"Training complete :: "
    f"{time.time() - start:.1f}s"
)


# =======================
# Validation
# =======================

print("\nValidation...")


val_pred = model.predict_proba(
    X_val
)[:, 1]


r = y_val.mean()


brier = (
    (val_pred - y_val) ** 2
).mean()


baseline_brier = r * (1 - r)


score = max(
    0,
    100000 * (
        1 - brier / baseline_brier
    )
)


print(
    f"Brier: "
    f"{brier:.6f}"
)

print(
    f"Baseline Brier r(1-r): "
    f"{baseline_brier:.6f}"
)

print(
    f"Validation Score: "
    f"{score:.2f}"
)


# =======================
# 예측 분포
# =======================

print("\nPrediction statistics:")


print(
    pd.Series(val_pred).describe()
)


print("\nEnsemble weights:")

print(
    f"CatBoost = "
    f"{CAT_WEIGHT:.7f}"
)

print(
    f"XGBoost  = "
    f"{XGB_WEIGHT:.7f}"
)


# =======================
# 전체 데이터 재학습
# =======================

print("\nRetrain with full data...")


start = time.time()


model.fit(
    train[FEATURES],
    train[TARGET_COL]
)


print(
    f"Retraining complete :: "
    f"{time.time() - start:.1f}s"
)


# =======================
# 모델 저장
# =======================

os.makedirs(
    "./model",
    exist_ok=True
)


MODEL_PATH = "./model/rf.pkl"


joblib.dump(
    model,
    MODEL_PATH,
    compress=3
)


print(
    f"\nSaved: {MODEL_PATH}"
)

print(
    "Model: CatBoost + XGBoost Ensemble"
)

print(
    f"CatBoost weight: {CAT_WEIGHT:.7f}"
)

print(
    f"XGBoost weight: {XGB_WEIGHT:.7f}"
)

print(
    f"Features: {len(FEATURES)}"
)