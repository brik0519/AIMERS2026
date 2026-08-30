# error_analysis.py

# ============================================================
# STEP 0. 현재 기본 앙상블 Error Analysis
#
# 목적:
#   현재 CatBoost + XGBoost 기본 앙상블이
#   어떤 야구 상황에서 상대적으로 예측을 잘 못하는지 확인한다.
#
# 학습:
#   2019 ~ 2022
#
# 검증 / 오류 분석:
#   2023
#
# 주의:
#   - Feature Engineering 실험 코드가 아님
#   - 최종 모델을 저장하지 않음
#   - 기존 ./model/rf.pkl 을 덮어쓰지 않음
#   - 2024 데이터는 사용하지 않음
# ============================================================


import os
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import VotingClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. 기본 설정
# ============================================================

DATA_DIR = "./data"

ID_COL = "row_id"
TARGET_COL = "control_success"

# 현재 기본 앙상블에서 범주형으로 처리하는 컬럼
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
]

# 현재 기본 앙상블 weight 그대로 사용
CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315

# 분석 결과 저장 위치
# 실험 관련 결과는 ./model 하위에 저장
EXP_DIR = "./model/error_analysis_v001"

os.makedirs(
    EXP_DIR,
    exist_ok=True
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

print("Load data...")


# test.csv의 컬럼 구조를 기준으로
# 실제 예측에 사용할 Feature 목록을 만든다.
#
# row_id는 식별자이므로 Feature에서 제외한다.
test_cols = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


FEATURES = [
    col
    for col in test_cols
    if col != ID_COL
]


# 범주형을 제외한 나머지는 수치형으로 처리
NUM_COLS = [
    col
    for col in FEATURES
    if col not in CAT_COLS
]


# train.csv에서는
# Feature + target만 불러온다.
train = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="utf-8-sig",
    usecols=FEATURES + [TARGET_COL],
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


# ============================================================
# 3. Error Analysis에 필요한 컬럼 확인
# ============================================================

# 아래 컬럼들은 모델 학습용 추가 Feature가 아니라
# 검증 결과를 상황별로 분석할 때 사용한다.

ANALYSIS_REQUIRED_COLS = [
    "balls_before",
    "strikes_before",
    "base_state",
    "li",
    "inning",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev3_game_success_rate",
]


missing_cols = [
    col
    for col in ANALYSIS_REQUIRED_COLS
    if col not in train.columns
]


if missing_cols:
    raise ValueError(
        "Error Analysis에 필요한 컬럼이 없습니다: "
        f"{missing_cols}"
    )


# ============================================================
# 4. 전처리
# ============================================================

print("\nBuild preprocessing...")


# 범주형:
#   결측치는 최빈값
#   문자열 category는 숫자로 encoding
cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    ),
])


# 수치형:
#   결측치를 training data의 median으로 채운다.
num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
])


preprocessor = ColumnTransformer([
    (
        "cat",
        cat_pipeline,
        CAT_COLS,
    ),
    (
        "num",
        num_pipeline,
        NUM_COLS,
    ),
])


# ============================================================
# 5. CatBoost
# ============================================================

print("Build CatBoost...")


# 현재 기본 앙상블과 동일한 설정
catboost = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    thread_count=-1,

    # catboost_info 폴더가 자동 생성되지 않도록 한다.
    allow_writing_files=False,
)


# ============================================================
# 6. XGBoost
# ============================================================

print("Build XGBoost...")


# 현재 기본 앙상블과 동일한 설정
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


# ============================================================
# 7. CatBoost + XGBoost Soft Voting
# ============================================================

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
    ],
)


# 전처리와 모델을 하나의 Pipeline으로 묶는다.
model = Pipeline([
    ("pre", preprocessor),
    ("clf", ensemble),
])


# ============================================================
# 8. 2019~2022 Train / 2023 Validation
# ============================================================

print("\nSplit train / validation...")


# Feature 실험을 설계할 개발용 validation
#
# 2019~2022 → 학습
# 2023      → Error Analysis
#
# 2024는 여기서 사용하지 않는다.

is_train = train["season"] <= 2022
is_val = train["season"] == 2023


X_train = train.loc[
    is_train,
    FEATURES,
].copy()

y_train = train.loc[
    is_train,
    TARGET_COL,
].copy()


X_val = train.loc[
    is_val,
    FEATURES,
].copy()

y_val = train.loc[
    is_val,
    TARGET_COL,
].copy()


print(
    f"train={len(X_train)} | "
    f"val={len(X_val)}"
)

print(
    "Train seasons: "
    f"{train.loc[is_train, 'season'].min()} ~ "
    f"{train.loc[is_train, 'season'].max()}"
)

print(
    "Validation season: "
    f"{train.loc[is_val, 'season'].unique().tolist()}"
)


# ============================================================
# 9. 모델 학습
# ============================================================

print("\nTrain Ensemble...")


start = time.time()


model.fit(
    X_train,
    y_train,
)


print(
    f"Training complete :: "
    f"{time.time() - start:.1f}s"
)


# ============================================================
# 10. 2023 Validation 예측
# ============================================================

print("\nValidation...")


val_pred = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 11. 전체 Validation Brier / BSS 확인
# ============================================================

r = y_val.mean()


brier = (
    (val_pred - y_val.to_numpy()) ** 2
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


# ============================================================
# 12. 각 투구별 Error 데이터 생성
# ============================================================

print("\nBuild error analysis data...")


# X_val을 복사해서
# 실제 target / prediction / Brier error를 붙인다.
analysis = X_val.copy()


analysis["target"] = (
    y_val.to_numpy()
)


analysis["pred"] = (
    val_pred
)


# Brier Score는 각 행에서:
#
# (예측확률 - 실제값)^2
#
# 이 값이 높을수록 해당 투구를 잘못 예측한 것.
analysis["brier_error"] = (
    analysis["pred"]
    - analysis["target"]
) ** 2


# ============================================================
# 13. 분석용 Feature 생성
# ============================================================
#
# 주의:
# 이 Feature들은 아직 모델에 넣지 않는다.
#
# 현재 모델이 어떤 상황에서 약한지
# 분석하기 위한 용도로만 사용한다.
# ============================================================


# ------------------------------------------------------------
# 13-1. Ball-Strike Count
# ------------------------------------------------------------
#
# balls_before=3
# strikes_before=2
#
# → "3-2"
#
# xStrikes 4등 솔루션의
# Ball-Strike Count 아이디어와 연결되는 분석.

analysis["count_state"] = (
    analysis["balls_before"]
    .astype("Int64")
    .astype(str)
    + "-"
    + analysis["strikes_before"]
    .astype("Int64")
    .astype(str)
)


# ------------------------------------------------------------
# 13-2. 최근 3경기 제구력 - 장기 제구력
# ------------------------------------------------------------
#
# 양수:
#   최근 제구 성공률이 평소보다 좋음
#
# 음수:
#   최근 제구 성공률이 평소보다 나쁨
#
# KBO OPS 글에서 가져온
# "최근 기록과 장기 수준을 함께 본다"는 아이디어.

analysis["recent_vs_career"] = (
    analysis[
        "asof_pitcher_prev3_game_success_rate"
    ]
    -
    analysis[
        "asof_pitcher_success_rate"
    ]
)


# ------------------------------------------------------------
# 13-3. LI를 5개 구간으로 나누기
# ------------------------------------------------------------
#
# LI는 연속형 숫자이므로
# 값 하나하나를 groupby하면 분석하기 어렵다.
#
# 2023 validation 데이터를
# roughly 같은 수의 5개 그룹으로 나눈다.

analysis["li_bin"] = pd.qcut(
    analysis["li"],
    q=5,
    duplicates="drop",
)


# ------------------------------------------------------------
# 13-4. recent_vs_career 역시 5개 구간으로 나누기
# ------------------------------------------------------------

analysis["recent_vs_career_bin"] = pd.qcut(
    analysis["recent_vs_career"],
    q=5,
    duplicates="drop",
)


# ============================================================
# 14. Error Report 함수
# ============================================================

def make_error_report(
    df,
    group_col,
):
    """
    특정 상황별로 현재 모델의 오류를 요약한다.

    결과 컬럼:

    n
        해당 상황의 투구 수

    mean_brier
        해당 상황에서의 평균 Brier Error
        높을수록 모델이 상대적으로 못 맞힘

    error_sum
        해당 상황에서 발생한 Brier Error의 총합

    error_contribution
        전체 validation Brier Error 중
        해당 상황이 차지한 비율

    target_rate
        해당 상황의 실제 control_success 비율

    pred_mean
        해당 상황에서 모델이 예측한
        평균 control_success 확률

    calibration_gap
        pred_mean - target_rate

        양수 → 모델이 성공확률을 과대예측
        음수 → 모델이 성공확률을 과소예측
    """

    report = (
        df
        .groupby(
            group_col,
            observed=False,
            dropna=False,
        )
        .agg(
            n=(
                "brier_error",
                "size",
            ),
            mean_brier=(
                "brier_error",
                "mean",
            ),
            error_sum=(
                "brier_error",
                "sum",
            ),
            target_rate=(
                "target",
                "mean",
            ),
            pred_mean=(
                "pred",
                "mean",
            ),
        )
        .reset_index()
    )


    # 전체 오류 중 각 그룹이 차지하는 비율
    report["error_contribution"] = (
        report["error_sum"]
        / df["brier_error"].sum()
    )


    # 실제 성공률과 예측확률의 차이
    report["calibration_gap"] = (
        report["pred_mean"]
        - report["target_rate"]
    )


    # 우선 평균 Brier가 높은 그룹부터 출력
    report = report.sort_values(
        "mean_brier",
        ascending=False,
    )


    return report


# ============================================================
# 15. 분석할 그룹
# ============================================================

GROUPS = [
    "count_state",
    "base_state",
    "li_bin",
    "inning",
    "recent_vs_career_bin",
]


# ============================================================
# 16. 그룹별 Error Analysis 실행
# ============================================================

print("\n")
print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)


for col in GROUPS:

    report = make_error_report(
        analysis,
        col,
    )


    print(
        f"\n===== {col} ====="
    )

    print(
        report.to_string(
            index=False
        )
    )


    # 각 분석 결과를 CSV로 저장
    report.to_csv(
        os.path.join(
            EXP_DIR,
            f"error_{col}.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 17. 전체 2023 Validation Prediction 저장
# ============================================================
#
# 이후 EXP-1 / EXP-2를 비교할 때
# 다시 사용할 수 있도록 보관한다.

analysis.to_csv(
    os.path.join(
        EXP_DIR,
        "validation_predictions.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. 분석 요약 저장
# ============================================================

summary = pd.DataFrame([
    {
        "train_season": "2019-2022",
        "validation_season": 2023,
        "n_train": len(X_train),
        "n_validation": len(X_val),

        "brier": brier,
        "baseline_brier": baseline_brier,
        "validation_score": score,

        "catboost_weight": CAT_WEIGHT,
        "xgboost_weight": XGB_WEIGHT,
    }
])


summary.to_csv(
    os.path.join(
        EXP_DIR,
        "summary.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 19. 종료
# ============================================================
#
# 중요:
# 여기서는 전체 데이터로 다시 학습하지 않는다.
# 모델을 ./model/rf.pkl 로 저장하지도 않는다.
#
# 따라서 기존 최고 모델은 전혀 건드리지 않는다.
# ============================================================

print("\n")
print("=" * 60)
print("Error Analysis Complete")
print("=" * 60)

print(
    f"Results saved to: "
    f"{EXP_DIR}"
)

print("\nGenerated files:")

print(
    "  summary.csv"
)

print(
    "  validation_predictions.csv"
)

for col in GROUPS:
    print(
        f"  error_{col}.csv"
    )

print(
    "\nNo model was saved or overwritten."
)