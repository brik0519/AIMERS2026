# exp1_core_fe_2024.py

# ============================================================
# EXP-1. Core Feature Engineering - 2024 Validation
#
# 목적:
#   STEP 0 / EXP-1에서 2023 기준으로 확인했던 Core FE 11개가
#   다음 연도인 2024에서도 실제로 성능을 개선하는지 확인한다.
#
#
# 비교 모델
# ------------------------------------------------------------
# 1. BASE
#    Raw Features
#    + CatBoost 56.56%
#    + XGBoost 43.44%
#
# 2. EXP-1
#    Raw Features
#    + Core Feature 11개
#    + CatBoost 56.56%
#    + XGBoost 43.44%
#
#
# 학습 / 검증
# ------------------------------------------------------------
# Train      : 2019 ~ 2023
# Validation : 2024
#
#
# 중요한 원칙
# ------------------------------------------------------------
# - BASE와 EXP-1의 모델 설정은 완전히 동일
# - Feature만 다르게 해서 공정하게 비교
# - Ensemble weight도 동일
# - 하이퍼파라미터도 동일
# - 모델 저장하지 않음
# - 기존 ./model/rf.pkl 덮어쓰지 않음
# - test.csv 예측하지 않음
#
#
# 기존 2023 결과
# ------------------------------------------------------------
# BASE Brier  : 0.253595
# EXP-1 Brier : 0.253491
# Improvement : +0.000104
#
# 따라서 이번 실험의 목적은:
#
# "이 작은 개선이 2024에서도 같은 방향으로 재현되는가?"
#
# 를 확인하는 것이다.
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


# ------------------------------------------------------------
# 현재 최고 기본 앙상블의 weight
#
# EXP-1에서도 그대로 유지한다.
# ------------------------------------------------------------

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# ------------------------------------------------------------
# 결과 저장 폴더
# ------------------------------------------------------------

EXP_DIR = "./model/exp1_core_fe_2024_v001"

os.makedirs(
    EXP_DIR,
    exist_ok=True
)


# ============================================================
# 2. Core Feature Engineering 함수
# ============================================================

def build_features(df):
    """
    Raw 데이터에 EXP-1 Core Feature 11개를 추가한다.

    ----------------------------------------------------------
    A. Recent vs Career 6개
    ----------------------------------------------------------

    최근 1/3/5경기 기록을
    투수 자신의 장기 기록과 비교한다.

    예:

        장기 success = 0.55
        최근 3경기 success = 0.48

        recent3_success_vs_career = -0.07

    즉 단순히 최근 성공률 자체가 아니라

        "평소보다 최근 상태가 좋아졌는가 / 나빠졌는가"

    를 표현한다.


    ----------------------------------------------------------
    B. Trend 4개
    ----------------------------------------------------------

    최근 1경기 / 3경기 / 5경기 기록의 차이를 통해

        상승
        하락
        안정

    같은 최근 흐름을 모델에게 직접 제공한다.


    ----------------------------------------------------------
    C. Count State 1개
    ----------------------------------------------------------

    balls_before와 strikes_before를 합쳐

        0_0
        1_2
        3_0
        3_2

    같은 하나의 야구 상황으로 표현한다.
    """

    df = df.copy()


    # ========================================================
    # A. Recent vs Career
    # ========================================================


    # --------------------------------------------------------
    # Success Rate
    # --------------------------------------------------------

    df["recent1_success_vs_career"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )


    df["recent3_success_vs_career"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )


    df["recent5_success_vs_career"] = (
        df["asof_pitcher_prev5_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )


    # --------------------------------------------------------
    # Middle Rate
    # --------------------------------------------------------

    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )


    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )


    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )


    # ========================================================
    # B. Trend
    # ========================================================


    # --------------------------------------------------------
    # Success Trend
    # --------------------------------------------------------

    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )


    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )


    # --------------------------------------------------------
    # Middle Trend
    # --------------------------------------------------------

    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )


    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )


    # ========================================================
    # C. Ball-Strike Count State
    # ========================================================
    #
    # 원래 balls_before / strikes_before는 그대로 유지한다.
    #
    # 추가로 두 값을 합친 count_state를 하나 더 만든다.
    # ========================================================

    balls = (
        df["balls_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    strikes = (
        df["strikes_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    df["count_state"] = (
        balls
        + "_"
        + strikes
    )


    return df


# ============================================================
# 3. 모델 생성 함수
# ============================================================
#
# BASE와 EXP-1이 반드시 동일한 모델 설정을 사용해야 하므로
# 모델 생성 코드를 함수로 하나만 만든다.
#
# 차이는:
#
# BASE:
#   cat_cols = 기존 범주형 3개
#
# EXP-1:
#   cat_cols = 기존 범주형 3개 + count_state
#
# 뿐이다.
# ============================================================

def build_model(
    features,
    cat_cols,
):
    """
    주어진 Feature 목록으로
    CatBoost + XGBoost soft-voting ensemble을 만든다.
    """

    # 범주형을 제외한 Feature는 모두 수치형 처리
    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]


    # ========================================================
    # 범주형 전처리
    # ========================================================
    #
    # 1. 결측값 → 최빈값
    # 2. category → 숫자 encoding
    #
    # validation에 학습에서 보지 못한 값이 나오면 -1
    # ========================================================

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


    # ========================================================
    # 수치형 전처리
    # ========================================================
    #
    # 결측값 → 학습 데이터의 median
    # ========================================================

    num_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
    ])


    # ========================================================
    # ColumnTransformer
    # ========================================================

    preprocessor = ColumnTransformer([
        (
            "cat",
            cat_pipeline,
            cat_cols,
        ),
        (
            "num",
            num_pipeline,
            num_cols,
        ),
    ])


    # ========================================================
    # CatBoost
    # ========================================================
    #
    # 현재 BASE와 완전히 동일한 설정.
    # ========================================================

    catboost = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,

        loss_function="Logloss",
        l2_leaf_reg=3.0,

        random_seed=42,

        verbose=False,
        thread_count=-1,

        # catboost_info 폴더 생성 방지
        allow_writing_files=False,
    )


    # ========================================================
    # XGBoost
    # ========================================================
    #
    # 역시 BASE와 완전히 동일한 설정.
    # ========================================================

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


    # ========================================================
    # Soft Voting Ensemble
    # ========================================================
    #
    # CatBoost : 56.56%
    # XGBoost  : 43.44%
    # ========================================================

    ensemble = VotingClassifier(
        estimators=[
            (
                "catboost",
                catboost,
            ),
            (
                "xgboost",
                xgboost,
            ),
        ],

        voting="soft",

        weights=[
            CAT_WEIGHT,
            XGB_WEIGHT,
        ],
    )


    # ========================================================
    # 최종 Pipeline
    # ========================================================

    model = Pipeline([
        (
            "pre",
            preprocessor,
        ),
        (
            "clf",
            ensemble,
        ),
    ])


    return model


# ============================================================
# 4. Brier 평가 함수
# ============================================================

def evaluate_prediction(
    y_true,
    pred,
):
    """
    Validation prediction을 받아서

    Brier Score
    Baseline Brier
    Brier Skill Score

    를 계산한다.
    """

    y_array = y_true.to_numpy()


    # --------------------------------------------------------
    # Brier Score
    #
    # 낮을수록 좋음
    # --------------------------------------------------------

    brier = (
        (pred - y_array) ** 2
    ).mean()


    # --------------------------------------------------------
    # Baseline Brier
    #
    # 모든 샘플에 validation의 평균 성공률 r을
    # 예측했을 때의 Brier.
    # --------------------------------------------------------

    r = y_true.mean()

    baseline_brier = (
        r * (1 - r)
    )


    # --------------------------------------------------------
    # 공식 Brier Skill Score
    # --------------------------------------------------------

    score = max(
        0,
        100000
        * (
            1
            - brier / baseline_brier
        )
    )


    return (
        brier,
        baseline_brier,
        score,
    )


# ============================================================
# 5. Raw Feature 목록 불러오기
# ============================================================

print("=" * 60)
print("EXP-1 Core FE - 2024 Validation")
print("=" * 60)

print("\nLoad data...")


# test.csv의 헤더만 읽는다.
#
# 실제 test에서 사용할 수 있는 Feature만
# 학습 Feature로 사용하기 위함.

test_cols = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "test.csv"
    ),
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


# row_id 제외
RAW_FEATURES = [
    col
    for col in test_cols
    if col != ID_COL
]


# ============================================================
# 6. Core Feature 목록
# ============================================================

CORE_FEATURES = [

    # Recent vs Career - Success
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",

    # Recent vs Career - Middle
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",

    # Success Trend
    "success_trend_1_3",
    "success_trend_3_5",

    # Middle Trend
    "middle_trend_1_3",
    "middle_trend_3_5",

    # Count
    "count_state",
]


# BASE는 Raw Feature만 사용
BASE_FEATURES = (
    RAW_FEATURES
)


# EXP-1은 Raw + 11개 사용
EXP1_FEATURES = (
    RAW_FEATURES
    + CORE_FEATURES
)


# ============================================================
# 7. Feature Engineering에 필요한 컬럼 확인
# ============================================================

FE_REQUIRED_COLS = [
    "balls_before",
    "strikes_before",

    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",

    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",

    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]


missing_cols = [
    col
    for col in FE_REQUIRED_COLS
    if col not in RAW_FEATURES
]


if missing_cols:

    raise ValueError(
        "Core Feature 생성에 필요한 "
        f"컬럼이 없습니다: {missing_cols}"
    )


# ============================================================
# 8. Train 데이터 불러오기
# ============================================================

train = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "train.csv"
    ),
    encoding="utf-8-sig",
    usecols=(
        RAW_FEATURES
        + [TARGET_COL]
    ),
)


print(
    f"Raw train shape: "
    f"{train.shape}"
)


print(
    f"Raw features: "
    f"{len(RAW_FEATURES)}"
)


# ============================================================
# 9. EXP-1 Feature 생성
# ============================================================

print("\nBuild EXP-1 Core Features...")


train_fe = build_features(
    train
)


print(
    f"Added Core Features: "
    f"{len(CORE_FEATURES)}"
)


print(
    f"EXP-1 Total Features: "
    f"{len(EXP1_FEATURES)}"
)


# Feature가 정상 생성됐는지 확인
missing_after_fe = [
    col
    for col in EXP1_FEATURES
    if col not in train_fe.columns
]


if missing_after_fe:

    raise ValueError(
        "Feature Engineering 후에도 없는 "
        f"컬럼이 있습니다: {missing_after_fe}"
    )


# ============================================================
# 10. 2019~2023 Train / 2024 Validation
# ============================================================

print("\nSplit Train / Validation...")


is_train = (
    train["season"] <= 2023
)

is_val = (
    train["season"] == 2024
)


# ------------------------------------------------------------
# BASE 데이터
# ------------------------------------------------------------

X_train_base = train.loc[
    is_train,
    BASE_FEATURES,
].copy()


X_val_base = train.loc[
    is_val,
    BASE_FEATURES,
].copy()


# ------------------------------------------------------------
# EXP-1 데이터
# ------------------------------------------------------------

X_train_exp1 = train_fe.loc[
    is_train,
    EXP1_FEATURES,
].copy()


X_val_exp1 = train_fe.loc[
    is_val,
    EXP1_FEATURES,
].copy()


# ------------------------------------------------------------
# target
#
# BASE와 EXP-1이 같은 행을 사용하므로 target도 동일하다.
# ------------------------------------------------------------

y_train = train.loc[
    is_train,
    TARGET_COL,
].copy()


y_val = train.loc[
    is_val,
    TARGET_COL,
].copy()


print(
    f"Train rows      : "
    f"{len(y_train):,}"
)

print(
    f"Validation rows : "
    f"{len(y_val):,}"
)

print(
    "Train seasons   : "
    "2019 ~ 2023"
)

print(
    "Validation      : "
    "2024"
)


# ============================================================
# 11. BASE 모델 생성
# ============================================================

BASE_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
]


print("\nBuild BASE model...")


base_model = build_model(
    features=BASE_FEATURES,
    cat_cols=BASE_CAT_COLS,
)


# ============================================================
# 12. BASE 모델 학습
# ============================================================

print("\n")
print("=" * 60)
print("Train BASE")
print("=" * 60)


base_start = time.time()


base_model.fit(
    X_train_base,
    y_train,
)


base_train_time = (
    time.time()
    - base_start
)


print(
    f"BASE training complete :: "
    f"{base_train_time:.1f}s"
)


# ============================================================
# 13. BASE Validation 예측
# ============================================================

print("\nBASE Validation...")


base_pred = (
    base_model
    .predict_proba(
        X_val_base
    )[:, 1]
)


(
    base_brier,
    baseline_brier,
    base_score,
) = evaluate_prediction(
    y_val,
    base_pred,
)


print(
    f"BASE Brier: "
    f"{base_brier:.6f}"
)


print(
    f"BASE Validation Score: "
    f"{base_score:.2f}"
)


# ============================================================
# 14. EXP-1 모델 생성
# ============================================================

EXP1_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",

    # EXP-1에서 추가된 categorical Feature
    "count_state",
]


print("\nBuild EXP-1 model...")


exp1_model = build_model(
    features=EXP1_FEATURES,
    cat_cols=EXP1_CAT_COLS,
)


# ============================================================
# 15. EXP-1 모델 학습
# ============================================================

print("\n")
print("=" * 60)
print("Train EXP-1")
print("=" * 60)


exp1_start = time.time()


exp1_model.fit(
    X_train_exp1,
    y_train,
)


exp1_train_time = (
    time.time()
    - exp1_start
)


print(
    f"EXP-1 training complete :: "
    f"{exp1_train_time:.1f}s"
)


# ============================================================
# 16. EXP-1 Validation 예측
# ============================================================

print("\nEXP-1 Validation...")


exp1_pred = (
    exp1_model
    .predict_proba(
        X_val_exp1
    )[:, 1]
)


(
    exp1_brier,
    _,
    exp1_score,
) = evaluate_prediction(
    y_val,
    exp1_pred,
)


print(
    f"EXP-1 Brier: "
    f"{exp1_brier:.6f}"
)


print(
    f"EXP-1 Validation Score: "
    f"{exp1_score:.2f}"
)


# ============================================================
# 17. BASE vs EXP-1 비교
# ============================================================
#
# Brier는 낮을수록 좋다.
#
# 따라서:
#
# improvement = BASE - EXP1
#
# 값이 양수이면 EXP-1이 개선된 것이다.
# ============================================================

brier_improvement = (
    base_brier
    - exp1_brier
)


brier_delta = (
    exp1_brier
    - base_brier
)


if exp1_brier < base_brier:

    result = "IMPROVED"


elif exp1_brier > base_brier:

    result = "WORSE"


else:

    result = "SAME"


# ============================================================
# 18. 최종 결과 출력
# ============================================================

print("\n")
print("=" * 60)
print("2024 VALIDATION RESULT")
print("=" * 60)


print(
    f"BASE Brier : "
    f"{base_brier:.6f}"
)


print(
    f"EXP-1 Brier: "
    f"{exp1_brier:.6f}"
)


print(
    f"Brier delta: "
    f"{brier_delta:+.6f}"
)


print(
    f"Improvement: "
    f"{brier_improvement:+.6f}"
)


print(
    f"Result: "
    f"{result}"
)


print(
    f"\nBaseline Brier r(1-r): "
    f"{baseline_brier:.6f}"
)


print(
    f"BASE BSS : "
    f"{base_score:.2f}"
)


print(
    f"EXP-1 BSS: "
    f"{exp1_score:.2f}"
)


print("\nPrediction statistics")


print("\nBASE:")

print(
    pd.Series(
        base_pred
    ).describe()
)


print("\nEXP-1:")

print(
    pd.Series(
        exp1_pred
    ).describe()
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
# 19. Validation Prediction 저장
# ============================================================

validation_result = pd.DataFrame({
    "target": y_val.to_numpy(),

    "base_pred": base_pred,
    "exp1_pred": exp1_pred,
})


# 각 모델의 행 단위 Brier Error도 저장
validation_result["base_brier_error"] = (
    (
        validation_result["base_pred"]
        - validation_result["target"]
    ) ** 2
)


validation_result["exp1_brier_error"] = (
    (
        validation_result["exp1_pred"]
        - validation_result["target"]
    ) ** 2
)


# EXP-1이 각 행에서 BASE보다 얼마나 좋아졌는지
validation_result["error_improvement"] = (
    validation_result["base_brier_error"]
    - validation_result["exp1_brier_error"]
)


validation_result.to_csv(
    os.path.join(
        EXP_DIR,
        "validation_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 20. 실험 Summary 저장
# ============================================================

summary = pd.DataFrame([
    {
        "experiment":
            "EXP-1 Core FE - 2024 Validation",

        "train_seasons":
            "2019-2023",

        "validation_season":
            2024,

        "raw_features":
            len(RAW_FEATURES),

        "added_features":
            len(CORE_FEATURES),

        "exp1_total_features":
            len(EXP1_FEATURES),

        "base_brier":
            base_brier,

        "exp1_brier":
            exp1_brier,

        "brier_delta":
            brier_delta,

        "brier_improvement":
            brier_improvement,

        "result":
            result,

        "baseline_brier":
            baseline_brier,

        "base_bss":
            base_score,

        "exp1_bss":
            exp1_score,

        "catboost_weight":
            CAT_WEIGHT,

        "xgboost_weight":
            XGB_WEIGHT,

        "base_training_seconds":
            base_train_time,

        "exp1_training_seconds":
            exp1_train_time,
    }
])


summary.to_csv(
    os.path.join(
        EXP_DIR,
        "summary.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 21. Core Feature 목록 저장
# ============================================================

feature_info = pd.DataFrame({
    "feature": CORE_FEATURES
})


feature_info.to_csv(
    os.path.join(
        EXP_DIR,
        "core_features.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 22. 종료
# ============================================================
#
# 이번 파일에서는:
#
#   모델 저장 X
#   ./model/rf.pkl 수정 X
#   Test 예측 X
#   Submission 생성 X
#
# BASE와 EXP-1의 2024 Validation만 비교한다.
# ============================================================

print("\n")
print("=" * 60)
print("EXP-1 2024 Validation Complete")
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

print(
    "  core_features.csv"
)


print(
    "\nNo model was saved or overwritten."
)


# ============================================================
# 23. 최종 판단 문구
# ============================================================

print("\n")
print("=" * 60)


if result == "IMPROVED":

    print(
        "EXP-1 also improved the 2024 Brier."
    )

    print(
        "→ 2023과 2024 모두 개선 방향."
    )

    print(
        "→ EXP-1 Feature Set 채택 후보."
    )


elif result == "WORSE":

    print(
        "EXP-1 worsened the 2024 Brier."
    )

    print(
        "→ 2023 개선이 2024에서 재현되지 않음."
    )

    print(
        "→ 아직 제출하지 않음."
    )


else:

    print(
        "EXP-1 produced the same 2024 Brier."
    )

    print(
        "→ 추가 판단 필요."
    )


print("=" * 60)