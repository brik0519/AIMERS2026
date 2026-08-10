# exp3_temporal_only.py

# ============================================================
# EXP-3. Temporal Feature Only
#
# 목적:
#   EXP-1에서 추가했던 Core Feature 11개 중
#
#       1. Recent vs Career 6개
#       2. Trend 4개
#
#   즉 "투수의 시간축"과 관련된 Feature 10개만 따로 넣어서
#   실제로 이 Feature들이 성능 개선에 기여했는지 확인한다.
#
#
# 비교
# ------------------------------------------------------------
#
# BASE
#   Raw Feature 47개
#
# EXP-3
#   Raw Feature 47개
#   + Temporal Feature 10개
#
#
# 이번에는 count_state를 넣지 않는다.
#
# 따라서:
#
#   EXP-3이 좋아진다
#       → Temporal Feature 자체가 유효할 가능성
#
#   EXP-3이 안 좋아지는데 EXP-1은 좋았다
#       → count_state와의 조합 효과일 가능성
#
#
# Validation
# ------------------------------------------------------------
#
# 1차
#   2019 ~ 2022 → 2023
#
# 2차
#   2019 ~ 2023 → 2024
#
#
# 이미 같은 모델 설정으로 측정한 BASE Brier:
#
#   2023 BASE = 0.253595
#   2024 BASE = 0.248270
#
# 따라서 BASE 모델을 다시 학습하지 않고,
# EXP-3만 학습해서 기존 BASE와 비교한다.
#
#
# 중요한 통제 조건
# ------------------------------------------------------------
#
# - CatBoost 설정 동일
# - XGBoost 설정 동일
# - Ensemble weight 동일
# - preprocessing 동일
#
# 오직 Temporal Feature 10개만 추가한다.
#
#
# 이번 파일에서는
# ------------------------------------------------------------
#
# - 모델 저장 X
# - rf.pkl 덮어쓰기 X
# - test 예측 X
# - submission 생성 X
#
# 로컬 Validation 실험만 수행한다.
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
# 현재 기본 앙상블 weight
# ------------------------------------------------------------
#
# Feature 효과만 보기 위해 그대로 유지한다.
#
# CatBoost : 56.56%
# XGBoost  : 43.44%
# ------------------------------------------------------------

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# ------------------------------------------------------------
# 기존 BASE Validation 결과
#
# 앞 실험에서 동일한 모델 설정으로 직접 측정한 값.
# ------------------------------------------------------------

BASE_BRIER = {
    2023: 0.253595,
    2024: 0.248270,
}


# ------------------------------------------------------------
# 결과 저장 폴더
# ------------------------------------------------------------

EXP_DIR = "./model/exp3_temporal_only_v001"

os.makedirs(
    EXP_DIR,
    exist_ok=True
)


# ============================================================
# 2. Temporal Feature 생성 함수
# ============================================================

def build_temporal_features(df):
    """
    투수의 시간 변화와 관련된 Feature 10개를 만든다.


    ==========================================================
    A. Recent vs Career 6개
    ==========================================================

    최근 1/3/5경기 기록이
    자신의 장기 기록과 얼마나 차이 나는지 표현한다.

    예:

        career success = 0.55
        prev3 success  = 0.48

        recent3_success_vs_career = -0.07

    즉:

        최근 상태가 평소보다 좋아졌는가?
        최근 상태가 평소보다 떨어졌는가?

    를 모델에게 직접 알려준다.


    ==========================================================
    B. Trend 4개
    ==========================================================

    최근 1/3/5경기 기록 사이의 차이를 이용해서

        최근 상승
        최근 하락
        안정

    같은 변화 방향을 표현한다.
    """

    df = df.copy()


    # ========================================================
    # A-1. Recent vs Career - Success Rate
    # ========================================================

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


    # ========================================================
    # A-2. Recent vs Career - Middle Rate
    # ========================================================

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
    # B-1. Success Trend
    # ========================================================

    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )


    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )


    # ========================================================
    # B-2. Middle Trend
    # ========================================================

    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )


    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )


    return df


# ============================================================
# 3. 모델 생성 함수
# ============================================================

def build_model(
    features,
    cat_cols,
):
    """
    현재 BASE와 동일한

        preprocessing
        + CatBoost
        + XGBoost
        + Soft Voting

    모델을 만든다.

    이번 EXP-3에서 모델 설정은 전혀 변경하지 않는다.
    """

    # --------------------------------------------------------
    # 범주형을 제외한 Feature는 모두 수치형
    # --------------------------------------------------------

    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]


    # ========================================================
    # 범주형 전처리
    # ========================================================
    #
    # 결측값
    #   → 최빈값
    #
    # category
    #   → ordinal encoding
    #
    # Train에서 보지 못한 값
    #   → -1
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
    # 결측값
    #   → Train median
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
    # 전체 전처리
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
    # 기존 BASE와 동일.
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

        # catboost_info 폴더 자동 생성 방지
        allow_writing_files=False,
    )


    # ========================================================
    # XGBoost
    # ========================================================
    #
    # 기존 BASE와 동일.
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
    # CatBoost + XGBoost Soft Voting
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
# 4. Validation 평가 함수
# ============================================================

def evaluate_prediction(
    y_true,
    pred,
):
    """
    Prediction으로

        Brier Score
        Baseline Brier
        Brier Skill Score

    를 계산한다.

    Brier:
        낮을수록 좋음

    BSS:
        높을수록 좋음
    """

    y_array = (
        y_true.to_numpy()
    )


    # --------------------------------------------------------
    # Brier Score
    # --------------------------------------------------------

    brier = (
        (pred - y_array) ** 2
    ).mean()


    # --------------------------------------------------------
    # Validation 성공률
    # --------------------------------------------------------

    r = (
        y_true.mean()
    )


    # --------------------------------------------------------
    # Baseline Brier
    # --------------------------------------------------------

    baseline_brier = (
        r * (1 - r)
    )


    # --------------------------------------------------------
    # 공식 BSS
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
# 5. 데이터 불러오기
# ============================================================

print("=" * 60)
print("EXP-3 Temporal Feature Only")
print("=" * 60)

print("\nLoad data...")


# ------------------------------------------------------------
# test.csv header로 Raw Feature 목록 생성
# ------------------------------------------------------------

test_cols = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "test.csv"
    ),
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


RAW_FEATURES = [
    col
    for col in test_cols
    if col != ID_COL
]


# ============================================================
# 6. Temporal Feature 목록
# ============================================================

TEMPORAL_FEATURES = [

    # Recent vs Career - Success
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",

    # Recent vs Career - Middle
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",

    # Trend - Success
    "success_trend_1_3",
    "success_trend_3_5",

    # Trend - Middle
    "middle_trend_1_3",
    "middle_trend_3_5",
]


# 최종 EXP-3 Feature
EXP3_FEATURES = (
    RAW_FEATURES
    + TEMPORAL_FEATURES
)


# ============================================================
# 7. 필요한 Raw 컬럼 확인
# ============================================================

FE_REQUIRED_COLS = [

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
        "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
        f"{missing_cols}"
    )


# ============================================================
# 8. train.csv 불러오기
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
# 9. Temporal Feature Engineering 적용
# ============================================================

print("\nBuild Temporal Features...")


train_fe = build_temporal_features(
    train
)


print(
    f"Added Temporal Features: "
    f"{len(TEMPORAL_FEATURES)}"
)


print(
    f"EXP-3 Total Features: "
    f"{len(EXP3_FEATURES)}"
)


print("\nTemporal Features:")

for feature in TEMPORAL_FEATURES:

    print(
        f"  {feature}"
    )


# ============================================================
# 10. Feature 생성 확인
# ============================================================

missing_after_fe = [
    col
    for col in EXP3_FEATURES
    if col not in train_fe.columns
]


if missing_after_fe:

    raise ValueError(
        "Feature Engineering 후 없는 컬럼: "
        f"{missing_after_fe}"
    )


# ============================================================
# 11. 범주형 Feature
# ============================================================
#
# EXP-3에는 count_state가 없다.
#
# 따라서 기존 BASE와 같은 범주형 3개만 사용.
# ============================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
]


# ============================================================
# 12. 한 연도 Validation을 실행하는 함수
# ============================================================

def run_validation(
    train_df,
    train_max_year,
    val_year,
):
    """
    EXP-3 모델을 한 번 학습하고

    기존 BASE Brier와 비교한다.
    """

    print("\n")
    print("=" * 60)

    print(
        f"VALIDATION {val_year}"
    )

    print("=" * 60)


    # ========================================================
    # Train / Validation Split
    # ========================================================

    is_train = (
        train_df["season"]
        <= train_max_year
    )


    is_val = (
        train_df["season"]
        == val_year
    )


    X_train = train_df.loc[
        is_train,
        EXP3_FEATURES,
    ].copy()


    X_val = train_df.loc[
        is_val,
        EXP3_FEATURES,
    ].copy()


    y_train = train_df.loc[
        is_train,
        TARGET_COL,
    ].copy()


    y_val = train_df.loc[
        is_val,
        TARGET_COL,
    ].copy()


    print(
        f"Train seasons   : "
        f"2019 ~ {train_max_year}"
    )


    print(
        f"Validation      : "
        f"{val_year}"
    )


    print(
        f"Train rows      : "
        f"{len(y_train):,}"
    )


    print(
        f"Validation rows : "
        f"{len(y_val):,}"
    )


    # ========================================================
    # 모델 생성
    # ========================================================

    print("\nBuild EXP-3 model...")


    model = build_model(
        features=EXP3_FEATURES,
        cat_cols=CAT_COLS,
    )


    # ========================================================
    # 모델 학습
    # ========================================================

    print("\nTrain EXP-3...")


    start = time.time()


    model.fit(
        X_train,
        y_train,
    )


    training_time = (
        time.time()
        - start
    )


    print(
        f"Training complete :: "
        f"{training_time:.1f}s"
    )


    # ========================================================
    # Validation Prediction
    # ========================================================

    print("\nValidation...")


    pred = (
        model
        .predict_proba(
            X_val
        )[:, 1]
    )


    # ========================================================
    # 평가
    # ========================================================

    (
        exp3_brier,
        baseline_brier,
        exp3_bss,
    ) = evaluate_prediction(
        y_val,
        pred,
    )


    # 기존 BASE Brier
    base_brier = (
        BASE_BRIER[val_year]
    )


    # --------------------------------------------------------
    # Brier는 낮을수록 좋음
    #
    # 양수이면 EXP-3 개선
    # --------------------------------------------------------

    improvement = (
        base_brier
        - exp3_brier
    )


    delta = (
        exp3_brier
        - base_brier
    )


    if exp3_brier < base_brier:

        result = "IMPROVED"


    elif exp3_brier > base_brier:

        result = "WORSE"


    else:

        result = "SAME"


    # ========================================================
    # 결과 출력
    # ========================================================

    print("\n")
    print("-" * 60)

    print(
        f"{val_year} RESULT"
    )

    print("-" * 60)


    print(
        f"BASE Brier : "
        f"{base_brier:.6f}"
    )


    print(
        f"EXP-3 Brier: "
        f"{exp3_brier:.6f}"
    )


    print(
        f"Brier delta: "
        f"{delta:+.6f}"
    )


    print(
        f"Improvement: "
        f"{improvement:+.6f}"
    )


    print(
        f"Result: "
        f"{result}"
    )


    print(
        f"\nEXP-3 BSS: "
        f"{exp3_bss:.2f}"
    )


    print("\nPrediction statistics:")


    print(
        pd.Series(
            pred
        ).describe()
    )


    # ========================================================
    # Validation Prediction 저장
    # ========================================================

    prediction_result = pd.DataFrame({
        "target":
            y_val.to_numpy(),

        "pred":
            pred,
    })


    prediction_result[
        "brier_error"
    ] = (
        (
            prediction_result["pred"]
            - prediction_result["target"]
        ) ** 2
    )


    prediction_result.to_csv(
        os.path.join(
            EXP_DIR,
            f"validation_predictions_{val_year}.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # 결과 반환
    # ========================================================

    return {
        "validation_year":
            val_year,

        "train_seasons":
            f"2019-{train_max_year}",

        "base_brier":
            base_brier,

        "exp3_brier":
            exp3_brier,

        "brier_delta":
            delta,

        "improvement":
            improvement,

        "result":
            result,

        "baseline_brier":
            baseline_brier,

        "exp3_bss":
            exp3_bss,

        "training_seconds":
            training_time,
    }


# ============================================================
# 13. 2023 Validation
# ============================================================
#
# 2019 ~ 2022 → 2023
# ============================================================

result_2023 = run_validation(
    train_df=train_fe,
    train_max_year=2022,
    val_year=2023,
)


# ============================================================
# 14. 2024 Validation
# ============================================================
#
# 이번 EXP-3는 단순 방향성 확인 실험이므로
# 2023 결과와 관계없이 2024도 바로 확인한다.
#
# 2019 ~ 2023 → 2024
# ============================================================

result_2024 = run_validation(
    train_df=train_fe,
    train_max_year=2023,
    val_year=2024,
)


# ============================================================
# 15. 두 연도 결과 저장
# ============================================================

summary = pd.DataFrame([
    result_2023,
    result_2024,
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
# 16. Temporal Feature 목록 저장
# ============================================================

feature_info = pd.DataFrame({
    "feature":
        TEMPORAL_FEATURES
})


feature_info.to_csv(
    os.path.join(
        EXP_DIR,
        "temporal_features.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. 최종 결과 출력
# ============================================================

print("\n")
print("=" * 60)
print("EXP-3 FINAL SUMMARY")
print("=" * 60)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 18. 간단한 최종 판단
# ============================================================

improved_2023 = (
    result_2023["result"]
    == "IMPROVED"
)


improved_2024 = (
    result_2024["result"]
    == "IMPROVED"
)


print("\n")
print("=" * 60)


if (
    improved_2023
    and
    improved_2024
):

    print(
        "EXP-3 TEMPORAL FEATURES: POSITIVE"
    )

    print(
        "2023과 2024 모두 BASE보다 개선."
    )

    print(
        "→ Temporal Feature 방향이 유효한 후보."
    )


elif (
    improved_2023
    or
    improved_2024
):

    print(
        "EXP-3 TEMPORAL FEATURES: MIXED"
    )

    print(
        "한 연도에서만 개선."
    )

    print(
        "→ 안정적인 효과라고 보기 어려움."
    )


else:

    print(
        "EXP-3 TEMPORAL FEATURES: NEGATIVE"
    )

    print(
        "2023과 2024 모두 BASE보다 개선되지 않음."
    )

    print(
        "→ Temporal Feature 단독 효과는 약함."
    )


print("=" * 60)


# ============================================================
# 19. 종료
# ============================================================
#
# 이번 실험에서는 모델을 저장하지 않는다.
# ============================================================

print("\n")
print("=" * 60)
print("EXP-3 Complete")
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
    "  temporal_features.csv"
)

print(
    "  validation_predictions_2023.csv"
)

print(
    "  validation_predictions_2024.csv"
)


print(
    "\nNo model was saved or overwritten."
)