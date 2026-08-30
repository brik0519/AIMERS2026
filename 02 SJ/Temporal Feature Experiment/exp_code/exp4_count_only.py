# exp4_count_only.py

# ============================================================
# EXP-4. Count State Only
#
# 목적:
#   EXP-1에서 추가한 Feature 11개 중
#
#       count_state 1개
#
#   만 따로 추가해서,
#   Ball-Strike Count 표현 자체가 성능에 도움이 되는지 확인한다.
#
#
# 비교
# ------------------------------------------------------------
#
# BASE
#   Raw Feature 47개
#
# EXP-4
#   Raw Feature 47개
#   + count_state 1개
#
#   총 48개 Feature
#
#
# count_state 예시
# ------------------------------------------------------------
#
# balls_before = 0, strikes_before = 0
# → "0_0"
#
# balls_before = 1, strikes_before = 2
# → "1_2"
#
# balls_before = 3, strikes_before = 2
# → "3_2"
#
#
# 왜 실험하는가?
# ------------------------------------------------------------
#
# 기존 모델에도 balls_before와 strikes_before는 각각 존재한다.
#
# 하지만:
#
#   balls = 3
#   strikes = 2
#
# 를 별개의 숫자 두 개로 보는 것과
#
#   count_state = "3_2"
#
# 라는 하나의 야구 상황으로 보는 것은 다르다.
#
# xStrikes 사례에서 Ball-Strike Count가 유용한 Feature였기 때문에
# 우리 문제에서도 이 표현이 도움이 되는지 단독으로 확인한다.
#
#
# Validation
# ------------------------------------------------------------
#
# 2023:
#   2019 ~ 2022 Train
#   2023 Validation
#
# 2024:
#   2019 ~ 2023 Train
#   2024 Validation
#
#
# 기존 BASE Brier
# ------------------------------------------------------------
#
# 2023 = 0.253595
# 2024 = 0.248270
#
#
# 통제 조건
# ------------------------------------------------------------
#
# - CatBoost 설정 동일
# - XGBoost 설정 동일
# - Ensemble weight 동일
# - preprocessing 동일
#
# 오직 count_state 1개만 추가한다.
#
#
# 이 파일에서는
# ------------------------------------------------------------
#
# - 모델 저장 X
# - rf.pkl 덮어쓰기 X
# - Test 예측 X
# - Submission 생성 X
#
# Validation 실험만 수행한다.
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
#
# Feature 효과만 비교해야 하므로
# 기존 값을 그대로 유지한다.
# ------------------------------------------------------------

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# ------------------------------------------------------------
# 앞 실험에서 같은 조건으로 측정한 BASE Brier
# ------------------------------------------------------------

BASE_BRIER = {
    2023: 0.253595,
    2024: 0.248270,
}


# ------------------------------------------------------------
# 결과 저장 폴더
# ------------------------------------------------------------

EXP_DIR = "./model/exp4_count_only_v001"

os.makedirs(
    EXP_DIR,
    exist_ok=True
)


# ============================================================
# 2. Count State Feature 생성 함수
# ============================================================

def build_count_feature(df):
    """
    balls_before와 strikes_before를 조합해서
    count_state라는 categorical Feature를 하나 만든다.

    예:

        balls_before   = 3
        strikes_before = 2

        count_state = "3_2"


    중요:
        기존 balls_before / strikes_before는 삭제하지 않는다.

    즉 모델에는:

        balls_before
        strikes_before
        count_state

    세 정보가 모두 들어간다.
    """

    df = df.copy()


    # --------------------------------------------------------
    # balls_before를 문자열로 변환
    #
    # 혹시 결측값이 있다면 -1로 표시한다.
    # --------------------------------------------------------

    balls = (
        df["balls_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )


    # --------------------------------------------------------
    # strikes_before도 동일하게 문자열로 변환
    # --------------------------------------------------------

    strikes = (
        df["strikes_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )


    # --------------------------------------------------------
    # 두 값을 하나의 Count State로 조합
    #
    # 예:
    #
    # 0 + "_" + 0 → "0_0"
    # 3 + "_" + 2 → "3_2"
    # --------------------------------------------------------

    df["count_state"] = (
        balls
        + "_"
        + strikes
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
    현재 BASE와 같은 구조의 모델을 생성한다.

    구조:

        preprocessing
              ↓
        CatBoost
        XGBoost
              ↓
        Soft Voting

    이번 EXP-4에서 바뀌는 것은
    count_state Feature 하나뿐이다.
    """


    # --------------------------------------------------------
    # categorical이 아닌 나머지는 수치형으로 처리
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
    # 1. 결측값 → 최빈값
    # 2. category → ordinal encoding
    # 3. Train에서 못 본 category → -1
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
    # 결측값 → Train median
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
    #
    # BASE와 동일
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
    #
    # BASE와 동일
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
    Validation prediction을 평가한다.

    반환:

        Brier Score
        Baseline Brier
        Brier Skill Score

    Brier:
        낮을수록 좋다.

    BSS:
        높을수록 좋다.
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
    # Validation의 실제 성공률
    # --------------------------------------------------------

    r = (
        y_true.mean()
    )


    # --------------------------------------------------------
    # 평균 확률만 예측하는 Baseline의 Brier
    # --------------------------------------------------------

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
# 5. 데이터 불러오기
# ============================================================

print("=" * 60)
print("EXP-4 Count State Only")
print("=" * 60)

print("\nLoad data...")


# ------------------------------------------------------------
# test.csv header를 기준으로 Raw Feature 목록 생성
# ------------------------------------------------------------

test_cols = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "test.csv"
    ),
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


# row_id는 Feature에서 제외
RAW_FEATURES = [
    col
    for col in test_cols
    if col != ID_COL
]


# ============================================================
# 6. EXP-4 Feature 목록
# ============================================================

COUNT_FEATURES = [
    "count_state",
]


# Raw 47 + count_state 1
EXP4_FEATURES = (
    RAW_FEATURES
    + COUNT_FEATURES
)


print(
    f"Raw Features: "
    f"{len(RAW_FEATURES)}"
)


print(
    f"Added Features: "
    f"{len(COUNT_FEATURES)}"
)


print(
    f"EXP-4 Total Features: "
    f"{len(EXP4_FEATURES)}"
)


# ============================================================
# 7. 필요한 Raw 컬럼 확인
# ============================================================

FE_REQUIRED_COLS = [
    "balls_before",
    "strikes_before",
]


missing_cols = [
    col
    for col in FE_REQUIRED_COLS
    if col not in RAW_FEATURES
]


if missing_cols:

    raise ValueError(
        "Count State 생성에 필요한 컬럼이 없습니다: "
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
    f"\nRaw train shape: "
    f"{train.shape}"
)


# ============================================================
# 9. Count State 생성
# ============================================================

print("\nBuild count_state...")


train_fe = build_count_feature(
    train
)


# ------------------------------------------------------------
# 실제 어떤 Count가 생성됐는지 확인
# ------------------------------------------------------------

print("\ncount_state distribution:")


print(
    train_fe[
        "count_state"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# ============================================================
# 10. 생성 여부 확인
# ============================================================

missing_after_fe = [
    col
    for col in EXP4_FEATURES
    if col not in train_fe.columns
]


if missing_after_fe:

    raise ValueError(
        "Feature Engineering 후 없는 컬럼: "
        f"{missing_after_fe}"
    )


# ============================================================
# 11. 범주형 Feature 정의
# ============================================================
#
# BASE categorical:
#
#   top_bottom
#   game_type
#   base_state
#
# 여기에 count_state 하나만 추가한다.
# ============================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "count_state",
]


# ============================================================
# 12. 한 Validation 연도를 실행하는 함수
# ============================================================

def run_validation(
    train_df,
    train_max_year,
    val_year,
):
    """
    EXP-4를 한 번 학습하고
    같은 연도의 BASE Brier와 비교한다.
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
        EXP4_FEATURES,
    ].copy()


    X_val = train_df.loc[
        is_val,
        EXP4_FEATURES,
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

    print("\nBuild EXP-4 model...")


    model = build_model(
        features=EXP4_FEATURES,
        cat_cols=CAT_COLS,
    )


    # ========================================================
    # 학습
    # ========================================================

    print("\nTrain EXP-4...")


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
        exp4_brier,
        baseline_brier,
        exp4_bss,
    ) = evaluate_prediction(
        y_val,
        pred,
    )


    # 같은 연도의 기존 Raw BASE
    base_brier = (
        BASE_BRIER[val_year]
    )


    # ========================================================
    # BASE vs EXP-4
    # ========================================================
    #
    # Brier가 낮아지는 것이 개선.
    # ========================================================

    improvement = (
        base_brier
        - exp4_brier
    )


    delta = (
        exp4_brier
        - base_brier
    )


    if exp4_brier < base_brier:

        result = "IMPROVED"


    elif exp4_brier > base_brier:

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
        f"EXP-4 Brier: "
        f"{exp4_brier:.6f}"
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
        f"\nEXP-4 BSS: "
        f"{exp4_bss:.2f}"
    )


    print("\nPrediction statistics:")


    print(
        pd.Series(
            pred
        ).describe()
    )


    # ========================================================
    # Validation prediction 저장
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

        "exp4_brier":
            exp4_brier,

        "brier_delta":
            delta,

        "improvement":
            improvement,

        "result":
            result,

        "baseline_brier":
            baseline_brier,

        "exp4_bss":
            exp4_bss,

        "training_seconds":
            training_time,
    }


# ============================================================
# 13. 2023 Validation
# ============================================================
#
# Train:
#   2019 ~ 2022
#
# Validation:
#   2023
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
# Train:
#   2019 ~ 2023
#
# Validation:
#   2024
# ============================================================

result_2024 = run_validation(
    train_df=train_fe,
    train_max_year=2023,
    val_year=2024,
)


# ============================================================
# 15. Summary 저장
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
# 16. Feature 목록 저장
# ============================================================

feature_info = pd.DataFrame({
    "feature":
        COUNT_FEATURES
})


feature_info.to_csv(
    os.path.join(
        EXP_DIR,
        "count_features.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. 최종 Summary 출력
# ============================================================

print("\n")
print("=" * 60)
print("EXP-4 FINAL SUMMARY")
print("=" * 60)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 18. 간단 판단
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
        "EXP-4 COUNT STATE: POSITIVE"
    )

    print(
        "2023과 2024 모두 BASE보다 개선."
    )

    print(
        "→ count_state 단독으로도 안정적인 개선 후보."
    )


elif (
    improved_2023
    or
    improved_2024
):

    print(
        "EXP-4 COUNT STATE: MIXED"
    )

    print(
        "한 연도에서만 개선."
    )

    print(
        "→ 단독 효과는 안정적이지 않음."
    )


else:

    print(
        "EXP-4 COUNT STATE: NEGATIVE"
    )

    print(
        "두 연도 모두 BASE보다 개선되지 않음."
    )

    print(
        "→ count_state 단독 효과는 없음."
    )


print("=" * 60)


# ============================================================
# 19. 종료
# ============================================================
#
# 모델은 저장하지 않는다.
# ============================================================

print("\n")
print("=" * 60)
print("EXP-4 Complete")
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
    "  count_features.csv"
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