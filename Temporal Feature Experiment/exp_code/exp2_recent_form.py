# exp2_recent_form.py

# ============================================================
# EXP-2. Error-Driven Recent Form Feature
#
# 목적:
#   STEP 0 Error Analysis에서 발견된 가장 뚜렷한 약점인
#
#   "최근 제구력이 자신의 장기 평균보다 하락한 투수에서
#    현재 모델이 성공 확률을 과대예측하는 경향"
#
#   을 아주 단순한 categorical Feature 하나로 표현한다.
#
#
# 비교 모델
# ------------------------------------------------------------
# EXP-1:
#   Raw Features
#   + Core FE 11개
#
# EXP-2:
#   Raw Features
#   + Core FE 11개
#   + recent_form_state 1개
#
#
# recent_form_state
# ------------------------------------------------------------
#
#   missing
#       최근 3경기 success 정보가 없음
#
#   decline
#       최근 3경기 success < career success
#
#   improve_or_equal
#       최근 3경기 success >= career success
#
#
# 실험 순서
# ------------------------------------------------------------
#
# 1차:
#   Train      = 2019 ~ 2022
#   Validation = 2023
#
# 2차:
#   EXP-2가 2023에서 EXP-1보다 좋아졌을 때만 실행
#
#   Train      = 2019 ~ 2023
#   Validation = 2024
#
#
# 중요한 통제 조건
# ------------------------------------------------------------
#
# - CatBoost 설정 변경 X
# - XGBoost 설정 변경 X
# - Ensemble weight 변경 X
# - 전처리 방식 변경 X
#
# 오직 recent_form_state 1개만 추가한다.
#
#
# 이번 파일에서는
# ------------------------------------------------------------
#
# - 모델 저장 X
# - ./model/rf.pkl 덮어쓰기 X
# - test prediction X
# - submission 생성 X
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
# 현재 기본 CB + XGB 앙상블 weight
# ------------------------------------------------------------

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# ------------------------------------------------------------
# EXP-2 결과 저장 폴더
# ------------------------------------------------------------

EXP_DIR = "./model/exp2_recent_form_v001"

os.makedirs(
    EXP_DIR,
    exist_ok=True
)


# ============================================================
# 2. EXP-1 Core Feature 생성 함수
# ============================================================

def build_core_features(df):
    """
    EXP-1에서 검증한 Core Feature 11개를 생성한다.

    구성:

    A. Recent vs Career 6개
    B. Trend 4개
    C. Count State 1개
    """

    df = df.copy()


    # ========================================================
    # A. Recent vs Career - Success Rate
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
    # A. Recent vs Career - Middle Rate
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
    # B. Trend - Success
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
    # B. Trend - Middle
    # ========================================================

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
    # balls_before = 3
    # strikes_before = 2
    #
    # → count_state = "3_2"
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
# 3. EXP-2 Feature 생성 함수
# ============================================================

def build_exp2_features(df):
    """
    EXP-1 Core Feature 11개를 먼저 만든 뒤,
    EXP-2의 recent_form_state를 추가한다.

    recent_form_state는
    recent3_success_vs_career 값을 기준으로 한다.

    ----------------------------------------------------------

    missing
        최근 3경기 또는 career success 정보가 없어
        차이를 계산할 수 없는 경우

    decline
        recent3_success_vs_career < 0

        즉:
        최근 3경기 success가
        자신의 career success보다 낮음

    improve_or_equal
        recent3_success_vs_career >= 0

        즉:
        최근 상태가 career 수준 이상
    """

    # 먼저 EXP-1 Feature 생성
    df = build_core_features(df)


    # recent3_success_vs_career를 편하게 사용하기 위한 변수
    recent_diff = (
        df["recent3_success_vs_career"]
    )


    # --------------------------------------------------------
    # 기본값:
    #
    # 최근 상태가 career 수준 이상이라고 설정
    # --------------------------------------------------------

    df["recent_form_state"] = (
        "improve_or_equal"
    )


    # --------------------------------------------------------
    # 최근 상태가 career보다 낮으면 decline
    # --------------------------------------------------------

    df.loc[
        recent_diff < 0,
        "recent_form_state"
    ] = "decline"


    # --------------------------------------------------------
    # recent3 또는 career 값이 없어서
    # 차이를 계산할 수 없으면 missing
    #
    # STEP 0에서 이 missing 그룹의 Brier가 특히 높았기 때문에
    # missing 자체를 하나의 명시적인 상태로 보존한다.
    # --------------------------------------------------------

    df.loc[
        recent_diff.isna(),
        "recent_form_state"
    ] = "missing"


    return df


# ============================================================
# 4. 모델 생성 함수
# ============================================================
#
# EXP-1과 EXP-2가 같은 모델 설정을 사용하도록
# 하나의 함수에서 모델을 생성한다.
# ============================================================

def build_model(
    features,
    cat_cols,
):
    """
    CatBoost + XGBoost Soft Voting Ensemble 생성.

    EXP-1 / EXP-2 모두 같은 함수 사용.

    따라서 모델 차이가 아니라
    Feature 차이만 성능에 영향을 주도록 한다.
    """

    # 범주형을 제외하면 전부 수치형
    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]


    # ========================================================
    # 범주형 전처리
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
    # 기존 BASE / EXP-1과 동일
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
    # 기존 BASE / EXP-1과 동일
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
# 5. Prediction 평가 함수
# ============================================================

def evaluate_prediction(
    y_true,
    pred,
):
    """
    Brier Score와 Brier Skill Score 계산.

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
    # Validation 데이터의 평균 성공률
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
# 6. 데이터 불러오기
# ============================================================

print("=" * 60)
print("EXP-2 Error-Driven Recent Form")
print("=" * 60)

print("\nLoad data...")


# ------------------------------------------------------------
# test.csv의 header를 기준으로 Raw Feature 목록 생성
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
# 7. EXP-1 / EXP-2 Feature 목록
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


# EXP-2에서 딱 하나 추가
EXP2_EXTRA_FEATURES = [
    "recent_form_state",
]


# ------------------------------------------------------------
# EXP-1 Feature
# ------------------------------------------------------------

EXP1_FEATURES = (
    RAW_FEATURES
    + CORE_FEATURES
)


# ------------------------------------------------------------
# EXP-2 Feature
# ------------------------------------------------------------

EXP2_FEATURES = (
    EXP1_FEATURES
    + EXP2_EXTRA_FEATURES
)


# ============================================================
# 8. 필요한 Raw Column 확인
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
        "Feature 생성에 필요한 컬럼이 없습니다: "
        f"{missing_cols}"
    )


# ============================================================
# 9. train.csv 불러오기
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
# 10. EXP-2 Feature Engineering
# ============================================================

print("\nBuild Features...")


# 이 데이터에는:
#
# Raw
# + EXP-1 Core 11개
# + EXP-2 recent_form_state
#
# 가 모두 들어 있다.

train_fe = build_exp2_features(
    train
)


print(
    f"EXP-1 Core Features: "
    f"{len(CORE_FEATURES)}"
)


print(
    f"EXP-2 Extra Features: "
    f"{len(EXP2_EXTRA_FEATURES)}"
)


print(
    f"EXP-1 Total Features: "
    f"{len(EXP1_FEATURES)}"
)


print(
    f"EXP-2 Total Features: "
    f"{len(EXP2_FEATURES)}"
)


# ============================================================
# 11. recent_form_state 분포 확인
# ============================================================
#
# Feature가 정상적으로 만들어졌는지 확인한다.
# ============================================================

print("\nrecent_form_state distribution:")


print(
    train_fe[
        "recent_form_state"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 12. 생성된 Feature 존재 여부 확인
# ============================================================

missing_after_fe = [
    col
    for col in EXP2_FEATURES
    if col not in train_fe.columns
]


if missing_after_fe:

    raise ValueError(
        "Feature Engineering 후 없는 컬럼: "
        f"{missing_after_fe}"
    )


# ============================================================
# 13. 범주형 Feature
# ============================================================

EXP1_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",

    # EXP-1
    "count_state",
]


EXP2_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",

    # EXP-1
    "count_state",

    # EXP-2
    "recent_form_state",
]


# ============================================================
# 14. 한 Validation 구간을 실험하는 함수
# ============================================================

def run_validation(
    train_df,
    train_max_year,
    val_year,
):
    """
    같은 Train/Validation 구간에서

    EXP-1
    vs
    EXP-2

    두 모델을 직접 비교한다.
    """

    print("\n")
    print("=" * 60)

    print(
        f"VALIDATION {val_year}"
    )

    print("=" * 60)


    # ========================================================
    # Train / Validation split
    # ========================================================

    is_train = (
        train_df["season"]
        <= train_max_year
    )


    is_val = (
        train_df["season"]
        == val_year
    )


    y_train = train_df.loc[
        is_train,
        TARGET_COL,
    ].copy()


    y_val = train_df.loc[
        is_val,
        TARGET_COL,
    ].copy()


    # --------------------------------------------------------
    # EXP-1 데이터
    # --------------------------------------------------------

    X_train_exp1 = train_df.loc[
        is_train,
        EXP1_FEATURES,
    ].copy()


    X_val_exp1 = train_df.loc[
        is_val,
        EXP1_FEATURES,
    ].copy()


    # --------------------------------------------------------
    # EXP-2 데이터
    # --------------------------------------------------------

    X_train_exp2 = train_df.loc[
        is_train,
        EXP2_FEATURES,
    ].copy()


    X_val_exp2 = train_df.loc[
        is_val,
        EXP2_FEATURES,
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
    # EXP-1 Model
    # ========================================================

    print("\nBuild EXP-1 model...")


    exp1_model = build_model(
        features=EXP1_FEATURES,
        cat_cols=EXP1_CAT_COLS,
    )


    print("\nTrain EXP-1...")


    start = time.time()


    exp1_model.fit(
        X_train_exp1,
        y_train,
    )


    exp1_time = (
        time.time()
        - start
    )


    print(
        f"EXP-1 training complete :: "
        f"{exp1_time:.1f}s"
    )


    # --------------------------------------------------------
    # EXP-1 Prediction
    # --------------------------------------------------------

    exp1_pred = (
        exp1_model
        .predict_proba(
            X_val_exp1
        )[:, 1]
    )


    (
        exp1_brier,
        baseline_brier,
        exp1_bss,
    ) = evaluate_prediction(
        y_val,
        exp1_pred,
    )


    print(
        f"EXP-1 Brier: "
        f"{exp1_brier:.6f}"
    )


    print(
        f"EXP-1 BSS  : "
        f"{exp1_bss:.2f}"
    )


    # ========================================================
    # EXP-2 Model
    # ========================================================

    print("\nBuild EXP-2 model...")


    exp2_model = build_model(
        features=EXP2_FEATURES,
        cat_cols=EXP2_CAT_COLS,
    )


    print("\nTrain EXP-2...")


    start = time.time()


    exp2_model.fit(
        X_train_exp2,
        y_train,
    )


    exp2_time = (
        time.time()
        - start
    )


    print(
        f"EXP-2 training complete :: "
        f"{exp2_time:.1f}s"
    )


    # --------------------------------------------------------
    # EXP-2 Prediction
    # --------------------------------------------------------

    exp2_pred = (
        exp2_model
        .predict_proba(
            X_val_exp2
        )[:, 1]
    )


    (
        exp2_brier,
        _,
        exp2_bss,
    ) = evaluate_prediction(
        y_val,
        exp2_pred,
    )


    print(
        f"EXP-2 Brier: "
        f"{exp2_brier:.6f}"
    )


    print(
        f"EXP-2 BSS  : "
        f"{exp2_bss:.2f}"
    )


    # ========================================================
    # EXP-1 vs EXP-2 비교
    # ========================================================

    improvement = (
        exp1_brier
        - exp2_brier
    )


    delta = (
        exp2_brier
        - exp1_brier
    )


    if exp2_brier < exp1_brier:

        result = "IMPROVED"


    elif exp2_brier > exp1_brier:

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
        f"EXP-1 Brier: "
        f"{exp1_brier:.6f}"
    )


    print(
        f"EXP-2 Brier: "
        f"{exp2_brier:.6f}"
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
        f"\nEXP-1 BSS: "
        f"{exp1_bss:.2f}"
    )


    print(
        f"EXP-2 BSS: "
        f"{exp2_bss:.2f}"
    )


    # ========================================================
    # Validation Prediction 저장
    # ========================================================

    prediction_result = pd.DataFrame({
        "target":
            y_val.to_numpy(),

        "exp1_pred":
            exp1_pred,

        "exp2_pred":
            exp2_pred,
    })


    prediction_result[
        "exp1_brier_error"
    ] = (
        (
            prediction_result["exp1_pred"]
            - prediction_result["target"]
        ) ** 2
    )


    prediction_result[
        "exp2_brier_error"
    ] = (
        (
            prediction_result["exp2_pred"]
            - prediction_result["target"]
        ) ** 2
    )


    # 양수이면 해당 행에서 EXP-2가 더 잘한 것
    prediction_result[
        "error_improvement"
    ] = (
        prediction_result["exp1_brier_error"]
        - prediction_result["exp2_brier_error"]
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
    # Summary 반환
    # ========================================================

    return {
        "validation_year":
            val_year,

        "train_seasons":
            f"2019-{train_max_year}",

        "exp1_brier":
            exp1_brier,

        "exp2_brier":
            exp2_brier,

        "brier_delta":
            delta,

        "improvement":
            improvement,

        "result":
            result,

        "baseline_brier":
            baseline_brier,

        "exp1_bss":
            exp1_bss,

        "exp2_bss":
            exp2_bss,

        "exp1_training_seconds":
            exp1_time,

        "exp2_training_seconds":
            exp2_time,
    }


# ============================================================
# 15. 2023 Validation
# ============================================================
#
# Feature 설계에 사용하고 있는 개발 구간.
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


# 결과를 담을 리스트
all_results = [
    result_2023
]


# ============================================================
# 16. 2023에서 좋아졌을 때만 2024 실행
# ============================================================
#
# 2023부터 안 좋아진 Feature를 굳이
# 2024까지 반복해서 확인하지 않는다.
# ============================================================

if result_2023["result"] == "IMPROVED":

    print("\n")
    print("=" * 60)

    print(
        "EXP-2 improved 2023."
    )

    print(
        "→ Run 2024 Validation."
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Train:
    #   2019 ~ 2023
    #
    # Validation:
    #   2024
    # --------------------------------------------------------

    result_2024 = run_validation(
        train_df=train_fe,
        train_max_year=2023,
        val_year=2024,
    )


    all_results.append(
        result_2024
    )


else:

    print("\n")
    print("=" * 60)

    print(
        "EXP-2 did NOT improve 2023."
    )

    print(
        "→ Stop before 2024 Validation."
    )

    print("=" * 60)


# ============================================================
# 17. 전체 Summary 저장
# ============================================================

summary = pd.DataFrame(
    all_results
)


summary.to_csv(
    os.path.join(
        EXP_DIR,
        "summary.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. Feature 목록 저장
# ============================================================

feature_info = pd.DataFrame({
    "feature":
        CORE_FEATURES
        + EXP2_EXTRA_FEATURES,

    "stage":
        (
            ["EXP-1 Core"] * len(CORE_FEATURES)
            +
            ["EXP-2 Extra"] * len(EXP2_EXTRA_FEATURES)
        ),
})


feature_info.to_csv(
    os.path.join(
        EXP_DIR,
        "features.csv"
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 19. 최종 판단
# ============================================================

print("\n")
print("=" * 60)
print("EXP-2 FINAL SUMMARY")
print("=" * 60)


print(
    summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 2023만 실행된 경우
# ------------------------------------------------------------

if len(all_results) == 1:

    print("\nEXP-2 rejected.")

    print(
        "2023에서 EXP-1보다 좋아지지 않았으므로 "
        "EXP-2 Feature를 채택하지 않는다."
    )


# ------------------------------------------------------------
# 2023 + 2024 둘 다 실행된 경우
# ------------------------------------------------------------

else:

    result_2024 = all_results[1]


    if (
        result_2023["result"] == "IMPROVED"
        and
        result_2024["result"] == "IMPROVED"
    ):

        print("\nEXP-2 CANDIDATE ACCEPTED.")

        print(
            "2023과 2024 모두 EXP-1보다 개선."
        )

        print(
            "→ EXP-2를 현재 Feature 후보로 채택할 근거가 있음."
        )


    else:

        print("\nEXP-2 NOT ACCEPTED.")

        print(
            "2023 개선이 2024에서 재현되지 않음."
        )

        print(
            "→ EXP-1을 유지하는 것이 더 안전함."
        )


# ============================================================
# 20. 종료
# ============================================================

print("\n")
print("=" * 60)
print("EXP-2 Complete")
print("=" * 60)


print(
    f"Results saved to: "
    f"{EXP_DIR}"
)


print(
    "\nNo model was saved or overwritten."
)