# exp1_core_fe.py

# ============================================================
# EXP-1. Core Feature Engineering
#
# 목적:
#   현재 CatBoost + XGBoost 기본 앙상블의 모델 설정은 그대로 두고,
#   세 참고 글에서 가져온 핵심 Feature 11개만 추가했을 때
#   성능이 개선되는지 확인한다.
#
# 추가 Feature:
#
#   A. Recent vs Career 6개
#      최근 1/3/5경기 기록 - 투수 장기 기록
#
#   B. Trend 4개
#      최근 1경기 - 최근 3경기
#      최근 3경기 - 최근 5경기
#
#   C. Count State 1개
#      balls_before + strikes_before
#
# 학습:
#   2019 ~ 2022
#
# 검증:
#   2023
#
# 비교 기준:
#   기존 Raw CB+XGB의 2023 Brier = 0.253595
#
# 주의:
#   - 모델 하이퍼파라미터는 BASE와 동일
#   - Ensemble weight도 BASE와 동일
#   - 이번 실험에서는 모델 저장하지 않음
#   - 기존 ./model/rf.pkl 덮어쓰지 않음
#   - 2024 데이터는 아직 사용하지 않음
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
# 기존 BASE 모델의 2023 Validation Brier
#
# STEP 0에서:
#
# 2019~2022 → Train
# 2023      → Validation
#
# 으로 얻은 결과.
#
# EXP-1의 Brier가 이것보다 낮아지면 개선이다.
# ------------------------------------------------------------

BASE_BRIER_2023 = 0.253595


# ------------------------------------------------------------
# 기존 최고 앙상블 weight 그대로 사용
# ------------------------------------------------------------

CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# ------------------------------------------------------------
# EXP-1 결과 저장 폴더
#
# 모델 자체를 저장하는 것이 아니라
# validation 결과와 실험 summary만 저장한다.
# ------------------------------------------------------------

EXP_DIR = "./model/exp1_core_fe_v001"

os.makedirs(
    EXP_DIR,
    exist_ok=True
)


# ============================================================
# 2. EXP-1 Feature 생성 함수
# ============================================================

def build_features(df):
    """
    Raw Feature에 EXP-1 Core Feature 11개를 추가한다.

    추가되는 Feature:

    [A] Recent vs Career 6개

        최근 1경기 success - career success
        최근 3경기 success - career success
        최근 5경기 success - career success

        최근 1경기 middle - career middle
        최근 3경기 middle - career middle
        최근 5경기 middle - career middle


    [B] Trend 4개

        최근 1경기 success - 최근 3경기 success
        최근 3경기 success - 최근 5경기 success

        최근 1경기 middle - 최근 3경기 middle
        최근 3경기 middle - 최근 5경기 middle


    [C] Count State 1개

        balls_before + strikes_before

        예:
            balls=3, strikes=2
            → "3_2"
    """

    df = df.copy()


    # ========================================================
    # A. Recent vs Career
    # ========================================================
    #
    # 핵심 아이디어:
    #
    # "최근 성공률이 0.50이다"
    #
    # 보다
    #
    # "원래 0.55인 투수가 최근 0.50으로 떨어졌다"
    #
    # 와 같이 최근 상태를 자신의 장기 수준과 비교해서
    # 모델에게 직접 알려준다.
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
    #
    # Recent vs Career가
    # "최근 상태가 평소보다 어떤가?"
    # 를 표현한다면,
    #
    # Trend는
    # "최근 상태가 어느 방향으로 움직이고 있는가?"
    # 를 표현한다.
    #
    # 예:
    #
    # prev1 > prev3
    # → 최근 쪽으로 갈수록 상승
    #
    # prev1 < prev3
    # → 최근 쪽으로 갈수록 하락
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
    # 기존에는:
    #
    # balls_before = 3
    # strikes_before = 2
    #
    # 를 서로 다른 숫자 Feature로 사용했다.
    #
    # 여기서는 추가로:
    #
    # count_state = "3_2"
    #
    # 를 만들어 하나의 야구 상황으로도 표현한다.
    #
    # 원래 balls / strikes Feature는 삭제하지 않는다.
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
# 3. 데이터 컬럼 구조 확인
# ============================================================

print("Load data...")


# test.csv의 헤더만 읽어서
# 실제 inference 시 존재하는 Raw Feature 목록을 만든다.
#
# 이렇게 하면 train에서만 존재하는 target이
# Feature에 들어가는 것을 방지할 수 있다.

test_cols = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "test.csv"
    ),
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()


# row_id는 식별자이므로 모델 Feature에서 제외
RAW_FEATURES = [
    col
    for col in test_cols
    if col != ID_COL
]


# ============================================================
# 4. EXP-1에서 새로 추가하는 Feature 목록
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

    # Ball-Strike Count
    "count_state",
]


# 최종 EXP-1 모델이 사용할 Feature
FEATURES = (
    RAW_FEATURES
    + CORE_FEATURES
)


# ============================================================
# 5. 필요한 Raw Column 확인
# ============================================================
#
# build_features()가 사용하는 원본 컬럼이
# 실제 데이터에 존재하는지 먼저 확인한다.
#
# 이름이 틀렸다면 학습을 시작하기 전에
# 바로 에러를 발생시킨다.
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
        "EXP-1 Feature 생성에 필요한 컬럼이 없습니다: "
        f"{missing_cols}"
    )


# ============================================================
# 6. train.csv 불러오기
# ============================================================
#
# 중요:
#
# CORE_FEATURES는 CSV에 실제로 존재하는 컬럼이 아니다.
#
# 따라서 처음에는 RAW_FEATURES만 불러온 뒤,
# build_features()를 실행해서 새 Feature를 만든다.
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


# ============================================================
# 7. Core Feature Engineering 적용
# ============================================================

print("\nBuild Core Features...")


train = build_features(train)


print(
    f"Raw features : "
    f"{len(RAW_FEATURES)}"
)

print(
    f"Core features: "
    f"{len(CORE_FEATURES)}"
)

print(
    f"Total features: "
    f"{len(FEATURES)}"
)


print("\nAdded Core Features:")

for col in CORE_FEATURES:
    print(
        f"  {col}"
    )


# ============================================================
# 8. 생성된 Feature 확인
# ============================================================

missing_after_fe = [
    col
    for col in FEATURES
    if col not in train.columns
]


if missing_after_fe:
    raise ValueError(
        "Feature 생성 후에도 없는 컬럼이 있습니다: "
        f"{missing_after_fe}"
    )


# ============================================================
# 9. 범주형 / 수치형 Feature 구분
# ============================================================

# 기존 범주형 3개에
# 새 count_state를 추가한다.

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "count_state",
]


# 나머지는 모두 수치형 처리
NUM_COLS = [
    col
    for col in FEATURES
    if col not in CAT_COLS
]


print(
    f"\nCategorical features: "
    f"{len(CAT_COLS)}"
)

print(
    f"Numerical features: "
    f"{len(NUM_COLS)}"
)


# ============================================================
# 10. 전처리
# ============================================================

print("\nBuild preprocessing...")


# ------------------------------------------------------------
# 범주형 전처리
#
# 1. 결측값 → 최빈값
# 2. 문자열 category → 정수값
#
# validation에 train에서 보지 못한 category가 등장하면
# -1로 encoding
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 수치형 전처리
#
# 결측값 → Train median
# ------------------------------------------------------------

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
# 11. CatBoost
# ============================================================
#
# BASE 모델 설정 그대로 유지한다.
#
# EXP-1에서 바꾸는 것은 Feature뿐이다.
# ============================================================

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

    # catboost_info 디렉터리가
    # 자동 생성되지 않도록 함
    allow_writing_files=False,
)


# ============================================================
# 12. XGBoost
# ============================================================
#
# 역시 BASE 모델 설정 그대로 사용.
# ============================================================

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


# ============================================================
# 13. CatBoost + XGBoost Ensemble
# ============================================================
#
# STEP 0과 동일한 weight를 그대로 사용한다.
#
# CatBoost : 56.56%
# XGBoost  : 43.44%
# ============================================================

print("Build Ensemble...")


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


# 전처리와 앙상블 모델을 하나의 Pipeline으로 구성
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


# ============================================================
# 14. 2019~2022 Train / 2023 Validation
# ============================================================
#
# STEP 0과 동일한 split을 사용해야
# BASE vs EXP-1을 공정하게 비교할 수 있다.
#
# 2024는 여기서 사용하지 않는다.
# ============================================================

print("\nSplit train / validation...")


is_train = (
    train["season"] <= 2022
)

is_val = (
    train["season"] == 2023
)


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
    f"Train rows: "
    f"{len(X_train):,}"
)

print(
    f"Validation rows: "
    f"{len(X_val):,}"
)


print(
    "Train seasons: "
    f"{train.loc[is_train, 'season'].min()} "
    f"~ "
    f"{train.loc[is_train, 'season'].max()}"
)

print(
    "Validation season: 2023"
)


# ============================================================
# 15. 모델 학습
# ============================================================

print("\nTrain EXP-1 Ensemble...")


start = time.time()


model.fit(
    X_train,
    y_train,
)


train_time = (
    time.time()
    - start
)


print(
    f"Training complete :: "
    f"{train_time:.1f}s"
)


# ============================================================
# 16. Validation Prediction
# ============================================================

print("\nValidation...")


val_pred = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 17. Brier Score 계산
# ============================================================

y_val_array = (
    y_val.to_numpy()
)


brier = (
    (
        val_pred
        - y_val_array
    ) ** 2
).mean()


# ============================================================
# 18. Brier Skill Score 계산
# ============================================================

r = (
    y_val.mean()
)


baseline_brier = (
    r
    * (1 - r)
)


score = max(
    0,
    100000
    * (
        1
        - brier / baseline_brier
    )
)


# ============================================================
# 19. BASE와 EXP-1 직접 비교
# ============================================================
#
# Brier는 낮을수록 좋다.
#
# 예:
#
# BASE   = 0.253595
# EXP-1  = 0.252500
#
# delta  = -0.001095
#
# → EXP-1 개선
# ============================================================

brier_delta = (
    brier
    - BASE_BRIER_2023
)


brier_improvement = (
    BASE_BRIER_2023
    - brier
)


if brier < BASE_BRIER_2023:
    result = "IMPROVED"

elif brier > BASE_BRIER_2023:
    result = "WORSE"

else:
    result = "SAME"


# ============================================================
# 20. 결과 출력
# ============================================================

print("\n")
print("=" * 60)
print("EXP-1 RESULT")
print("=" * 60)


print(
    f"BASE Brier : "
    f"{BASE_BRIER_2023:.6f}"
)


print(
    f"EXP-1 Brier: "
    f"{brier:.6f}"
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
    f"Validation Score: "
    f"{score:.2f}"
)


print("\nPrediction statistics:")


print(
    pd.Series(
        val_pred
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
# 21. Validation Prediction 저장
# ============================================================
#
# 나중에 BASE / EXP-1 또는 Error Analysis를
# 비교할 수 있도록 validation prediction을 저장한다.
# ============================================================

validation_result = pd.DataFrame({
    "target": y_val_array,
    "pred": val_pred,
})


validation_result["brier_error"] = (
    (
        validation_result["pred"]
        - validation_result["target"]
    ) ** 2
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
# 22. 실험 Summary 저장
# ============================================================

summary = pd.DataFrame([
    {
        "experiment": "EXP-1 Core FE",

        "train_season": "2019-2022",
        "validation_season": 2023,

        "raw_features": len(RAW_FEATURES),
        "added_features": len(CORE_FEATURES),
        "total_features": len(FEATURES),

        "base_brier": BASE_BRIER_2023,
        "exp1_brier": brier,

        "brier_delta": brier_delta,
        "brier_improvement": brier_improvement,

        "result": result,

        "baseline_brier": baseline_brier,
        "validation_score": score,

        "catboost_weight": CAT_WEIGHT,
        "xgboost_weight": XGB_WEIGHT,

        "training_seconds": train_time,
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
# 23. 추가 Feature 목록도 저장
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
# 24. 종료
# ============================================================
#
# 중요:
#
# 이번 EXP-1에서는:
#
#   - 전체 데이터 재학습 X
#   - 모델 저장 X
#   - rf.pkl 덮어쓰기 X
#   - test prediction X
#   - 제출 X
#
# 오직:
#
#   BASE Raw Features
#        vs
#   Raw + Core FE 11개
#
# 의 2023 Validation 성능을 비교한다.
# ============================================================

print("\n")
print("=" * 60)
print("EXP-1 Complete")
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


# 최종적으로 사람이 바로 판단할 수 있게
# 한 번 더 간단히 출력한다.

print("\n")
print("=" * 60)

if result == "IMPROVED":

    print(
        "EXP-1 improved the 2023 Brier."
    )

    print(
        "→ 다음 단계: 2024 Validation 확인"
    )

else:

    print(
        "EXP-1 did NOT improve the 2023 Brier."
    )

    print(
        "→ 아직 2024 Validation / 제출로 넘어가지 않음"
    )

print("=" * 60)