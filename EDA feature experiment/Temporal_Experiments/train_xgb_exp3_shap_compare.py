# train_xgb_exp3_shap_compare.py
"""
LG Aimers Phase2
CatBoost vs XGBoost - Temporal 10개 SHAP 중요도 비교

목적
----
동일한 EXP3 Temporal Feature 10개를
CatBoost와 XGBoost가 서로 어떤 순서로 중요하게 사용하는지 비교한다.

이번 실험에서 확인하는 것
--------------------------
1) XGBoost EXP3를 현재 표준 Expanding Window 4-Fold로 학습
2) 각 Fold Validation에서 Temporal 10개의 XGBoost SHAP 계산
3) 기존 CatBoost EXP3 SHAP 결과를 불러옴
4) CatBoost vs XGBoost Temporal 10개 중요도 순위 비교
5) Temporal 10개 내부 중요도 비중(%) 비교
6) Success vs Middle / Level vs Trend 그룹 비교
7) 두 모델 순위의 Spearman rank correlation 계산

중요한 해석 원칙
----------------
- SHAP은 "모델이 얼마나 사용했는가"를 나타낸다.
- SHAP 순위가 높다고 해당 Feature가 Brier를 개선했다는 뜻은 아니다.
- 서로 다른 모델의 raw SHAP 절대값 자체보다
  "모델 내부 순위"와 "Temporal 10개 내부 비중"을 중심으로 비교한다.

CatBoost 2023 Fold 주의
----------------------
기존 CatBoost EXP3의 2023 Fold는 best iteration이 사실상 0이고
Temporal SHAP이 모두 0이었다.

따라서:
- 4개 Fold 결과는 모두 저장한다.
- CatBoost vs XGBoost의 주 비교는 동일한 2021 / 2022 / 2024 3개 Fold로 한다.

XGBoost 설정
------------
기존 Temporal Feature Experiment의 exp3_xgboost/train.py에서 사용한
XGBoost 단독 EXP3 설정과 전처리를 그대로 유지한다.

- CAT_COLS 5개: top_bottom, game_type, base_state, pitcher_hand, batter_hand
- OrdinalEncoder
- ID-like 컬럼은 numeric 처리
- objective=binary:logistic
- learning_rate=0.03
- max_depth=8
- min_child_weight=10
- subsample=0.85
- colsample_bytree=0.85
- reg_alpha=1
- reg_lambda=10
- tree_method=hist
- n_estimators 최대 5000
- Brier custom eval_metric
- early_stopping_rounds=200
- random_state=42

누수 / 독립성
-------------
Temporal Feature는 각 row에 이미 존재하는 공식 asof_* 값끼리만 계산한다.
test 내부 다른 row를 이용한 groupby / rolling / lag / frequency / 분포 계산은 없다.
"""

import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "xgb_exp3_expanding_shap_v001"

DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# 기존 CatBoost EXP3 SHAP 결과
CATBOOST_EXPERIMENT_NAME = "cb_v001_ctr1_expanding_exp3"
CATBOOST_ANALYSIS_NAME = "shap_residual_v001"

CATBOOST_SHAP_PATH = (
    VALIDATION_ROOT
    / CATBOOST_EXPERIMENT_NAME
    / "analysis"
    / CATBOOST_ANALYSIS_NAME
    / "shap_temporal_by_fold.csv"
)


# =========================================================
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

VALID_SEASONS = [
    2021,
    2022,
    2023,
    2024,
]

# CatBoost 2023 Fold는 SHAP이 모두 0이므로
# 모델 간 주 비교에서는 2021 / 2022 / 2024만 사용
PRIMARY_COMPARE_SEASONS = [
    2021,
    2022,
    2024,
]

FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2019, 2020],
        "valid_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "valid_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "valid_season": 2024,
    },
]


# =========================================================
# XGBoost 기존 정책
# =========================================================

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42

XGB_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# SHAP 설정
# 기존 CatBoost SHAP과 동일한 row 위치를 뽑기 위한 설정
# =========================================================

SHAP_SAMPLE_SIZE = 10_000
SHAP_RANDOM_SEED = 42


# =========================================================
# Temporal Feature
# =========================================================

TEMPORAL_FEATURES = [
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
    "success_trend_1_3",
    "success_trend_3_5",
    "middle_trend_1_3",
    "middle_trend_3_5",
]

TEMPORAL_GROUPS = {
    "success_level": [
        "recent1_success_vs_career",
        "recent3_success_vs_career",
        "recent5_success_vs_career",
    ],
    "middle_level": [
        "recent1_middle_vs_career",
        "recent3_middle_vs_career",
        "recent5_middle_vs_career",
    ],
    "success_trend": [
        "success_trend_1_3",
        "success_trend_3_5",
    ],
    "middle_trend": [
        "middle_trend_1_3",
        "middle_trend_3_5",
    ],
    "success_all": [
        "recent1_success_vs_career",
        "recent3_success_vs_career",
        "recent5_success_vs_career",
        "success_trend_1_3",
        "success_trend_3_5",
    ],
    "middle_all": [
        "recent1_middle_vs_career",
        "recent3_middle_vs_career",
        "recent5_middle_vs_career",
        "middle_trend_1_3",
        "middle_trend_3_5",
    ],
    "level_all": [
        "recent1_success_vs_career",
        "recent3_success_vs_career",
        "recent5_success_vs_career",
        "recent1_middle_vs_career",
        "recent3_middle_vs_career",
        "recent5_middle_vs_career",
    ],
    "trend_all": [
        "success_trend_1_3",
        "success_trend_3_5",
        "middle_trend_1_3",
        "middle_trend_3_5",
    ],
}


# =========================================================
# Temporal Feature 생성
# =========================================================

def add_temporal_features(df):
    """
    기존 EXP3 Temporal Feature 10개와 동일한 계산.

    *_vs_career는 기존 변수명을 유지하지만 실제 의미는
    Recent vs 해당 시점까지 Cumulative(as-of)이다.
    """
    df = df.copy()

    required_cols = [
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    # Recent vs Cumulative - Success
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

    # Recent vs Cumulative - Middle
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

    # Success Trend
    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )

    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    # Middle Trend
    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )

    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


# =========================================================
# Data QA
# =========================================================

def validate_temporal_features(
    before_df,
    after_df,
    raw_features,
):
    """
    Temporal 생성 QA
    - row 수 보존
    - row_id 순서 보존 / unique
    - target 결측 없음
    - Raw Feature 47개
    - 수식 검증
    - inf 없음
    - NaN 기록
    """
    assert len(before_df) == len(after_df), (
        "Feature 생성 후 row 수가 변했습니다."
    )

    assert before_df[ID_COL].equals(
        after_df[ID_COL]
    ), (
        "Feature 생성 후 row_id 순서가 변했습니다."
    )

    assert after_df[ID_COL].is_unique, (
        "row_id 중복이 있습니다."
    )

    assert after_df[TARGET_COL].notna().all(), (
        "Target에 결측이 있습니다."
    )

    if len(raw_features) != 47:
        raise ValueError(
            f"Raw Feature 수가 예상(47)과 다릅니다: "
            f"{len(raw_features)}"
        )

    expected_map = {
        "recent1_success_vs_career": (
            after_df["asof_pitcher_prev1_game_success_rate"]
            - after_df["asof_pitcher_success_rate"]
        ),
        "recent3_success_vs_career": (
            after_df["asof_pitcher_prev3_game_success_rate"]
            - after_df["asof_pitcher_success_rate"]
        ),
        "recent5_success_vs_career": (
            after_df["asof_pitcher_prev5_game_success_rate"]
            - after_df["asof_pitcher_success_rate"]
        ),
        "recent1_middle_vs_career": (
            after_df["asof_pitcher_prev1_game_middle_rate"]
            - after_df["asof_pitcher_middle_rate"]
        ),
        "recent3_middle_vs_career": (
            after_df["asof_pitcher_prev3_game_middle_rate"]
            - after_df["asof_pitcher_middle_rate"]
        ),
        "recent5_middle_vs_career": (
            after_df["asof_pitcher_prev5_game_middle_rate"]
            - after_df["asof_pitcher_middle_rate"]
        ),
        "success_trend_1_3": (
            after_df["asof_pitcher_prev1_game_success_rate"]
            - after_df["asof_pitcher_prev3_game_success_rate"]
        ),
        "success_trend_3_5": (
            after_df["asof_pitcher_prev3_game_success_rate"]
            - after_df["asof_pitcher_prev5_game_success_rate"]
        ),
        "middle_trend_1_3": (
            after_df["asof_pitcher_prev1_game_middle_rate"]
            - after_df["asof_pitcher_prev3_game_middle_rate"]
        ),
        "middle_trend_3_5": (
            after_df["asof_pitcher_prev3_game_middle_rate"]
            - after_df["asof_pitcher_prev5_game_middle_rate"]
        ),
    }

    for feature, expected in expected_map.items():
        if not np.allclose(
            expected.to_numpy(dtype=np.float64),
            after_df[feature].to_numpy(dtype=np.float64),
            equal_nan=True,
        ):
            raise AssertionError(
                f"Temporal Feature 수식 불일치: {feature}"
            )

    values = (
        after_df[TEMPORAL_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    inf_counts = dict(
        zip(
            TEMPORAL_FEATURES,
            np.isinf(values)
            .sum(axis=0)
            .astype(int)
            .tolist(),
        )
    )

    if any(
        count > 0
        for count in inf_counts.values()
    ):
        raise ValueError(
            f"Temporal Feature에 inf/-inf가 있습니다: "
            f"{inf_counts}"
        )

    nan_counts = {
        feature: int(
            after_df[feature]
            .isna()
            .sum()
        )
        for feature in TEMPORAL_FEATURES
    }

    return {
        "raw_feature_count": int(
            len(raw_features)
        ),
        "temporal_feature_count": int(
            len(TEMPORAL_FEATURES)
        ),
        "total_feature_count": int(
            len(raw_features)
            + len(TEMPORAL_FEATURES)
        ),
        "temporal_nan_counts": nan_counts,
        "temporal_inf_counts": inf_counts,
    }


# =========================================================
# XGBoost 전처리
# 기존 exp3_xgboost/train.py와 동일
# =========================================================

def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "입력 데이터에 Feature가 없습니다: "
            + ", ".join(missing)
        )

    X = df[features].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


def create_preprocessor(
    cat_cols,
    num_cols,
):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ],
        remainder="drop",
    )


def transformed_feature_names(
    cat_cols,
    num_cols,
):
    """
    ColumnTransformer 출력 순서는
    cat transformer -> num transformer 순서이므로
    실제 XGBoost 입력 column 순서를 명시적으로 반환한다.
    """
    return (
        list(cat_cols)
        + list(num_cols)
    )


# =========================================================
# Brier
# 기존 XGBoost EXP3 코드와 동일한 custom metric
# =========================================================

def brier_metric(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    r = float(
        y_true.mean()
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    baseline_brier = float(
        r * (1.0 - r)
    )

    bss = float(
        max(
            0.0,
            100000.0
            * (
                1.0
                - brier / baseline_brier
            ),
        )
    )

    return {
        "positive_rate": r,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": bss,
        "prediction_mean": float(
            y_pred.mean()
        ),
        "prediction_std": float(
            y_pred.std()
        ),
    }


def create_validation_model():
    """
    기존 XGBoost 단독 EXP3 모델 설정 그대로.
    """
    return XGBClassifier(
        n_estimators=MAX_ITERATIONS,
        eval_metric=brier_metric,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **MODEL_PARAMS,
    )


# =========================================================
# SHAP sampling
# CatBoost 분석과 동일한 위치 sampling
# =========================================================

def select_shap_sample(
    validation_df,
    valid_season,
):
    n = len(
        validation_df
    )

    if (
        SHAP_SAMPLE_SIZE is None
        or SHAP_SAMPLE_SIZE >= n
    ):
        return validation_df.copy()

    rng = np.random.default_rng(
        SHAP_RANDOM_SEED
        + int(valid_season)
    )

    positions = np.sort(
        rng.choice(
            n,
            size=SHAP_SAMPLE_SIZE,
            replace=False,
        )
    )

    return (
        validation_df
        .iloc[positions]
        .copy()
    )


# =========================================================
# XGBoost native SHAP
# =========================================================

def calculate_xgb_shap(
    model,
    X_transformed,
    feature_names,
    valid_season,
):
    """
    XGBoost Booster의 pred_contribs=True를 이용해
    native SHAP contribution을 계산한다.

    마지막 column은 bias(expected value)이므로 제외한다.
    """
    X_transformed = np.asarray(
        X_transformed,
        dtype=np.float32,
    )

    dmatrix = xgb.DMatrix(
        X_transformed,
        feature_names=list(
            feature_names
        ),
    )

    booster = model.get_booster()

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    if (
        best_iteration is not None
        and int(best_iteration) >= 0
    ):
        iteration_range = (
            0,
            int(best_iteration) + 1,
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True,
            iteration_range=iteration_range,
        )
    else:
        contributions = booster.predict(
            dmatrix,
            pred_contribs=True,
        )

    # 마지막 column = Bias term
    feature_contributions = (
        contributions[:, :-1]
    )

    if (
        feature_contributions.shape[1]
        != len(feature_names)
    ):
        raise AssertionError(
            "XGBoost SHAP Feature 수와 입력 Feature 수가 다릅니다."
        )

    mean_abs_shap = np.mean(
        np.abs(
            feature_contributions
        ),
        axis=0,
    )

    result = pd.DataFrame(
        {
            "model": "xgboost",
            "validation_season": int(
                valid_season
            ),
            "feature": list(
                feature_names
            ),
            "mean_abs_shap": (
                mean_abs_shap
            ),
            "n_shap_sample": int(
                len(X_transformed)
            ),
        }
    )

    return result


# =========================================================
# XGBoost Expanding 학습 + Fold SHAP
# =========================================================

def train_xgb_expanding_and_shap(
    train,
    raw_features,
    feature_qa,
):
    """
    XGBoost EXP3 4-Fold Expanding Window 학습.
    Fold별 model / preprocessor / prediction / SHAP 저장.
    """
    features = (
        list(raw_features)
        + TEMPORAL_FEATURES
    )

    cat_cols = [
        col
        for col in XGB_CAT_COLS
        if col in features
    ]

    id_like_cols = [
        col
        for col in ID_LIKE_COLS
        if col in features
    ]

    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]

    transformed_names = (
        transformed_feature_names(
            cat_cols,
            num_cols,
        )
    )

    # Temporal 10개는 수치형이므로 변환 후에도 같은 이름으로 존재해야 함
    missing_temporal = [
        feature
        for feature in TEMPORAL_FEATURES
        if feature not in transformed_names
    ]

    if missing_temporal:
        raise AssertionError(
            "변환된 XGBoost Feature 이름에서 Temporal Feature 누락: "
            + ", ".join(missing_temporal)
        )

    experiment_model_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
    )

    if experiment_model_dir.exists():
        raise FileExistsError(
            f"기존 XGBoost Model experiment가 있습니다: "
            f"{experiment_model_dir}\n"
            "기존 결과 보호를 위해 자동 덮어쓰지 않습니다."
        )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 XGBoost Validation experiment가 있습니다: "
            f"{VALIDATION_DIR}\n"
            "기존 결과 보호를 위해 자동 덮어쓰지 않습니다."
        )

    experiment_model_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        VALIDATION_DIR
        / "train_xgb_exp3_shap_compare.py",
    )

    fold_results = []
    oof_frames = []
    shap_frames = []

    print()
    print("=" * 92)
    print("XGBoost EXP3 Expanding Window + SHAP")
    print("=" * 92)
    print("XGBoost version      :", xgb.__version__)
    print("Raw Feature          :", len(raw_features))
    print("Temporal Feature     :", len(TEMPORAL_FEATURES))
    print("Total Feature        :", len(features))
    print("XGB categorical      :", cat_cols)
    print("Primary SHAP seasons :", PRIMARY_COMPARE_SEASONS)
    print("=" * 92)

    for fold_config in FOLDS:
        fold_number = int(
            fold_config["fold"]
        )

        train_seasons = list(
            fold_config[
                "train_seasons"
            ]
        )

        valid_season = int(
            fold_config[
                "valid_season"
            ]
        )

        # -------------------------------------------------
        # 시간축 QA
        # -------------------------------------------------

        assert (
            max(train_seasons)
            < valid_season
        )

        is_train = (
            train["season"]
            .isin(train_seasons)
        )

        is_valid = (
            train["season"]
            == valid_season
        )

        train_ids = train.loc[
            is_train,
            ID_COL,
        ]

        valid_ids = train.loc[
            is_valid,
            ID_COL,
        ]

        if not set(
            train_ids
        ).isdisjoint(
            set(valid_ids)
        ):
            raise AssertionError(
                f"Fold {fold_number}: "
                "Train/Validation row_id가 겹칩니다."
            )

        # -------------------------------------------------
        # Raw 입력
        # -------------------------------------------------

        train_df = train.loc[
            is_train
        ]

        valid_df = train.loc[
            is_valid
        ]

        y_train = train_df[
            TARGET_COL
        ].copy()

        y_valid = valid_df[
            TARGET_COL
        ].copy()

        X_train_raw = (
            prepare_raw_features(
                train_df,
                features,
                id_like_cols,
            )
        )

        X_valid_raw = (
            prepare_raw_features(
                valid_df,
                features,
                id_like_cols,
            )
        )

        # -------------------------------------------------
        # Fold마다 train만으로 preprocessor fit
        # -------------------------------------------------

        preprocessor = (
            create_preprocessor(
                cat_cols,
                num_cols,
            )
        )

        X_train = (
            preprocessor
            .fit_transform(
                X_train_raw
            )
        )

        X_valid = (
            preprocessor
            .transform(
                X_valid_raw
            )
        )

        # -------------------------------------------------
        # XGBoost 학습
        # -------------------------------------------------

        model = (
            create_validation_model()
        )

        print()
        print("-" * 92)
        print(
            f"Fold {fold_number}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]}"
            f" -> {valid_season}"
        )
        print(
            "Train:",
            X_train.shape,
            "| Valid:",
            X_valid.shape,
        )
        print("-" * 92)

        start = time.time()

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_valid,
                    y_valid,
                )
            ],
            verbose=100,
        )

        elapsed = float(
            time.time()
            - start
        )

        valid_pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        metrics = (
            calculate_metrics(
                y_valid.to_numpy(),
                valid_pred,
            )
        )

        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        if (
            best_iteration is None
            or int(best_iteration) < 0
        ):
            final_iterations = int(
                MAX_ITERATIONS
            )
        else:
            final_iterations = int(
                best_iteration
                + 1
            )

        print(
            f"Brier={metrics['brier']:.8f} | "
            f"BSS={metrics['bss']:.2f} | "
            f"iter={final_iterations} | "
            f"time={elapsed:.1f}s"
        )

        # -------------------------------------------------
        # Fold 모델 / 전처리 저장
        # -------------------------------------------------

        fold_model_dir = (
            experiment_model_dir
            / f"fold_{valid_season}"
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        model_path = (
            fold_model_dir
            / (
                f"{EXPERIMENT_NAME}"
                f"_fold_{valid_season}.pkl"
            )
        )

        preprocessor_path = (
            fold_model_dir
            / (
                f"{EXPERIMENT_NAME}"
                f"_preprocessor_{valid_season}.pkl"
            )
        )

        joblib.dump(
            model,
            model_path,
            compress=3,
        )

        joblib.dump(
            preprocessor,
            preprocessor_path,
            compress=3,
        )

        # -------------------------------------------------
        # Fold prediction 저장
        # -------------------------------------------------

        fold_validation_dir = (
            VALIDATION_DIR
            / f"fold_{valid_season}"
        )

        fold_validation_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_pred = pd.DataFrame(
            {
                ID_COL: (
                    valid_ids
                    .to_numpy()
                ),
                "season": int(
                    valid_season
                ),
                "target": (
                    y_valid
                    .to_numpy()
                ),
                "prediction": (
                    valid_pred
                ),
                "fold": int(
                    fold_number
                ),
            }
        )

        fold_pred.to_csv(
            fold_validation_dir
            / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # -------------------------------------------------
        # CatBoost SHAP과 동일 위치의 10,000 row sample
        # -------------------------------------------------

        sample = (
            select_shap_sample(
                valid_df,
                valid_season,
            )
        )

        X_shap_raw = (
            prepare_raw_features(
                sample,
                features,
                id_like_cols,
            )
        )

        X_shap = (
            preprocessor
            .transform(
                X_shap_raw
            )
        )

        fold_shap = (
            calculate_xgb_shap(
                model=model,
                X_transformed=X_shap,
                feature_names=(
                    transformed_names
                ),
                valid_season=(
                    valid_season
                ),
            )
        )

        fold_shap.to_csv(
            fold_validation_dir
            / "shap_all_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

        temporal_shap = (
            fold_shap.loc[
                fold_shap[
                    "feature"
                ].isin(
                    TEMPORAL_FEATURES
                )
            ]
            .copy()
        )

        temporal_shap.to_csv(
            fold_validation_dir
            / "shap_temporal.csv",
            index=False,
            encoding="utf-8-sig",
        )

        shap_frames.append(
            temporal_shap
        )

        # -------------------------------------------------
        # Fold metadata
        # -------------------------------------------------

        fold_metadata = {
            "experiment_name": (
                EXPERIMENT_NAME
            ),
            "fold": fold_number,
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "valid_season": (
                valid_season
            ),
            "n_train": int(
                is_train.sum()
            ),
            "n_valid": int(
                is_valid.sum()
            ),
            "raw_feature_count": int(
                len(raw_features)
            ),
            "temporal_feature_count": int(
                len(TEMPORAL_FEATURES)
            ),
            "total_feature_count": int(
                len(features)
            ),
            "cat_cols": cat_cols,
            "id_like_cols": (
                id_like_cols
            ),
            "transformed_feature_names": (
                transformed_names
            ),
            "model_params": (
                MODEL_PARAMS
            ),
            "max_iterations": (
                MAX_ITERATIONS
            ),
            "early_stopping_rounds": (
                EARLY_STOPPING_ROUNDS
            ),
            "best_iteration_zero_based": (
                None
                if best_iteration is None
                else int(
                    best_iteration
                )
            ),
            "final_iterations": (
                final_iterations
            ),
            "elapsed_seconds": (
                elapsed
            ),
            "shap_sample_size": int(
                len(sample)
            ),
            **metrics,
        }

        with (
            fold_validation_dir
            / "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                fold_metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        fold_results.append(
            fold_metadata
        )

        oof_frames.append(
            fold_pred
        )

        del (
            X_train_raw,
            X_valid_raw,
            X_train,
            X_valid,
            y_train,
            y_valid,
            model,
            preprocessor,
            valid_pred,
            fold_pred,
            sample,
            X_shap_raw,
            X_shap,
            fold_shap,
            temporal_shap,
        )

        gc.collect()

    # -----------------------------------------------------
    # OOF 저장
    # -----------------------------------------------------

    oof = pd.concat(
        oof_frames,
        axis=0,
        ignore_index=True,
    )

    if not oof[ID_COL].is_unique:
        raise AssertionError(
            "XGBoost OOF row_id 중복"
        )

    oof.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_metrics = (
        calculate_metrics(
            oof[
                "target"
            ].to_numpy(),
            oof[
                "prediction"
            ].to_numpy(),
        )
    )

    # -----------------------------------------------------
    # XGB Temporal SHAP 전체 저장
    # -----------------------------------------------------

    xgb_temporal_by_fold = (
        pd.concat(
            shap_frames,
            axis=0,
            ignore_index=True,
        )
    )

    xgb_temporal_by_fold.to_csv(
        VALIDATION_DIR
        / "xgb_temporal_shap_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4 Fold 평균
    xgb_4fold = (
        xgb_temporal_by_fold
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            std_abs_shap=(
                "mean_abs_shap",
                "std",
            ),
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
    )

    xgb_4fold.to_csv(
        VALIDATION_DIR
        / "xgb_temporal_shap_4fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 주 비교 2021/2022/2024
    xgb_primary = (
        xgb_temporal_by_fold.loc[
            xgb_temporal_by_fold[
                "validation_season"
            ].isin(
                PRIMARY_COMPARE_SEASONS
            )
        ]
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            std_abs_shap=(
                "mean_abs_shap",
                "std",
            ),
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
    )

    xgb_primary.to_csv(
        VALIDATION_DIR
        / "xgb_temporal_shap_primary3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "purpose": (
            "XGBoost EXP3 Expanding Window 학습 및 "
            "CatBoost vs XGBoost Temporal 10 SHAP 순위 비교"
        ),
        "feature_qa": (
            feature_qa
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "max_iterations": (
            MAX_ITERATIONS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "fold_results": (
            fold_results
        ),
        "pooled_oof_metrics": (
            pooled_metrics
        ),
        "primary_compare_seasons": (
            PRIMARY_COMPARE_SEASONS
        ),
    }

    with (
        VALIDATION_DIR
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall_metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return (
        xgb_temporal_by_fold,
        pooled_metrics,
    )


# =========================================================
# CatBoost SHAP 로드
# =========================================================

def load_catboost_temporal_shap():
    """
    기존 analyze_temporal_exp3.py에서 저장한
    CatBoost EXP3 Temporal Fold별 SHAP을 불러온다.
    """
    if not CATBOOST_SHAP_PATH.exists():
        raise FileNotFoundError(
            "기존 CatBoost Temporal SHAP 파일이 없습니다:\n"
            f"{CATBOOST_SHAP_PATH}\n\n"
            "먼저 analyze_temporal_exp3.py를 실행해 "
            "CatBoost SHAP 결과를 생성해야 합니다."
        )

    cat = pd.read_csv(
        CATBOOST_SHAP_PATH,
        encoding="utf-8-sig",
    )

    required = {
        "validation_season",
        "feature",
        "mean_abs_shap",
    }

    missing = (
        required
        - set(cat.columns)
    )

    if missing:
        raise ValueError(
            "CatBoost SHAP 파일 필수 컬럼 누락: "
            + ", ".join(
                sorted(missing)
            )
        )

    cat = cat.loc[
        cat["feature"].isin(
            TEMPORAL_FEATURES
        )
    ].copy()

    cat["model"] = (
        "catboost"
    )

    return cat


# =========================================================
# Rank correlation
# =========================================================

def spearman_rank_correlation(
    left_rank,
    right_rank,
):
    """
    두 rank vector의 Pearson correlation을 계산하면
    Spearman rank correlation과 동일하다.
    """
    left = np.asarray(
        left_rank,
        dtype=np.float64,
    )

    right = np.asarray(
        right_rank,
        dtype=np.float64,
    )

    if (
        len(left) < 2
        or np.std(left) == 0
        or np.std(right) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            left,
            right,
        )[0, 1]
    )


# =========================================================
# 모델 간 SHAP 비교
# =========================================================

def build_model_comparison(
    cat_by_fold,
    xgb_by_fold,
):
    """
    CatBoost와 XGBoost의 Temporal 10개 SHAP 순위를 비교한다.

    주 결과:
    - 2021 / 2022 / 2024 평균
    - 모델 내부 normalized temporal share
    - rank difference
    - Spearman rank correlation
    """
    analysis_dir = (
        VALIDATION_DIR
        / "model_comparison"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Primary 3-Fold 평균
    # -----------------------------------------------------

    cat_primary = (
        cat_by_fold.loc[
            cat_by_fold[
                "validation_season"
            ].isin(
                PRIMARY_COMPARE_SEASONS
            )
        ]
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            catboost_mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            catboost_std_abs_shap=(
                "mean_abs_shap",
                "std",
            ),
        )
    )

    xgb_primary = (
        xgb_by_fold.loc[
            xgb_by_fold[
                "validation_season"
            ].isin(
                PRIMARY_COMPARE_SEASONS
            )
        ]
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            xgboost_mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            xgboost_std_abs_shap=(
                "mean_abs_shap",
                "std",
            ),
        )
    )

    comparison = (
        cat_primary.merge(
            xgb_primary,
            on="feature",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(comparison) != len(
        TEMPORAL_FEATURES
    ):
        raise AssertionError(
            "모델 비교에 Temporal 10개가 모두 포함되지 않았습니다."
        )

    # 각 모델 내부 rank
    comparison[
        "catboost_rank"
    ] = (
        comparison[
            "catboost_mean_abs_shap"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    comparison[
        "xgboost_rank"
    ] = (
        comparison[
            "xgboost_mean_abs_shap"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # Temporal 10개 안에서의 중요도 비중
    cat_sum = float(
        comparison[
            "catboost_mean_abs_shap"
        ].sum()
    )

    xgb_sum = float(
        comparison[
            "xgboost_mean_abs_shap"
        ].sum()
    )

    comparison[
        "catboost_temporal_share_pct"
    ] = (
        comparison[
            "catboost_mean_abs_shap"
        ]
        / cat_sum
        * 100.0
    )

    comparison[
        "xgboost_temporal_share_pct"
    ] = (
        comparison[
            "xgboost_mean_abs_shap"
        ]
        / xgb_sum
        * 100.0
    )

    comparison[
        "rank_difference_cat_minus_xgb"
    ] = (
        comparison[
            "catboost_rank"
        ]
        - comparison[
            "xgboost_rank"
        ]
    )

    comparison[
        "abs_rank_difference"
    ] = np.abs(
        comparison[
            "rank_difference_cat_minus_xgb"
        ]
    )

    comparison = (
        comparison
        .sort_values(
            [
                "catboost_rank",
                "xgboost_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    comparison.to_csv(
        analysis_dir
        / "catboost_vs_xgboost_temporal_rank.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Overall Spearman
    # -----------------------------------------------------

    overall_spearman = (
        spearman_rank_correlation(
            comparison[
                "catboost_rank"
            ],
            comparison[
                "xgboost_rank"
            ],
        )
    )

    # -----------------------------------------------------
    # Fold별 ranking / correlation
    # -----------------------------------------------------

    fold_compare_frames = []
    fold_corr_rows = []

    for season in (
        PRIMARY_COMPARE_SEASONS
    ):
        cat_fold = (
            cat_by_fold.loc[
                cat_by_fold[
                    "validation_season"
                ]
                == season,
                [
                    "feature",
                    "mean_abs_shap",
                ],
            ]
            .rename(
                columns={
                    "mean_abs_shap":
                    "catboost_mean_abs_shap"
                }
            )
        )

        xgb_fold = (
            xgb_by_fold.loc[
                xgb_by_fold[
                    "validation_season"
                ]
                == season,
                [
                    "feature",
                    "mean_abs_shap",
                ],
            ]
            .rename(
                columns={
                    "mean_abs_shap":
                    "xgboost_mean_abs_shap"
                }
            )
        )

        fold = cat_fold.merge(
            xgb_fold,
            on="feature",
            how="inner",
            validate="one_to_one",
        )

        fold[
            "catboost_rank"
        ] = (
            fold[
                "catboost_mean_abs_shap"
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

        fold[
            "xgboost_rank"
        ] = (
            fold[
                "xgboost_mean_abs_shap"
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

        fold[
            "validation_season"
        ] = int(
            season
        )

        fold[
            "abs_rank_difference"
        ] = np.abs(
            fold[
                "catboost_rank"
            ]
            - fold[
                "xgboost_rank"
            ]
        )

        corr = (
            spearman_rank_correlation(
                fold[
                    "catboost_rank"
                ],
                fold[
                    "xgboost_rank"
                ],
            )
        )

        fold_corr_rows.append(
            {
                "validation_season": int(
                    season
                ),
                "spearman_rank_correlation": (
                    corr
                ),
            }
        )

        fold_compare_frames.append(
            fold
        )

    fold_compare = pd.concat(
        fold_compare_frames,
        axis=0,
        ignore_index=True,
    )

    fold_compare.to_csv(
        analysis_dir
        / "catboost_vs_xgboost_temporal_rank_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlation_df = pd.DataFrame(
        fold_corr_rows
        + [
            {
                "validation_season": (
                    "primary_3fold_mean"
                ),
                "spearman_rank_correlation": (
                    overall_spearman
                ),
            }
        ]
    )

    correlation_df.to_csv(
        analysis_dir
        / "rank_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Group summary
    # -----------------------------------------------------

    model_frames = []

    for model_name, source, value_col in [
        (
            "catboost",
            comparison,
            "catboost_mean_abs_shap",
        ),
        (
            "xgboost",
            comparison,
            "xgboost_mean_abs_shap",
        ),
    ]:
        total_temporal = float(
            source[
                value_col
            ].sum()
        )

        for (
            group_name,
            group_features,
        ) in TEMPORAL_GROUPS.items():
            part = source.loc[
                source[
                    "feature"
                ].isin(
                    group_features
                )
            ]

            group_sum = float(
                part[
                    value_col
                ].sum()
            )

            model_frames.append(
                {
                    "model": model_name,
                    "group": group_name,
                    "n_features": int(
                        len(part)
                    ),
                    "sum_mean_abs_shap": (
                        group_sum
                    ),
                    "mean_abs_shap_per_feature": float(
                        part[
                            value_col
                        ].mean()
                    ),
                    "temporal_share_pct": float(
                        group_sum
                        / total_temporal
                        * 100.0
                    ),
                }
            )

    group_summary = pd.DataFrame(
        model_frames
    )

    group_summary.to_csv(
        analysis_dir
        / "temporal_group_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "primary_compare_seasons": (
            PRIMARY_COMPARE_SEASONS
        ),
        "catboost_2023_excluded_reason": (
            "Existing CatBoost EXP3 2023 Fold stopped at "
            "best iteration 0 and all Temporal SHAP values were 0."
        ),
        "overall_primary_3fold_spearman": (
            overall_spearman
        ),
        "interpretation": (
            "Compare within-model ranks and normalized Temporal shares. "
            "SHAP is model usage, not predictive usefulness."
        ),
    }

    with (
        analysis_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return (
        comparison,
        correlation_df,
        group_summary,
    )


# =========================================================
# Console 출력
# =========================================================

def print_results(
    comparison,
    correlation_df,
    group_summary,
    pooled_metrics,
):
    print()
    print("=" * 112)
    print(
        "CatBoost vs XGBoost - Temporal 10 SHAP 순위 "
        "(2021 / 2022 / 2024 평균)"
    )
    print("=" * 112)

    display = comparison[
        [
            "feature",
            "catboost_rank",
            "catboost_temporal_share_pct",
            "xgboost_rank",
            "xgboost_temporal_share_pct",
            "abs_rank_difference",
        ]
    ].copy()

    display[
        "catboost_temporal_share_pct"
    ] = (
        display[
            "catboost_temporal_share_pct"
        ]
        .map(
            lambda x: f"{x:.2f}%"
        )
    )

    display[
        "xgboost_temporal_share_pct"
    ] = (
        display[
            "xgboost_temporal_share_pct"
        ]
        .map(
            lambda x: f"{x:.2f}%"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print("Rank correlation")
    print("=" * 112)
    print(
        correlation_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print("Success vs Middle / Level vs Trend")
    print("=" * 112)

    important_groups = [
        "success_all",
        "middle_all",
        "level_all",
        "trend_all",
        "success_level",
        "success_trend",
        "middle_level",
        "middle_trend",
    ]

    group_display = (
        group_summary.loc[
            group_summary[
                "group"
            ].isin(
                important_groups
            )
        ]
        .copy()
    )

    group_display[
        "temporal_share_pct"
    ] = (
        group_display[
            "temporal_share_pct"
        ]
        .map(
            lambda x: f"{x:.2f}%"
        )
    )

    print(
        group_display[
            [
                "model",
                "group",
                "n_features",
                "mean_abs_shap_per_feature",
                "temporal_share_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print("XGBoost EXP3 Expanding 성능 참고")
    print("=" * 112)
    print(
        "Pooled OOF Brier:",
        f"{pooled_metrics['brier']:.8f}",
    )
    print(
        "Pooled OOF BSS  :",
        f"{pooled_metrics['bss']:.2f}",
    )

    print()
    print(
        "주의: SHAP 순위는 모델 내부 활용도입니다. "
        "순위가 높다고 Brier 개선 Feature라는 뜻은 아닙니다."
    )
    print(
        "또한 CatBoost/XGBoost raw SHAP 절대값을 직접 크기 비교하기보다 "
        "rank와 Temporal 10개 내부 비중을 중심으로 해석하세요."
    )
    print("=" * 112)


# =========================================================
# Main
# =========================================================

def main():
    # -----------------------------------------------------
    # CatBoost SHAP 존재 여부를 학습 전에 확인
    # -----------------------------------------------------

    if not CATBOOST_SHAP_PATH.exists():
        raise FileNotFoundError(
            "CatBoost SHAP 결과가 없습니다:\n"
            f"{CATBOOST_SHAP_PATH}\n\n"
            "기존 analyze_temporal_exp3.py를 먼저 실행하세요."
        )

    # -----------------------------------------------------
    # Raw 47 구성
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID_COL not in test_cols:
        raise ValueError(
            f"test.csv에 '{ID_COL}'가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID_COL
    ]

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR
        / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID_COL]
            + raw_features
            + [TARGET_COL]
        ),
    )

    if "season" not in train.columns:
        raise ValueError(
            "train.csv에 season 컬럼이 없습니다."
        )

    before = train.copy()

    train = add_temporal_features(
        train
    )

    feature_qa = (
        validate_temporal_features(
            before_df=before,
            after_df=train,
            raw_features=raw_features,
        )
    )

    del before
    gc.collect()

    # -----------------------------------------------------
    # XGBoost EXP3 학습 + SHAP
    # -----------------------------------------------------

    (
        xgb_temporal_by_fold,
        pooled_metrics,
    ) = train_xgb_expanding_and_shap(
        train=train,
        raw_features=raw_features,
        feature_qa=feature_qa,
    )

    # -----------------------------------------------------
    # 기존 CatBoost SHAP 로드
    # -----------------------------------------------------

    catboost_temporal_by_fold = (
        load_catboost_temporal_shap()
    )

    # -----------------------------------------------------
    # 모델 간 비교
    # -----------------------------------------------------

    (
        comparison,
        correlation_df,
        group_summary,
    ) = build_model_comparison(
        cat_by_fold=(
            catboost_temporal_by_fold
        ),
        xgb_by_fold=(
            xgb_temporal_by_fold
        ),
    )

    # -----------------------------------------------------
    # 최종 출력
    # -----------------------------------------------------

    print_results(
        comparison=comparison,
        correlation_df=correlation_df,
        group_summary=group_summary,
        pooled_metrics=pooled_metrics,
    )

    print()
    print(
        "결과 저장:",
        VALIDATION_DIR
        / "model_comparison"
        / "catboost_vs_xgboost_temporal_rank.csv",
    )


if __name__ == "__main__":
    main()
