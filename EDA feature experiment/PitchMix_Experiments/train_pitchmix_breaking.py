# train_pitchmix_breaking.py
"""
LG Aimers Phase2
cb_v002 - Breaking Pitch Mix × Count Feature Experiment

목적
----
현재 기준 모델 cb_v002 (Raw 47)와 모델 조건을 완전히 동일하게 유지하고,
Breaking 성향 × Count interaction 2개만 추가하여
Feature의 predictive usefulness를 검증한다.

비교
----
Baseline:
    cb_v002
    Raw Feature 47개

Experiment:
    Raw Feature 47개
    + breaking_x_balls
    + breaking_x_strikes

추가 Feature
------------
breaking_x_balls
    = asof_pitcher_breaking_rate * balls_before

breaking_x_strikes
    = asof_pitcher_breaking_rate * strikes_before

고정 조건
---------
- CatBoostClassifier
- iterations 최대 800
- learning_rate = 0.07
- depth = 8
- loss_function = Logloss
- l2_leaf_reg = 10
- random_strength = 1
- random_seed = 42
- max_ctr_complexity = 1
- Early Stopping = 80
- 기존 categorical feature 9개 동일
- 기존 범주형 결측 처리 동일
- Expanding Window 4 Fold 동일

주의
----
- Feature Set만 변경한다.
- 현재 투구의 실제 구종은 사용하지 않는다.
- asof_pitcher_breaking_rate는 해당 행의 투구 직전까지 계산된
  공식 asof_* Feature이므로 사용 가능하다.
- test의 다른 row를 이용한 집계 / rolling / lag / 빈도 계산은 없다.
- Feature screening 단계이므로 2019~2024 Final Model 재학습은 하지 않는다.
"""

import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_v002_pitchmix_breaking_v001"

DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

FOLD_MODEL_DIR = MODEL_DIR / EXPERIMENT_NAME

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기준 실험
# =========================================================

BASELINE_EXPERIMENT_NAME = "cb_v002"

BASELINE_VALIDATION_DIR = (
    VALIDATION_ROOT
    / BASELINE_EXPERIMENT_NAME
)

BASELINE_OOF_PATH = (
    BASELINE_VALIDATION_DIR
    / "oof_pred.csv"
)


# =========================================================
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"


# =========================================================
# CatBoost 범주형 Feature
# cb_v002와 동일
# =========================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
]


# =========================================================
# Expanding Window
# cb_v002와 동일
# =========================================================

EXPANDING_FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2019, 2020],
        "validation_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "validation_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "validation_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [
            2019,
            2020,
            2021,
            2022,
            2023,
        ],
        "validation_season": 2024,
    },
]


# =========================================================
# CatBoost 설정
# cb_v002와 동일
# =========================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# =========================================================
# 신규 Feature
# =========================================================

BREAKING_COUNT_FEATURES = [
    "breaking_x_balls",
    "breaking_x_strikes",
]


# =========================================================
# Feature 생성
# =========================================================

def add_breaking_count_features(df):
    """
    Breaking 성향 × 현재 Count interaction 2개 생성.

    모든 연산은 현재 row 안에 존재하는 공식 입력값만 사용한다.

    breaking_x_balls
        = asof_pitcher_breaking_rate × balls_before

    breaking_x_strikes
        = asof_pitcher_breaking_rate × strikes_before
    """

    df = df.copy()

    required_cols = [
        "asof_pitcher_breaking_rate",
        "balls_before",
        "strikes_before",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Breaking × Count Feature 생성에 필요한 "
            "컬럼이 없습니다: "
            + ", ".join(missing_cols)
        )

    df["breaking_x_balls"] = (
        df["asof_pitcher_breaking_rate"]
        * df["balls_before"]
    )

    df["breaking_x_strikes"] = (
        df["asof_pitcher_breaking_rate"]
        * df["strikes_before"]
    )

    return df


# =========================================================
# 신규 Feature QA
# =========================================================

def validate_breaking_features(
    before_df,
    after_df,
    raw_features,
):
    """
    Feature 생성 과정 QA.

    검사
    ----
    1. row 수 보존
    2. row_id 순서 보존
    3. row_id unique
    4. target 결측 없음
    5. Raw Feature = 47개
    6. 신규 Feature 수식 검증
    7. inf / -inf 없음
    8. NaN 개수 기록
    """

    if len(before_df) != len(after_df):
        raise AssertionError(
            "Feature 생성 후 row 수가 변했습니다."
        )

    if not before_df[ID_COL].equals(
        after_df[ID_COL]
    ):
        raise AssertionError(
            "Feature 생성 후 row_id 순서가 변했습니다."
        )

    if not after_df[ID_COL].is_unique:
        raise AssertionError(
            "row_id 중복이 있습니다."
        )

    if not after_df[TARGET_COL].notna().all():
        raise AssertionError(
            "Target에 결측이 있습니다."
        )

    if len(raw_features) != 47:
        raise ValueError(
            "Raw Feature 수가 예상(47)과 다릅니다: "
            f"{len(raw_features)}"
        )

    # -----------------------------------------------------
    # 수식 검증
    # -----------------------------------------------------

    expected_balls = (
        after_df["asof_pitcher_breaking_rate"]
        * after_df["balls_before"]
    )

    expected_strikes = (
        after_df["asof_pitcher_breaking_rate"]
        * after_df["strikes_before"]
    )

    if not np.allclose(
        expected_balls.to_numpy(
            dtype=np.float64
        ),
        after_df[
            "breaking_x_balls"
        ].to_numpy(
            dtype=np.float64
        ),
        equal_nan=True,
    ):
        raise AssertionError(
            "breaking_x_balls 수식이 일치하지 않습니다."
        )

    if not np.allclose(
        expected_strikes.to_numpy(
            dtype=np.float64
        ),
        after_df[
            "breaking_x_strikes"
        ].to_numpy(
            dtype=np.float64
        ),
        equal_nan=True,
    ):
        raise AssertionError(
            "breaking_x_strikes 수식이 일치하지 않습니다."
        )

    # -----------------------------------------------------
    # inf 확인
    # -----------------------------------------------------

    values = (
        after_df[BREAKING_COUNT_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    inf_counts = dict(
        zip(
            BREAKING_COUNT_FEATURES,
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
            "신규 Feature에 inf/-inf가 있습니다: "
            f"{inf_counts}"
        )

    nan_counts = {
        feature: int(
            after_df[feature]
            .isna()
            .sum()
        )
        for feature in BREAKING_COUNT_FEATURES
    }

    return {
        "n_rows_before": int(
            len(before_df)
        ),
        "n_rows_after": int(
            len(after_df)
        ),
        "row_id_unique": bool(
            after_df[ID_COL].is_unique
        ),
        "raw_feature_count": int(
            len(raw_features)
        ),
        "extra_feature_count": int(
            len(BREAKING_COUNT_FEATURES)
        ),
        "total_feature_count": int(
            len(raw_features)
            + len(BREAKING_COUNT_FEATURES)
        ),
        "extra_feature_nan_counts": (
            nan_counts
        ),
        "extra_feature_inf_counts": (
            inf_counts
        ),
    }


# =========================================================
# CatBoost 전처리
# cb_v002와 동일
# =========================================================

def prepare_features(df, cat_cols):
    """
    cb_v002와 동일.

    - 수치형 NaN: CatBoost가 직접 처리
    - 범주형 NaN: "__MISSING__"
    - 범주형 dtype: str
    """

    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Metric
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Brier / Baseline Brier / BSS 계산.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    positive_rate = float(
        np.mean(y_true)
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    baseline_brier = float(
        positive_rate
        * (1.0 - positive_rate)
    )

    if baseline_brier <= 0:
        bss_raw = 0.0
        bss = 0.0

    else:
        bss_raw = float(
            100000.0
            * (
                1.0
                - brier
                / baseline_brier
            )
        )

        bss = max(
            0.0,
            bss_raw,
        )

    return {
        "positive_rate": (
            positive_rate
        ),
        "brier": brier,
        "baseline_brier": (
            baseline_brier
        ),
        "bss_raw": (
            bss_raw
        ),
        "bss": bss,
        "prediction_mean": float(
            np.mean(y_pred)
        ),
        "prediction_std": float(
            np.std(y_pred)
        ),
        "prediction_min": float(
            np.min(y_pred)
        ),
        "prediction_max": float(
            np.max(y_pred)
        ),
    }


# =========================================================
# Expanding Window Validation
# =========================================================

def run_expanding_validation(
    train,
    raw_features,
):
    """
    Breaking × Count Feature Set을
    4-Fold Expanding Window로 검증한다.
    """

    features = (
        list(raw_features)
        + BREAKING_COUNT_FEATURES
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    fold_results = []
    oof_frames = []

    for fold_config in EXPANDING_FOLDS:

        fold = int(
            fold_config["fold"]
        )

        train_seasons = (
            fold_config[
                "train_seasons"
            ]
        )

        validation_season = int(
            fold_config[
                "validation_season"
            ]
        )

        # -------------------------------------------------
        # 시간축 QA
        # -------------------------------------------------

        if (
            max(train_seasons)
            >= validation_season
        ):
            raise AssertionError(
                "Expanding Window 시간 순서가 잘못되었습니다."
            )

        train_mask = (
            train["season"]
            .isin(train_seasons)
        )

        val_mask = (
            train["season"]
            == validation_season
        )

        if not train_mask.any():
            raise ValueError(
                f"Fold {fold}: Train 데이터가 없습니다."
            )

        if not val_mask.any():
            raise ValueError(
                f"Fold {fold}: Validation 데이터가 없습니다."
            )

        train_ids = train.loc[
            train_mask,
            ID_COL,
        ]

        val_ids = train.loc[
            val_mask,
            ID_COL,
        ]

        if not set(
            train_ids
        ).isdisjoint(
            set(val_ids)
        ):
            raise AssertionError(
                f"Fold {fold}: "
                "Train/Validation row_id가 겹칩니다."
            )

        # -------------------------------------------------
        # 데이터
        # -------------------------------------------------

        X_train = prepare_features(
            train.loc[
                train_mask,
                features,
            ],
            cat_cols,
        )

        y_train = train.loc[
            train_mask,
            TARGET_COL,
        ]

        X_val = prepare_features(
            train.loc[
                val_mask,
                features,
            ],
            cat_cols,
        )

        y_val = train.loc[
            val_mask,
            TARGET_COL,
        ]

        # -------------------------------------------------
        # 모델
        # -------------------------------------------------

        estimator = CatBoostClassifier(
            **MODEL_PARAMS
        )

        print()
        print("=" * 74)

        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]}"
            f" -> {validation_season}"
        )

        print("=" * 74)

        print(
            "Train:",
            X_train.shape,
        )

        print(
            "Valid:",
            X_val.shape,
        )

        print(
            "Features:",
            len(features),
        )

        print(
            "Extra:",
            BREAKING_COUNT_FEATURES,
        )

        # -------------------------------------------------
        # 학습
        # -------------------------------------------------

        start_time = time.time()

        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(
                X_val,
                y_val,
            ),
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            use_best_model=True,
        )

        elapsed = float(
            time.time()
            - start_time
        )

        # -------------------------------------------------
        # 예측
        # -------------------------------------------------

        val_pred = (
            estimator.predict_proba(
                X_val
            )[:, 1]
        )

        metrics = (
            calculate_metrics(
                y_val,
                val_pred,
            )
        )

        best_iteration = (
            estimator.get_best_iteration()
        )

        if (
            best_iteration is None
            or best_iteration < 0
        ):
            final_iterations = int(
                MODEL_PARAMS[
                    "iterations"
                ]
            )

        else:
            final_iterations = int(
                best_iteration + 1
            )

        # -------------------------------------------------
        # Fold Model 저장
        # -------------------------------------------------

        fold_model_dir = (
            FOLD_MODEL_DIR
            / f"fold_{validation_season}"
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_model_path = (
            fold_model_dir
            / (
                f"{EXPERIMENT_NAME}"
                f"_fold_{validation_season}.pkl"
            )
        )

        joblib.dump(
            estimator,
            fold_model_path,
            compress=3,
        )

        # -------------------------------------------------
        # Fold Prediction 저장
        # -------------------------------------------------

        fold_validation_dir = (
            VALIDATION_DIR
            / f"fold_{validation_season}"
        )

        fold_validation_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        pred_df = pd.DataFrame(
            {
                ID_COL: (
                    val_ids.to_numpy()
                ),
                "target": (
                    y_val.to_numpy()
                ),
                "prediction": (
                    val_pred
                ),
                "validation_season": (
                    validation_season
                ),
            }
        )

        pred_df.to_csv(
            fold_validation_dir
            / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # -------------------------------------------------
        # Fold Metadata
        # -------------------------------------------------

        fold_metadata = {
            "experiment_name": (
                EXPERIMENT_NAME
            ),
            "fold": fold,
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "validation_season": (
                validation_season
            ),
            "n_train": int(
                train_mask.sum()
            ),
            "n_validation": int(
                val_mask.sum()
            ),
            "raw_feature_count": int(
                len(raw_features)
            ),
            "extra_feature_count": int(
                len(
                    BREAKING_COUNT_FEATURES
                )
            ),
            "total_feature_count": int(
                len(features)
            ),
            "extra_features": list(
                BREAKING_COUNT_FEATURES
            ),
            "categorical_features": list(
                cat_cols
            ),
            "model_params": (
                MODEL_PARAMS
            ),
            "early_stopping_rounds": (
                EARLY_STOPPING_ROUNDS
            ),
            "best_iteration_zero_based": (
                None
                if (
                    best_iteration is None
                    or best_iteration < 0
                )
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
            "model_path": str(
                fold_model_path
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
            pred_df
        )

        print(
            f"Brier Score      : "
            f"{metrics['brier']:.8f}"
        )

        print(
            f"BSS              : "
            f"{metrics['bss']:.2f}"
        )

        print(
            f"Best iteration   : "
            f"{best_iteration}"
        )

        print(
            f"Final iterations : "
            f"{final_iterations}"
        )

        print(
            f"학습 시간         : "
            f"{elapsed:.1f}초"
        )

        del (
            X_train,
            X_val,
            y_train,
            y_val,
            estimator,
            val_pred,
            pred_df,
        )

        gc.collect()

    # -----------------------------------------------------
    # OOF 통합
    # -----------------------------------------------------

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    if not oof[ID_COL].is_unique:
        raise AssertionError(
            "OOF row_id가 중복되었습니다."
        )

    oof.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_metrics = calculate_metrics(
        oof["target"],
        oof["prediction"],
    )

    return (
        fold_results,
        oof,
        oof_metrics,
    )


# =========================================================
# 기존 cb_v002와 직접 비교
# =========================================================

def compare_with_baseline(
    experiment_oof,
):
    """
    cb_v002 OOF와 신규 실험 OOF를
    동일 row_id 기준으로 맞춰 직접 비교한다.

    delta_brier > 0:
        신규 Feature 개선

    delta_brier < 0:
        신규 Feature 악화
    """

    if not BASELINE_OOF_PATH.exists():
        raise FileNotFoundError(
            "cb_v002 OOF가 없습니다:\n"
            f"{BASELINE_OOF_PATH}\n\n"
            "먼저 cb_v002 Expanding Validation 결과가 "
            "정상 저장되어 있는지 확인하세요."
        )

    baseline = pd.read_csv(
        BASELINE_OOF_PATH,
        encoding="utf-8-sig",
    )

    required_cols = {
        ID_COL,
        "target",
        "prediction",
        "validation_season",
    }

    missing_cols = (
        required_cols
        - set(
            baseline.columns
        )
    )

    if missing_cols:
        raise ValueError(
            "cb_v002 OOF 필수 컬럼이 없습니다: "
            f"{sorted(missing_cols)}"
        )

    baseline = (
        baseline[
            [
                ID_COL,
                "target",
                "prediction",
                "validation_season",
            ]
        ]
        .rename(
            columns={
                "target": (
                    "target_baseline"
                ),
                "prediction": (
                    "prediction_baseline"
                ),
                "validation_season": (
                    "validation_season_baseline"
                ),
            }
        )
    )

    experiment = (
        experiment_oof[
            [
                ID_COL,
                "target",
                "prediction",
                "validation_season",
            ]
        ]
        .rename(
            columns={
                "target": (
                    "target_experiment"
                ),
                "prediction": (
                    "prediction_experiment"
                ),
                "validation_season": (
                    "validation_season_experiment"
                ),
            }
        )
    )

    merged = baseline.merge(
        experiment,
        on=ID_COL,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (
        merged["_merge"]
        == "both"
    ).all():
        raise AssertionError(
            "cb_v002와 신규 실험의 "
            "OOF row 구성이 다릅니다."
        )

    merged = merged.drop(
        columns="_merge"
    )

    if not np.array_equal(
        merged[
            "target_baseline"
        ].to_numpy(),
        merged[
            "target_experiment"
        ].to_numpy(),
    ):
        raise AssertionError(
            "cb_v002와 신규 실험의 "
            "target이 일치하지 않습니다."
        )

    if not np.array_equal(
        merged[
            "validation_season_baseline"
        ].to_numpy(),
        merged[
            "validation_season_experiment"
        ].to_numpy(),
    ):
        raise AssertionError(
            "cb_v002와 신규 실험의 "
            "Validation season이 일치하지 않습니다."
        )

    merged["target"] = (
        merged[
            "target_baseline"
        ]
    )

    merged[
        "validation_season"
    ] = (
        merged[
            "validation_season_baseline"
        ]
    )

    # -----------------------------------------------------
    # Row-level squared error
    # -----------------------------------------------------

    merged[
        "baseline_squared_error"
    ] = (
        merged[
            "prediction_baseline"
        ]
        - merged[
            "target"
        ]
    ) ** 2

    merged[
        "experiment_squared_error"
    ] = (
        merged[
            "prediction_experiment"
        ]
        - merged[
            "target"
        ]
    ) ** 2

    # 양수 = 신규 Feature 개선
    merged[
        "delta_squared_error"
    ] = (
        merged[
            "baseline_squared_error"
        ]
        - merged[
            "experiment_squared_error"
        ]
    )

    # -----------------------------------------------------
    # Fold별 비교
    # -----------------------------------------------------

    comparison_rows = []

    for validation_season in [
        2021,
        2022,
        2023,
        2024,
    ]:

        part = merged.loc[
            merged[
                "validation_season"
            ]
            == validation_season
        ]

        baseline_brier = float(
            part[
                "baseline_squared_error"
            ].mean()
        )

        experiment_brier = float(
            part[
                "experiment_squared_error"
            ].mean()
        )

        delta_brier = float(
            baseline_brier
            - experiment_brier
        )

        comparison_rows.append(
            {
                "scope": str(
                    validation_season
                ),
                "n": int(
                    len(part)
                ),
                "baseline_brier": (
                    baseline_brier
                ),
                "experiment_brier": (
                    experiment_brier
                ),
                "delta_brier": (
                    delta_brier
                ),
                "experiment_better": bool(
                    delta_brier > 0
                ),
            }
        )

    # -----------------------------------------------------
    # Pooled OOF
    # -----------------------------------------------------

    baseline_pooled = float(
        merged[
            "baseline_squared_error"
        ].mean()
    )

    experiment_pooled = float(
        merged[
            "experiment_squared_error"
        ].mean()
    )

    pooled_delta = float(
        baseline_pooled
        - experiment_pooled
    )

    comparison_rows.append(
        {
            "scope": "pooled_oof",
            "n": int(
                len(merged)
            ),
            "baseline_brier": (
                baseline_pooled
            ),
            "experiment_brier": (
                experiment_pooled
            ),
            "delta_brier": (
                pooled_delta
            ),
            "experiment_better": bool(
                pooled_delta > 0
            ),
        }
    )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        VALIDATION_DIR
        / "baseline_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    merged.to_csv(
        VALIDATION_DIR
        / "oof_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        comparison,
        merged,
    )


# =========================================================
# 전체 Metadata
# =========================================================

def save_experiment_metadata(
    raw_features,
    feature_qa,
    fold_results,
    oof_metrics,
    comparison,
):
    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "baseline_experiment": (
            BASELINE_EXPERIMENT_NAME
        ),
        "purpose": (
            "Breaking Pitch Mix x Count "
            "Feature Experiment"
        ),
        "raw_feature_count": int(
            len(raw_features)
        ),
        "extra_features": list(
            BREAKING_COUNT_FEATURES
        ),
        "total_feature_count": int(
            len(raw_features)
            + len(
                BREAKING_COUNT_FEATURES
            )
        ),
        "categorical_features": list(
            CAT_COLS
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "feature_qa": (
            feature_qa
        ),
        "folds": (
            fold_results
        ),
        "experiment_pooled_oof": (
            oof_metrics
        ),
        "baseline_comparison": (
            comparison.to_dict(
                orient="records"
            )
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
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            "Validation experiment already exists: "
            f"{VALIDATION_DIR}\n"
            "기존 결과 보호를 위해 "
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if FOLD_MODEL_DIR.exists():
        raise FileExistsError(
            "Model experiment already exists: "
            f"{FOLD_MODEL_DIR}\n"
            "기존 결과 보호를 위해 "
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # Raw Feature
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID_COL not in test_cols:
        raise ValueError(
            f"test.csv에 {ID_COL}가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID_COL
    ]

    if len(raw_features) != 47:
        raise ValueError(
            "Raw Feature 수가 47이 아닙니다: "
            f"{len(raw_features)}"
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID_COL]
            + raw_features
            + [TARGET_COL]
        ),
    )

    print("=" * 74)
    print("데이터 로드 완료")
    print("=" * 74)

    print(
        "Train shape:",
        train.shape,
    )

    print(
        "Raw Features:",
        len(raw_features),
    )

    print(
        "Season:",
        int(
            train["season"].min()
        ),
        "~",
        int(
            train["season"].max()
        ),
    )

    print(
        "Control success rate:",
        f"{train[TARGET_COL].mean():.6f}",
    )

    # -----------------------------------------------------
    # 신규 Feature 생성
    # -----------------------------------------------------

    before = train.copy()

    train = (
        add_breaking_count_features(
            train
        )
    )

    feature_qa = (
        validate_breaking_features(
            before_df=before,
            after_df=train,
            raw_features=raw_features,
        )
    )

    del before
    gc.collect()

    print()
    print("=" * 74)
    print("Feature QA")
    print("=" * 74)

    print(
        "Extra Features:",
        BREAKING_COUNT_FEATURES,
    )

    print(
        "Total Features:",
        len(raw_features)
        + len(
            BREAKING_COUNT_FEATURES
        ),
    )

    print(
        "NaN:",
        feature_qa[
            "extra_feature_nan_counts"
        ],
    )

    print(
        "Inf:",
        feature_qa[
            "extra_feature_inf_counts"
        ],
    )

    # -----------------------------------------------------
    # 디렉터리
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FOLD_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # 현재 실행 코드 복사
    shutil.copy2(
        Path(__file__).resolve(),
        FOLD_MODEL_DIR
        / "train.py",
    )

    # -----------------------------------------------------
    # Expanding Validation
    # -----------------------------------------------------

    (
        fold_results,
        oof,
        oof_metrics,
    ) = run_expanding_validation(
        train=train,
        raw_features=raw_features,
    )

    # -----------------------------------------------------
    # 기존 cb_v002와 비교
    # -----------------------------------------------------

    (
        comparison,
        merged,
    ) = compare_with_baseline(
        experiment_oof=oof,
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    save_experiment_metadata(
        raw_features=raw_features,
        feature_qa=feature_qa,
        fold_results=fold_results,
        oof_metrics=oof_metrics,
        comparison=comparison,
    )

    # -----------------------------------------------------
    # 결과 출력
    # -----------------------------------------------------

    print()
    print("=" * 86)
    print(
        "cb_v002 vs Breaking × Count"
    )
    print("=" * 86)

    for _, row in comparison.iterrows():

        direction = (
            "개선"
            if row[
                "delta_brier"
            ] > 0
            else "악화"
        )

        print(
            f"{row['scope']:>10} | "
            f"Basic="
            f"{row['baseline_brier']:.8f} | "
            f"EXP="
            f"{row['experiment_brier']:.8f} | "
            f"Delta="
            f"{row['delta_brier']:+.8f} | "
            f"{direction}"
        )

    print("=" * 86)

    print(
        "Validation 저장:",
        VALIDATION_DIR,
    )

    print(
        "Fold Model 저장:",
        FOLD_MODEL_DIR,
    )

    print()
    print(
        "※ Delta = cb_v002 Brier - "
        "Breaking Experiment Brier"
    )

    print(
        "※ Delta > 0 : Breaking Feature 개선"
    )

    print(
        "※ Delta < 0 : Breaking Feature 악화"
    )


if __name__ == "__main__":
    main()