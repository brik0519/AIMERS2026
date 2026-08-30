# train_pitchmix_both.py
"""
LG Aimers Phase2
cb_v002 - Breaking + Offspeed × Count Feature Experiment

목적
----
현재 기준 모델 cb_v002 (Raw 47)의 모든 학습 조건을 유지하고,
Pitch Mix × Count interaction 4개만 추가한다.

비교
----
Baseline:
    cb_v002
    Raw Feature 47개

Experiment:
    Raw Feature 47개
    + breaking_x_balls
    + breaking_x_strikes
    + offspeed_x_balls
    + offspeed_x_strikes

판단
----
Delta Brier = cb_v002 Brier - Experiment Brier

Delta > 0 : 신규 Feature 개선
Delta < 0 : 신규 Feature 악화

주의
----
- Feature만 변경한다.
- 현재 투구의 실제 구종은 사용하지 않는다.
- 공식 asof_*와 동일 row의 count 정보만 이용한다.
- test 내부 다른 row를 이용한 집계/rolling/lag 등은 없다.
- Feature screening 단계이므로 최종 2019~2024 재학습은 하지 않는다.
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

EXPERIMENT_NAME = "cb_v002_pitchmix_both_v001"

DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기준 모델
# =========================================================

BASELINE_EXPERIMENT_NAME = "cb_v002"

BASELINE_OOF_PATH = (
    VALIDATION_ROOT
    / BASELINE_EXPERIMENT_NAME
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
# 신규 Feature 4개
# =========================================================

PITCHMIX_COUNT_FEATURES = [
    "breaking_x_balls",
    "breaking_x_strikes",
    "offspeed_x_balls",
    "offspeed_x_strikes",
]


# =========================================================
# Feature 생성
# =========================================================

def add_pitchmix_count_features(df):
    """
    Pitch Mix × Count interaction 4개 생성.

    breaking_x_balls
        = asof_pitcher_breaking_rate * balls_before

    breaking_x_strikes
        = asof_pitcher_breaking_rate * strikes_before

    offspeed_x_balls
        = asof_pitcher_offspeed_rate * balls_before

    offspeed_x_strikes
        = asof_pitcher_offspeed_rate * strikes_before
    """

    df = df.copy()

    required_cols = [
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
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
            "Pitch Mix × Count Feature 생성에 "
            "필요한 컬럼이 없습니다: "
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

    df["offspeed_x_balls"] = (
        df["asof_pitcher_offspeed_rate"]
        * df["balls_before"]
    )

    df["offspeed_x_strikes"] = (
        df["asof_pitcher_offspeed_rate"]
        * df["strikes_before"]
    )

    return df


# =========================================================
# Feature QA
# =========================================================

def validate_pitchmix_features(
    before_df,
    after_df,
    raw_features,
):
    """
    신규 Feature 생성 QA.

    - row 수 보존
    - row_id 순서 보존
    - row_id unique
    - target 결측 없음
    - Raw Feature 47개
    - Feature 수식 검증
    - inf/-inf 없음
    - NaN 개수 기록
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

    expected_map = {
        "breaking_x_balls": (
            after_df["asof_pitcher_breaking_rate"]
            * after_df["balls_before"]
        ),
        "breaking_x_strikes": (
            after_df["asof_pitcher_breaking_rate"]
            * after_df["strikes_before"]
        ),
        "offspeed_x_balls": (
            after_df["asof_pitcher_offspeed_rate"]
            * after_df["balls_before"]
        ),
        "offspeed_x_strikes": (
            after_df["asof_pitcher_offspeed_rate"]
            * after_df["strikes_before"]
        ),
    }

    for feature, expected in expected_map.items():

        actual = after_df[feature]

        if not np.allclose(
            expected.to_numpy(
                dtype=np.float64
            ),
            actual.to_numpy(
                dtype=np.float64
            ),
            equal_nan=True,
        ):
            raise AssertionError(
                f"Feature 수식 불일치: {feature}"
            )

    values = (
        after_df[PITCHMIX_COUNT_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    inf_counts = dict(
        zip(
            PITCHMIX_COUNT_FEATURES,
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
        for feature in PITCHMIX_COUNT_FEATURES
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
            len(PITCHMIX_COUNT_FEATURES)
        ),
        "total_feature_count": int(
            len(raw_features)
            + len(PITCHMIX_COUNT_FEATURES)
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

    - 수치형 NaN은 CatBoost가 처리
    - 범주형 NaN만 "__MISSING__"
    - 범주형은 str
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

    if baseline_brier <= 0.0:
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
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
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

    features = (
        list(raw_features)
        + PITCHMIX_COUNT_FEATURES
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    fold_results = []
    oof_frames = []

    experiment_model_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
    )

    for fold_config in EXPANDING_FOLDS:

        fold = int(
            fold_config["fold"]
        )

        train_seasons = list(
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
                "Expanding Window 시간 순서가 "
                "잘못되었습니다."
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
                f"Fold {fold}: "
                "Train 데이터가 없습니다."
            )

        if not val_mask.any():
            raise ValueError(
                f"Fold {fold}: "
                "Validation 데이터가 없습니다."
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
        # Data
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
        # Model
        # -------------------------------------------------

        estimator = CatBoostClassifier(
            **MODEL_PARAMS
        )

        print()
        print("=" * 80)

        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]}"
            f" -> {validation_season}"
        )

        print("=" * 80)

        print(
            "Train:",
            X_train.shape,
        )

        print(
            "Valid:",
            X_val.shape,
        )

        print(
            "Raw Features:",
            len(raw_features),
        )

        print(
            "Extra Features:",
            PITCHMIX_COUNT_FEATURES,
        )

        print(
            "Total Features:",
            len(features),
        )

        # -------------------------------------------------
        # Fit
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
        # Prediction
        # -------------------------------------------------

        val_pred = (
            estimator.predict_proba(
                X_val
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_val,
            val_pred,
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
            experiment_model_dir
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
        # Fold Validation 저장
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
        # Metadata
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
                len(PITCHMIX_COUNT_FEATURES)
            ),
            "total_feature_count": int(
                len(features)
            ),
            "extra_features": list(
                PITCHMIX_COUNT_FEATURES
            ),
            "categorical_features": list(
                cat_cols
            ),
            "model_params": (
                MODEL_PARAMS
            ),
            "early_stopping_rounds": int(
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
# cb_v002와 비교
# =========================================================

def compare_with_baseline(
    experiment_oof,
):

    if not BASELINE_OOF_PATH.exists():
        raise FileNotFoundError(
            "cb_v002 OOF가 없습니다: "
            f"{BASELINE_OOF_PATH}"
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

    missing = (
        required_cols
        - set(baseline.columns)
    )

    if missing:
        raise ValueError(
            "cb_v002 OOF 필수 컬럼 누락: "
            f"{sorted(missing)}"
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

    # -----------------------------------------------------
    # OOF QA
    # -----------------------------------------------------

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
            "Validation season이 다릅니다."
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
    # Row-level Brier
    # -----------------------------------------------------

    merged[
        "baseline_squared_error"
    ] = (
        merged[
            "prediction_baseline"
        ]
        - merged["target"]
    ) ** 2

    merged[
        "experiment_squared_error"
    ] = (
        merged[
            "prediction_experiment"
        ]
        - merged["target"]
    ) ** 2

    # 양수 = 신규 실험 개선
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

    comparison_rows = []

    # -----------------------------------------------------
    # Fold별
    # -----------------------------------------------------

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

    return comparison


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
            "Breaking + Offspeed "
            "Pitch Mix x Count Experiment"
        ),
        "raw_feature_count": int(
            len(raw_features)
        ),
        "extra_features": list(
            PITCHMIX_COUNT_FEATURES
        ),
        "total_feature_count": int(
            len(raw_features)
            + len(PITCHMIX_COUNT_FEATURES)
        ),
        "categorical_features": list(
            CAT_COLS
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "early_stopping_rounds": int(
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

    experiment_model_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
    )

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            "Validation experiment already exists: "
            f"{VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if experiment_model_dir.exists():
        raise FileExistsError(
            "Model experiment already exists: "
            f"{experiment_model_dir}\n"
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

    print("=" * 80)
    print("데이터 로드 완료")
    print("=" * 80)

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
    # Feature 생성
    # -----------------------------------------------------

    before = train.copy()

    train = (
        add_pitchmix_count_features(
            train
        )
    )

    feature_qa = (
        validate_pitchmix_features(
            before_df=before,
            after_df=train,
            raw_features=raw_features,
        )
    )

    del before
    gc.collect()

    print()
    print("=" * 80)
    print("Feature QA")
    print("=" * 80)

    print(
        "Extra Features:",
        PITCHMIX_COUNT_FEATURES,
    )

    print(
        "Total Features:",
        len(raw_features)
        + len(
            PITCHMIX_COUNT_FEATURES
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
        experiment_model_dir
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
    # cb_v002 비교
    # -----------------------------------------------------

    comparison = compare_with_baseline(
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
    # 출력
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "cb_v002 Raw47 vs "
        "Breaking + Offspeed × Count"
    )
    print("=" * 100)

    for _, row in comparison.iterrows():

        if row["delta_brier"] > 0:
            direction = "개선"

        elif row["delta_brier"] < 0:
            direction = "악화"

        else:
            direction = "동일"

        print(
            f"{row['scope']:>10} | "
            f"Basic="
            f"{row['baseline_brier']:.8f} | "
            f"Both="
            f"{row['experiment_brier']:.8f} | "
            f"Delta="
            f"{row['delta_brier']:+.8f} | "
            f"{direction}"
        )

    print("=" * 100)

    print(
        "Validation 저장:",
        VALIDATION_DIR,
    )

    print(
        "Fold Model 저장:",
        experiment_model_dir,
    )

    print()

    print(
        "Delta = cb_v002 Brier "
        "- Both Brier"
    )

    print(
        "Delta > 0 : 신규 Feature 개선"
    )

    print(
        "Delta < 0 : 신규 Feature 악화"
    )


if __name__ == "__main__":
    main()