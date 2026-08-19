# train_basic_vs_middle5_multiseed.py
"""
LG Aimers Phase2 - Basic vs Middle Temporal 5 Multi-Seed 안정성 검증

목적
----
현재 Expanding Window OOF에서 Middle Temporal 5개가 Basic보다
Brier를 약 0.00002052 개선했다.

하지만 차이가 매우 작으므로, 이 개선이 특정 random_seed에만 나타난
우연한 결과인지 확인한다.

실험
----
각 seed마다 동일한 4-Fold Expanding Window에서 아래 두 모델을 비교한다.

1) Basic
   - Raw Feature 47개

2) Middle 5
   - Raw Feature 47개
   - recent1_middle_vs_career
   - recent3_middle_vs_career
   - recent5_middle_vs_career
   - middle_trend_1_3
   - middle_trend_3_5

고정 조건
---------
- CatBoost hyperparameter 동일
- Logloss 학습 동일
- Early Stopping 동일
- max_ctr_complexity=1 동일
- categorical 처리 동일
- Expanding Window Fold 동일
- Feature Set 차이는 Basic vs Middle5뿐
- seed만 변경

주의
----
이 실험은 통계적 유의성 검정 자체가 아니다.
목적은 "Middle 5의 작은 개선 방향이 seed가 달라도 반복되는가?"를 확인하는 것이다.

Temporal Feature는 동일 row의 공식 asof_* 값만 사용한다.
test 내부 다른 row를 이용하는 groupby / rolling / lag / 집계는 사용하지 않는다.
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

EXPERIMENT_NAME = "basic_vs_middle5_multiseed_v001"

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
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

# 현재 Expanding 코드와 동일한 범주형 컬럼
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

# 5개 seed로 방향 안정성 확인
SEEDS = [42, 7, 2026, 123, 999]

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

EARLY_STOPPING_ROUNDS = 80


# =========================================================
# Feature 정의
# =========================================================

MIDDLE_TEMPORAL_FEATURES = [
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
    "middle_trend_1_3",
    "middle_trend_3_5",
]


# =========================================================
# Temporal Feature 생성
# =========================================================

def add_middle_temporal_features(df):
    """
    Middle Temporal 5개를 생성한다.

    변수명 *_vs_career는 기존 코드 호환을 위해 유지하지만,
    실제 의미는 Recent vs 해당 시점까지 Cumulative(as-of)이다.

    모든 계산은 동일 row 내부의 공식 asof_* 값끼리만 수행한다.
    """
    df = df.copy()

    required_cols = [
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Middle Temporal 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing_cols)
        )

    # Recent vs Cumulative
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

    # Recent Trend
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

def validate_middle_temporal_features(
    before_df,
    after_df,
    raw_features,
):
    """
    Feature 생성 QA
    - row 수 보존
    - row_id 순서 보존 / unique
    - target 결측 없음
    - Raw Feature 47개 확인
    - Middle Temporal 5개 수식 검증
    - inf/-inf 없음
    - NaN 개수 기록
    """
    assert len(before_df) == len(after_df), (
        "Feature 생성 후 row 수가 변했습니다."
    )

    assert before_df[ID_COL].equals(after_df[ID_COL]), (
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
            f"Raw Feature 수가 예상(47)과 다릅니다: {len(raw_features)}"
        )

    expected_map = {
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
        actual = after_df[feature]

        if not np.allclose(
            expected.to_numpy(dtype=np.float64),
            actual.to_numpy(dtype=np.float64),
            equal_nan=True,
        ):
            raise AssertionError(
                f"Middle Temporal 수식 불일치: {feature}"
            )

    middle_array = (
        after_df[MIDDLE_TEMPORAL_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    inf_counts = dict(
        zip(
            MIDDLE_TEMPORAL_FEATURES,
            np.isinf(middle_array)
            .sum(axis=0)
            .astype(int)
            .tolist(),
        )
    )

    if any(count > 0 for count in inf_counts.values()):
        raise ValueError(
            f"Middle Temporal에 inf/-inf가 있습니다: {inf_counts}"
        )

    nan_counts = {
        feature: int(after_df[feature].isna().sum())
        for feature in MIDDLE_TEMPORAL_FEATURES
    }

    return {
        "n_rows_before": int(len(before_df)),
        "n_rows_after": int(len(after_df)),
        "row_id_unique": bool(after_df[ID_COL].is_unique),
        "raw_feature_count": int(len(raw_features)),
        "middle_temporal_feature_count": int(
            len(MIDDLE_TEMPORAL_FEATURES)
        ),
        "middle_temporal_nan_counts": nan_counts,
        "middle_temporal_inf_counts": inf_counts,
    }


# =========================================================
# CatBoost 전처리 / Metric
# =========================================================

def prepare_features(df, cat_cols):
    """
    기존 Expanding 실험과 동일하게
    범주형 결측값만 문자열로 처리한다.
    """
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


def calculate_brier(y_true, y_pred):
    """Brier Score 계산."""
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


def calculate_metrics(y_true, y_pred):
    """Brier / baseline Brier / BSS 및 prediction 통계."""
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

    brier = calculate_brier(
        y_true,
        y_pred,
    )

    baseline_brier = float(
        positive_rate
        * (1.0 - positive_rate)
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
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": bss,
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


# =========================================================
# 모델 파라미터
# =========================================================

def get_model_params(seed):
    """
    기존 모델 설정은 유지하고 random_seed만 바꾼다.
    """
    return {
        "iterations": 800,
        "learning_rate": 0.07,
        "depth": 8,
        "loss_function": "Logloss",
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
        "random_seed": int(seed),
        "max_ctr_complexity": 1,
        "thread_count": -1,
        "verbose": 100,
        "allow_writing_files": False,
    }


# =========================================================
# 단일 모델(Basic 또는 Middle5) 4-Fold 실행
# =========================================================

def run_one_model(
    train,
    raw_features,
    feature_key,
    extra_features,
    seed,
):
    """
    하나의 seed에서 Basic 또는 Middle5 모델을
    4-Fold Expanding Window로 학습한다.
    """
    features = (
        list(raw_features)
        + list(extra_features)
    )

    cat_cols = [
        col for col in CAT_COLS
        if col in features
    ]

    model_params = get_model_params(
        seed
    )

    model_run_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
        / f"seed_{seed}"
        / feature_key
    )

    validation_run_dir = (
        VALIDATION_DIR
        / f"seed_{seed}"
        / feature_key
    )

    model_run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    validation_run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    fold_rows = []
    oof_frames = []

    print()
    print("=" * 86)
    print(
        f"seed={seed} | {feature_key} | "
        f"Feature={len(features)}"
    )
    print("=" * 86)

    for fold_config in FOLDS:
        fold_number = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        valid_season = fold_config["valid_season"]

        # -------------------------------------------------
        # 시간축 QA
        # -------------------------------------------------

        assert max(train_seasons) < valid_season

        is_train = train["season"].isin(
            train_seasons
        )

        is_valid = (
            train["season"]
            == valid_season
        )

        train_row_ids = train.loc[
            is_train,
            ID_COL,
        ]

        valid_row_ids = train.loc[
            is_valid,
            ID_COL,
        ]

        if not set(train_row_ids).isdisjoint(
            set(valid_row_ids)
        ):
            raise AssertionError(
                f"seed={seed}, {feature_key}, "
                f"fold={fold_number}: "
                "Train/Validation row_id가 겹칩니다."
            )

        X_train = prepare_features(
            train.loc[
                is_train,
                features,
            ],
            cat_cols,
        )

        y_train = train.loc[
            is_train,
            TARGET_COL,
        ]

        X_valid = prepare_features(
            train.loc[
                is_valid,
                features,
            ],
            cat_cols,
        )

        y_valid = train.loc[
            is_valid,
            TARGET_COL,
        ]

        estimator = CatBoostClassifier(
            **model_params
        )

        print()
        print("-" * 86)
        print(
            f"{feature_key} | seed={seed} | "
            f"{train_seasons[0]}~{train_seasons[-1]}"
            f" -> {valid_season}"
        )
        print(
            "Train:",
            X_train.shape,
            "| Valid:",
            X_valid.shape,
        )
        print("-" * 86)

        start_time = time.time()

        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )

        elapsed_seconds = float(
            time.time() - start_time
        )

        valid_pred = (
            estimator.predict_proba(
                X_valid
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_valid.to_numpy(),
            valid_pred,
        )

        best_iteration = (
            estimator.get_best_iteration()
        )

        if (
            best_iteration is None
            or best_iteration < 0
        ):
            final_iterations = int(
                model_params["iterations"]
            )
        else:
            final_iterations = int(
                best_iteration + 1
            )

        fold_model_path = (
            model_run_dir
            / f"fold_{valid_season}.pkl"
        )

        joblib.dump(
            estimator,
            fold_model_path,
            compress=3,
        )

        fold_pred = pd.DataFrame(
            {
                ID_COL: valid_row_ids.to_numpy(),
                "season": int(valid_season),
                "target": y_valid.to_numpy(),
                "prediction": valid_pred,
                "fold": int(fold_number),
            }
        )

        fold_pred.to_csv(
            validation_run_dir
            / f"fold_{valid_season}_pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fold_row = {
            "seed": int(seed),
            "feature_set": feature_key,
            "fold": int(fold_number),
            "valid_season": int(valid_season),
            "n_train": int(is_train.sum()),
            "n_valid": int(is_valid.sum()),
            "total_feature_count": int(len(features)),
            "temporal_feature_count": int(
                len(extra_features)
            ),
            "best_iteration": int(
                final_iterations
            ),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            **metrics,
        }

        fold_rows.append(
            fold_row
        )

        oof_frames.append(
            fold_pred
        )

        print(
            f"Brier={metrics['brier']:.8f} | "
            f"BSS={metrics['bss']:.2f} | "
            f"iter={final_iterations} | "
            f"time={elapsed_seconds:.1f}s"
        )

        del (
            X_train,
            X_valid,
            y_train,
            y_valid,
            estimator,
            valid_pred,
            fold_pred,
        )

        gc.collect()

    oof = pd.concat(
        oof_frames,
        axis=0,
        ignore_index=True,
    )

    if not oof[ID_COL].is_unique:
        raise AssertionError(
            f"seed={seed}, {feature_key}: "
            "OOF row_id가 unique하지 않습니다."
        )

    oof.to_csv(
        validation_run_dir
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_brier = calculate_brier(
        oof["target"].to_numpy(),
        oof["prediction"].to_numpy(),
    )

    print(
        f"[완료] seed={seed} | {feature_key} | "
        f"pooled OOF Brier={pooled_brier:.8f}"
    )

    return (
        pd.DataFrame(fold_rows),
        oof,
        float(pooled_brier),
    )


# =========================================================
# Seed별 Basic vs Middle5 비교
# =========================================================

def compare_seed(
    seed,
    basic_fold_df,
    middle_fold_df,
    basic_oof,
    middle_oof,
    basic_pooled,
    middle_pooled,
):
    """
    동일 seed의 Basic과 Middle5를 직접 비교한다.
    delta = Basic Brier - Middle5 Brier

    delta > 0 : Middle5 개선
    delta < 0 : Middle5 악화
    """
    basic_key = (
        basic_fold_df[
            [
                "valid_season",
                "brier",
            ]
        ]
        .rename(
            columns={
                "brier": "basic_brier"
            }
        )
    )

    middle_key = (
        middle_fold_df[
            [
                "valid_season",
                "brier",
            ]
        ]
        .rename(
            columns={
                "brier": "middle5_brier"
            }
        )
    )

    fold_compare = basic_key.merge(
        middle_key,
        on="valid_season",
        how="inner",
        validate="one_to_one",
    )

    fold_compare.insert(
        0,
        "seed",
        int(seed),
    )

    fold_compare[
        "delta_brier_basic_minus_middle5"
    ] = (
        fold_compare["basic_brier"]
        - fold_compare["middle5_brier"]
    )

    fold_compare[
        "middle5_better"
    ] = (
        fold_compare[
            "delta_brier_basic_minus_middle5"
        ]
        > 0
    )

    # OOF row가 정확히 같은지 검증
    basic_check = (
        basic_oof[
            [
                ID_COL,
                "season",
                "target",
            ]
        ]
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )

    middle_check = (
        middle_oof[
            [
                ID_COL,
                "season",
                "target",
            ]
        ]
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )

    if not basic_check.equals(
        middle_check
    ):
        raise AssertionError(
            f"seed={seed}: "
            "Basic과 Middle5 OOF row/target 구성이 다릅니다."
        )

    seed_summary = {
        "seed": int(seed),
        "basic_pooled_brier": float(
            basic_pooled
        ),
        "middle5_pooled_brier": float(
            middle_pooled
        ),
        "delta_brier_basic_minus_middle5": float(
            basic_pooled
            - middle_pooled
        ),
        "middle5_better": bool(
            basic_pooled
            - middle_pooled
            > 0
        ),
    }

    return (
        fold_compare,
        seed_summary,
    )


# =========================================================
# 전체 안정성 요약
# =========================================================

def build_stability_summary(
    seed_summary_df,
    fold_summary_df,
):
    """
    seed별 결과를 평균/표준편차/승리 횟수로 요약한다.

    이것은 공식적인 통계적 유의성 검정이 아니라
    random_seed에 대한 방향 안정성 확인이다.
    """
    deltas = (
        seed_summary_df[
            "delta_brier_basic_minus_middle5"
        ]
        .to_numpy(dtype=np.float64)
    )

    stability = {
        "n_seeds": int(
            len(seed_summary_df)
        ),
        "middle5_win_count": int(
            seed_summary_df[
                "middle5_better"
            ].sum()
        ),
        "middle5_loss_count": int(
            (
                ~seed_summary_df[
                    "middle5_better"
                ]
            ).sum()
        ),
        "mean_basic_pooled_brier": float(
            seed_summary_df[
                "basic_pooled_brier"
            ].mean()
        ),
        "mean_middle5_pooled_brier": float(
            seed_summary_df[
                "middle5_pooled_brier"
            ].mean()
        ),
        "mean_delta_basic_minus_middle5": float(
            deltas.mean()
        ),
        "std_delta_across_seeds": float(
            deltas.std(ddof=0)
        ),
        "min_delta_across_seeds": float(
            deltas.min()
        ),
        "max_delta_across_seeds": float(
            deltas.max()
        ),
        "median_delta_across_seeds": float(
            np.median(deltas)
        ),
    }

    # 시즌별로 seed 평균 delta도 계산
    season_summary = (
        fold_summary_df
        .groupby(
            "valid_season",
            as_index=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            mean_basic_brier=(
                "basic_brier",
                "mean",
            ),
            mean_middle5_brier=(
                "middle5_brier",
                "mean",
            ),
            mean_delta=(
                "delta_brier_basic_minus_middle5",
                "mean",
            ),
            std_delta=(
                "delta_brier_basic_minus_middle5",
                "std",
            ),
            middle5_win_count=(
                "middle5_better",
                "sum",
            ),
        )
    )

    return (
        stability,
        season_summary,
    )


# =========================================================
# Main
# =========================================================

def main():
    # -----------------------------------------------------
    # 출력 폴더 중복 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 Validation experiment가 있습니다: {VALIDATION_DIR}\n"
            "기존 결과 보호를 위해 자동 덮어쓰지 않습니다."
        )

    model_experiment_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
    )

    if model_experiment_dir.exists():
        raise FileExistsError(
            f"기존 Model experiment가 있습니다: {model_experiment_dir}\n"
            "기존 결과 보호를 위해 자동 덮어쓰지 않습니다."
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        VALIDATION_DIR
        / "train_basic_vs_middle5_multiseed.py",
    )

    # -----------------------------------------------------
    # Raw Feature 구성
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID_COL not in test_cols:
        raise ValueError(
            f"test.csv에 '{ID_COL}' 컬럼이 없습니다."
        )

    raw_features = [
        col for col in test_cols
        if col != ID_COL
    ]

    # -----------------------------------------------------
    # Train 로드 + Middle Temporal 생성 + QA
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

    if "season" not in train.columns:
        raise ValueError(
            "train.csv에 season 컬럼이 없습니다."
        )

    before = train.copy()

    train = add_middle_temporal_features(
        train
    )

    feature_qa = (
        validate_middle_temporal_features(
            before_df=before,
            after_df=train,
            raw_features=raw_features,
        )
    )

    del before
    gc.collect()

    print("=" * 86)
    print("Basic vs Middle5 Multi-Seed Stability")
    print("=" * 86)
    print("Seeds              :", SEEDS)
    print("Raw Feature        :", len(raw_features))
    print(
        "Middle Temporal    :",
        len(MIDDLE_TEMPORAL_FEATURES),
    )
    print(
        "Basic Feature      :",
        len(raw_features),
    )
    print(
        "Middle5 Feature    :",
        len(raw_features)
        + len(MIDDLE_TEMPORAL_FEATURES),
    )
    print(
        "Temporal NaN counts:",
        feature_qa[
            "middle_temporal_nan_counts"
        ],
    )
    print("=" * 86)

    all_model_fold_rows = []
    all_fold_compare_rows = []
    seed_summary_rows = []

    # -----------------------------------------------------
    # Seed 반복
    # -----------------------------------------------------

    for seed in SEEDS:
        # Basic
        (
            basic_fold_df,
            basic_oof,
            basic_pooled,
        ) = run_one_model(
            train=train,
            raw_features=raw_features,
            feature_key="basic",
            extra_features=[],
            seed=seed,
        )

        # Middle 5
        (
            middle_fold_df,
            middle_oof,
            middle_pooled,
        ) = run_one_model(
            train=train,
            raw_features=raw_features,
            feature_key="middle5",
            extra_features=MIDDLE_TEMPORAL_FEATURES,
            seed=seed,
        )

        basic_fold_df[
            "model_group"
        ] = "basic"

        middle_fold_df[
            "model_group"
        ] = "middle5"

        all_model_fold_rows.extend(
            [
                basic_fold_df,
                middle_fold_df,
            ]
        )

        (
            fold_compare,
            seed_summary,
        ) = compare_seed(
            seed=seed,
            basic_fold_df=basic_fold_df,
            middle_fold_df=middle_fold_df,
            basic_oof=basic_oof,
            middle_oof=middle_oof,
            basic_pooled=basic_pooled,
            middle_pooled=middle_pooled,
        )

        all_fold_compare_rows.append(
            fold_compare
        )

        seed_summary_rows.append(
            seed_summary
        )

        print()
        print("#" * 86)
        print(
            f"SEED {seed} 비교"
        )
        print("#" * 86)
        print(
            f"Basic   : {basic_pooled:.8f}"
        )
        print(
            f"Middle5 : {middle_pooled:.8f}"
        )
        print(
            "Delta    : "
            f"{basic_pooled - middle_pooled:+.8f} "
            "(+이면 Middle5 개선)"
        )
        print("#" * 86)

    # -----------------------------------------------------
    # 전체 결과 저장
    # -----------------------------------------------------

    model_fold_df = pd.concat(
        all_model_fold_rows,
        axis=0,
        ignore_index=True,
    )

    fold_summary_df = pd.concat(
        all_fold_compare_rows,
        axis=0,
        ignore_index=True,
    )

    seed_summary_df = pd.DataFrame(
        seed_summary_rows
    )

    (
        stability,
        season_summary_df,
    ) = build_stability_summary(
        seed_summary_df=seed_summary_df,
        fold_summary_df=fold_summary_df,
    )

    model_fold_df.to_csv(
        VALIDATION_DIR
        / "all_model_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_summary_df.to_csv(
        VALIDATION_DIR
        / "fold_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary_df.to_csv(
        VALIDATION_DIR
        / "seed_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    season_summary_df.to_csv(
        VALIDATION_DIR
        / "season_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "purpose": (
            "Basic vs Middle Temporal 5의 작은 Brier 개선이 "
            "random_seed 변화에서도 반복되는지 확인"
        ),
        "seeds": [
            int(seed)
            for seed in SEEDS
        ],
        "folds": FOLDS,
        "middle_temporal_features": (
            MIDDLE_TEMPORAL_FEATURES
        ),
        "early_stopping_rounds": int(
            EARLY_STOPPING_ROUNDS
        ),
        "fixed_model_params_except_seed": {
            key: value
            for key, value
            in get_model_params(SEEDS[0]).items()
            if key != "random_seed"
        },
        "delta_definition": (
            "Basic Brier - Middle5 Brier; "
            "positive means Middle5 is better"
        ),
        "feature_qa": feature_qa,
        "stability_summary": stability,
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

    # -----------------------------------------------------
    # 콘솔 최종 출력
    # -----------------------------------------------------

    print()
    print("=" * 86)
    print("SEED별 최종 비교")
    print("=" * 86)

    display_seed = (
        seed_summary_df.copy()
    )

    for col in [
        "basic_pooled_brier",
        "middle5_pooled_brier",
        "delta_brier_basic_minus_middle5",
    ]:
        display_seed[col] = (
            display_seed[col]
            .map(
                lambda value: f"{value:.8f}"
            )
        )

    print(
        display_seed.to_string(
            index=False
        )
    )

    print()
    print("=" * 86)
    print("안정성 요약")
    print("=" * 86)
    print(
        "Middle5 승리 seed :",
        f"{stability['middle5_win_count']}"
        f"/{stability['n_seeds']}",
    )
    print(
        "Basic 평균 Brier  :",
        f"{stability['mean_basic_pooled_brier']:.8f}",
    )
    print(
        "Middle5 평균 Brier:",
        f"{stability['mean_middle5_pooled_brier']:.8f}",
    )
    print(
        "평균 Delta        :",
        f"{stability['mean_delta_basic_minus_middle5']:+.8f}",
    )
    print(
        "Seed간 Delta Std  :",
        f"{stability['std_delta_across_seeds']:.8f}",
    )
    print(
        "최소 Delta        :",
        f"{stability['min_delta_across_seeds']:+.8f}",
    )
    print(
        "최대 Delta        :",
        f"{stability['max_delta_across_seeds']:+.8f}",
    )

    print()
    print("=" * 86)
    print("시즌별 Seed 안정성")
    print("=" * 86)

    display_season = (
        season_summary_df.copy()
    )

    for col in [
        "mean_basic_brier",
        "mean_middle5_brier",
        "mean_delta",
        "std_delta",
    ]:
        display_season[col] = (
            display_season[col]
            .map(
                lambda value: f"{value:.8f}"
            )
        )

    print(
        display_season.to_string(
            index=False
        )
    )

    print()
    print(
        "해석 기준: delta > 0이면 해당 비교에서 Middle5가 Basic보다 좋습니다."
    )
    print(
        "이 결과는 formal significance test가 아니라 random_seed robustness 확인입니다."
    )
    print(
        "결과 저장:",
        VALIDATION_DIR,
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
