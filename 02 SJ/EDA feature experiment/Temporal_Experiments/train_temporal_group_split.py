# train_temporal_group_split.py
"""
LG Aimers Phase2 - Temporal Feature Group Split Experiment

목적
----
기존 EXP3 Temporal 10개를
1) Success Temporal 5개
2) Middle Temporal 5개
로 분리하여, 어느 계열이 실제 CatBoost Expanding Window Brier에
도움이 되는지 확인한다.

비교 대상
---------
- Basic       : Raw 47
- All Temporal: Raw 47 + Temporal 10
- Success 5   : Raw 47 + Success Temporal 5
- Middle 5    : Raw 47 + Middle Temporal 5

중요
----
- Basic / All Temporal은 이미 생성된 OOF 결과를 재사용한다.
- 이 스크립트는 Success 5 / Middle 5만 새로 학습한다.
- CatBoost 설정, seed, categorical 처리, Early Stopping,
  Expanding Window Fold는 기존 cb_v001_ctr1_expanding과 동일하게 유지한다.
- Feature Set만 변경한다.
- Temporal Feature는 동일 row의 공식 asof_* 값끼리만 계산한다.
- test 내부 다른 row를 이용한 groupby / rolling / lag / 집계를 수행하지 않는다.
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

EXPERIMENT_NAME = "temporal_group_split"

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
# CatBoost 설정
# 기존 Basic / EXP3 Expanding과 동일
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
# Temporal Feature 정의
# =========================================================

SUCCESS_TEMPORAL_FEATURES = [
    # Recent vs Cumulative - Success
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",

    # Success Trend
    "success_trend_1_3",
    "success_trend_3_5",
]

MIDDLE_TEMPORAL_FEATURES = [
    # Recent vs Cumulative - Middle
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",

    # Middle Trend
    "middle_trend_1_3",
    "middle_trend_3_5",
]

ALL_TEMPORAL_FEATURES = (
    SUCCESS_TEMPORAL_FEATURES
    + MIDDLE_TEMPORAL_FEATURES
)


# 기존 결과와 비교할 Experiment 이름
EXPERIMENTS = {
    "basic": {
        "experiment_name": "cb_v001_ctr1_expanding",
        "feature_group": [],
        "train": False,
    },
    "all_temporal_10": {
        "experiment_name": "cb_v001_ctr1_expanding_exp3",
        "feature_group": ALL_TEMPORAL_FEATURES,
        "train": False,
    },
    "success_5": {
        "experiment_name": "cb_v001_ctr1_expanding_success5",
        "feature_group": SUCCESS_TEMPORAL_FEATURES,
        "train": True,
    },
    "middle_5": {
        "experiment_name": "cb_v001_ctr1_expanding_middle5",
        "feature_group": MIDDLE_TEMPORAL_FEATURES,
        "train": True,
    },
}


# =========================================================
# Temporal Feature 생성
# =========================================================

def add_temporal_features(df):
    """
    기존 EXP3과 동일한 Temporal Feature 10개를 생성한다.

    모든 계산은 동일 row에 이미 존재하는 공식 asof_* 값끼리 수행한다.
    변수명은 기존 코드 호환을 위해 *_vs_career를 유지하지만,
    실제 의미는 '해당 시점까지 cumulative(as-of)'이다.
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

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing_cols)
        )

    # -----------------------------------------------------
    # Success: Recent vs Cumulative
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Middle: Recent vs Cumulative
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Success Trend
    # -----------------------------------------------------

    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )
    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    # -----------------------------------------------------
    # Middle Trend
    # -----------------------------------------------------

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

def validate_temporal_features(before_df, after_df, raw_features):
    """
    Temporal Feature 생성이 의도대로 되었는지 확인한다.

    검사
    ----
    - row 수 보존
    - row_id 순서 보존
    - row_id unique
    - target 결측 없음
    - Raw 47개 확인
    - Temporal 10개 수식 확인
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

    # -----------------------------------------------------
    # 수식 검증
    # -----------------------------------------------------

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
        actual = after_df[feature]

        if not np.allclose(
            expected.to_numpy(dtype=np.float64),
            actual.to_numpy(dtype=np.float64),
            equal_nan=True,
        ):
            raise AssertionError(
                f"Temporal Feature 수식 불일치: {feature}"
            )

    temporal_array = (
        after_df[ALL_TEMPORAL_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    inf_counts = dict(
        zip(
            ALL_TEMPORAL_FEATURES,
            np.isinf(temporal_array)
            .sum(axis=0)
            .astype(int)
            .tolist(),
        )
    )

    if any(count > 0 for count in inf_counts.values()):
        raise ValueError(
            f"Temporal Feature에 inf/-inf가 있습니다: {inf_counts}"
        )

    nan_counts = {
        feature: int(after_df[feature].isna().sum())
        for feature in ALL_TEMPORAL_FEATURES
    }

    return {
        "n_rows_before": int(len(before_df)),
        "n_rows_after": int(len(after_df)),
        "row_id_unique": bool(after_df[ID_COL].is_unique),
        "raw_feature_count": int(len(raw_features)),
        "all_temporal_feature_count": int(
            len(ALL_TEMPORAL_FEATURES)
        ),
        "success_feature_count": int(
            len(SUCCESS_TEMPORAL_FEATURES)
        ),
        "middle_feature_count": int(
            len(MIDDLE_TEMPORAL_FEATURES)
        ),
        "temporal_nan_counts": nan_counts,
        "temporal_inf_counts": inf_counts,
    }


# =========================================================
# CatBoost 전처리 / Metric
# =========================================================

def prepare_features(df, cat_cols):
    """
    기존 Basic / EXP3 Expanding과 동일하게
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


def calculate_metrics(y_true, y_pred):
    """Brier / baseline Brier / BSS를 계산한다."""
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    positive_rate = float(np.mean(y_true))
    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
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


def save_fold_validation(
    fold_dir,
    row_ids,
    y_true,
    y_pred,
    metadata,
):
    """Fold prediction / npz / metadata를 저장한다."""
    fold_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert (
        len(row_ids)
        == len(y_true)
        == len(y_pred)
    )

    pd.DataFrame(
        {
            ID_COL: row_ids,
            "target": y_true,
            "prediction": y_pred,
        }
    ).to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        fold_dir / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction=y_pred,
    )

    with (
        fold_dir / "metadata.json"
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
# 단일 Feature Group 실험 실행
# =========================================================

def run_experiment(
    experiment_key,
    experiment_config,
    train,
    raw_features,
    feature_qa,
):
    """
    Success 5 또는 Middle 5 하나의 Experiment를
    4-Fold Expanding Window로 실행한다.
    """
    experiment_name = (
        experiment_config["experiment_name"]
    )
    temporal_features = (
        experiment_config["feature_group"]
    )

    features = (
        list(raw_features)
        + list(temporal_features)
    )

    expected_feature_count = (
        len(raw_features)
        + len(temporal_features)
    )

    if len(features) != expected_feature_count:
        raise AssertionError(
            "Feature 수 계산이 맞지 않습니다."
        )

    # 이번 두 실험은 47 + 5 = 52여야 함
    if expected_feature_count != 52:
        raise ValueError(
            f"{experiment_key}: 예상 Feature 수 52가 아닙니다: "
            f"{expected_feature_count}"
        )

    cat_cols = [
        col for col in CAT_COLS
        if col in features
    ]

    experiment_model_dir = (
        MODEL_DIR / experiment_name
    )
    experiment_validation_dir = (
        VALIDATION_ROOT / experiment_name
    )

    if experiment_model_dir.exists():
        raise FileExistsError(
            f"기존 Model experiment가 있습니다: "
            f"{experiment_model_dir}\n"
            "기존 결과를 보존하려면 폴더를 지우지 말고 "
            "EXPERIMENTS의 experiment_name을 변경하세요."
        )

    if experiment_validation_dir.exists():
        raise FileExistsError(
            f"기존 Validation experiment가 있습니다: "
            f"{experiment_validation_dir}\n"
            "기존 결과를 보존하려면 폴더를 지우지 말고 "
            "EXPERIMENTS의 experiment_name을 변경하세요."
        )

    experiment_model_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    experiment_validation_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        experiment_model_dir
        / "train_temporal_group_split.py",
    )

    print()
    print("=" * 78)
    print(
        f"[{experiment_key}] {experiment_name}"
    )
    print("=" * 78)
    print(
        "Temporal Feature:",
        temporal_features,
    )
    print(
        "Raw / Temporal / Total:",
        len(raw_features),
        "/",
        len(temporal_features),
        "/",
        len(features),
    )
    print("=" * 78)

    fold_results = []
    oof_frames = []

    for fold_config in FOLDS:
        fold_number = (
            fold_config["fold"]
        )
        train_seasons = (
            fold_config["train_seasons"]
        )
        valid_season = (
            fold_config["valid_season"]
        )

        # -------------------------------------------------
        # 시간축 QA
        # -------------------------------------------------

        assert (
            max(train_seasons)
            < valid_season
        )

        is_train = train["season"].isin(
            train_seasons
        )
        is_valid = (
            train["season"]
            == valid_season
        )

        if (
            not is_train.any()
            or not is_valid.any()
        ):
            raise ValueError(
                f"{experiment_key} Fold {fold_number}: "
                "Train 또는 Validation이 비었습니다."
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
                f"{experiment_key} Fold {fold_number}: "
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
            **MODEL_PARAMS
        )

        print()
        print("-" * 78)
        print(
            f"{experiment_key} | Fold {fold_number}: "
            f"{train_seasons[0]}~{train_seasons[-1]} "
            f"-> {valid_season}"
        )
        print(
            "Train:",
            X_train.shape,
            "| Valid:",
            X_valid.shape,
        )
        print("-" * 78)

        start_time = time.time()

        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(
                X_valid,
                y_valid,
            ),
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            use_best_model=True,
        )

        elapsed_seconds = float(
            time.time() - start_time
        )

        valid_pred = estimator.predict_proba(
            X_valid
        )[:, 1]

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
                MODEL_PARAMS["iterations"]
            )
        else:
            final_iterations = int(
                best_iteration + 1
            )

        fold_model_dir = (
            experiment_model_dir
            / f"fold_{valid_season}"
        )
        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_model_path = (
            fold_model_dir
            / (
                f"{experiment_name}"
                f"_fold_{valid_season}.pkl"
            )
        )

        joblib.dump(
            estimator,
            fold_model_path,
            compress=3,
        )

        fold_metadata = {
            "experiment_key": experiment_key,
            "experiment_name": experiment_name,
            "fold": int(fold_number),
            "train_seasons": [
                int(x)
                for x in train_seasons
            ],
            "valid_season": int(
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
                len(temporal_features)
            ),
            "total_feature_count": int(
                len(features)
            ),
            "temporal_features": list(
                temporal_features
            ),
            "cat_cols": list(cat_cols),
            "model_params": MODEL_PARAMS,
            "early_stopping_rounds": int(
                EARLY_STOPPING_ROUNDS
            ),
            "best_iteration_zero_based": (
                int(best_iteration)
                if (
                    best_iteration is not None
                    and best_iteration >= 0
                )
                else None
            ),
            "final_iterations": int(
                final_iterations
            ),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            **metrics,
        }

        fold_validation_dir = (
            experiment_validation_dir
            / f"fold_{valid_season}"
        )

        save_fold_validation(
            fold_dir=fold_validation_dir,
            row_ids=valid_row_ids.to_numpy(),
            y_true=y_valid.to_numpy(),
            y_pred=valid_pred,
            metadata=fold_metadata,
        )

        fold_results.append(
            fold_metadata
        )

        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COL: (
                        valid_row_ids
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
            f"{experiment_key}: OOF row_id 중복"
        )

    oof.to_csv(
        experiment_validation_dir
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        experiment_validation_dir
        / "oof_validation.npz",
        row_id=oof[ID_COL].to_numpy(),
        season=oof["season"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction=oof["prediction"].to_numpy(),
        fold=oof["fold"].to_numpy(),
    )

    pooled_metrics = calculate_metrics(
        oof["target"].to_numpy(),
        oof["prediction"].to_numpy(),
    )

    fold_briers = np.asarray(
        [
            result["brier"]
            for result in fold_results
        ],
        dtype=np.float64,
    )

    overall_metadata = {
        "experiment_key": experiment_key,
        "experiment_name": experiment_name,
        "raw_feature_count": int(
            len(raw_features)
        ),
        "temporal_feature_count": int(
            len(temporal_features)
        ),
        "total_feature_count": int(
            len(features)
        ),
        "temporal_features": list(
            temporal_features
        ),
        "feature_qa": feature_qa,
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": int(
            EARLY_STOPPING_ROUNDS
        ),
        "fold_results": fold_results,
        "fold_mean_brier": float(
            fold_briers.mean()
        ),
        "fold_std_brier": float(
            fold_briers.std()
        ),
        "pooled_oof_metrics": (
            pooled_metrics
        ),
    }

    with (
        experiment_validation_dir
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

    print()
    print("=" * 78)
    print(
        f"{experiment_key} 완료"
    )
    print("=" * 78)

    for result in fold_results:
        print(
            f"{result['valid_season']}: "
            f"Brier={result['brier']:.8f}, "
            f"iter={result['final_iterations']}"
        )

    print(
        "Pooled OOF Brier:",
        f"{pooled_metrics['brier']:.8f}",
    )
    print("=" * 78)


# =========================================================
# 기존 + 신규 OOF 결과 비교
# =========================================================

def load_oof(experiment_name):
    path = (
        VALIDATION_ROOT
        / experiment_name
        / "oof_pred.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"OOF prediction이 없습니다: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required_cols = {
        ID_COL,
        "season",
        "target",
        "prediction",
        "fold",
    }

    missing = (
        required_cols
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path}에 필요한 컬럼이 없습니다: "
            f"{sorted(missing)}"
        )

    if not df[ID_COL].is_unique:
        raise AssertionError(
            f"{experiment_name}: "
            "OOF row_id가 unique하지 않습니다."
        )

    return df


def build_comparison():
    """
    Basic / All10 / Success5 / Middle5의
    Fold별 Brier 및 Pooled OOF Brier를 한 표로 만든다.
    """
    loaded = {}

    for key, config in EXPERIMENTS.items():
        loaded[key] = load_oof(
            config["experiment_name"]
        )

    # -----------------------------------------------------
    # 네 실험이 정확히 같은 OOF row/target인지 검증
    # -----------------------------------------------------

    reference = (
        loaded["basic"][
            [
                ID_COL,
                "season",
                "target",
                "fold",
            ]
        ]
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )

    for key, df in loaded.items():
        current = (
            df[
                [
                    ID_COL,
                    "season",
                    "target",
                    "fold",
                ]
            ]
            .sort_values(ID_COL)
            .reset_index(drop=True)
        )

        if not reference.equals(current):
            raise AssertionError(
                f"{key}: Basic과 OOF row/target 구성이 다릅니다."
            )

    # -----------------------------------------------------
    # Fold별 + pooled Brier
    # -----------------------------------------------------

    rows = []

    for key, df in loaded.items():
        config = EXPERIMENTS[key]

        for season in [
            2021,
            2022,
            2023,
            2024,
        ]:
            part = df.loc[
                df["season"] == season
            ]

            brier = float(
                np.mean(
                    (
                        part["prediction"]
                        - part["target"]
                    ) ** 2
                )
            )

            rows.append(
                {
                    "experiment": key,
                    "experiment_name": (
                        config[
                            "experiment_name"
                        ]
                    ),
                    "scope": str(season),
                    "n": int(len(part)),
                    "brier": brier,
                }
            )

        pooled_brier = float(
            np.mean(
                (
                    df["prediction"]
                    - df["target"]
                ) ** 2
            )
        )

        rows.append(
            {
                "experiment": key,
                "experiment_name": (
                    config["experiment_name"]
                ),
                "scope": "pooled_oof",
                "n": int(len(df)),
                "brier": pooled_brier,
            }
        )

    long_df = pd.DataFrame(rows)

    basic_map = (
        long_df.loc[
            long_df["experiment"] == "basic",
            ["scope", "brier"],
        ]
        .set_index("scope")["brier"]
        .to_dict()
    )

    long_df[
        "delta_brier_basic_minus_experiment"
    ] = long_df.apply(
        lambda row: (
            basic_map[row["scope"]]
            - row["brier"]
        ),
        axis=1,
    )

    # delta > 0 이면 Basic보다 개선
    long_df["better_than_basic"] = (
        long_df[
            "delta_brier_basic_minus_experiment"
        ]
        > 0
    )

    # -----------------------------------------------------
    # 보기 쉬운 Wide table
    # -----------------------------------------------------

    wide_df = (
        long_df.pivot(
            index="experiment",
            columns="scope",
            values="brier",
        )
        .reset_index()
    )

    desired_cols = [
        "experiment",
        "2021",
        "2022",
        "2023",
        "2024",
        "pooled_oof",
    ]

    wide_df = wide_df[
        desired_cols
    ]

    basic_pooled = float(
        wide_df.loc[
            wide_df["experiment"] == "basic",
            "pooled_oof",
        ].iloc[0]
    )

    wide_df[
        "delta_pooled_basic_minus_experiment"
    ] = (
        basic_pooled
        - wide_df["pooled_oof"]
    )

    order = [
        "basic",
        "all_temporal_10",
        "success_5",
        "middle_5",
    ]

    wide_df["__order"] = (
        wide_df["experiment"]
        .map(
            {
                name: i
                for i, name
                in enumerate(order)
            }
        )
    )

    wide_df = (
        wide_df
        .sort_values("__order")
        .drop(columns="__order")
        .reset_index(drop=True)
    )

    return (
        long_df,
        wide_df,
    )


# =========================================================
# Main
# =========================================================

def main():
    # -----------------------------------------------------
    # 기존 결과 존재 여부 먼저 확인
    # -----------------------------------------------------

    for key in [
        "basic",
        "all_temporal_10",
    ]:
        exp_name = (
            EXPERIMENTS[key][
                "experiment_name"
            ]
        )

        path = (
            VALIDATION_ROOT
            / exp_name
            / "oof_pred.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"기존 {key} OOF 결과가 없습니다: {path}\n"
                "Basic / EXP3 Expanding 실험을 먼저 완료해야 합니다."
            )

    # -----------------------------------------------------
    # Raw Feature는 공식 test.csv 구조로 결정
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
    # Train 로드 + Temporal 생성 + QA
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

    print("=" * 78)
    print("Temporal Group Split Experiment")
    print("=" * 78)
    print("Train shape       :", train.shape)
    print("Raw Feature       :", len(raw_features))
    print(
        "Success Temporal  :",
        len(SUCCESS_TEMPORAL_FEATURES),
    )
    print(
        "Middle Temporal   :",
        len(MIDDLE_TEMPORAL_FEATURES),
    )
    print(
        "Temporal NaN count:",
        feature_qa["temporal_nan_counts"],
    )
    print("=" * 78)

    # -----------------------------------------------------
    # Success 5 / Middle 5만 신규 학습
    # -----------------------------------------------------

    for key in [
        "success_5",
        "middle_5",
    ]:
        run_experiment(
            experiment_key=key,
            experiment_config=(
                EXPERIMENTS[key]
            ),
            train=train,
            raw_features=raw_features,
            feature_qa=feature_qa,
        )

    # -----------------------------------------------------
    # 4개 실험 결과 비교
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        VALIDATION_DIR
        / "train_temporal_group_split.py",
    )

    long_df, wide_df = (
        build_comparison()
    )

    long_df.to_csv(
        VALIDATION_DIR
        / "temporal_group_comparison_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wide_df.to_csv(
        VALIDATION_DIR
        / "temporal_group_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "purpose": (
            "Basic / All Temporal 10 / "
            "Success Temporal 5 / Middle Temporal 5 "
            "Expanding Window Brier comparison"
        ),
        "delta_definition": (
            "Basic Brier - Experiment Brier; "
            "positive means Experiment is better"
        ),
        "success_temporal_features": (
            SUCCESS_TEMPORAL_FEATURES
        ),
        "middle_temporal_features": (
            MIDDLE_TEMPORAL_FEATURES
        ),
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "feature_qa": feature_qa,
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

    print()
    print("=" * 78)
    print("최종 비교")
    print("=" * 78)

    display_df = wide_df.copy()

    for col in [
        "2021",
        "2022",
        "2023",
        "2024",
        "pooled_oof",
        "delta_pooled_basic_minus_experiment",
    ]:
        display_df[col] = (
            display_df[col]
            .map(
                lambda value: f"{value:.8f}"
            )
        )

    print(
        display_df.to_string(
            index=False
        )
    )

    print()
    print(
        "delta_pooled_basic_minus_experiment > 0 "
        "이면 Basic보다 개선입니다."
    )
    print(
        "결과 저장:",
        VALIDATION_DIR
        / "temporal_group_comparison.csv",
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
