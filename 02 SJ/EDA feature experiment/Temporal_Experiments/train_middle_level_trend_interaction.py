# train_middle_level_trend_interaction.py
"""
LG Aimers Phase2 - Middle Level × Trend Interaction 실험

목적
----
2024 원인 분석에서 Middle Temporal의 효과가 단순 Level 또는 Trend 단독보다
Level과 Trend가 함께 특정 상태에 있을 때 더 크게 나타나는 패턴이 관찰되었다.

하지만 2024에서 발견한 특정 구간(L2/T3 등)을 그대로 Feature로 만들면
Validation에 과적합될 위험이 있다.

따라서 threshold를 사용하지 않고, 모든 시즌에 동일하게 적용 가능한
연속형 interaction 1개를 만든다.

    middle_level_mean
        = mean(
            recent1_middle_vs_career,
            recent3_middle_vs_career,
            recent5_middle_vs_career
          )

    middle_trend_mean
        = mean(
            middle_trend_1_3,
            middle_trend_3_5
          )

    middle_level_trend_interaction
        = middle_level_mean * middle_trend_mean

비교
----
기존 결과 재사용:
1) Basic
   - Raw 47

2) Middle 5
   - Raw 47 + Middle Temporal 5

신규 학습:
3) Interaction Only
   - Raw 47 + middle_level_trend_interaction 1개

4) Middle 5 + Interaction
   - Raw 47 + Middle Temporal 5 + interaction 1개

해석
----
- Interaction Only가 Basic보다 좋으면:
  Level × Trend 조합 자체가 압축된 신호로 쓸모 있을 가능성.

- Middle5 + Interaction이 Middle5보다 좋으면:
  CatBoost가 원래 5개 Feature만으로 학습한 관계보다
  explicit interaction을 추가하는 것이 도움이 될 가능성.

- 둘 다 개선되지 않으면:
  2024에서 관찰한 Level × Trend 패턴이 전체 시즌에서
  일반화되는 Feature 신호라는 근거는 약해진다.

중요
----
- 2024에서 관찰한 특정 threshold / bin은 사용하지 않는다.
- 모든 Feature는 동일 row의 공식 asof_* 값끼리만 계산한다.
- test 내부 다른 row를 이용한 groupby / rolling / lag / 집계는 없다.
- CatBoost 설정, seed, Fold, Early Stopping은 기존 실험과 동일하다.
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

EXPERIMENT_NAME = "middle_level_trend_interaction_v001"

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
# Feature 정의
# =========================================================

MIDDLE_LEVEL_FEATURES = [
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
]

MIDDLE_TREND_FEATURES = [
    "middle_trend_1_3",
    "middle_trend_3_5",
]

MIDDLE_TEMPORAL_FEATURES = (
    MIDDLE_LEVEL_FEATURES
    + MIDDLE_TREND_FEATURES
)

INTERACTION_FEATURE = (
    "middle_level_trend_interaction"
)


# 기존 결과 + 신규 결과
EXPERIMENTS = {
    "basic": {
        "experiment_name": "cb_v001_ctr1_expanding",
        "extra_features": [],
        "train": False,
    },
    "middle_5": {
        "experiment_name": "cb_v001_ctr1_expanding_middle5",
        "extra_features": MIDDLE_TEMPORAL_FEATURES,
        "train": False,
    },
    "interaction_only": {
        "experiment_name": "cb_v001_ctr1_expanding_middle_interaction1",
        "extra_features": [INTERACTION_FEATURE],
        "train": True,
    },
    "middle5_plus_interaction": {
        "experiment_name": "cb_v001_ctr1_expanding_middle5_interaction1",
        "extra_features": (
            MIDDLE_TEMPORAL_FEATURES
            + [INTERACTION_FEATURE]
        ),
        "train": True,
    },
}


# =========================================================
# Feature 생성
# =========================================================

def add_middle_features(df):
    """
    기존 Middle Temporal 5개와 Level × Trend interaction 1개를 생성한다.

    interaction은 2024 특정 threshold를 사용하지 않고
    모든 row에서 동일한 연속형 공식으로 계산한다.
    """
    df = df.copy()

    required_cols = [
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Middle Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    # -----------------------------------------------------
    # Middle Level 3
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
    # Middle Trend 2
    # -----------------------------------------------------

    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )

    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    # -----------------------------------------------------
    # Group-level 평균
    #
    # 단순 산술식으로 계산해 NaN이 있는 row는 interaction도 NaN이 되도록 한다.
    # 기존 asof_* 결측을 임의로 보완하지 않는다.
    # -----------------------------------------------------

    middle_level_mean = (
        df["recent1_middle_vs_career"]
        + df["recent3_middle_vs_career"]
        + df["recent5_middle_vs_career"]
    ) / 3.0

    middle_trend_mean = (
        df["middle_trend_1_3"]
        + df["middle_trend_3_5"]
    ) / 2.0

    # -----------------------------------------------------
    # Level × Trend Interaction
    # -----------------------------------------------------

    df[INTERACTION_FEATURE] = (
        middle_level_mean
        * middle_trend_mean
    )

    return df


# =========================================================
# Data QA
# =========================================================

def validate_features(
    before_df,
    after_df,
    raw_features,
):
    """
    생성 Feature QA
    - row 수 보존
    - row_id 순서 보존 / unique
    - target 결측 없음
    - Raw Feature 47개
    - Middle 5 수식 검증
    - Interaction 수식 검증
    - inf/-inf 없음
    - NaN 개수 기록
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
        "recent1_middle_vs_career": (
            after_df[
                "asof_pitcher_prev1_game_middle_rate"
            ]
            - after_df[
                "asof_pitcher_middle_rate"
            ]
        ),
        "recent3_middle_vs_career": (
            after_df[
                "asof_pitcher_prev3_game_middle_rate"
            ]
            - after_df[
                "asof_pitcher_middle_rate"
            ]
        ),
        "recent5_middle_vs_career": (
            after_df[
                "asof_pitcher_prev5_game_middle_rate"
            ]
            - after_df[
                "asof_pitcher_middle_rate"
            ]
        ),
        "middle_trend_1_3": (
            after_df[
                "asof_pitcher_prev1_game_middle_rate"
            ]
            - after_df[
                "asof_pitcher_prev3_game_middle_rate"
            ]
        ),
        "middle_trend_3_5": (
            after_df[
                "asof_pitcher_prev3_game_middle_rate"
            ]
            - after_df[
                "asof_pitcher_prev5_game_middle_rate"
            ]
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
                f"Feature 수식 불일치: {feature}"
            )

    expected_level_mean = (
        after_df["recent1_middle_vs_career"]
        + after_df["recent3_middle_vs_career"]
        + after_df["recent5_middle_vs_career"]
    ) / 3.0

    expected_trend_mean = (
        after_df["middle_trend_1_3"]
        + after_df["middle_trend_3_5"]
    ) / 2.0

    expected_interaction = (
        expected_level_mean
        * expected_trend_mean
    )

    if not np.allclose(
        expected_interaction.to_numpy(dtype=np.float64),
        after_df[
            INTERACTION_FEATURE
        ].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise AssertionError(
            "Middle Level × Trend interaction 수식 불일치"
        )

    check_features = (
        MIDDLE_TEMPORAL_FEATURES
        + [INTERACTION_FEATURE]
    )

    values = (
        after_df[check_features]
        .to_numpy(dtype=np.float64)
    )

    inf_counts = dict(
        zip(
            check_features,
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
            f"생성 Feature에 inf/-inf가 있습니다: {inf_counts}"
        )

    nan_counts = {
        feature: int(
            after_df[feature]
            .isna()
            .sum()
        )
        for feature in check_features
    }

    return {
        "n_rows_before": int(len(before_df)),
        "n_rows_after": int(len(after_df)),
        "row_id_unique": bool(
            after_df[ID_COL].is_unique
        ),
        "raw_feature_count": int(
            len(raw_features)
        ),
        "middle_temporal_count": int(
            len(MIDDLE_TEMPORAL_FEATURES)
        ),
        "interaction_count": 1,
        "generated_nan_counts": nan_counts,
        "generated_inf_counts": inf_counts,
    }


# =========================================================
# CatBoost 유틸
# =========================================================

def prepare_features(
    df,
    cat_cols,
):
    """
    기존 CatBoost 실험과 동일하게
    범주형 결측만 문자열 처리한다.
    """
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Brier / baseline Brier / BSS 계산.
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

    bss = float(
        max(
            0.0,
            100000.0
            * (
                1.0
                - brier
                / baseline_brier
            ),
        )
    )

    return {
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": bss,
    }


# =========================================================
# 신규 Experiment 실행
# =========================================================

def run_experiment(
    experiment_key,
    experiment_config,
    train,
    raw_features,
    feature_qa,
):
    """
    신규 Feature Set 하나를 4-Fold Expanding Window로 학습한다.
    """
    experiment_name = (
        experiment_config[
            "experiment_name"
        ]
    )

    extra_features = (
        experiment_config[
            "extra_features"
        ]
    )

    features = (
        list(raw_features)
        + list(extra_features)
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    experiment_model_dir = (
        MODEL_DIR
        / experiment_name
    )

    experiment_validation_dir = (
        VALIDATION_ROOT
        / experiment_name
    )

    if experiment_model_dir.exists():
        raise FileExistsError(
            f"기존 Model 폴더가 있습니다: "
            f"{experiment_model_dir}"
        )

    if experiment_validation_dir.exists():
        raise FileExistsError(
            f"기존 Validation 폴더가 있습니다: "
            f"{experiment_validation_dir}"
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
        / "train_middle_level_trend_interaction.py",
    )

    print()
    print("=" * 82)
    print(
        f"[{experiment_key}] "
        f"{experiment_name}"
    )
    print("=" * 82)
    print(
        "Extra Feature:",
        extra_features,
    )
    print(
        "Raw / Extra / Total:",
        len(raw_features),
        "/",
        len(extra_features),
        "/",
        len(features),
    )
    print("=" * 82)

    fold_results = []
    oof_frames = []

    for fold_config in FOLDS:
        fold_number = (
            fold_config["fold"]
        )

        train_seasons = (
            fold_config[
                "train_seasons"
            ]
        )

        valid_season = (
            fold_config[
                "valid_season"
            ]
        )

        # 시간축 QA
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

        if not set(train_ids).isdisjoint(
            set(valid_ids)
        ):
            raise AssertionError(
                f"{experiment_key} "
                f"Fold {fold_number}: "
                "Train/Validation row_id 겹침"
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

        model = CatBoostClassifier(
            **MODEL_PARAMS
        )

        print()
        print("-" * 82)
        print(
            f"{experiment_key} | "
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
        print("-" * 82)

        start_time = time.time()

        model.fit(
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

        elapsed = float(
            time.time()
            - start_time
        )

        valid_pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_valid.to_numpy(),
            valid_pred,
        )

        best_iteration = (
            model.get_best_iteration()
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

        fold_model_dir = (
            experiment_model_dir
            / f"fold_{valid_season}"
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        joblib.dump(
            model,
            fold_model_dir
            / (
                f"{experiment_name}"
                f"_fold_{valid_season}.pkl"
            ),
            compress=3,
        )

        fold_validation_dir = (
            experiment_validation_dir
            / f"fold_{valid_season}"
        )

        fold_validation_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        pd.DataFrame(
            {
                ID_COL: (
                    valid_ids
                    .to_numpy()
                ),
                "target": (
                    y_valid
                    .to_numpy()
                ),
                "prediction": (
                    valid_pred
                ),
            }
        ).to_csv(
            fold_validation_dir
            / "pred.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fold_metadata = {
            "experiment_key": (
                experiment_key
            ),
            "experiment_name": (
                experiment_name
            ),
            "fold": int(
                fold_number
            ),
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
            "extra_feature_count": int(
                len(extra_features)
            ),
            "total_feature_count": int(
                len(features)
            ),
            "extra_features": list(
                extra_features
            ),
            "best_iteration": int(
                final_iterations
            ),
            "elapsed_seconds": (
                elapsed
            ),
            "model_params": (
                MODEL_PARAMS
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
            pd.DataFrame(
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
        )

        print(
            f"Brier="
            f"{metrics['brier']:.8f}"
            f" | BSS="
            f"{metrics['bss']:.2f}"
            f" | iter="
            f"{final_iterations}"
            f" | time="
            f"{elapsed:.1f}s"
        )

        del (
            X_train,
            X_valid,
            y_train,
            y_valid,
            model,
            valid_pred,
        )

        gc.collect()

    # OOF
    oof = pd.concat(
        oof_frames,
        axis=0,
        ignore_index=True,
    )

    if not oof[ID_COL].is_unique:
        raise AssertionError(
            f"{experiment_key}: "
            "OOF row_id 중복"
        )

    oof.to_csv(
        experiment_validation_dir
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_metrics = (
        calculate_metrics(
            oof["target"]
            .to_numpy(),
            oof["prediction"]
            .to_numpy(),
        )
    )

    overall_metadata = {
        "experiment_key": (
            experiment_key
        ),
        "experiment_name": (
            experiment_name
        ),
        "raw_feature_count": int(
            len(raw_features)
        ),
        "extra_feature_count": int(
            len(extra_features)
        ),
        "total_feature_count": int(
            len(features)
        ),
        "extra_features": list(
            extra_features
        ),
        "feature_qa": feature_qa,
        "fold_results": (
            fold_results
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
    print(
        f"[완료] {experiment_key}"
        f" pooled OOF Brier = "
        f"{pooled_metrics['brier']:.8f}"
    )


# =========================================================
# OOF 비교
# =========================================================

def load_oof(
    experiment_name,
):
    path = (
        VALIDATION_ROOT
        / experiment_name
        / "oof_pred.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"OOF prediction 없음: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = {
        ID_COL,
        "season",
        "target",
        "prediction",
        "fold",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path} 필수 컬럼 누락: "
            f"{sorted(missing)}"
        )

    if not df[ID_COL].is_unique:
        raise AssertionError(
            f"{experiment_name}: "
            "OOF row_id 중복"
        )

    return df


def build_comparison():
    """
    Basic / Middle5 / Interaction Only /
    Middle5 + Interaction을 같은 OOF row에서 비교한다.
    """
    loaded = {
        key: load_oof(
            config[
                "experiment_name"
            ]
        )
        for key, config
        in EXPERIMENTS.items()
    }

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

        if not reference.equals(
            current
        ):
            raise AssertionError(
                f"{key}: "
                "Basic과 OOF row/target 구성이 다릅니다."
            )

    rows = []

    for key, df in loaded.items():
        for scope in [
            2021,
            2022,
            2023,
            2024,
        ]:
            part = df.loc[
                df["season"]
                == scope
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
                    "scope": str(
                        scope
                    ),
                    "n": int(
                        len(part)
                    ),
                    "brier": brier,
                }
            )

        pooled = float(
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
                "scope": (
                    "pooled_oof"
                ),
                "n": int(
                    len(df)
                ),
                "brier": pooled,
            }
        )

    long_df = pd.DataFrame(
        rows
    )

    wide_df = (
        long_df
        .pivot(
            index="experiment",
            columns="scope",
            values="brier",
        )
        .reset_index()
    )

    wide_df = wide_df[
        [
            "experiment",
            "2021",
            "2022",
            "2023",
            "2024",
            "pooled_oof",
        ]
    ]

    basic_brier = float(
        wide_df.loc[
            wide_df[
                "experiment"
            ]
            == "basic",
            "pooled_oof",
        ].iloc[0]
    )

    middle5_brier = float(
        wide_df.loc[
            wide_df[
                "experiment"
            ]
            == "middle_5",
            "pooled_oof",
        ].iloc[0]
    )

    wide_df[
        "delta_basic_minus_experiment"
    ] = (
        basic_brier
        - wide_df[
            "pooled_oof"
        ]
    )

    wide_df[
        "delta_middle5_minus_experiment"
    ] = (
        middle5_brier
        - wide_df[
            "pooled_oof"
        ]
    )

    order = [
        "basic",
        "middle_5",
        "interaction_only",
        "middle5_plus_interaction",
    ]

    wide_df["__order"] = (
        wide_df[
            "experiment"
        ].map(
            {
                name: i
                for i, name
                in enumerate(order)
            }
        )
    )

    wide_df = (
        wide_df
        .sort_values(
            "__order"
        )
        .drop(
            columns="__order"
        )
        .reset_index(
            drop=True
        )
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
    # 기존 OOF 존재 확인
    # -----------------------------------------------------

    for key in [
        "basic",
        "middle_5",
    ]:
        experiment_name = (
            EXPERIMENTS[key][
                "experiment_name"
            ]
        )

        path = (
            VALIDATION_ROOT
            / experiment_name
            / "oof_pred.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"기존 {key} OOF가 없습니다: "
                f"{path}"
            )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 비교 폴더가 있습니다: "
            f"{VALIDATION_DIR}"
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
            f"test.csv에 {ID_COL}가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID_COL
    ]

    # -----------------------------------------------------
    # Train 로드 + Feature 생성 + QA
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

    before = train.copy()

    train = add_middle_features(
        train
    )

    feature_qa = validate_features(
        before_df=before,
        after_df=train,
        raw_features=raw_features,
    )

    del before
    gc.collect()

    print("=" * 82)
    print(
        "Middle Level × Trend Interaction Experiment"
    )
    print("=" * 82)
    print(
        "Raw Feature:",
        len(raw_features),
    )
    print(
        "Middle5:",
        MIDDLE_TEMPORAL_FEATURES,
    )
    print(
        "Interaction:",
        INTERACTION_FEATURE,
    )
    print(
        "NaN counts:",
        feature_qa[
            "generated_nan_counts"
        ],
    )
    print("=" * 82)

    # -----------------------------------------------------
    # 신규 2개만 학습
    # -----------------------------------------------------

    for key in [
        "interaction_only",
        "middle5_plus_interaction",
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
    # 최종 비교
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        VALIDATION_DIR
        / "train_middle_level_trend_interaction.py",
    )

    (
        long_df,
        wide_df,
    ) = build_comparison()

    long_df.to_csv(
        VALIDATION_DIR
        / "interaction_comparison_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wide_df.to_csv(
        VALIDATION_DIR
        / "interaction_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "hypothesis": (
            "Middle Level과 Middle Trend의 결합 관계를 "
            "연속형 interaction으로 명시하면 "
            "Basic 또는 Middle5보다 out-of-time Brier가 "
            "개선될 수 있다."
        ),
        "interaction_definition": {
            "middle_level_mean": (
                "(recent1_middle_vs_career + "
                "recent3_middle_vs_career + "
                "recent5_middle_vs_career) / 3"
            ),
            "middle_trend_mean": (
                "(middle_trend_1_3 + "
                "middle_trend_3_5) / 2"
            ),
            INTERACTION_FEATURE: (
                "middle_level_mean * "
                "middle_trend_mean"
            ),
        },
        "validation": FOLDS,
        "model_params": (
            MODEL_PARAMS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "feature_qa": (
            feature_qa
        ),
        "warning": (
            "2024 specific bin thresholds are not used. "
            "This experiment tests a general continuous interaction."
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

    print()
    print("=" * 82)
    print("최종 비교")
    print("=" * 82)

    display = wide_df.copy()

    numeric_cols = [
        "2021",
        "2022",
        "2023",
        "2024",
        "pooled_oof",
        "delta_basic_minus_experiment",
        "delta_middle5_minus_experiment",
    ]

    for col in numeric_cols:
        display[col] = (
            display[col]
            .map(
                lambda x: f"{x:.8f}"
            )
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "delta_basic_minus_experiment > 0 "
        "→ Basic보다 개선"
    )
    print(
        "delta_middle5_minus_experiment > 0 "
        "→ 기존 Middle5보다 개선"
    )
    print()
    print(
        "결과 저장:",
        VALIDATION_DIR
        / "interaction_comparison.csv",
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
