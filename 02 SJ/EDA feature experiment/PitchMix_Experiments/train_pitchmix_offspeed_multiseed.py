# train_pitchmix_offspeed_multiseed.py
"""
LG Aimers Phase2
Offspeed × Count 2개 Multi-Seed 안정성 검증

목적
----
seed=42 단일 실험에서:

cb_v002 Raw47
    pooled OOF = 0.24680316

Raw47 + Offspeed × Count 2개
    pooled OOF = 0.24679230

Delta = +0.00001086

로 소폭 개선되었다.

하지만 개선폭이 작고 2024 Validation에서는 악화했으므로,
여러 random_seed에서 개선 방향이 반복되는지 확인한다.

비교 Feature
------------
Basic
    Raw Feature 47개

Offspeed × Count
    Raw Feature 47개
    + offspeed_x_balls
    + offspeed_x_strikes

Multi-Seed
----------
42, 7, 2026, 123, 999

seed=42:
    기존 cb_v002 / cb_v002_pitchmix_offspeed_v001 OOF 재사용

나머지 seed:
    Basic과 Offspeed 모델을 같은 seed로 각각 새로 학습

고정 조건
---------
- CatBoostClassifier
- iterations 최대 800
- learning_rate = 0.07
- depth = 8
- loss_function = Logloss
- l2_leaf_reg = 10
- random_strength = 1
- max_ctr_complexity = 1
- Early Stopping = 80
- categorical feature 동일
- 4-Fold Expanding Window 동일
- Feature Set 외 조건 동일

판정
----
Delta = Basic Brier - Offspeed Brier

Delta > 0 : Offspeed 개선
Delta < 0 : Offspeed 악화

주의
----
- 현재 투구의 실제 구종은 사용하지 않는다.
- 공식 asof_*와 동일 row의 count만 이용한다.
- test 내부 다른 row를 이용한 집계/rolling/lag는 없다.
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

EXPERIMENT_NAME = "cb_v002_pitchmix_offspeed_multiseed_v001"

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

SEEDS = [
    42,
    7,
    2026,
    123,
    999,
]


# =========================================================
# 기존 seed=42 결과
# =========================================================

SEED42_BASIC_EXPERIMENT = "cb_v002"

SEED42_OFFSPEED_EXPERIMENT = (
    "cb_v002_pitchmix_offspeed_v001"
)


# =========================================================
# CatBoost 범주형
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
        "train_seasons": [
            2019,
            2020,
            2021,
            2022,
        ],
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
# 고정 모델 설정
# random_seed만 seed별 변경
# =========================================================

EARLY_STOPPING_ROUNDS = 80


def get_model_params(seed):
    """
    cb_v002 설정 그대로 유지하고 random_seed만 변경.
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
# Offspeed × Count Feature
# =========================================================

OFFSPEED_COUNT_FEATURES = [
    "offspeed_x_balls",
    "offspeed_x_strikes",
]


def add_offspeed_count_features(df):
    """
    Offspeed × Count 2개 생성.

    offspeed_x_balls
        = asof_pitcher_offspeed_rate * balls_before

    offspeed_x_strikes
        = asof_pitcher_offspeed_rate * strikes_before
    """

    df = df.copy()

    required_cols = [
        "asof_pitcher_offspeed_rate",
        "balls_before",
        "strikes_before",
    ]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Offspeed Feature 생성 필수 컬럼 누락: "
            + ", ".join(missing)
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

def validate_features(
    before_df,
    after_df,
    raw_features,
):

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
            "Raw Feature 수가 47이 아닙니다: "
            f"{len(raw_features)}"
        )

    expected = {
        "offspeed_x_balls": (
            after_df[
                "asof_pitcher_offspeed_rate"
            ]
            * after_df["balls_before"]
        ),
        "offspeed_x_strikes": (
            after_df[
                "asof_pitcher_offspeed_rate"
            ]
            * after_df["strikes_before"]
        ),
    }

    for feature, expected_value in (
        expected.items()
    ):

        if not np.allclose(
            expected_value.to_numpy(
                dtype=np.float64
            ),
            after_df[
                feature
            ].to_numpy(
                dtype=np.float64
            ),
            equal_nan=True,
        ):
            raise AssertionError(
                f"수식 불일치: {feature}"
            )

    values = (
        after_df[
            OFFSPEED_COUNT_FEATURES
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if np.isinf(values).any():
        raise ValueError(
            "Offspeed Feature에 "
            "inf/-inf가 있습니다."
        )


# =========================================================
# CatBoost 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):

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

def calculate_brier(
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


# =========================================================
# seed=42 기존 OOF 로드
# =========================================================

def load_seed42_oof():

    basic_path = (
        VALIDATION_ROOT
        / SEED42_BASIC_EXPERIMENT
        / "oof_pred.csv"
    )

    offspeed_path = (
        VALIDATION_ROOT
        / SEED42_OFFSPEED_EXPERIMENT
        / "oof_pred.csv"
    )

    if not basic_path.exists():
        raise FileNotFoundError(
            f"seed42 Basic OOF 없음: "
            f"{basic_path}"
        )

    if not offspeed_path.exists():
        raise FileNotFoundError(
            f"seed42 Offspeed OOF 없음: "
            f"{offspeed_path}"
        )

    basic = pd.read_csv(
        basic_path,
        encoding="utf-8-sig",
    )

    offspeed = pd.read_csv(
        offspeed_path,
        encoding="utf-8-sig",
    )

    return (
        basic,
        offspeed,
    )


# =========================================================
# 단일 Feature Set 학습
# =========================================================

def run_one_model(
    train,
    raw_features,
    seed,
    feature_key,
    extra_features,
):
    """
    특정 seed에서 Basic 또는 Offspeed를
    4-Fold Expanding Window로 실행.
    """

    features = (
        list(raw_features)
        + list(extra_features)
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    model_params = (
        get_model_params(
            seed
        )
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
    print("=" * 90)
    print(
        f"seed={seed} | "
        f"{feature_key} | "
        f"Feature={len(features)}"
    )
    print("=" * 90)

    for config in EXPANDING_FOLDS:

        fold = int(
            config["fold"]
        )

        train_seasons = list(
            config[
                "train_seasons"
            ]
        )

        validation_season = int(
            config[
                "validation_season"
            ]
        )

        if (
            max(train_seasons)
            >= validation_season
        ):
            raise AssertionError(
                "시간 순서 오류"
            )

        train_mask = (
            train["season"]
            .isin(train_seasons)
        )

        valid_mask = (
            train["season"]
            == validation_season
        )

        train_ids = train.loc[
            train_mask,
            ID_COL,
        ]

        valid_ids = train.loc[
            valid_mask,
            ID_COL,
        ]

        if not set(
            train_ids
        ).isdisjoint(
            set(valid_ids)
        ):
            raise AssertionError(
                "Train / Validation "
                "row_id overlap"
            )

        X_train = (
            prepare_features(
                train.loc[
                    train_mask,
                    features,
                ],
                cat_cols,
            )
        )

        y_train = train.loc[
            train_mask,
            TARGET_COL,
        ]

        X_valid = (
            prepare_features(
                train.loc[
                    valid_mask,
                    features,
                ],
                cat_cols,
            )
        )

        y_valid = train.loc[
            valid_mask,
            TARGET_COL,
        ]

        model = (
            CatBoostClassifier(
                **model_params
            )
        )

        print()
        print("-" * 90)

        print(
            f"{feature_key} | "
            f"seed={seed} | "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]}"
            f" -> {validation_season}"
        )

        print(
            "Train:",
            X_train.shape,
            "| Valid:",
            X_valid.shape,
        )

        print("-" * 90)

        start = time.time()

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
            - start
        )

        valid_pred = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        brier = (
            calculate_brier(
                y_valid,
                valid_pred,
            )
        )

        best_iteration = (
            model.get_best_iteration()
        )

        if (
            best_iteration is None
            or best_iteration < 0
        ):
            final_iterations = int(
                model_params[
                    "iterations"
                ]
            )

        else:
            final_iterations = int(
                best_iteration + 1
            )

        # ---------------------------------------------
        # Model 저장
        # ---------------------------------------------

        fold_model_dir = (
            model_run_dir
            / f"fold_{validation_season}"
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        joblib.dump(
            model,
            fold_model_dir
            / (
                f"{feature_key}"
                f"_seed_{seed}"
                f"_fold_{validation_season}"
                f".pkl"
            ),
            compress=3,
        )

        # ---------------------------------------------
        # OOF
        # ---------------------------------------------

        fold_pred = pd.DataFrame(
            {
                ID_COL: (
                    valid_ids
                    .to_numpy()
                ),
                "validation_season": (
                    validation_season
                ),
                "target": (
                    y_valid
                    .to_numpy()
                ),
                "prediction": (
                    valid_pred
                ),
                "fold": fold,
            }
        )

        fold_pred.to_csv(
            validation_run_dir
            / (
                f"fold_"
                f"{validation_season}"
                f"_pred.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        oof_frames.append(
            fold_pred
        )

        fold_rows.append(
            {
                "seed": int(seed),
                "feature_set": (
                    feature_key
                ),
                "fold": fold,
                "validation_season": (
                    validation_season
                ),
                "n_train": int(
                    train_mask.sum()
                ),
                "n_valid": int(
                    valid_mask.sum()
                ),
                "feature_count": int(
                    len(features)
                ),
                "best_iteration": (
                    final_iterations
                ),
                "brier": brier,
                "elapsed_seconds": (
                    elapsed
                ),
            }
        )

        print(
            f"Brier="
            f"{brier:.8f} | "
            f"iter="
            f"{final_iterations} | "
            f"time="
            f"{elapsed:.1f}s"
        )

        del (
            X_train,
            X_valid,
            y_train,
            y_valid,
            model,
            valid_pred,
            fold_pred,
        )

        gc.collect()

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    if not oof[
        ID_COL
    ].is_unique:

        raise AssertionError(
            "OOF row_id 중복"
        )

    oof.to_csv(
        validation_run_dir
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_brier = (
        calculate_brier(
            oof["target"],
            oof["prediction"],
        )
    )

    print(
        f"[완료] seed={seed} | "
        f"{feature_key} | "
        f"Pooled={pooled_brier:.8f}"
    )

    return (
        pd.DataFrame(
            fold_rows
        ),
        oof,
        pooled_brier,
    )


# =========================================================
# 동일 seed Basic vs Offspeed 비교
# =========================================================

def compare_seed(
    seed,
    basic_oof,
    offspeed_oof,
):

    basic = (
        basic_oof[
            [
                ID_COL,
                "validation_season",
                "target",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": (
                    "basic_prediction"
                )
            }
        )
    )

    offspeed = (
        offspeed_oof[
            [
                ID_COL,
                "validation_season",
                "target",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": (
                    "offspeed_prediction"
                )
            }
        )
    )

    merged = basic.merge(
        offspeed[
            [
                ID_COL,
                "offspeed_prediction",
            ]
        ],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(basic):
        raise AssertionError(
            f"seed={seed}: "
            "OOF row 수 불일치"
        )

    merged[
        "basic_se"
    ] = (
        merged[
            "basic_prediction"
        ]
        - merged["target"]
    ) ** 2

    merged[
        "offspeed_se"
    ] = (
        merged[
            "offspeed_prediction"
        ]
        - merged["target"]
    ) ** 2

    rows = []

    for season in [
        2021,
        2022,
        2023,
        2024,
    ]:

        part = merged.loc[
            merged[
                "validation_season"
            ]
            == season
        ]

        basic_brier = float(
            part[
                "basic_se"
            ].mean()
        )

        offspeed_brier = float(
            part[
                "offspeed_se"
            ].mean()
        )

        delta = float(
            basic_brier
            - offspeed_brier
        )

        rows.append(
            {
                "seed": int(seed),
                "validation_season": (
                    season
                ),
                "basic_brier": (
                    basic_brier
                ),
                "offspeed_brier": (
                    offspeed_brier
                ),
                "delta_brier": (
                    delta
                ),
                "offspeed_better": bool(
                    delta > 0
                ),
            }
        )

    basic_pooled = float(
        merged[
            "basic_se"
        ].mean()
    )

    offspeed_pooled = float(
        merged[
            "offspeed_se"
        ].mean()
    )

    pooled_delta = float(
        basic_pooled
        - offspeed_pooled
    )

    seed_summary = {
        "seed": int(seed),
        "basic_pooled_brier": (
            basic_pooled
        ),
        "offspeed_pooled_brier": (
            offspeed_pooled
        ),
        "delta_brier": (
            pooled_delta
        ),
        "offspeed_better": bool(
            pooled_delta > 0
        ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        seed_summary,
    )


# =========================================================
# Main
# =========================================================

def main():

    experiment_model_dir = (
        MODEL_DIR
        / EXPERIMENT_NAME
    )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 Validation 존재: "
            f"{VALIDATION_DIR}"
        )

    if experiment_model_dir.exists():
        raise FileExistsError(
            f"기존 Model 존재: "
            f"{experiment_model_dir}"
        )

    # -----------------------------------------------------
    # Raw Feature
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    raw_features = [
        col
        for col in test_cols
        if col != ID_COL
    ]

    if len(raw_features) != 47:
        raise ValueError(
            "Raw Feature != 47: "
            f"{len(raw_features)}"
        )

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

    before = train.copy()

    train = (
        add_offspeed_count_features(
            train
        )
    )

    validate_features(
        before_df=before,
        after_df=train,
        raw_features=raw_features,
    )

    del before

    gc.collect()

    # -----------------------------------------------------
    # 디렉터리
    # -----------------------------------------------------

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

    all_fold_comparisons = []

    all_seed_summaries = []

    # -----------------------------------------------------
    # Seed 42 기존 결과 재사용
    # -----------------------------------------------------

    print()
    print("=" * 90)
    print(
        "seed=42 기존 결과 재사용"
    )
    print("=" * 90)

    (
        seed42_basic,
        seed42_offspeed,
    ) = load_seed42_oof()

    (
        fold_compare,
        seed_summary,
    ) = compare_seed(
        seed=42,
        basic_oof=seed42_basic,
        offspeed_oof=seed42_offspeed,
    )

    fold_compare[
        "source"
    ] = "reused"

    all_fold_comparisons.append(
        fold_compare
    )

    seed_summary[
        "source"
    ] = "reused"

    all_seed_summaries.append(
        seed_summary
    )

    print(
        "seed=42 | "
        f"Basic="
        f"{seed_summary['basic_pooled_brier']:.8f} | "
        f"Offspeed="
        f"{seed_summary['offspeed_pooled_brier']:.8f} | "
        f"Delta="
        f"{seed_summary['delta_brier']:+.8f}"
    )

    # -----------------------------------------------------
    # 나머지 Seed 학습
    # -----------------------------------------------------

    for seed in [
        7,
        2026,
        123,
        999,
    ]:

        basic_fold_df, basic_oof, _ = (
            run_one_model(
                train=train,
                raw_features=raw_features,
                seed=seed,
                feature_key="basic",
                extra_features=[],
            )
        )

        (
            offspeed_fold_df,
            offspeed_oof,
            _,
        ) = run_one_model(
            train=train,
            raw_features=raw_features,
            seed=seed,
            feature_key="offspeed_count_2",
            extra_features=(
                OFFSPEED_COUNT_FEATURES
            ),
        )

        fold_compare, seed_summary = (
            compare_seed(
                seed=seed,
                basic_oof=basic_oof,
                offspeed_oof=(
                    offspeed_oof
                ),
            )
        )

        fold_compare[
            "source"
        ] = "trained"

        all_fold_comparisons.append(
            fold_compare
        )

        seed_summary[
            "source"
        ] = "trained"

        all_seed_summaries.append(
            seed_summary
        )

        del (
            basic_fold_df,
            basic_oof,
            offspeed_fold_df,
            offspeed_oof,
        )

        gc.collect()

    # -----------------------------------------------------
    # 전체 결과
    # -----------------------------------------------------

    fold_summary = pd.concat(
        all_fold_comparisons,
        ignore_index=True,
    )

    seed_summary_df = pd.DataFrame(
        all_seed_summaries
    )

    fold_summary.to_csv(
        VALIDATION_DIR
        / "fold_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_summary_df.to_csv(
        VALIDATION_DIR
        / "seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 전체 안정성
    # -----------------------------------------------------

    deltas = (
        seed_summary_df[
            "delta_brier"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    stability = {
        "n_seeds": int(
            len(
                seed_summary_df
            )
        ),
        "offspeed_win_count": int(
            seed_summary_df[
                "offspeed_better"
            ].sum()
        ),
        "offspeed_loss_count": int(
            (
                ~seed_summary_df[
                    "offspeed_better"
                ]
            ).sum()
        ),
        "mean_basic_brier": float(
            seed_summary_df[
                "basic_pooled_brier"
            ].mean()
        ),
        "mean_offspeed_brier": float(
            seed_summary_df[
                "offspeed_pooled_brier"
            ].mean()
        ),
        "mean_delta": float(
            deltas.mean()
        ),
        "std_delta": float(
            deltas.std(
                ddof=0
            )
        ),
        "min_delta": float(
            deltas.min()
        ),
        "max_delta": float(
            deltas.max()
        ),
        "median_delta": float(
            np.median(
                deltas
            )
        ),
    }

    # -----------------------------------------------------
    # 시즌별 Seed 안정성
    # -----------------------------------------------------

    season_summary = (
        fold_summary
        .groupby(
            "validation_season",
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
            mean_offspeed_brier=(
                "offspeed_brier",
                "mean",
            ),
            mean_delta=(
                "delta_brier",
                "mean",
            ),
            std_delta=(
                "delta_brier",
                "std",
            ),
            offspeed_win_count=(
                "offspeed_better",
                "sum",
            ),
        )
    )

    season_summary.to_csv(
        VALIDATION_DIR
        / "season_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        VALIDATION_DIR
        / "stability_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            stability,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # 콘솔 출력
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "Offspeed × Count Multi-Seed"
    )
    print("=" * 100)

    for _, row in (
        seed_summary_df
        .sort_values("seed")
        .iterrows()
    ):

        direction = (
            "개선"
            if row[
                "delta_brier"
            ] > 0
            else "악화"
        )

        print(
            f"seed={int(row['seed']):4d} | "
            f"Basic="
            f"{row['basic_pooled_brier']:.8f} | "
            f"Offspeed="
            f"{row['offspeed_pooled_brier']:.8f} | "
            f"Delta="
            f"{row['delta_brier']:+.8f} | "
            f"{direction}"
        )

    print("=" * 100)

    print(
        "Win count     :",
        stability[
            "offspeed_win_count"
        ],
        "/",
        stability[
            "n_seeds"
        ],
    )

    print(
        "Mean Basic    :",
        f"{stability['mean_basic_brier']:.8f}",
    )

    print(
        "Mean Offspeed :",
        f"{stability['mean_offspeed_brier']:.8f}",
    )

    print(
        "Mean Delta    :",
        f"{stability['mean_delta']:+.8f}",
    )

    print(
        "Delta Std     :",
        f"{stability['std_delta']:.8f}",
    )

    print()
    print(
        "Season별 안정성"
    )

    print(
        season_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "결과 저장:",
        VALIDATION_DIR,
    )


if __name__ == "__main__":
    main()