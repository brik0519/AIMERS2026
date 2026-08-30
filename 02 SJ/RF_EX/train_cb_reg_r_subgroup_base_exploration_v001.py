# train_cb_reg_r_subgroup_base_exploration_v001.py
# ============================================================
# R Internal Subgroup Base Exploration
#
# 목적
# ----
# R 전체를 specialist로 처리하는 방식은 temporal generalization이 약했다.
# Residual discovery에서는 R 내부의 특정 상황에서 반복적인 오차가 발견됐다.
#
# 따라서 Base 단계에서 R 전체가 아니라 R 내부 subgroup만 specialist 처리했을 때
# 실제로 안정적인 개선이 생기는지 확인한다.
#
# 비교
# ----
# Control:
#   Global model
#   - train: fold train 전체 R+F
#   - ES: fold validation 전체
#
# 각 R_CORE candidate마다 두 Treatment를 비교:
#
# 1) Shared-Train CORE-ES
#   - train: fold train 전체 R+F
#   - ES: 해당 R_CORE validation row만
#   - routing: R_CORE만 이 모델, 나머지는 Global
#
# 2) CORE-Only Specialist
#   - train: 해당 R_CORE train row만
#   - ES: 해당 R_CORE validation row만
#   - routing: R_CORE만 이 모델, 나머지는 Global
#
# Candidate subgroup
# ------------------
# CORE_A:
#   R & hand_matchup == "1_1"
#
# CORE_B:
#   R & hand_matchup == "1_1"
#     & dominant_pitchmix == "fastball"
#
# CORE_C:
#   R & hand_matchup == "1_1"
#     & base_state == "___"
#
# CORE_D:
#   R & hand_matchup == "1_1"
#     & outs_before == 1
#
# 주의
# ----
# 이 subgroup들은 앞선 2021~2023 residual discovery에서 나온
# 단순/고표본 candidate를 기반으로 한다.
#
# 2024 residual 결과도 이미 확인한 상태이므로,
# 이번 실험에서 2024를 "완전한 untouched holdout"이라고 부르지는 않는다.
# 따라서 최종 판단은:
#   - fold별 방향 일관성
#   - pooled gain
#   - 2021/2022/2023/2024 안정성
# 을 함께 본다.
#
# 대회 규칙
# ---------
# 모든 subgroup은 현재 row 자체의 game_type / hand / base_state /
# outs / 공식 asof pitchmix rate만 이용해 정의한다.
# 다른 test row, test 집계, rolling, lag, test 분포는 사용하지 않는다.
# ============================================================

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Experiment
# ============================================================

EXPERIMENT_NAME = "cb_reg_r_subgroup_base_exploration_v001"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_DIR = (
    Path("./validation")
    / EXPERIMENT_NAME
)

MODEL_DIR = (
    Path("./model")
    / EXPERIMENT_NAME
)


# ============================================================
# Columns
# ============================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"

EXCLUDE_FEATURES = [
    "asof_pitcher_pitchmix_n",
]

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


# ============================================================
# Folds
# ============================================================

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
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "validation_season": 2024,
    },
]


# ============================================================
# HB-EB
# ============================================================

HBEB_CONFIGS = [
    {
        "group_cols": ["pitcher_hand"],
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "new_col": "hbeb_pitcher_success_rate",
    },
    {
        "group_cols": ["batter_hand"],
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "new_col": "hbeb_batter_success_rate",
    },
]


# ============================================================
# CatBoost
# ============================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 0,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# ============================================================
# JSON helper
# ============================================================

def to_python(obj):
    if isinstance(obj, dict):
        return {
            str(k): to_python(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            to_python(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    rate = pd.to_numeric(
        df_train[rate_col],
        errors="coerce",
    )

    g_mu = float(rate.mean())
    g_var = float(rate.var())

    max_bernoulli_var = (
        g_mu * (1.0 - g_mu)
    )

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = (
            max_bernoulli_var * 0.5
        )

    g_m = (
        max_bernoulli_var / g_var
    ) - 1.0

    g_alpha = (
        g_mu * g_m
    )

    g_beta = (
        (1.0 - g_mu) * g_m
    )

    temp = (
        df_train[
            group_cols + [rate_col]
        ]
        .copy()
    )

    temp[rate_col] = pd.to_numeric(
        temp[rate_col],
        errors="coerce",
    )

    stats = (
        temp.groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            [
                "mean",
                "var",
                "count",
            ]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0 - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(stats["var"])
        & (stats["var"] > 0.0)
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"] >= 30
        )
    )

    stats["alpha"] = (
        g_alpha
    )

    stats["beta"] = (
        g_beta
    )

    if valid_mask.any():
        mu = stats.loc[
            valid_mask,
            "mean",
        ]

        var = stats.loc[
            valid_mask,
            "var",
        ]

        m = (
            mu
            * (
                1.0 - mu
            )
            / var
        ) - 1.0

        stats.loc[
            valid_mask,
            "alpha",
        ] = (
            mu * m
        )

        stats.loc[
            valid_mask,
            "beta",
        ] = (
            (1.0 - mu) * m
        )

    prior_df = stats[
        group_cols
        + [
            "alpha",
            "beta",
        ]
    ].copy()

    return (
        prior_df,
        (
            float(g_alpha),
            float(g_beta),
        ),
    )


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    original_index = (
        df.index
    )

    lookup_df = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup_df["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup_df["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n * rate
    )

    denominator = (
        n + alpha + beta
    )

    eb_rate = np.divide(
        success_count + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=(
            denominator > 0.0
        ),
    )

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def fit_hbeb_priors(
    df_train,
):
    priors = {}

    for config in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=(
                config["group_cols"]
            ),
            rate_col=(
                config["rate_col"]
            ),
        )

        priors[
            config["new_col"]
        ] = {
            "prior_df": (
                prior_df
            ),
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config["group_cols"]
            ),
            "n_col": (
                config["n_col"]
            ),
            "rate_col": (
                config["rate_col"]
            ),
        }

    return priors


def apply_hbeb_with_priors(
    df,
    base_features,
    priors,
):
    transformed = (
        df.copy()
    )

    current_features = (
        list(base_features)
    )

    for config in HBEB_CONFIGS:
        info = (
            priors[
                config["new_col"]
            ]
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=(
                config["group_cols"]
            ),
            n_col=(
                config["n_col"]
            ),
            rate_col=(
                config["rate_col"]
            ),
            prior_df=(
                info["prior_df"]
            ),
            global_prior=(
                info["global_prior"]
            ),
        )

        if (
            config["rate_col"]
            in current_features
        ):
            current_features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in current_features
        ):
            current_features.append(
                config["new_col"]
            )

    return (
        transformed,
        current_features,
    )


# ============================================================
# Feature
# ============================================================

def prepare_features(
    df,
    cat_cols,
):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )

    return df


# ============================================================
# Derived subgroup features
# ============================================================

def add_subgroup_features(
    df,
):
    out = (
        df.copy()
    )

    pitcher_hand = (
        out["pitcher_hand"]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
    )

    batter_hand = (
        out["batter_hand"]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
    )

    out[
        "_hand_matchup"
    ] = (
        pitcher_hand
        + "_"
        + batter_hand
    )

    # dominant pitch mix:
    # 현재 row의 공식 asof rate 3개만 사용
    pitch_cols = {
        "fastball": (
            "asof_pitcher_fastball_rate"
        ),
        "breaking": (
            "asof_pitcher_breaking_rate"
        ),
        "offspeed": (
            "asof_pitcher_offspeed_rate"
        ),
    }

    if all(
        col in out.columns
        for col in pitch_cols.values()
    ):
        mix = pd.DataFrame(
            {
                name: pd.to_numeric(
                    out[col],
                    errors="coerce",
                )
                for name, col
                in pitch_cols.items()
            },
            index=out.index,
        )

        all_missing = (
            mix.isna().all(
                axis=1
            )
        )

        dominant = (
            mix.fillna(
                -np.inf
            )
            .idxmax(
                axis=1
            )
            .astype(str)
        )

        dominant.loc[
            all_missing
        ] = "__MISSING__"

        out[
            "_dominant_pitchmix"
        ] = (
            dominant
        )
    else:
        out[
            "_dominant_pitchmix"
        ] = (
            "__MISSING__"
        )

    return out


# ============================================================
# Subgroup definitions
# ============================================================

def subgroup_mask(
    df,
    subgroup_name,
):
    """
    모든 조건은 현재 row 자체 feature만 사용.
    """
    game_type = (
        df[GROUP_COL]
        .astype(str)
    )

    hand_matchup = (
        df[
            "_hand_matchup"
        ]
        .astype(str)
    )

    core_a = (
        game_type.eq(
            R_GROUP
        )
        & hand_matchup.eq(
            "1_1"
        )
    )

    if (
        subgroup_name
        == "CORE_A_HAND_1_1"
    ):
        return (
            core_a.to_numpy()
        )

    if (
        subgroup_name
        == "CORE_B_HAND_1_1_FASTBALL"
    ):
        return (
            (
                core_a
                & df[
                    "_dominant_pitchmix"
                ]
                .astype(str)
                .eq(
                    "fastball"
                )
            )
            .to_numpy()
        )

    if (
        subgroup_name
        == "CORE_C_HAND_1_1_BASE_EMPTY"
    ):
        return (
            (
                core_a
                & df[
                    "base_state"
                ]
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
                .eq(
                    "___"
                )
            )
            .to_numpy()
        )

    if (
        subgroup_name
        == "CORE_D_HAND_1_1_OUT1"
    ):
        outs = (
            pd.to_numeric(
                df[
                    "outs_before"
                ],
                errors="coerce",
            )
        )

        return (
            (
                core_a
                & outs.eq(1)
            )
            .to_numpy()
        )

    raise ValueError(
        f"Unknown subgroup: "
        f"{subgroup_name}"
    )


SUBGROUPS = [
    "CORE_A_HAND_1_1",
    "CORE_B_HAND_1_1_FASTBALL",
    "CORE_C_HAND_1_1_BASE_EMPTY",
    "CORE_D_HAND_1_1_OUT1",
]


# ============================================================
# Metrics
# ============================================================

def brier(
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
            (
                y_pred - y_true
            ) ** 2
        )
    )


def metrics_for_mask(
    y,
    pred,
    mask,
):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.sum() == 0:
        return {
            "n": 0,
            "brier": None,
        }

    return {
        "n": int(
            mask.sum()
        ),
        "brier": brier(
            np.asarray(y)[mask],
            np.asarray(pred)[mask],
        ),
    }


# ============================================================
# Model fit
# ============================================================

def fit_model(
    X_train,
    y_train,
    X_eval,
    y_eval,
    cat_cols,
):
    model = (
        CatBoostRegressor(
            **MODEL_PARAMS
        )
    )

    start = (
        time.time()
    )

    model.fit(
        X_train,
        y_train,
        cat_features=(
            cat_cols
        ),
        eval_set=(
            X_eval,
            y_eval,
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

    best_iteration = (
        model.get_best_iteration()
    )

    n_iterations = int(
        best_iteration + 1
        if (
            best_iteration
            is not None
            and best_iteration >= 0
        )
        else MODEL_PARAMS[
            "iterations"
        ]
    )

    return (
        model,
        n_iterations,
        elapsed,
    )


# ============================================================
# Fold
# ============================================================

def run_fold(
    train,
    fold_config,
    base_features,
    cat_cols,
):
    fold = int(
        fold_config[
            "fold"
        ]
    )

    train_seasons = list(
        fold_config[
            "train_seasons"
        ]
    )

    valid_season = int(
        fold_config[
            "validation_season"
        ]
    )

    raw_train = (
        train.loc[
            train[
                SEASON_COL
            ].isin(
                train_seasons
            )
        ]
        .copy()
    )

    raw_valid = (
        train.loc[
            train[
                SEASON_COL
            ].eq(
                valid_season
            )
        ]
        .copy()
    )

    priors = (
        fit_hbeb_priors(
            raw_train
        )
    )

    (
        fold_train,
        features,
    ) = apply_hbeb_with_priors(
        raw_train,
        base_features,
        priors,
    )

    (
        fold_valid,
        valid_features,
    ) = apply_hbeb_with_priors(
        raw_valid,
        base_features,
        priors,
    )

    if (
        features
        != valid_features
    ):
        raise ValueError(
            "Feature mismatch"
        )

    # subgroup 정의용 derived column은 model feature에 넣지 않는다.
    fold_train = (
        add_subgroup_features(
            fold_train
        )
    )

    fold_valid = (
        add_subgroup_features(
            fold_valid
        )
    )

    X_train = (
        prepare_features(
            fold_train[
                features
            ],
            cat_cols,
        )
    )

    y_train = (
        fold_train[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    X_valid = (
        prepare_features(
            fold_valid[
                features
            ],
            cat_cols,
        )
    )

    y_valid = (
        fold_valid[
            TARGET
        ]
        .astype(
            np.float64
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        f"Fold {fold} | "
        f"{train_seasons[0]}~"
        f"{train_seasons[-1]} "
        f"-> {valid_season}"
    )

    print(
        "=" * 110
    )

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    (
        global_model,
        global_iter,
        global_elapsed,
    ) = fit_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        cat_cols,
    )

    global_pred = np.clip(
        np.asarray(
            global_model.predict(
                X_valid
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    global_brier = (
        brier(
            y_valid,
            global_pred,
        )
    )

    print(
        f"Global | "
        f"Brier={global_brier:.10f} | "
        f"iter={global_iter}"
    )

    rows = []

    for subgroup in (
        SUBGROUPS
    ):
        train_core_mask = (
            subgroup_mask(
                fold_train,
                subgroup,
            )
        )

        valid_core_mask = (
            subgroup_mask(
                fold_valid,
                subgroup,
            )
        )

        n_train_core = int(
            train_core_mask.sum()
        )

        n_valid_core = int(
            valid_core_mask.sum()
        )

        if (
            n_train_core < 1000
            or n_valid_core < 500
        ):
            print(
                f"{subgroup} skipped | "
                f"train={n_train_core:,}, "
                f"valid={n_valid_core:,}"
            )

            continue

        global_core = (
            metrics_for_mask(
                y_valid,
                global_pred,
                valid_core_mask,
            )
        )

        # ----------------------------------------------------
        # Method 1:
        # Shared Train + CORE-specific ES
        # ----------------------------------------------------

        (
            shared_model,
            shared_iter,
            shared_elapsed,
        ) = fit_model(
            X_train,
            y_train,
            X_valid.loc[
                valid_core_mask
            ],
            y_valid.loc[
                valid_core_mask
            ],
            cat_cols,
        )

        shared_core_pred = np.clip(
            np.asarray(
                shared_model.predict(
                    X_valid.loc[
                        valid_core_mask
                    ]
                ),
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )

        shared_pred = (
            global_pred.copy()
        )

        shared_pred[
            valid_core_mask
        ] = (
            shared_core_pred
        )

        shared_brier = (
            brier(
                y_valid,
                shared_pred,
            )
        )

        shared_core = (
            metrics_for_mask(
                y_valid,
                shared_pred,
                valid_core_mask,
            )
        )

        # ----------------------------------------------------
        # Method 2:
        # CORE-only specialist
        # ----------------------------------------------------

        (
            core_model,
            core_iter,
            core_elapsed,
        ) = fit_model(
            X_train.loc[
                train_core_mask
            ],
            y_train.loc[
                train_core_mask
            ],
            X_valid.loc[
                valid_core_mask
            ],
            y_valid.loc[
                valid_core_mask
            ],
            cat_cols,
        )

        core_only_pred = np.clip(
            np.asarray(
                core_model.predict(
                    X_valid.loc[
                        valid_core_mask
                    ]
                ),
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )

        specialist_pred = (
            global_pred.copy()
        )

        specialist_pred[
            valid_core_mask
        ] = (
            core_only_pred
        )

        specialist_brier = (
            brier(
                y_valid,
                specialist_pred,
            )
        )

        specialist_core = (
            metrics_for_mask(
                y_valid,
                specialist_pred,
                valid_core_mask,
            )
        )

        shared_gain = float(
            global_brier
            - shared_brier
        )

        specialist_gain = float(
            global_brier
            - specialist_brier
        )

        shared_core_gain = float(
            global_core[
                "brier"
            ]
            - shared_core[
                "brier"
            ]
        )

        specialist_core_gain = float(
            global_core[
                "brier"
            ]
            - specialist_core[
                "brier"
            ]
        )

        print(
            f"{subgroup} | "
            f"n={n_valid_core:,}"
        )

        print(
            f"  Shared-CORE-ES | "
            f"Gain={shared_gain:+.10f} | "
            f"CoreGain={shared_core_gain:+.10f} | "
            f"iter={shared_iter}"
        )

        print(
            f"  CORE-only      | "
            f"Gain={specialist_gain:+.10f} | "
            f"CoreGain={specialist_core_gain:+.10f} | "
            f"iter={core_iter}"
        )

        rows.extend(
            [
                {
                    "fold": (
                        fold
                    ),
                    "validation_season": (
                        valid_season
                    ),
                    "subgroup": (
                        subgroup
                    ),
                    "method": (
                        "shared_train_core_es"
                    ),
                    "n_train_core": (
                        n_train_core
                    ),
                    "n_valid_core": (
                        n_valid_core
                    ),
                    "global_iteration": (
                        global_iter
                    ),
                    "specialist_iteration": (
                        shared_iter
                    ),
                    "global_brier": (
                        global_brier
                    ),
                    "treatment_brier": (
                        shared_brier
                    ),
                    "overall_gain": (
                        shared_gain
                    ),
                    "global_core_brier": (
                        global_core[
                            "brier"
                        ]
                    ),
                    "treatment_core_brier": (
                        shared_core[
                            "brier"
                        ]
                    ),
                    "core_gain": (
                        shared_core_gain
                    ),
                    "elapsed_seconds": (
                        shared_elapsed
                    ),
                },
                {
                    "fold": (
                        fold
                    ),
                    "validation_season": (
                        valid_season
                    ),
                    "subgroup": (
                        subgroup
                    ),
                    "method": (
                        "core_only_specialist"
                    ),
                    "n_train_core": (
                        n_train_core
                    ),
                    "n_valid_core": (
                        n_valid_core
                    ),
                    "global_iteration": (
                        global_iter
                    ),
                    "specialist_iteration": (
                        core_iter
                    ),
                    "global_brier": (
                        global_brier
                    ),
                    "treatment_brier": (
                        specialist_brier
                    ),
                    "overall_gain": (
                        specialist_gain
                    ),
                    "global_core_brier": (
                        global_core[
                            "brier"
                        ]
                    ),
                    "treatment_core_brier": (
                        specialist_core[
                            "brier"
                        ]
                    ),
                    "core_gain": (
                        specialist_core_gain
                    ),
                    "elapsed_seconds": (
                        core_elapsed
                    ),
                },
            ]
        )

    return (
        rows,
        {
            "fold": fold,
            "validation_season": (
                valid_season
            ),
            "global_brier": (
                global_brier
            ),
            "global_iteration": (
                global_iter
            ),
            "global_elapsed_seconds": (
                global_elapsed
            ),
        },
        features,
    )


# ============================================================
# Summary
# ============================================================

def summarize_results(
    result_df,
):
    summaries = []

    for (
        subgroup,
        method,
    ), group in result_df.groupby(
        [
            "subgroup",
            "method",
        ]
    ):
        group = (
            group.sort_values(
                "validation_season"
            )
        )

        gains = (
            group[
                "overall_gain"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        core_gains = (
            group[
                "core_gain"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # pooled equivalent:
        # fold별 overall Brier의 단순 평균이 아니라,
        # 이 파일에서는 우선 season별 gain 안정성을 비교하는 목적.
        # 각 fold row 수가 유사하므로 mean gain도 같이 제공.
        summaries.append(
            {
                "subgroup": (
                    subgroup
                ),
                "method": (
                    method
                ),
                "fold_wins": int(
                    (gains > 0.0)
                    .sum()
                ),
                "fold_losses": int(
                    (gains < 0.0)
                    .sum()
                ),
                "fold_ties": int(
                    (gains == 0.0)
                    .sum()
                ),
                "mean_overall_gain": float(
                    gains.mean()
                ),
                "min_overall_gain": float(
                    gains.min()
                ),
                "max_overall_gain": float(
                    gains.max()
                ),
                "mean_core_gain": float(
                    core_gains.mean()
                ),
                "gain_2021": (
                    float(
                        group.loc[
                            group[
                                "validation_season"
                            ].eq(
                                2021
                            ),
                            "overall_gain",
                        ].iloc[0]
                    )
                    if (
                        group[
                            "validation_season"
                        ]
                        .eq(2021)
                        .any()
                    )
                    else None
                ),
                "gain_2022": (
                    float(
                        group.loc[
                            group[
                                "validation_season"
                            ].eq(
                                2022
                            ),
                            "overall_gain",
                        ].iloc[0]
                    )
                    if (
                        group[
                            "validation_season"
                        ]
                        .eq(2022)
                        .any()
                    )
                    else None
                ),
                "gain_2023": (
                    float(
                        group.loc[
                            group[
                                "validation_season"
                            ].eq(
                                2023
                            ),
                            "overall_gain",
                        ].iloc[0]
                    )
                    if (
                        group[
                            "validation_season"
                        ]
                        .eq(2023)
                        .any()
                    )
                    else None
                ),
                "gain_2024": (
                    float(
                        group.loc[
                            group[
                                "validation_season"
                            ].eq(
                                2024
                            ),
                            "overall_gain",
                        ].iloc[0]
                    )
                    if (
                        group[
                            "validation_season"
                        ]
                        .eq(2024)
                        .any()
                    )
                    else None
                ),
                "iterations": (
                    ",".join(
                        str(int(v))
                        for v in group[
                            "specialist_iteration"
                        ].tolist()
                    )
                ),
            }
        )

    summary_df = (
        pd.DataFrame(
            summaries
        )
        .sort_values(
            [
                "fold_wins",
                "min_overall_gain",
                "mean_overall_gain",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return (
        summary_df
    )


# ============================================================
# Main
# ============================================================

def main():
    if (
        VALIDATION_DIR.exists()
        or MODEL_DIR.exists()
    ):
        raise FileExistsError(
            "Experiment output already exists. "
            "Change EXPERIMENT_NAME."
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    base_features = [
        col
        for col in test_cols
        if (
            col != ID
            and col
            not in EXCLUDE_FEATURES
        )
    ]

    required_subgroup_cols = [
        GROUP_COL,
        "pitcher_hand",
        "batter_hand",
        "base_state",
        "outs_before",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]

    missing_required = [
        col
        for col in (
            required_subgroup_cols
        )
        if col not in base_features
    ]

    if missing_required:
        raise ValueError(
            "Subgroup feature missing: "
            f"{missing_required}"
        )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in base_features
    ]

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + base_features
            + [TARGET]
        ),
        low_memory=False,
    )

    print(
        "=" * 110
    )

    print(
        "R Internal Subgroup Base Exploration"
    )

    print(
        "=" * 110
    )

    print(
        "Control: Global"
    )

    print(
        "Methods:"
    )

    print(
        "  1) shared_train_core_es"
    )

    print(
        "  2) core_only_specialist"
    )

    print(
        "Subgroups:"
    )

    for subgroup in (
        SUBGROUPS
    ):
        print(
            f"  - {subgroup}"
        )

    print(
        "=" * 110
    )

    all_rows = []
    fold_meta = []
    final_features = None

    for fold_config in (
        EXPANDING_FOLDS
    ):
        (
            rows,
            meta,
            features,
        ) = run_fold(
            train,
            fold_config,
            base_features,
            cat_cols,
        )

        all_rows.extend(
            rows
        )

        fold_meta.append(
            meta
        )

        if (
            final_features
            is None
        ):
            final_features = (
                list(features)
            )
        elif (
            final_features
            != features
        ):
            raise ValueError(
                "Fold feature mismatch"
            )

    result_df = pd.DataFrame(
        all_rows
    )

    result_df.to_csv(
        VALIDATION_DIR
        / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = (
        summarize_results(
            result_df
        )
    )

    summary_df.to_csv(
        VALIDATION_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base": (
            "cb_reg_v003_test-style Global"
        ),
        "subgroups": (
            SUBGROUPS
        ),
        "methods": [
            "shared_train_core_es",
            "core_only_specialist",
        ],
        "fold_meta": (
            fold_meta
        ),
        "features": (
            final_features
        ),
        "categorical_features": (
            cat_cols
        ),
        "model_params": (
            MODEL_PARAMS
        ),
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "important_note": (
            "2024 residual holdout has already been inspected in prior analysis. "
            "Therefore 2024 is not treated as a pristine untouched holdout here. "
            "Judge subgroup base value mainly by cross-season consistency."
        ),
    }

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_python(
                metadata
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    shutil.copy2(
        Path(
            __file__
        ).resolve(),
        MODEL_DIR
        / "train.py",
    )

    print()
    print(
        "=" * 110
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 110
    )

    display_cols = [
        "subgroup",
        "method",
        "fold_wins",
        "fold_losses",
        "fold_ties",
        "mean_overall_gain",
        "min_overall_gain",
        "gain_2021",
        "gain_2022",
        "gain_2023",
        "gain_2024",
        "iterations",
    ]

    print(
        summary_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "판정 기준:"
    )

    print(
        "- 4/4 또는 3/4 wins + 큰 음수 fold 없음 "
        "=> Base subgroup 후보"
    )

    print(
        "- 2023만 크게 +이고 다른 시즌 0/- "
        "=> 2023 regime-specific 가능성, Base 채택 보류"
    )

    print(
        "- CORE-only가 불안정하고 Shared-CORE-ES가 안정적 "
        "=> 데이터를 자르지 말고 shared training 유지"
    )

    print(
        "- 둘 다 불안정 "
        "=> Base는 Global로 확정하고 residual correction으로 이동"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()
