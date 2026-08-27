# train.py
# cb_reg_v019_asof_residual

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "cb_reg_v019_asof_residual"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# ============================================================
# 기본 컬럼
# ============================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

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
# Exact-ASOF
# ============================================================

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
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
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# ============================================================
# Conditional Residual Correction 설정
# ============================================================

MIN_N_PARENT = 200
MIN_N_DETAIL = 200

SHRINK_PARENT = 300.0
SHRINK_DETAIL = 100.0

PARENT_ONLY_SCALE = 0.5
PARENT_WEIGHT = 0.5
DETAIL_WEIGHT = 0.5

ALPHA = 0.25
CAP = 0.02


# ============================================================
# 공통
# ============================================================

def prepare_features(df, cat_cols):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return df


def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def calculate_bss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    brier = brier_score(y_true, y_pred)
    p = float(np.mean(y_true))
    baseline = p * (1.0 - p)

    if baseline <= 0.0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier / baseline
        )
    )


# ============================================================
# Exact-ASOF용 pitcher × season 정적 통계
# ============================================================

def build_pitcher_season_stats(df_train):
    """
    공식 train label만으로 pitcher × season 완결 통계를 만든다.

    Exact-ASOF 계산 시에는 target row의 season보다
    과거 season만 사용한다.
    """

    required = {
        "pitcher_id",
        SEASON_COL,
        TARGET,
    }

    missing = sorted(
        required - set(df_train.columns)
    )

    if missing:
        raise KeyError(
            f"pitcher season stats 필수 컬럼 없음: {missing}"
        )

    stats = (
        df_train
        .groupby(
            ["pitcher_id", SEASON_COL],
            dropna=False,
            sort=False,
        )[TARGET]
        .agg(
            n="size",
            success="sum",
        )
        .reset_index()
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats["n"] = (
        stats["n"]
        .astype(np.float64)
    )

    stats["success"] = (
        stats["success"]
        .astype(np.float64)
    )

    return stats


def add_pitcher_exact_asof_features(
    df,
    pitcher_season_stats,
):
    """
    해당 row의 official cumulative ASOF에서
    과거 완결 시즌 누적치를 빼서
    현재 시즌 시점 직전 상태를 복원한다.

    다른 validation/test row는 사용하지 않는다.
    """

    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 컬럼 없음: {missing}"
        )

    out = df.copy()

    for col in EXACT_FEATURES:
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    diagnostics = []

    for target_year in sorted(
        seasons.unique()
    ):
        row_mask = (
            seasons == int(target_year)
        )

        rows = out.loc[row_mask]

        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist) > 0:
            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[["n", "success"]]
                .sum()
            )

            hist_n = (
                rows["pitcher_id"]
                .map(hist_agg["n"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

            hist_success = (
                rows["pitcher_id"]
                .map(hist_agg["success"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            hist_n = np.zeros(
                len(rows),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(rows),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                rows["asof_pitcher_n"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64)
        )

        asof_rate = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(dtype=np.float64)
        )

        asof_success = (
            asof_n * asof_rate
        )

        raw_current_n = (
            asof_n - hist_n
        )

        current_n = np.maximum(
            raw_current_n,
            0.0,
        )

        raw_current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                raw_current_success
            ),
            np.clip(
                raw_current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_current = (
            (current_n > 0.0)
            & np.isfinite(
                current_success
            )
        )

        current_rate[
            valid_current
        ] = (
            current_success[
                valid_current
            ]
            / current_n[
                valid_current
            ]
        )

        past_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_past = (
            hist_n > 0.0
        )

        past_rate[
            valid_past
        ] = (
            hist_success[
                valid_past
            ]
            / hist_n[
                valid_past
            ]
        )

        idx = rows.index

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = current_n

        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = np.log1p(
            current_n
        )

        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = current_rate

        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = past_rate

        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = (
            current_rate
            - past_rate
        )

        diagnostics.append(
            {
                "season": int(target_year),
                "n_rows": int(len(rows)),
                "negative_current_n": int(
                    (
                        raw_current_n
                        < -1e-8
                    ).sum()
                ),
                "current_rate_available": int(
                    np.isfinite(
                        current_rate
                    ).sum()
                ),
                "past_rate_available": int(
                    np.isfinite(
                        past_rate
                    ).sum()
                ),
            }
        )

    return (
        out,
        pd.DataFrame(diagnostics),
    )


# ============================================================
# HB-EB
# ============================================================

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    train subset만으로 HB-EB prior를 추정한다.
    """

    g_mu = float(
        df_train[rate_col].mean()
    )

    g_var = float(
        df_train[rate_col].var()
    )

    max_bernoulli_var = (
        g_mu * (1.0 - g_mu)
    )

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = (
            max_bernoulli_var
            * 0.5
        )

    g_m = (
        max_bernoulli_var
        / g_var
    ) - 1.0

    g_alpha = (
        g_mu * g_m
    )

    g_beta = (
        (1.0 - g_mu)
        * g_m
    )

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            ["mean", "var", "count"]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0
            - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(
            stats["var"]
        )
        & (
            stats["var"] > 0.0
        )
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"] >= 30
        )
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

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
            * (1.0 - mu)
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
            (1.0 - mu)
            * m
        )

    prior_df = stats[
        group_cols
        + ["alpha", "beta"]
    ].copy()

    global_prior = (
        float(g_alpha),
        float(g_beta),
    )

    return (
        prior_df,
        global_prior,
    )


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    lookup = (
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
        lookup["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(dtype=np.float64)
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .to_numpy(dtype=np.float64)
    )

    success_count = (
        n * rate
    )

    denominator = (
        n + alpha + beta
    )

    return np.divide(
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


def apply_hbeb_pair(
    df_train,
    df_apply,
    base_features,
):
    """
    df_train만으로 prior를 만들고
    df_apply에 적용한다.
    """

    transformed = (
        df_apply.copy()
    )

    features = list(
        base_features
    )

    priors = {}

    for config in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config[
                "group_cols"
            ],
            rate_col=config[
                "rate_col"
            ],
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=config[
                "group_cols"
            ],
            n_col=config[
                "n_col"
            ],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            config["rate_col"]
            in features
        ):
            features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in features
        ):
            features.append(
                config["new_col"]
            )

        priors[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config[
                    "group_cols"
                ]
            ),
            "n_col": config[
                "n_col"
            ],
            "rate_col": config[
                "rate_col"
            ],
        }

    return (
        transformed,
        features,
        priors,
    )


# ============================================================
# pressure state
# ============================================================

def make_pressure_state(
    balls,
    strikes,
):
    balls = pd.to_numeric(
        balls,
        errors="coerce",
    )

    strikes = pd.to_numeric(
        strikes,
        errors="coerce",
    )

    full_mask = (
        (balls == 3)
        & (strikes == 2)
    )

    high_mask = (
        (
            (balls == 3)
            | (strikes == 2)
        )
        & (~full_mask)
    )

    result = np.full(
        len(balls),
        "normal",
        dtype=object,
    )

    result[
        high_mask.to_numpy()
    ] = "high"

    result[
        full_mask.to_numpy()
    ] = "full"

    return result


# ============================================================
# Residual table
# ============================================================

def build_effect_table(
    source_df,
    group_cols,
    min_n,
    shrink_k,
):
    grouped = (
        source_df
        .groupby(
            group_cols,
            dropna=False,
            sort=False,
        )[
            "centered_residual"
        ]
        .agg(
            n="size",
            mean_residual="mean",
        )
        .reset_index()
    )

    grouped = (
        grouped.loc[
            grouped["n"]
            >= int(min_n)
        ]
        .copy()
    )

    grouped[
        "reliability"
    ] = (
        grouped["n"]
        / (
            grouped["n"]
            + float(shrink_k)
        )
    )

    grouped["effect"] = (
        grouped[
            "mean_residual"
        ]
        * grouped[
            "reliability"
        ]
    )

    return grouped


def build_final_residual_artifact(
    oof_2024,
):
    """
    2024 Exact-ASOF OOF residual로
    최종 train-only correction table 생성.
    """

    source = (
        oof_2024.copy()
    )

    source["residual"] = (
        source[TARGET]
        - source["prediction"]
    )

    domain_mean = (
        source
        .groupby(
            GAME_TYPE_COL,
            dropna=False,
        )["residual"]
        .transform("mean")
    )

    source[
        "centered_residual"
    ] = (
        source["residual"]
        - domain_mean
    )

    source[
        "pressure_state"
    ] = make_pressure_state(
        source["balls_before"],
        source["strikes_before"],
    )

    parent_cols = [
        "pitcher_id",
        "batter_hand",
    ]

    detail_cols = [
        "pitcher_id",
        "pressure_state",
        "batter_hand",
    ]

    parent_table = (
        build_effect_table(
            source_df=source,
            group_cols=parent_cols,
            min_n=MIN_N_PARENT,
            shrink_k=SHRINK_PARENT,
        )
    )

    detail_table = (
        build_effect_table(
            source_df=source,
            group_cols=detail_cols,
            min_n=MIN_N_DETAIL,
            shrink_k=SHRINK_DETAIL,
        )
    )

    return {
        "enabled": True,
        "source_validation_season": 2024,
        "parent_group_cols": parent_cols,
        "detail_group_cols": detail_cols,
        "parent_table": parent_table,
        "detail_table": detail_table,
        "min_n_parent": MIN_N_PARENT,
        "min_n_detail": MIN_N_DETAIL,
        "shrink_parent": SHRINK_PARENT,
        "shrink_detail": SHRINK_DETAIL,
        "parent_only_scale": PARENT_ONLY_SCALE,
        "parent_weight": PARENT_WEIGHT,
        "detail_weight": DETAIL_WEIGHT,
        "alpha": ALPHA,
        "cap": CAP,
        "pressure_definition": {
            "full": (
                "balls_before == 3 "
                "and strikes_before == 2"
            ),
            "high": (
                "balls_before == 3 "
                "or strikes_before == 2, "
                "excluding full"
            ),
            "normal": "otherwise",
        },
        "uses_test_distribution": False,
        "row_independent_application": True,
    }


# ============================================================
# Expanding Window
# ============================================================

def run_expanding_validation(
    train,
    base_features,
):
    fold_results = []
    oof_frames = []
    exact_diag_frames = []

    for config in EXPANDING_FOLDS:
        fold = int(
            config["fold"]
        )

        train_seasons = (
            config[
                "train_seasons"
            ]
        )

        validation_season = int(
            config[
                "validation_season"
            ]
        )

        train_mask = (
            train[SEASON_COL]
            .isin(train_seasons)
        )

        val_mask = (
            train[SEASON_COL]
            == validation_season
        )

        df_train = (
            train.loc[
                train_mask
            ]
            .copy()
        )

        df_val = (
            train.loc[
                val_mask
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Exact-ASOF
        # ----------------------------------------------------

        pitcher_season_stats = (
            build_pitcher_season_stats(
                df_train
            )
        )

        (
            train_exact,
            train_diag,
        ) = (
            add_pitcher_exact_asof_features(
                df_train,
                pitcher_season_stats,
            )
        )

        (
            val_exact,
            val_diag,
        ) = (
            add_pitcher_exact_asof_features(
                df_val,
                pitcher_season_stats,
            )
        )

        train_diag[
            "scope"
        ] = "train"

        val_diag[
            "scope"
        ] = "validation"

        diag = pd.concat(
            [train_diag, val_diag],
            ignore_index=True,
        )

        diag["fold"] = fold
        diag[
            "validation_season"
        ] = validation_season

        exact_diag_frames.append(
            diag
        )

        exact_features = list(
            base_features
        )

        for col in EXACT_FEATURES:
            if col not in exact_features:
                exact_features.append(col)

        # ----------------------------------------------------
        # HB-EB
        # ----------------------------------------------------

        (
            train_final,
            final_features,
            _,
        ) = apply_hbeb_pair(
            df_train=train_exact,
            df_apply=train_exact,
            base_features=exact_features,
        )

        (
            val_final,
            val_features,
            _,
        ) = apply_hbeb_pair(
            df_train=train_exact,
            df_apply=val_exact,
            base_features=exact_features,
        )

        if (
            final_features
            != val_features
        ):
            raise ValueError(
                "train/validation feature 목록 불일치"
            )

        cat_cols = [
            c
            for c in CAT_COLS
            if c in final_features
        ]

        X_train = prepare_features(
            train_final[
                final_features
            ],
            cat_cols,
        )

        y_train = (
            train_final[
                TARGET
            ]
            .astype(np.float64)
        )

        X_val = prepare_features(
            val_final[
                final_features
            ],
            cat_cols,
        )

        y_val = (
            val_final[
                TARGET
            ]
            .astype(np.float64)
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]}"
            f" -> {validation_season}"
        )
        print(
            f"Features: {len(final_features)}"
        )
        print("=" * 90)

        estimator = (
            CatBoostRegressor(
                **MODEL_PARAMS
            )
        )

        start = time.time()

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

        elapsed = (
            time.time()
            - start
        )

        pred_raw = np.asarray(
            estimator.predict(
                X_val
            ),
            dtype=np.float64,
        )

        pred = np.clip(
            pred_raw,
            0.0,
            1.0,
        )

        brier = brier_score(
            y_val,
            pred,
        )

        bss = calculate_bss(
            y_val,
            pred,
        )

        best_iteration = (
            estimator
            .get_best_iteration()
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
            final_iterations = (
                int(best_iteration) + 1
            )

        print(
            f"Brier: {brier:.10f}"
        )
        print(
            f"BSS  : {bss:.2f}"
        )
        print(
            f"iter : {final_iterations}"
        )
        print(
            f"time : {elapsed:.1f}s"
        )

        fold_results.append(
            {
                "fold": fold,
                "train_seasons": list(
                    map(
                        int,
                        train_seasons,
                    )
                ),
                "validation_season": (
                    validation_season
                ),
                "n_train": int(
                    len(df_train)
                ),
                "n_validation": int(
                    len(df_val)
                ),
                "brier": float(brier),
                "bss": float(bss),
                "best_iteration": (
                    int(best_iteration)
                ),
                "final_iterations": (
                    int(final_iterations)
                ),
                "elapsed_seconds": float(
                    elapsed
                ),
            }
        )

        oof_fold = pd.DataFrame(
            {
                ID: (
                    df_val[ID]
                    .to_numpy()
                ),
                TARGET: (
                    y_val.to_numpy()
                ),
                "prediction_raw": (
                    pred_raw
                ),
                "prediction": pred,
                SEASON_COL: (
                    validation_season
                ),
                GAME_TYPE_COL: (
                    df_val[
                        GAME_TYPE_COL
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                "pitcher_id": (
                    df_val[
                        "pitcher_id"
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                "batter_hand": (
                    df_val[
                        "batter_hand"
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                "balls_before": (
                    df_val[
                        "balls_before"
                    ]
                    .to_numpy()
                ),
                "strikes_before": (
                    df_val[
                        "strikes_before"
                    ]
                    .to_numpy()
                ),
            }
        )

        oof_frames.append(
            oof_fold
        )

    fold_results_df = (
        pd.DataFrame(
            fold_results
        )
    )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    exact_diag = pd.concat(
        exact_diag_frames,
        ignore_index=True,
    )

    return (
        fold_results_df,
        oof,
        exact_diag,
    )


# ============================================================
# Full train
# ============================================================

def train_final_model(
    train,
    base_features,
    final_iterations,
):
    """
    전체 train(2019~2024)에서
    Exact-ASOF + HB-EB를 생성하고
    최종 CatBoost를 학습한다.
    """

    pitcher_season_stats = (
        build_pitcher_season_stats(
            train
        )
    )

    (
        train_exact,
        final_exact_diag,
    ) = (
        add_pitcher_exact_asof_features(
            train,
            pitcher_season_stats,
        )
    )

    exact_features = list(
        base_features
    )

    for col in EXACT_FEATURES:
        if col not in exact_features:
            exact_features.append(col)

    (
        train_final,
        final_features,
        hbeb_priors,
    ) = apply_hbeb_pair(
        df_train=train_exact,
        df_apply=train_exact,
        base_features=exact_features,
    )

    cat_cols = [
        c
        for c in CAT_COLS
        if c in final_features
    ]

    X_full = prepare_features(
        train_final[
            final_features
        ],
        cat_cols,
    )

    y_full = (
        train_final[
            TARGET
        ]
        .astype(np.float64)
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    model = (
        CatBoostRegressor(
            **final_params
        )
    )

    print()
    print("=" * 90)
    print(
        "Final full training"
    )
    print(
        f"iterations: {final_iterations}"
    )
    print(
        f"features  : {len(final_features)}"
    )
    print("=" * 90)

    start = time.time()

    model.fit(
        X_full,
        y_full,
        cat_features=cat_cols,
    )

    elapsed = (
        time.time()
        - start
    )

    return {
        "model": model,
        "features": final_features,
        "cat_cols": cat_cols,
        "hbeb_priors": hbeb_priors,
        "pitcher_season_stats": (
            pitcher_season_stats
        ),
        "final_exact_diagnostics": (
            final_exact_diag
        ),
        "final_params": final_params,
        "elapsed_seconds": float(
            elapsed
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation exists: {VALIDATION_DIR}\n"
            "기존 결과를 보존하려면 "
            "EXPERIMENT_NAME을 변경하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"Model exists: {MODEL_DIR}\n"
            "기존 결과를 보존하려면 "
            "EXPERIMENT_NAME을 변경하세요."
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
    # canonical v003 base feature schema
    # --------------------------------------------------------

    test_cols = (
        pd.read_csv(
            TEST_PATH,
            encoding="utf-8-sig",
            nrows=0,
        )
        .columns
        .tolist()
    )

    base_features = [
        c
        for c in test_cols
        if c != ID
    ]

    # canonical v003에서 반드시 유지
    if (
        "asof_pitcher_pitchmix_n"
        not in base_features
    ):
        raise ValueError(
            "asof_pitcher_pitchmix_n이 없습니다. "
            "canonical cb_reg_v003 schema를 확인하세요."
        )

    required_exact_sources = [
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    ]

    missing_exact = [
        c
        for c in required_exact_sources
        if c not in base_features
    ]

    if missing_exact:
        raise ValueError(
            f"Exact-ASOF source 누락: {missing_exact}"
        )

    required_residual_sources = [
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    missing_residual = [
        c
        for c in required_residual_sources
        if c not in base_features
    ]

    if missing_residual:
        raise ValueError(
            f"Residual source 누락: {missing_residual}"
        )

    usecols = list(
        dict.fromkeys(
            [
                ID,
                *base_features,
                TARGET,
            ]
        )
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=usecols,
    )

    if not train[ID].is_unique:
        raise ValueError(
            "train row_id가 unique하지 않습니다."
        )

    if train[TARGET].isna().any():
        raise ValueError(
            "target에 NaN이 있습니다."
        )

    print("=" * 90)
    print(
        f"Experiment: {EXPERIMENT_NAME}"
    )
    print(
        "Base      : canonical cb_reg_v003"
    )
    print(
        "Exact-ASOF: ON"
    )
    print(
        "Residual  : conditional gated correction"
    )
    print(
        "Weighting : OFF"
    )
    print(
        "pitchmix_n retained: True"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Expanding Window
    # --------------------------------------------------------

    (
        fold_results,
        oof,
        exact_diag,
    ) = run_expanding_validation(
        train=train,
        base_features=base_features,
    )

    fold_results.to_csv(
        VALIDATION_DIR
        / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof.to_csv(
        VALIDATION_DIR
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    exact_diag.to_csv(
        VALIDATION_DIR
        / "exact_asof_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_brier = brier_score(
        oof[TARGET],
        oof["prediction"],
    )

    pooled_bss = calculate_bss(
        oof[TARGET],
        oof["prediction"],
    )

    print()
    print("=" * 90)
    print("Exact-ASOF OOF summary")
    print("=" * 90)

    print(
        fold_results[
            [
                "validation_season",
                "brier",
                "bss",
                "final_iterations",
            ]
        ]
        .to_string(index=False)
    )

    print(
        f"\nPooled OOF Brier: "
        f"{pooled_brier:.10f}"
    )

    print(
        f"Pooled OOF BSS  : "
        f"{pooled_bss:.2f}"
    )

    # --------------------------------------------------------
    # 2024 OOF -> final residual table
    # --------------------------------------------------------

    oof_2024 = (
        oof.loc[
            oof[SEASON_COL]
            == 2024
        ]
        .copy()
    )

    if len(oof_2024) == 0:
        raise ValueError(
            "2024 OOF가 없습니다."
        )

    residual_artifact = (
        build_final_residual_artifact(
            oof_2024
        )
    )

    print()
    print("=" * 90)
    print("Final residual tables")
    print("=" * 90)
    print(
        "Parent groups:",
        len(
            residual_artifact[
                "parent_table"
            ]
        ),
    )
    print(
        "Detail groups:",
        len(
            residual_artifact[
                "detail_table"
            ]
        ),
    )

    # --------------------------------------------------------
    # latest fold iteration
    # --------------------------------------------------------

    latest_fold = (
        fold_results
        .sort_values(
            "validation_season"
        )
        .iloc[-1]
    )

    final_iterations = int(
        latest_fold[
            "final_iterations"
        ]
    )

    # --------------------------------------------------------
    # Full model
    # --------------------------------------------------------

    final = train_final_model(
        train=train,
        base_features=base_features,
        final_iterations=final_iterations,
    )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base_model": (
            "cb_reg_v003"
        ),
        "model": final["model"],
        "features": final["features"],
        "cat_cols": final["cat_cols"],
        "hbeb_priors": (
            final["hbeb_priors"]
        ),
        "pitcher_season_stats": (
            final[
                "pitcher_season_stats"
            ]
        ),
        "exact_features": list(
            EXACT_FEATURES
        ),
        "residual_correction": (
            residual_artifact
        ),
        "final_iterations": (
            final_iterations
        ),
        "model_params": (
            final["final_params"]
        ),
        "canonical_pitchmix_retained": (
            True
        ),
        "uses_test_distribution": False,
        "row_independent_inference": True,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "base_model": (
            "cb_reg_v003"
        ),
        "training_seasons": (
            sorted(
                int(x)
                for x
                in train[
                    SEASON_COL
                ].unique()
            )
        ),
        "n_train": int(
            len(train)
        ),
        "base_feature_count": int(
            len(base_features)
        ),
        "final_feature_count": int(
            len(final["features"])
        ),
        "exact_features": list(
            EXACT_FEATURES
        ),
        "final_iterations": int(
            final_iterations
        ),
        "pooled_oof_brier": float(
            pooled_brier
        ),
        "pooled_oof_bss": float(
            pooled_bss
        ),
        "residual_source_validation_season": (
            2024
        ),
        "parent_groups": int(
            len(
                residual_artifact[
                    "parent_table"
                ]
            )
        ),
        "detail_groups": int(
            len(
                residual_artifact[
                    "detail_table"
                ]
            )
        ),
        "residual_config": {
            "min_n_parent": (
                MIN_N_PARENT
            ),
            "min_n_detail": (
                MIN_N_DETAIL
            ),
            "shrink_parent": (
                SHRINK_PARENT
            ),
            "shrink_detail": (
                SHRINK_DETAIL
            ),
            "parent_only_scale": (
                PARENT_ONLY_SCALE
            ),
            "parent_weight": (
                PARENT_WEIGHT
            ),
            "detail_weight": (
                DETAIL_WEIGHT
            ),
            "alpha": ALPHA,
            "cap": CAP,
        },
        "uses_test_distribution": (
            False
        ),
        "row_independent_inference": (
            True
        ),
        "model_path": str(
            MODEL_PATH
        ),
    }

    with open(
        MODEL_DIR
        / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # source 보존
    shutil.copy2(
        Path(__file__).resolve(),
        MODEL_DIR / "train.py",
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(
        f"Model    : {MODEL_PATH}"
    )
    print(
        "Metadata : "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        f"Final iter: {final_iterations}"
    )
    print(
        "Exact-ASOF: True"
    )
    print(
        "Conditional residual correction: True"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
