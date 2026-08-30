#train.py

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v050_v42_l2_all_r_shrink015_submit"

ROOT = Path(".")
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = ROOT / "model" / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_DIR = ROOT / "validation" / EXPERIMENT_NAME

ID = "row_id"
TARGET = "control_success"
SEASON = "season"
GAME_TYPE = "game_type"

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


# =============================================================================
# Exact-ASOF5
# =============================================================================

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


# =============================================================================
# HB-EB
# =============================================================================

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


# =============================================================================
# PITCHER_CONDITIONAL FULL 21
# =============================================================================

PC_CONTEXTS = [
    ("v15_count", 28.0),
    ("batter_hand", 35.0),
    ("v15_inning_phase", 35.0),
    ("base_state", 35.0),
    ("v15_pressure", 28.0),
    ("v15_runners", 35.0),
    ("v15_li_bucket", 35.0),
]

PC_PARENT_STRENGTH = 80.0
PC_FIELDS = ("rate", "reliability", "delta")

PC_FEATURES = [
    f"pc__{ctx}__{field}"
    for ctx, _strength in PC_CONTEXTS
    for field in PC_FIELDS
]


# =============================================================================
# Current-season SUCCESS_FULL
# =============================================================================

SUCCESS_SPEC = {
    "pitcher": {
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "strengths": [40.0, 80.0, 160.0],
        "reliability_strength": 80.0,
    },
    "batter": {
        "id_col": "batter_id",
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "strengths": [40.0, 80.0, 160.0],
        "reliability_strength": 100.0,
    },
}

SUCCESS_FEATURES = [
    # pitcher
    "gseason__pitcher_rate_k40",
    "gseason__pitcher_rate_k80",
    "gseason__pitcher_rate_k160",
    "gseason__pitcher_reliability",
    "gseason__pitcher_minus_career",

    # batter
    "gseason__batter_rate_k40",
    "gseason__batter_rate_k80",
    "gseason__batter_rate_k160",
    "gseason__batter_raw",
    "gseason__batter_log_n",
    "gseason__batter_reliability",
    "gseason__batter_minus_career",

    # matchup current-state diff
    "gseason__pitcher_minus_batter_k80",
]


# =============================================================================
# Utility
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


def brier_score(
    y_true,
    y_pred,
) -> float:
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
            (y_true - y_pred) ** 2
        )
    )


def calculate_bss(
    y_true,
    y_pred,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    brier = brier_score(
        y_true,
        y_pred,
    )

    p = float(
        np.mean(
            y_true
        )
    )

    baseline = (
        p * (1.0 - p)
    )

    if baseline <= 0.0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier / baseline
        )
    )


def rounded_cumulative_count(
    n,
    rate,
):
    """
    v038 current-state semantics:
    cumulative event count ~= round(n * rate)
    """
    n = np.asarray(
        n,
        dtype=np.float64,
    )

    rate = np.asarray(
        rate,
        dtype=np.float64,
    )

    out = np.zeros(
        len(n),
        dtype=np.float64,
    )

    ok = (
        (n > 0.0)
        & np.isfinite(rate)
    )

    out[ok] = np.rint(
        n[ok]
        * rate[ok]
    )

    return out


# =============================================================================
# Exact-ASOF5
# =============================================================================

def build_pitcher_season_stats(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    공식 labeled train으로 pitcher x completed season 통계를 저장한다.

    Exact feature를 만들 때는 target row의 season보다
    과거 season만 합산한다.
    """
    required = {
        "pitcher_id",
        SEASON,
        TARGET,
    }

    missing = sorted(
        required
        - set(history.columns)
    )

    if missing:
        raise KeyError(
            f"Exact stats missing columns: {missing}"
        )

    stats = (
        history
        .groupby(
            ["pitcher_id", SEASON],
            sort=False,
            dropna=False,
        )[TARGET]
        .agg(
            n="size",
            success="sum",
        )
        .reset_index()
    )

    stats[SEASON] = (
        pd.to_numeric(
            stats[SEASON],
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
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    row-local official cumulative asof에서
    target season보다 과거의 labeled completed-season count를 빼서
    current-season pitcher state를 복원한다.

    다른 validation/test row 사용 없음.
    """
    required = {
        SEASON,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required
        - set(rows.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF missing columns: {missing}"
        )

    out = rows.copy()

    for col in EXACT_FEATURES:
        out[col] = np.nan

    row_seasons = (
        pd.to_numeric(
            out[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    stats = pitcher_season_stats.copy()

    stats[SEASON] = (
        pd.to_numeric(
            stats[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        row_seasons.unique()
    ):
        mask = (
            row_seasons
            == int(target_year)
        )

        part = out.loc[
            mask
        ]

        hist = stats.loc[
            stats[SEASON]
            < int(target_year)
        ]

        if len(hist):
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
                part["pitcher_id"]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                part["pitcher_id"]
                .map(
                    hist_agg["success"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

        else:
            hist_n = np.zeros(
                len(part),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(part),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                part["asof_pitcher_n"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_rate = (
            pd.to_numeric(
                part[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_success = (
            asof_n
            * asof_rate
        )

        current_n = np.maximum(
            asof_n - hist_n,
            0.0,
        )

        current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                current_success
            ),
            np.clip(
                current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(part),
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
            len(part),
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

        idx = part.index

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

    return out


# =============================================================================
# HB-EB
# =============================================================================

def compute_hbeb_priors(
    df_train: pd.DataFrame,
    group_cols: List[str],
    rate_col: str,
):
    rate = pd.to_numeric(
        df_train[
            rate_col
        ],
        errors="coerce",
    )

    g_mu = float(
        rate.mean()
    )

    g_var = float(
        rate.var()
    )

    max_bernoulli_var = (
        g_mu
        * (
            1.0 - g_mu
        )
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
        g_mu
        * g_m
    )

    g_beta = (
        (1.0 - g_mu)
        * g_m
    )

    tmp = (
        df_train[
            group_cols
        ]
        .copy()
    )

    tmp["_rate_"] = rate

    stats = (
        tmp
        .groupby(
            group_cols,
            dropna=False,
        )["_rate_"]
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
            1.0
            - stats["mean"]
        )
    )

    valid = (
        stats["var"].notna()
        & np.isfinite(
            stats["var"]
        )
        & (
            stats["var"]
            > 0.0
        )
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"]
            >= 30
        )
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

    if valid.any():
        mu = stats.loc[
            valid,
            "mean",
        ]

        var = stats.loc[
            valid,
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
            valid,
            "alpha",
        ] = (
            mu
            * m
        )

        stats.loc[
            valid,
            "beta",
        ] = (
            (1.0 - mu)
            * m
        )

    prior_df = (
        stats[
            group_cols
            + [
                "alpha",
                "beta",
            ]
        ]
        .copy()
    )

    global_prior = (
        float(g_alpha),
        float(g_beta),
    )

    return (
        prior_df,
        global_prior,
    )


def apply_hbeb(
    df: pd.DataFrame,
    group_cols: List[str],
    n_col: str,
    rate_col: str,
    prior_df: pd.DataFrame,
    global_prior,
):
    lookup = (
        df[
            group_cols
        ]
        .reset_index(
            drop=True
        )
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
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        pd.to_numeric(
            df[
                n_col
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        pd.to_numeric(
            df[
                rate_col
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n
        * rate
    )

    denominator = (
        n
        + alpha
        + beta
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
            denominator
            > 0.0
        ),
    )


def fit_hbeb_priors(
    df_train: pd.DataFrame,
) -> Dict[str, dict]:
    priors = {}

    for cfg in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=cfg[
                "group_cols"
            ],
            rate_col=cfg[
                "rate_col"
            ],
        )

        priors[
            cfg["new_col"]
        ] = {
            "prior_df": (
                prior_df
            ),
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                cfg[
                    "group_cols"
                ]
            ),
            "n_col": (
                cfg[
                    "n_col"
                ]
            ),
            "rate_col": (
                cfg[
                    "rate_col"
                ]
            ),
        }

    return priors


def apply_hbeb_with_priors(
    df_apply: pd.DataFrame,
    base_features: List[str],
    priors: Dict[str, dict],
):
    out = (
        df_apply.copy()
    )

    features = list(
        base_features
    )

    for cfg in HBEB_CONFIGS:
        info = priors[
            cfg[
                "new_col"
            ]
        ]

        out[
            cfg[
                "new_col"
            ]
        ] = apply_hbeb(
            df=out,
            group_cols=info[
                "group_cols"
            ],
            n_col=info[
                "n_col"
            ],
            rate_col=info[
                "rate_col"
            ],
            prior_df=info[
                "prior_df"
            ],
            global_prior=info[
                "global_prior"
            ],
        )

        # canonical HBEB replaces raw success-rate columns.
        if (
            cfg["rate_col"]
            in features
        ):
            features.remove(
                cfg[
                    "rate_col"
                ]
            )

        if (
            cfg["new_col"]
            not in features
        ):
            features.append(
                cfg[
                    "new_col"
                ]
            )

    return (
        out,
        features,
    )


# =============================================================================
# PC contexts / empirical Bayes
# =============================================================================

def add_pc_context_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    x = (
        frame.copy()
    )

    x[
        "v15_count"
    ] = (
        x[
            "balls_before"
        ]
        .astype(str)
        + "_"
        + x[
            "strikes_before"
        ]
        .astype(str)
    )

    x[
        "v15_inning_phase"
    ] = (
        pd.cut(
            pd.to_numeric(
                x["inning"],
                errors="coerce",
            ),
            [
                -np.inf,
                3,
                6,
                9,
                np.inf,
            ],
            labels=[
                "early",
                "middle",
                "late",
                "extra",
            ],
        )
        .astype(str)
    )

    x[
        "v15_pressure"
    ] = np.where(
        (
            x["balls_before"]
            == 3
        )
        & (
            x["strikes_before"]
            == 2
        ),
        "full",
        np.where(
            x[
                "balls_before"
            ]
            == 3,
            "three_ball",
            np.where(
                x[
                    "strikes_before"
                ]
                == 2,
                "two_strike",
                "normal",
            ),
        ),
    )

    x[
        "v15_runners"
    ] = np.where(
        (
            pd.to_numeric(
                x[
                    "num_runners_on"
                ],
                errors="coerce",
            )
            .fillna(0)
            > 0
        ),
        "on",
        "empty",
    )

    x[
        "v15_li_bucket"
    ] = (
        pd.cut(
            pd.to_numeric(
                x["li"],
                errors="coerce",
            ),
            [
                -np.inf,
                0.7,
                1.5,
                np.inf,
            ],
            labels=[
                "low",
                "mid",
                "high",
            ],
        )
        .astype(str)
    )

    return x


def pc_eb_table(
    history: pd.DataFrame,
    keys: List[str],
    parent_keys: List[str] | None,
    strength: float,
    parent_strength: float,
) -> pd.DataFrame:
    if len(
        history
    ) == 0:
        return pd.DataFrame(
            columns=(
                keys
                + [
                    "rate",
                    "reliability",
                    "delta",
                    "log_n",
                ]
            )
        )

    h = history

    league = float(
        h[
            TARGET
        ]
        .mean()
    )

    g = (
        h
        .groupby(
            keys,
            sort=False,
        )[TARGET]
        .agg(
            [
                "sum",
                "size",
            ]
        )
        .reset_index()
    )

    if parent_keys is None:
        g[
            "parent"
        ] = league

    else:
        pg = (
            h
            .groupby(
                parent_keys,
                sort=False,
            )[TARGET]
            .agg(
                [
                    "sum",
                    "size",
                ]
            )
            .reset_index()
        )

        pg[
            "parent_rate"
        ] = (
            pg["sum"]
            + parent_strength
            * league
        ) / (
            pg["size"]
            + parent_strength
        )

        g = (
            g.merge(
                pg[
                    parent_keys
                    + [
                        "parent_rate"
                    ]
                ],
                on=parent_keys,
                how="left",
            )
        )

        g[
            "parent"
        ] = (
            g[
                "parent_rate"
            ]
            .fillna(
                league
            )
        )

    g[
        "rate"
    ] = (
        g["sum"]
        + strength
        * g["parent"]
    ) / (
        g["size"]
        + strength
    )

    g[
        "reliability"
    ] = (
        g["size"]
        / (
            g["size"]
            + strength
        )
    )

    g[
        "delta"
    ] = (
        g["rate"]
        - g["parent"]
    )

    g[
        "log_n"
    ] = np.log1p(
        g["size"]
    )

    return g[
        keys
        + [
            "rate",
            "reliability",
            "delta",
            "log_n",
        ]
    ]


def pc_lookup(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    keys: List[str],
    field: str,
    default: float = 0.0,
):
    if len(
        table
    ) == 0:
        return np.full(
            len(rows),
            default,
            dtype=np.float32,
        )

    if len(
        keys
    ) == 1:
        s = (
            table
            .set_index(
                keys[0]
            )[
                field
            ]
        )

        return (
            rows[
                keys[0]
            ]
            .map(s)
            .fillna(default)
            .to_numpy(
                dtype=np.float32
            )
        )

    idx = (
        pd.MultiIndex
        .from_frame(
            rows[
                keys
            ]
        )
    )

    s = (
        table
        .set_index(
            keys
        )[
            field
        ]
    )

    return (
        s.reindex(
            idx
        )
        .fillna(default)
        .to_numpy(
            dtype=np.float32
        )
    )


def build_pc_for_rows(
    rows: pd.DataFrame,
    available_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    target season y:
        PC source = exactly y-1 labeled season.

    training row도 자기 season label을 사용하지 않는다.
    """
    required_rows = {
        SEASON,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
    }

    required_hist = (
        required_rows
        | {
            TARGET,
        }
    )

    missing_rows = sorted(
        required_rows
        - set(rows.columns)
    )

    missing_hist = sorted(
        required_hist
        - set(
            available_history.columns
        )
    )

    if missing_rows:
        raise KeyError(
            f"PC row columns missing: {missing_rows}"
        )

    if missing_hist:
        raise KeyError(
            f"PC history columns missing: {missing_hist}"
        )

    r_all = (
        add_pc_context_columns(
            rows
            .reset_index(
                drop=False
            )
            .rename(
                columns={
                    "index": (
                        "__orig_index"
                    ),
                }
            )
        )
    )

    h_all = (
        add_pc_context_columns(
            available_history
            .reset_index(
                drop=True
            )
        )
    )

    out = pd.DataFrame(
        0.0,
        index=np.arange(
            len(r_all)
        ),
        columns=PC_FEATURES,
        dtype=np.float32,
    )

    seasons = (
        pd.to_numeric(
            r_all[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    hist_seasons = (
        pd.to_numeric(
            h_all[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        target_mask = (
            seasons.eq(
                int(
                    target_year
                )
            )
            .to_numpy()
        )

        target_pos = np.flatnonzero(
            target_mask
        )

        rr = (
            r_all.loc[
                target_mask
            ]
            .reset_index(
                drop=True
            )
        )

        source_year = (
            int(
                target_year
            )
            - 1
        )

        hh = (
            h_all.loc[
                hist_seasons.eq(
                    source_year
                )
            ]
            .reset_index(
                drop=True
            )
        )

        if len(
            hh
        ) == 0:
            continue

        for (
            ctx,
            strength,
        ) in PC_CONTEXTS:
            keys = [
                "pitcher_id",
                ctx,
            ]

            table = (
                pc_eb_table(
                    history=hh,
                    keys=keys,
                    parent_keys=[
                        "pitcher_id",
                    ],
                    strength=float(
                        strength
                    ),
                    parent_strength=(
                        PC_PARENT_STRENGTH
                    ),
                )
            )

            prefix = (
                f"pc__{ctx}"
            )

            for field in PC_FIELDS:
                out.loc[
                    target_pos,
                    f"{prefix}__{field}",
                ] = pc_lookup(
                    table=table,
                    rows=rr,
                    keys=keys,
                    field=field,
                    default=0.0,
                )

    out.index = (
        rows.index
    )

    if (
        list(
            out.columns
        )
        != PC_FEATURES
    ):
        raise RuntimeError(
            "PC feature order mismatch."
        )

    if not np.isfinite(
        out.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "PC contains NaN/Inf."
        )

    return out


def mask_pc_to_r_only(
    pc_frame: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> pd.DataFrame:
    out = (
        pc_frame.copy()
    )

    gt = (
        source_rows[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    f_mask = (
        gt == "F"
    )

    out.loc[
        f_mask,
        PC_FEATURES,
    ] = 0.0

    values = (
        out[
            PC_FEATURES
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Masked PC contains NaN/Inf."
        )

    if f_mask.any():
        f_values = (
            out.loc[
                f_mask,
                PC_FEATURES,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.all(
            f_values
            == 0.0
        ):
            raise RuntimeError(
                "F PC masking failed."
            )

    return out


def build_pc_inference_tables(
    full_train: pd.DataFrame,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    script.py에서 test target season y에 대해
    source y-1 lookup만으로 PC를 만들 수 있게
    source season별 lookup table을 artifact에 저장한다.
    """
    h_all = (
        add_pc_context_columns(
            full_train
            .reset_index(
                drop=True
            )
        )
    )

    seasons = (
        pd.to_numeric(
            h_all[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    result = {}

    for source_year in sorted(
        seasons.unique()
    ):
        hh = (
            h_all.loc[
                seasons.eq(
                    int(
                        source_year
                    )
                )
            ]
            .reset_index(
                drop=True
            )
        )

        result[
            int(
                source_year
            )
        ] = {}

        for (
            ctx,
            strength,
        ) in PC_CONTEXTS:
            result[
                int(
                    source_year
                )
            ][
                ctx
            ] = pc_eb_table(
                history=hh,
                keys=[
                    "pitcher_id",
                    ctx,
                ],
                parent_keys=[
                    "pitcher_id",
                ],
                strength=float(
                    strength
                ),
                parent_strength=(
                    PC_PARENT_STRENGTH
                ),
            )

    return result


# =============================================================================
# Current-season SUCCESS_FULL
# =============================================================================

def build_success_bank(
    history: pd.DataFrame,
    entity: str,
):
    spec = (
        SUCCESS_SPEC[
            entity
        ]
    )

    id_col = (
        spec[
            "id_col"
        ]
    )

    if len(
        history
    ) == 0:
        return {
            "n": {},
            "success": {},
            "prior": 0.5,
        }

    agg = (
        history
        .groupby(
            id_col,
            sort=False,
            dropna=False,
        )[TARGET]
        .agg(
            [
                "size",
                "sum",
            ]
        )
    )

    max_year = int(
        pd.to_numeric(
            history[
                SEASON
            ],
            errors="raise",
        )
        .max()
    )

    prior = float(
        history.loc[
            pd.to_numeric(
                history[
                    SEASON
                ],
                errors="coerce",
            )
            .eq(
                max_year
            ),
            TARGET,
        ]
        .mean()
    )

    if not np.isfinite(
        prior
    ):
        prior = 0.5

    return {
        "n": (
            agg[
                "size"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "success": (
            agg[
                "sum"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "prior": float(
            prior
        ),
    }


def build_state_bank(
    available_history: pd.DataFrame,
    target_year: int,
):
    hist = (
        available_history.loc[
            pd.to_numeric(
                available_history[
                    SEASON
                ],
                errors="raise",
            )
            .astype(int)
            .lt(
                int(
                    target_year
                )
            )
        ]
        .copy()
    )

    return {
        "target_year": int(
            target_year
        ),
        "pitcher_success": (
            build_success_bank(
                hist,
                "pitcher",
            )
        ),
        "batter_success": (
            build_success_bank(
                hist,
                "batter",
            )
        ),
    }


def transform_success_state(
    rows: pd.DataFrame,
    bank: dict,
    entity: str,
    out: pd.DataFrame,
):
    spec = (
        SUCCESS_SPEC[
            entity
        ]
    )

    sb = (
        bank[
            f"{entity}_success"
        ]
    )

    id_col = (
        spec[
            "id_col"
        ]
    )

    n_col = (
        spec[
            "n_col"
        ]
    )

    rate_col = (
        spec[
            "rate_col"
        ]
    )

    ids = (
        rows[
            id_col
        ]
    )

    hist_n = (
        ids
        .map(
            sb[
                "n"
            ]
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    hist_success = (
        ids
        .map(
            sb[
                "success"
            ]
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    asof_n = (
        pd.to_numeric(
            rows[
                n_col
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    career = (
        pd.to_numeric(
            rows[
                rate_col
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    prior = float(
        sb[
            "prior"
        ]
    )

    career = np.where(
        np.isfinite(
            career
        ),
        career,
        prior,
    )

    cumulative_success = (
        rounded_cumulative_count(
            asof_n,
            career,
        )
    )

    current_n = np.maximum(
        asof_n
        - hist_n,
        0.0,
    )

    current_success = np.clip(
        cumulative_success
        - hist_success,
        0.0,
        current_n,
    )

    raw = np.divide(
        current_success,
        current_n,
        out=np.full(
            len(rows),
            prior,
            dtype=np.float64,
        ),
        where=(
            current_n
            > 0.0
        ),
    )

    prefix = (
        f"gseason__{entity}"
    )

    for k in (
        spec[
            "strengths"
        ]
    ):
        out[
            f"{prefix}_rate_k{int(k)}"
        ] = (
            (
                current_success
                + float(k)
                * prior
            )
            / (
                current_n
                + float(k)
            )
        ).astype(
            np.float32
        )

    out[
        f"{prefix}_raw"
    ] = (
        raw.astype(
            np.float32
        )
    )

    out[
        f"{prefix}_log_n"
    ] = (
        np.log1p(
            current_n
        )
        .astype(
            np.float32
        )
    )

    rk = float(
        spec[
            "reliability_strength"
        ]
    )

    out[
        f"{prefix}_reliability"
    ] = (
        current_n
        / (
            current_n
            + rk
        )
    ).astype(
        np.float32
    )

    out[
        f"{prefix}_minus_career"
    ] = (
        raw
        - career
    ).astype(
        np.float32
    )

    return {
        "current_n": (
            current_n
        ),
        "raw": (
            raw
        ),
        "k80": (
            out[
                f"{prefix}_rate_k80"
            ]
            .to_numpy(
                dtype=np.float64
            )
        ),
    }


def build_success_state_for_rows(
    rows: pd.DataFrame,
    available_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    SUCCESS_FULL 13 features.

    row target season y:
      historical constants = labeled history season < y
      current state = row-local official cumulative asof - historical constants
    """
    out = pd.DataFrame(
        index=rows.index
    )

    for col in (
        SUCCESS_FEATURES
    ):
        out[
            col
        ] = np.nan

    seasons = (
        pd.to_numeric(
            rows[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        mask = (
            seasons.eq(
                int(
                    target_year
                )
            )
        )

        rr = (
            rows.loc[
                mask
            ]
            .copy()
        )

        bank = (
            build_state_bank(
                available_history=(
                    available_history
                ),
                target_year=int(
                    target_year
                ),
            )
        )

        z = pd.DataFrame(
            index=rr.index
        )

        pitcher_state = (
            transform_success_state(
                rows=rr,
                bank=bank,
                entity="pitcher",
                out=z,
            )
        )

        batter_state = (
            transform_success_state(
                rows=rr,
                bank=bank,
                entity="batter",
                out=z,
            )
        )

        z[
            "gseason__pitcher_minus_batter_k80"
        ] = (
            pitcher_state[
                "k80"
            ]
            - batter_state[
                "k80"
            ]
        ).astype(
            np.float32
        )

        z = (
            z.reindex(
                columns=SUCCESS_FEATURES
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
            .astype(
                np.float32
            )
        )

        out.loc[
            rr.index,
            SUCCESS_FEATURES,
        ] = z[
            SUCCESS_FEATURES
        ]

    out = (
        out
        .fillna(0.0)
        .astype(
            np.float32
        )
    )

    if not np.isfinite(
        out.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "SUCCESS_FULL contains NaN/Inf."
        )

    return out


def build_success_history_artifact(
    full_train: pd.DataFrame,
) -> dict:
    """
    inference 시 raw train.csv 없이도 target season의
    historical n/success/prior를 복원할 수 있게
    season-level aggregated labeled history를 저장한다.
    """
    result = {}

    for entity in [
        "pitcher",
        "batter",
    ]:
        id_col = (
            SUCCESS_SPEC[
                entity
            ][
                "id_col"
            ]
        )

        stats = (
            full_train
            .groupby(
                [
                    id_col,
                    SEASON,
                ],
                sort=False,
                dropna=False,
            )[TARGET]
            .agg(
                n="size",
                success="sum",
            )
            .reset_index()
        )

        stats[
            SEASON
        ] = (
            pd.to_numeric(
                stats[
                    SEASON
                ],
                errors="raise",
            )
            .astype(int)
        )

        result[
            entity
        ] = stats

    season_prior = (
        full_train
        .groupby(
            SEASON,
            sort=False,
        )[TARGET]
        .mean()
    )

    result[
        "season_prior"
    ] = {
        int(k): float(v)
        for k, v in (
            season_prior
            .to_dict()
            .items()
        )
    }

    return result


# =============================================================================
# Feature-view construction
# =============================================================================

def get_base_features_from_test_header() -> List[str]:
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

    if (
        TARGET
        in base_features
    ):
        base_features.remove(
            TARGET
        )

    if (
        "asof_pitcher_pitchmix_n"
        not in base_features
    ):
        raise RuntimeError(
            "canonical feature asof_pitcher_pitchmix_n is missing."
        )

    return base_features


def build_control_view(
    rows: pd.DataFrame,
    available_history: pd.DataFrame,
    base_features: List[str],
    hbeb_prior_source: pd.DataFrame,
):
    pitcher_stats = (
        build_pitcher_season_stats(
            available_history
        )
    )

    exact = (
        add_pitcher_exact_asof_features(
            rows=rows,
            pitcher_season_stats=(
                pitcher_stats
            ),
        )
    )

    features = list(
        base_features
    )

    for col in (
        EXACT_FEATURES
    ):
        if (
            col
            not in features
        ):
            features.append(
                col
            )

    hbeb_priors = (
        fit_hbeb_priors(
            hbeb_prior_source
        )
    )

    (
        transformed,
        control_features,
    ) = apply_hbeb_with_priors(
        df_apply=exact,
        base_features=features,
        priors=hbeb_priors,
    )

    return (
        transformed,
        control_features,
        hbeb_priors,
        pitcher_stats,
    )


def build_fold_views(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    base_features: List[str],
):
    """
    One fold:
      control train/val
      treatment train/val

    Fold-train history only is used for validation feature creation.
    """

    # ---------------------------------------------------------------------
    # Exact-ASOF
    # ---------------------------------------------------------------------
    pitcher_stats = (
        build_pitcher_season_stats(
            df_train
        )
    )

    train_exact = (
        add_pitcher_exact_asof_features(
            rows=df_train,
            pitcher_season_stats=(
                pitcher_stats
            ),
        )
    )

    val_exact = (
        add_pitcher_exact_asof_features(
            rows=df_val,
            pitcher_season_stats=(
                pitcher_stats
            ),
        )
    )

    exact_plus = list(
        base_features
    )

    for col in (
        EXACT_FEATURES
    ):
        if (
            col
            not in exact_plus
        ):
            exact_plus.append(
                col
            )

    # ---------------------------------------------------------------------
    # HBEB prior: fold train only
    # ---------------------------------------------------------------------
    hbeb_priors = (
        fit_hbeb_priors(
            train_exact
        )
    )

    (
        train_control,
        control_features,
    ) = apply_hbeb_with_priors(
        df_apply=train_exact,
        base_features=exact_plus,
        priors=hbeb_priors,
    )

    (
        val_control,
        val_control_features,
    ) = apply_hbeb_with_priors(
        df_apply=val_exact,
        base_features=exact_plus,
        priors=hbeb_priors,
    )

    if (
        control_features
        != val_control_features
    ):
        raise RuntimeError(
            "control train/validation feature order mismatch."
        )

    # ---------------------------------------------------------------------
    # PC FULL
    # ---------------------------------------------------------------------
    pc_train = (
        build_pc_for_rows(
            rows=df_train,
            available_history=df_train,
        )
    )

    pc_val = (
        build_pc_for_rows(
            rows=df_val,
            available_history=df_train,
        )
    )

    pc_train = (
        mask_pc_to_r_only(
            pc_train,
            df_train,
        )
    )

    pc_val = (
        mask_pc_to_r_only(
            pc_val,
            df_val,
        )
    )

    # ---------------------------------------------------------------------
    # Current-season SUCCESS_FULL
    # ---------------------------------------------------------------------
    state_train = (
        build_success_state_for_rows(
            rows=df_train,
            available_history=df_train,
        )
    )

    state_val = (
        build_success_state_for_rows(
            rows=df_val,
            available_history=df_train,
        )
    )

    train_treatment = (
        train_control.copy()
    )

    val_treatment = (
        val_control.copy()
    )

    for col in (
        PC_FEATURES
    ):
        train_treatment[
            col
        ] = pc_train[
            col
        ]

        val_treatment[
            col
        ] = pc_val[
            col
        ]

    for col in (
        SUCCESS_FEATURES
    ):
        train_treatment[
            col
        ] = state_train[
            col
        ]

        val_treatment[
            col
        ] = state_val[
            col
        ]

    treatment_features = (
        list(
            control_features
        )
        + list(
            PC_FEATURES
        )
        + list(
            SUCCESS_FEATURES
        )
    )

    if (
        len(
            treatment_features
        )
        != len(
            set(
                treatment_features
            )
        )
    ):
        raise RuntimeError(
            "Duplicated treatment feature."
        )

    return (
        train_control,
        val_control,
        train_treatment,
        val_treatment,
        control_features,
        treatment_features,
    )


# =============================================================================
# Model fit
# =============================================================================

def fit_one_fold_model(
    train_view: pd.DataFrame,
    val_view: pd.DataFrame,
    y_train,
    y_val,
    features: List[str],
):
    cat_cols = [
        c
        for c in CAT_COLS
        if c in features
    ]

    X_train = (
        prepare_features(
            train_view[
                features
            ],
            cat_cols,
        )
    )

    X_val = (
        prepare_features(
            val_view[
                features
            ],
            cat_cols,
        )
    )

    model = (
        CatBoostRegressor(
            **MODEL_PARAMS
        )
    )

    started = (
        time.time()
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_val,
            y_val,
        ),
        cat_features=(
            cat_cols
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
        verbose=100,
    )

    elapsed = (
        time.time()
        - started
    )

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        best_tree_count = int(
            MODEL_PARAMS[
                "iterations"
            ]
        )
    else:
        best_tree_count = int(
            best_iteration
            + 1
        )

    pred = (
        np.clip(
            np.asarray(
                model.predict(
                    X_val
                ),
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )
    )

    return (
        pred,
        best_tree_count,
        float(
            elapsed
        ),
    )


# =============================================================================
# Validation
# =============================================================================

def run_validation(
    train: pd.DataFrame,
    base_features: List[str],
):
    fold_rows = []
    oof_frames = []

    for cfg in (
        EXPANDING_FOLDS
    ):
        fold = int(
            cfg[
                "fold"
            ]
        )

        train_seasons = list(
            cfg[
                "train_seasons"
            ]
        )

        val_year = int(
            cfg[
                "validation_season"
            ]
        )

        print()
        print(
            "=" * 112
        )
        print(
            f"Fold {fold}: "
            f"{train_seasons} -> {val_year}"
        )
        print(
            "=" * 112
        )

        df_train = (
            train.loc[
                train[
                    SEASON
                ]
                .isin(
                    train_seasons
                )
            ]
            .copy()
        )

        df_val = (
            train.loc[
                train[
                    SEASON
                ]
                .eq(
                    val_year
                )
            ]
            .copy()
        )

        if (
            len(
                df_train
            )
            == 0
            or len(
                df_val
            )
            == 0
        ):
            raise RuntimeError(
                f"Fold {fold}: empty train/validation."
            )

        (
            train_control,
            val_control,
            train_treatment,
            val_treatment,
            control_features,
            treatment_features,
        ) = build_fold_views(
            df_train=df_train,
            df_val=df_val,
            base_features=base_features,
        )

        y_train = (
            df_train[
                TARGET
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_val = (
            df_val[
                TARGET
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        print(
            f"CONTROL   features: "
            f"{len(control_features)}"
        )

        (
            control_pred,
            control_trees,
            control_seconds,
        ) = fit_one_fold_model(
            train_view=train_control,
            val_view=val_control,
            y_train=y_train,
            y_val=y_val,
            features=control_features,
        )

        print(
            f"TREATMENT features: "
            f"{len(treatment_features)} "
            f"(+{len(PC_FEATURES)} PC "
            f"+{len(SUCCESS_FEATURES)} state)"
        )

        (
            treatment_pred,
            treatment_trees,
            treatment_seconds,
        ) = fit_one_fold_model(
            train_view=train_treatment,
            val_view=val_treatment,
            y_train=y_train,
            y_val=y_val,
            features=treatment_features,
        )

        game_type = (
            df_val[
                GAME_TYPE
            ]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
            .to_numpy()
        )

        r_mask = (
            game_type
            != "F"
        )

        f_mask = (
            game_type
            == "F"
        )

        # -------------------------------------------------------------
        # Deployment routing
        # R = SUCCESS_FULL treatment
        # F = clean control
        # -------------------------------------------------------------
        routed_pred = (
            control_pred.copy()
        )

        routed_pred[
            r_mask
        ] = (
            treatment_pred[
                r_mask
            ]
        )

        # F must be exactly protected.
        if (
            f_mask.any()
            and not np.array_equal(
                routed_pred[
                    f_mask
                ],
                control_pred[
                    f_mask
                ],
            )
        ):
            raise RuntimeError(
                "F routing protection failed."
            )

        control_brier = (
            brier_score(
                y_val,
                control_pred,
            )
        )

        treatment_brier = (
            brier_score(
                y_val,
                treatment_pred,
            )
        )

        routed_brier = (
            brier_score(
                y_val,
                routed_pred,
            )
        )

        control_bss = (
            calculate_bss(
                y_val,
                control_pred,
            )
        )

        routed_bss = (
            calculate_bss(
                y_val,
                routed_pred,
            )
        )

        r_control_bss = (
            calculate_bss(
                y_val[
                    r_mask
                ],
                control_pred[
                    r_mask
                ],
            )
        )

        r_treatment_bss = (
            calculate_bss(
                y_val[
                    r_mask
                ],
                treatment_pred[
                    r_mask
                ],
            )
        )

        fold_rows.append({
            "fold": fold,
            "train_seasons": (
                ",".join(
                    map(
                        str,
                        train_seasons,
                    )
                )
            ),
            "validation_season": (
                val_year
            ),
            "n_train": int(
                len(
                    df_train
                )
            ),
            "n_validation": int(
                len(
                    df_val
                )
            ),
            "n_R": int(
                r_mask.sum()
            ),
            "n_F": int(
                f_mask.sum()
            ),
            "control_features": int(
                len(
                    control_features
                )
            ),
            "treatment_features": int(
                len(
                    treatment_features
                )
            ),
            "control_tree_count": int(
                control_trees
            ),
            "treatment_tree_count": int(
                treatment_trees
            ),
            "control_seconds": float(
                control_seconds
            ),
            "treatment_seconds": float(
                treatment_seconds
            ),
            "control_brier": float(
                control_brier
            ),
            "treatment_global_brier": float(
                treatment_brier
            ),
            "routed_brier": float(
                routed_brier
            ),
            "control_bss": float(
                control_bss
            ),
            "routed_bss": float(
                routed_bss
            ),
            "routed_gain_vs_control_bss": float(
                routed_bss
                - control_bss
            ),
            "R_control_bss": float(
                r_control_bss
            ),
            "R_treatment_bss": float(
                r_treatment_bss
            ),
            "R_gain_bss": float(
                r_treatment_bss
                - r_control_bss
            ),
        })

        oof_frames.append(
            pd.DataFrame({
                ID: (
                    df_val[
                        ID
                    ]
                    .to_numpy()
                ),
                SEASON: (
                    val_year
                ),
                GAME_TYPE: (
                    game_type
                ),
                "y_true": (
                    y_val
                ),
                "control_pred": (
                    control_pred
                ),
                "treatment_pred": (
                    treatment_pred
                ),
                "routed_pred": (
                    routed_pred
                ),
            })
        )

        print(
            f"RESULT | control BSS={control_bss:+.5f} "
            f"| routed BSS={routed_bss:+.5f} "
            f"| gain={routed_bss-control_bss:+.5f}"
        )

    fold_df = (
        pd.DataFrame(
            fold_rows
        )
    )

    oof = (
        pd.concat(
            oof_frames,
            ignore_index=True,
        )
    )

    return (
        fold_df,
        oof,
    )


# =============================================================================
# Final full-training view
# =============================================================================

def build_final_training_views(
    train: pd.DataFrame,
    base_features: List[str],
):
    """
    Full 2019~2024 train을 이용해 final two-model training view 생성.

    각 training row의 temporal feature 자체는 여전히
    row season보다 이전 labeled history만 사용한다.
    """

    # Exact
    pitcher_stats = (
        build_pitcher_season_stats(
            train
        )
    )

    exact = (
        add_pitcher_exact_asof_features(
            rows=train,
            pitcher_season_stats=(
                pitcher_stats
            ),
        )
    )

    exact_plus = list(
        base_features
    )

    for col in (
        EXACT_FEATURES
    ):
        if (
            col
            not in exact_plus
        ):
            exact_plus.append(
                col
            )

    # HBEB final priors: full labeled train
    hbeb_priors = (
        fit_hbeb_priors(
            exact
        )
    )

    (
        control_view,
        control_features,
    ) = apply_hbeb_with_priors(
        df_apply=exact,
        base_features=exact_plus,
        priors=hbeb_priors,
    )

    # PC
    pc = (
        build_pc_for_rows(
            rows=train,
            available_history=train,
        )
    )

    pc = (
        mask_pc_to_r_only(
            pc,
            train,
        )
    )

    # SUCCESS_FULL
    state = (
        build_success_state_for_rows(
            rows=train,
            available_history=train,
        )
    )

    treatment_view = (
        control_view.copy()
    )

    for col in (
        PC_FEATURES
    ):
        treatment_view[
            col
        ] = pc[
            col
        ]

    for col in (
        SUCCESS_FEATURES
    ):
        treatment_view[
            col
        ] = state[
            col
        ]

    treatment_features = (
        list(
            control_features
        )
        + list(
            PC_FEATURES
        )
        + list(
            SUCCESS_FEATURES
        )
    )

    # Inference preprocessing state.
    pc_tables = (
        build_pc_inference_tables(
            train
        )
    )

    success_history = (
        build_success_history_artifact(
            train
        )
    )

    return {
        "control_view": (
            control_view
        ),
        "treatment_view": (
            treatment_view
        ),
        "control_features": (
            control_features
        ),
        "treatment_features": (
            treatment_features
        ),
        "pitcher_season_stats": (
            pitcher_stats
        ),
        "hbeb_priors": (
            hbeb_priors
        ),
        "pc_tables_by_source_year": (
            pc_tables
        ),
        "success_history": (
            success_history
        ),
    }


def train_full_model(
    view: pd.DataFrame,
    y,
    features: List[str],
    iterations: int,
):
    cat_cols = [
        c
        for c in CAT_COLS
        if c in features
    ]

    X = (
        prepare_features(
            view[
                features
            ],
            cat_cols,
        )
    )

    params = {
        **MODEL_PARAMS,
        "iterations": int(
            iterations
        ),
    }

    model = (
        CatBoostRegressor(
            **params
        )
    )

    started = (
        time.time()
    )

    model.fit(
        X,
        y,
        cat_features=(
            cat_cols
        ),
        verbose=100,
    )

    elapsed = (
        time.time()
        - started
    )

    return (
        model,
        cat_cols,
        float(
            elapsed
        ),
    )




# =============================================================================
# L2 ExtraTrees residual adapter — exact recovered old schema
# =============================================================================

SOURCE_YEAR = 2024
EXPECTED_TARGET_YEAR = 2025

ADAPTER_SEEDS = [
    42,
    2026,
    314,
]

ADAPTER_SHRINK = 0.15

# v048/v049에서 선택 완료.
SELECTED_L2_ROUTE = "ALL_R"

VALIDATION_ADAPTER_CFG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}

FINAL_ADAPTER_CFG = {
    "n_estimators": 120,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}

N_JOBS = max(
    1,
    min(
        6,
        os.cpu_count() or 1,
    ),
)

REFERENCE_RAWNUM = [
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "pitcher_team_id",
    "batter_team_id",
]

REFERENCE_CAT_SMALL = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

ENG_COLS = [
    "eng_recent35",
    "eng_recent_delta",
    "eng_sb",
    "eng_same",
    "eng_close",
    "eng_late",
]

EXPECTED_ADAPTER_CORE_FEATURES = 44
EXPECTED_ADAPTER_EXACT_FEATURES = 5
EXPECTED_ADAPTER_TOTAL_FEATURES = 49


def numeric_series(
    frame: pd.DataFrame,
    col: str,
    default=np.nan,
) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(
            np.full(
                len(frame),
                default,
                dtype=np.float64,
            ),
            index=frame.index,
        )

    return pd.to_numeric(
        frame[col],
        errors="coerce",
    )


def add_eng(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Old L2와 동일한 6개 engineered feature.
    """

    x = frame.copy()

    career = numeric_series(
        x,
        "asof_pitcher_success_rate",
        0.5,
    )

    p1 = numeric_series(
        x,
        "asof_pitcher_prev1_game_success_rate",
        np.nan,
    )

    p3 = numeric_series(
        x,
        "asof_pitcher_prev3_game_success_rate",
        np.nan,
    )

    p5 = numeric_series(
        x,
        "asof_pitcher_prev5_game_success_rate",
        np.nan,
    )

    p1 = p1.fillna(
        career
    )

    p3 = p3.fillna(
        p1
    )

    p5 = p5.fillna(
        p3
    )

    x[
        "eng_recent35"
    ] = (
        0.5 * p1
        + 0.3 * p3
        + 0.2 * p5
    )

    x[
        "eng_recent_delta"
    ] = (
        x[
            "eng_recent35"
        ]
        - career
    )

    x[
        "eng_sb"
    ] = (
        numeric_series(
            x,
            "asof_pitcher_strike_rate",
            np.nan,
        )
        - numeric_series(
            x,
            "asof_pitcher_ball_rate",
            np.nan,
        )
    )

    if (
        "pitcher_hand"
        in x.columns
        and
        "batter_hand"
        in x.columns
    ):
        x[
            "eng_same"
        ] = (
            x[
                "pitcher_hand"
            ]
            .astype(str)
            .eq(
                x[
                    "batter_hand"
                ]
                .astype(str)
            )
            .astype(
                np.float32
            )
        )

    else:
        x[
            "eng_same"
        ] = 0.0

    x[
        "eng_close"
    ] = (
        numeric_series(
            x,
            "score_diff_pitcher_team",
            np.nan,
        )
        .abs()
        .le(1)
        .astype(
            np.float32
        )
    )

    x[
        "eng_late"
    ] = (
        numeric_series(
            x,
            "inning",
            np.nan,
        )
        .ge(7)
        .astype(
            np.float32
        )
    )

    return x


def fit_adapter_prep(
    map_rows: pd.DataFrame,
) -> dict:
    """
    해당 residual source season의 official rows에서만
    median/category mapping을 fit한다.
    """

    m = add_eng(
        map_rows.reset_index(
            drop=True
        )
    )

    numeric_cols = (
        [
            c
            for c in REFERENCE_RAWNUM
            if c in m.columns
        ]
        + ENG_COLS
    )

    missing_raw = [
        c
        for c in REFERENCE_RAWNUM
        if c not in m.columns
    ]

    if missing_raw:
        raise KeyError(
            f"L2 REFERENCE_RAWNUM missing: {missing_raw}"
        )

    med = (
        m[
            numeric_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .median()
        .fillna(
            0.0
        )
        .to_dict()
    )

    cat_cols = [
        c
        for c in REFERENCE_CAT_SMALL
        if c in m.columns
    ]

    missing_cat = [
        c
        for c in REFERENCE_CAT_SMALL
        if c not in m.columns
    ]

    if missing_cat:
        raise KeyError(
            f"L2 REFERENCE_CAT_SMALL missing: {missing_cat}"
        )

    maps = {}

    for col in cat_cols:
        s = (
            m[
                col
            ]
            .astype(
                "string"
            )
            .fillna(
                "<NA>"
            )
            .astype(str)
        )

        maps[
            col
        ] = {
            value: idx
            for idx, value in enumerate(
                pd.Index(
                    s.unique()
                )
            )
        }

    return {
        "numeric_cols": (
            numeric_cols
        ),
        "med": med,
        "maps": maps,
    }


def transform_adapter_core(
    prep: dict,
    rows: pd.DataFrame,
    base_pred,
) -> pd.DataFrame:
    """
    44-column old adapter core:
        29 raw numeric
        + 6 ENG
        + 8 category codes
        + base_pred
    """

    r = add_eng(
        rows.reset_index(
            drop=True
        )
    )

    A = (
        r[
            prep[
                "numeric_cols"
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .copy()
    )

    for col in prep[
        "numeric_cols"
    ]:
        A[
            col
        ] = (
            A[
                col
            ]
            .fillna(
                float(
                    prep[
                        "med"
                    ].get(
                        col,
                        0.0,
                    )
                )
            )
            .astype(
                np.float32
            )
        )

    for (
        col,
        mapping,
    ) in prep[
        "maps"
    ].items():
        A[
            "code_"
            + col
        ] = (
            r[
                col
            ]
            .astype(
                "string"
            )
            .fillna(
                "<NA>"
            )
            .astype(str)
            .map(
                mapping
            )
            .fillna(
                -1
            )
            .astype(
                np.float32
            )
        )

    A[
        "base_pred"
    ] = np.asarray(
        base_pred,
        dtype=np.float32,
    )

    if not np.isfinite(
        A.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "adapter core contains NaN/Inf"
        )

    if (
        A.shape[
            1
        ]
        != EXPECTED_ADAPTER_CORE_FEATURES
    ):
        raise RuntimeError(
            "adapter core feature count mismatch: "
            f"{A.shape[1]} != "
            f"{EXPECTED_ADAPTER_CORE_FEATURES}"
        )

    return A.reset_index(
        drop=True
    )


def exact5_matrix(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    L2에서도 L1과 동일 Exact-ASOF5 생성기를 재사용한다.
    """

    transformed = (
        add_pitcher_exact_asof_features(
            rows=rows,
            pitcher_season_stats=(
                pitcher_season_stats
            ),
        )
    )

    exact = (
        transformed[
            EXACT_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .astype(
            np.float32
        )
        .reset_index(
            drop=True
        )
    )

    if not np.isfinite(
        exact.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Exact-ASOF5 matrix contains NaN/Inf"
        )

    if (
        exact.shape[
            1
        ]
        != EXPECTED_ADAPTER_EXACT_FEATURES
    ):
        raise RuntimeError(
            "L2 Exact feature count mismatch."
        )

    return exact


def build_l2_matrix(
    rows: pd.DataFrame,
    base_pred,
    adapter_prep: dict,
    pitcher_season_stats: pd.DataFrame,
):
    core = (
        transform_adapter_core(
            prep=adapter_prep,
            rows=rows,
            base_pred=base_pred,
        )
    )

    exact = (
        exact5_matrix(
            rows=rows,
            pitcher_season_stats=(
                pitcher_season_stats
            ),
        )
    )

    X = pd.concat(
        [
            core,
            exact,
        ],
        axis=1,
    )

    if (
        X.columns
        .duplicated()
        .any()
    ):
        dup = (
            X.columns[
                X.columns
                .duplicated()
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate L2 feature columns: {dup}"
        )

    if not np.isfinite(
        X.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "L2 matrix contains NaN/Inf"
        )

    if (
        X.shape[
            1
        ]
        != EXPECTED_ADAPTER_TOTAL_FEATURES
    ):
        raise RuntimeError(
            "L2 total feature count mismatch: "
            f"{X.shape[1]} != "
            f"{EXPECTED_ADAPTER_TOTAL_FEATURES}"
        )

    return (
        X,
        list(
            core.columns
        ),
        list(
            exact.columns
        ),
    )


def get_r_mask(
    rows: pd.DataFrame,
):
    return (
        rows[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .ne(
            "F"
        )
        .to_numpy()
    )


def get_f_mask(
    rows: pd.DataFrame,
):
    return (
        rows[
            GAME_TYPE
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .eq(
            "F"
        )
        .to_numpy()
    )


def align_raw_year_with_oof(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    year: int,
):
    raw = (
        train.loc[
            train[
                SEASON
            ].eq(
                int(
                    year
                )
            )
        ]
        .copy()
    )

    pred = (
        oof.loc[
            oof[
                SEASON
            ].eq(
                int(
                    year
                )
            ),
            [
                ID,
                "y_true",
                "routed_pred",
            ],
        ]
        .copy()
        .rename(
            columns={
                "y_true": (
                    "_oof_y"
                ),
                "routed_pred": (
                    "_l1_pred"
                ),
            }
        )
    )

    if len(
        raw
    ) == 0:
        raise RuntimeError(
            f"No raw rows for year={year}"
        )

    if len(
        pred
    ) == 0:
        raise RuntimeError(
            f"No OOF rows for year={year}"
        )

    if pred[
        ID
    ].duplicated().any():
        raise RuntimeError(
            f"OOF row_id duplicate in year={year}"
        )

    merged = (
        raw.merge(
            pred,
            on=ID,
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        merged
    ) != len(
        raw
    ):
        raise RuntimeError(
            f"raw/OOF row count mismatch for year={year}: "
            f"{len(merged)} != {len(raw)}"
        )

    y = (
        pd.to_numeric(
            merged[
                TARGET
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    oof_y = (
        pd.to_numeric(
            merged[
                "_oof_y"
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    max_diff = float(
        np.max(
            np.abs(
                y
                - oof_y
            )
        )
    )

    if max_diff > 1e-12:
        raise RuntimeError(
            f"raw target != OOF y_true "
            f"for year={year}: {max_diff}"
        )

    merged[
        "_l1_pred"
    ] = (
        pd.to_numeric(
            merged[
                "_l1_pred"
            ],
            errors="raise",
        )
        .astype(
            np.float64
        )
    )

    if not np.isfinite(
        merged[
            "_l1_pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            f"L1 OOF NaN/Inf in year={year}"
        )

    return merged


def fit_l2_models(
    X: pd.DataFrame,
    raw_rows: pd.DataFrame,
    residual,
    cfg: dict,
):
    """
    Old L2와 동일:
    model 자체는 source-year R 전체 residual에 fit.
    selected routing은 inference 시 적용한다.
    """

    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    r_mask = get_r_mask(
        raw_rows
    )

    f_mask = get_f_mask(
        raw_rows
    )

    n_r = int(
        r_mask.sum()
    )

    minimum = int(
        cfg[
            "min_samples_leaf"
        ]
    ) * 2

    if n_r <= minimum:
        raise RuntimeError(
            f"L2 R rows insufficient: "
            f"{n_r} <= {minimum}"
        )

    models = {}
    diagnostics = []

    for seed in ADAPTER_SEEDS:
        model = (
            ExtraTreesRegressor(
                n_jobs=N_JOBS,
                random_state=(
                    16200
                    + int(
                        seed
                    )
                ),
                **cfg,
            )
        )

        t0 = (
            time.time()
        )

        model.fit(
            X.loc[
                r_mask
            ],
            residual[
                r_mask
            ],
        )

        elapsed = (
            time.time()
            - t0
        )

        corr = np.asarray(
            model.predict(
                X.loc[
                    r_mask
                ]
            ),
            dtype=np.float64,
        )

        models[
            int(
                seed
            )
        ] = model

        diagnostics.append({
            "adapter_seed": int(
                seed
            ),
            "n_train_R": (
                n_r
            ),
            "n_protected_F": int(
                f_mask.sum()
            ),
            "fit_elapsed_seconds": float(
                elapsed
            ),
            "train_correction_mean_R": float(
                corr.mean()
            ),
            "train_correction_std_R": float(
                corr.std()
            ),
            "train_correction_abs_mean_R": float(
                np.mean(
                    np.abs(
                        corr
                    )
                )
            ),
        })

    return (
        models,
        pd.DataFrame(
            diagnostics
        ),
    )


def predict_l2_models(
    models: dict,
    X: pd.DataFrame,
):
    per_seed = {}

    for seed in ADAPTER_SEEDS:
        corr = np.asarray(
            models[
                int(
                    seed
                )
            ].predict(
                X
            ),
            dtype=np.float64,
        )

        if not np.isfinite(
            corr
        ).all():
            raise RuntimeError(
                f"L2 correction NaN/Inf seed={seed}"
            )

        per_seed[
            int(
                seed
            )
        ] = corr

    mean_corr = (
        np.column_stack(
            [
                per_seed[
                    int(
                        seed
                    )
                ]
                for seed in ADAPTER_SEEDS
            ]
        )
        .mean(
            axis=1
        )
    )

    return (
        mean_corr,
        per_seed,
    )


def apply_selected_l2(
    rows: pd.DataFrame,
    base_pred,
    correction,
    shrink: float = ADAPTER_SHRINK,
):
    """
    Selected final route:
        ALL_R
        F protected exactly.
    """

    base = np.asarray(
        base_pred,
        dtype=np.float64,
    )

    corr = np.asarray(
        correction,
        dtype=np.float64,
    )

    pred = (
        base.copy()
    )

    r_mask = get_r_mask(
        rows
    )

    f_mask = get_f_mask(
        rows
    )

    pred[
        r_mask
    ] = np.clip(
        base[
            r_mask
        ]
        + float(
            shrink
        )
        * corr[
            r_mask
        ],
        0.0,
        1.0,
    )

    if (
        f_mask.any()
        and not np.array_equal(
            pred[
                f_mask
            ],
            base[
                f_mask
            ],
        )
    ):
        raise RuntimeError(
            "L2 F protection failed."
        )

    if not np.isfinite(
        pred
    ).all():
        raise RuntimeError(
            "L2 routed prediction contains NaN/Inf."
        )

    if (
        np.min(
            pred
        )
        < 0.0
        or
        np.max(
            pred
        )
        > 1.0
    ):
        raise RuntimeError(
            "L2 routed prediction outside [0,1]."
        )

    return pred


def run_latest_l2_sanity(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
):
    """
    재학습된 standalone v42 OOF로
    selected L2의 가장 최신 과거 transfer만 재확인:

        2023 residual -> 2024
        route = ALL_R
        shrink = 0.15

    목적:
    - v048/v049 선정 결과와 standalone train 구현이 틀어지지 않았는지 확인
    - 제출 모델 생성 전에 latest transfer가 양수인지 guard

    이 sanity model은 최종 artifact에 사용하지 않는다.
    """

    src = (
        align_raw_year_with_oof(
            train=train,
            oof=oof,
            year=2023,
        )
    )

    tgt = (
        align_raw_year_with_oof(
            train=train,
            oof=oof,
            year=2024,
        )
    )

    source_base = (
        src[
            "_l1_pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    source_y = (
        src[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    source_residual = (
        source_y
        - source_base
    )

    prep = (
        fit_adapter_prep(
            src
        )
    )

    (
        X_src,
        core_cols,
        exact_cols,
    ) = (
        build_l2_matrix(
            rows=src,
            base_pred=source_base,
            adapter_prep=prep,
            pitcher_season_stats=(
                pitcher_season_stats
            ),
        )
    )

    target_base = (
        tgt[
            "_l1_pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    (
        X_tgt,
        target_core_cols,
        target_exact_cols,
    ) = (
        build_l2_matrix(
            rows=tgt,
            base_pred=target_base,
            adapter_prep=prep,
            pitcher_season_stats=(
                pitcher_season_stats
            ),
        )
    )

    if list(
        X_src.columns
    ) != list(
        X_tgt.columns
    ):
        raise RuntimeError(
            "L2 sanity source/target feature order mismatch."
        )

    if (
        core_cols
        != target_core_cols
        or
        exact_cols
        != target_exact_cols
    ):
        raise RuntimeError(
            "L2 sanity core/exact feature mismatch."
        )

    (
        models,
        fit_diag,
    ) = (
        fit_l2_models(
            X=X_src,
            raw_rows=src,
            residual=(
                source_residual
            ),
            cfg=(
                VALIDATION_ADAPTER_CFG
            ),
        )
    )

    (
        corr_mean,
        corr_per_seed,
    ) = (
        predict_l2_models(
            models=models,
            X=X_tgt,
        )
    )

    y24 = (
        tgt[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    pred_mean = (
        apply_selected_l2(
            rows=tgt,
            base_pred=target_base,
            correction=corr_mean,
        )
    )

    base_bss = (
        calculate_bss(
            y24,
            target_base,
        )
    )

    selected_bss = (
        calculate_bss(
            y24,
            pred_mean,
        )
    )

    ensemble_gain = (
        selected_bss
        - base_bss
    )

    seed_rows = []

    for seed in ADAPTER_SEEDS:
        pred_seed = (
            apply_selected_l2(
                rows=tgt,
                base_pred=target_base,
                correction=(
                    corr_per_seed[
                        int(
                            seed
                        )
                    ]
                ),
            )
        )

        seed_gain = (
            calculate_bss(
                y24,
                pred_seed,
            )
            - base_bss
        )

        seed_rows.append({
            "seed": int(
                seed
            ),
            "gain_2024_vs_v42_bss": float(
                seed_gain
            ),
        })

    seed_df = (
        pd.DataFrame(
            seed_rows
        )
    )

    seed_wins = int(
        (
            seed_df[
                "gain_2024_vs_v42_bss"
            ]
            > 0.0
        )
        .sum()
    )

    print()
    print(
        "=" * 112
    )
    print(
        "L2 LATEST TRANSFER SANITY"
    )
    print(
        "=" * 112
    )
    print(
        "Source      : 2023 v42 OOT residual"
    )
    print(
        "Target      : 2024 v42 OOT"
    )
    print(
        f"Route       : {SELECTED_L2_ROUTE}"
    )
    print(
        f"Shrink      : {ADAPTER_SHRINK}"
    )
    print(
        f"v42 BSS     : {base_bss:+.5f}"
    )
    print(
        f"v42+L2 BSS  : {selected_bss:+.5f}"
    )
    print(
        f"Gain        : {ensemble_gain:+.5f}"
    )
    print(
        f"Seed wins   : {seed_wins}/{len(ADAPTER_SEEDS)}"
    )
    print(
        seed_df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )
    print(
        "=" * 112
    )

    # Conservative implementation guard.
    if ensemble_gain <= 0.0:
        raise RuntimeError(
            "Selected L2 failed latest-transfer sanity: "
            f"gain={ensemble_gain:+.5f}"
        )

    if seed_wins < 2:
        raise RuntimeError(
            "Selected L2 failed seed sanity: "
            f"{seed_wins}/{len(ADAPTER_SEEDS)}"
        )

    return {
        "source_year": 2023,
        "target_year": 2024,
        "route": (
            SELECTED_L2_ROUTE
        ),
        "shrink": float(
            ADAPTER_SHRINK
        ),
        "base_bss": float(
            base_bss
        ),
        "selected_bss": float(
            selected_bss
        ),
        "gain_bss": float(
            ensemble_gain
        ),
        "seed_wins": int(
            seed_wins
        ),
        "seed_metrics": (
            seed_df.to_dict(
                orient="records"
            )
        ),
        "fit_diagnostics": (
            fit_diag.to_dict(
                orient="records"
            )
        ),
    }


def fit_final_l2(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
):
    """
    최종 2025 inference용 L2:
        official 2024 y
        - standalone v42 2019~2023 -> 2024 OOT prediction

    위 residual을 2024 R rows에서 학습.
    """

    raw24 = (
        align_raw_year_with_oof(
            train=train,
            oof=oof,
            year=SOURCE_YEAR,
        )
    )

    y24 = (
        raw24[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    l1_oof24 = (
        raw24[
            "_l1_pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    residual24 = (
        y24
        - l1_oof24
    )

    adapter_prep = (
        fit_adapter_prep(
            raw24
        )
    )

    (
        X24,
        core_cols,
        exact_cols,
    ) = (
        build_l2_matrix(
            rows=raw24,
            base_pred=l1_oof24,
            adapter_prep=(
                adapter_prep
            ),
            pitcher_season_stats=(
                pitcher_season_stats
            ),
        )
    )

    print()
    print(
        "=" * 112
    )
    print(
        "FINAL L2 TRAINING"
    )
    print(
        "=" * 112
    )
    print(
        f"Source year : {SOURCE_YEAR}"
    )
    print(
        f"Rows        : {len(raw24):,}"
    )
    print(
        f"R rows      : {int(get_r_mask(raw24).sum()):,}"
    )
    print(
        f"F protected : {int(get_f_mask(raw24).sum()):,}"
    )
    print(
        f"Features    : {X24.shape[1]} "
        f"({len(core_cols)} core + {len(exact_cols)} Exact5)"
    )
    print(
        f"Residual mean : {residual24.mean():+.8f}"
    )
    print(
        f"Residual std  : {residual24.std():.8f}"
    )
    print(
        f"Final ET cfg  : {FINAL_ADAPTER_CFG}"
    )
    print(
        f"Seeds         : {ADAPTER_SEEDS}"
    )
    print(
        f"Deployment    : ALL_R × {ADAPTER_SHRINK}"
    )
    print(
        "=" * 112
    )

    (
        models,
        diagnostics,
    ) = (
        fit_l2_models(
            X=X24,
            raw_rows=raw24,
            residual=residual24,
            cfg=(
                FINAL_ADAPTER_CFG
            ),
        )
    )

    return {
        "source_year": int(
            SOURCE_YEAR
        ),
        "expected_target_year": int(
            EXPECTED_TARGET_YEAR
        ),
        "residual_definition": (
            "official_y_2024 - "
            "standalone_v42_2019_2023_to_2024_OOT_routed_prediction"
        ),
        "adapter_models": (
            models
        ),
        "adapter_seeds": list(
            ADAPTER_SEEDS
        ),
        "validation_adapter_cfg": dict(
            VALIDATION_ADAPTER_CFG
        ),
        "final_adapter_cfg": dict(
            FINAL_ADAPTER_CFG
        ),
        "adapter_shrink": float(
            ADAPTER_SHRINK
        ),
        "selected_route": (
            SELECTED_L2_ROUTE
        ),
        "adapter_prep": (
            adapter_prep
        ),
        "adapter_feature_cols": list(
            X24.columns
        ),
        "adapter_core_feature_cols": (
            core_cols
        ),
        "exact_cols": (
            exact_cols
        ),
        "reference_rawnum": list(
            REFERENCE_RAWNUM
        ),
        "reference_cat_small": list(
            REFERENCE_CAT_SMALL
        ),
        "eng_cols": list(
            ENG_COLS
        ),
        "fit_scope": (
            "all official 2024 R rows"
        ),
        "n_source_rows": int(
            len(
                raw24
            )
        ),
        "n_source_R": int(
            get_r_mask(
                raw24
            ).sum()
        ),
        "n_source_F_protected": int(
            get_f_mask(
                raw24
            ).sum()
        ),
        "residual_mean": float(
            residual24.mean()
        ),
        "residual_std": float(
            residual24.std()
        ),
        "fit_diagnostics": (
            diagnostics.to_dict(
                orient="records"
            )
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # ---------------------------------------------------------------------
    # Safety
    # ---------------------------------------------------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} already exists.\n"
            "기존 artifact 보호를 위해 자동 overwrite하지 않습니다."
        )

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} already exists.\n"
            "기존 validation 보호를 위해 자동 overwrite하지 않습니다."
        )

    total_started = (
        time.time()
    )

    print(
        "=" * 112
    )
    print(
        EXPERIMENT_NAME
    )
    print(
        "=" * 112
    )
    print(
        "Standalone : YES"
    )
    print(
        "External model/OOF reuse : NONE"
    )
    print(
        "L1 F : Exact-ASOF5 + HBEB CONTROL"
    )
    print(
        "L1 R : CONTROL + PC FULL21 + SUCCESS_FULL13"
    )
    print(
        "L1 routing : F=control_model / R=treatment_model"
    )
    print(
        "L2 : ExtraTrees residual adapter"
    )
    print(
        "L2 features : old exact core44 + Exact5 = 49"
    )
    print(
        f"L2 route : {SELECTED_L2_ROUTE}"
    )
    print(
        f"L2 shrink : {ADAPTER_SHRINK}"
    )
    print(
        f"L2 seeds : {ADAPTER_SEEDS}"
    )
    print(
        "L2 F protection : exact"
    )
    print(
        "Test usage : HEADER/SCHEMA ONLY"
    )
    print(
        "=" * 112
    )

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    train = (
        pd.read_csv(
            TRAIN_PATH,
            encoding="utf-8-sig",
            low_memory=False,
        )
    )

    train[
        SEASON
    ] = (
        pd.to_numeric(
            train[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    base_features = (
        get_base_features_from_test_header()
    )

    required_train = {
        ID,
        TARGET,
        SEASON,
        GAME_TYPE,
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "base_state",
        "game_dayofweek",
        "game_month",
        "top_bottom",
        "balls_before",
        "strikes_before",
        "outs_before",
        "inning",
        "score_diff_pitcher_team",
        "num_runners_on",
        "home_win_expectancy",
        "li",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    }

    missing = sorted(
        required_train
        - set(
            train.columns
        )
    )

    if missing:
        raise KeyError(
            f"train.csv missing required columns: {missing}"
        )

    # ---------------------------------------------------------------------
    # L1 expanding validation
    # ---------------------------------------------------------------------
    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        fold_df,
        oof,
    ) = run_validation(
        train=train,
        base_features=base_features,
    )

    fold_df.to_csv(
        VALIDATION_DIR
        / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Disk-safe.
    oof.to_csv(
        VALIDATION_DIR
        / "oof_predictions.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    y_oof = (
        oof[
            "y_true"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    control_oof = (
        oof[
            "control_pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    routed_oof = (
        oof[
            "routed_pred"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    pooled_control_bss = (
        calculate_bss(
            y_oof,
            control_oof,
        )
    )

    pooled_routed_bss = (
        calculate_bss(
            y_oof,
            routed_oof,
        )
    )

    pooled_gain_bss = (
        pooled_routed_bss
        - pooled_control_bss
    )

    latest = (
        fold_df
        .sort_values(
            "validation_season"
        )
        .iloc[
            -1
        ]
    )

    control_iterations = int(
        latest[
            "control_tree_count"
        ]
    )

    treatment_iterations = int(
        latest[
            "treatment_tree_count"
        ]
    )

    print()
    print(
        "=" * 112
    )
    print(
        "L1 VALIDATION SUMMARY"
    )
    print(
        "=" * 112
    )
    print(
        fold_df[
            [
                "validation_season",
                "control_bss",
                "routed_bss",
                "routed_gain_vs_control_bss",
                "R_gain_bss",
                "control_tree_count",
                "treatment_tree_count",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )
    print()
    print(
        f"Pooled control BSS : "
        f"{pooled_control_bss:+.5f}"
    )
    print(
        f"Pooled routed BSS  : "
        f"{pooled_routed_bss:+.5f}"
    )
    print(
        f"Pooled gain        : "
        f"{pooled_gain_bss:+.5f}"
    )
    print(
        f"Final control trees   : "
        f"{control_iterations}"
    )
    print(
        f"Final treatment trees : "
        f"{treatment_iterations}"
    )
    print(
        "=" * 112
    )

    # ---------------------------------------------------------------------
    # Full-data L1 transformation state
    # ---------------------------------------------------------------------
    final = (
        build_final_training_views(
            train=train,
            base_features=base_features,
        )
    )

    # ---------------------------------------------------------------------
    # Selected L2 latest-transfer sanity:
    # 2023 residual -> 2024
    # ---------------------------------------------------------------------
    l2_sanity = (
        run_latest_l2_sanity(
            train=train,
            oof=oof,
            pitcher_season_stats=(
                final[
                    "pitcher_season_stats"
                ]
            ),
        )
    )

    (
        VALIDATION_DIR
        / "l2_latest_transfer_sanity.json"
    ).write_text(
        json.dumps(
            l2_sanity,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Final 2025-use L2 fit on 2024 v42 OOT residual
    # ---------------------------------------------------------------------
    l2_state = (
        fit_final_l2(
            train=train,
            oof=oof,
            pitcher_season_stats=(
                final[
                    "pitcher_season_stats"
                ]
            ),
        )
    )

    # ---------------------------------------------------------------------
    # Final L1 full-data training
    # ---------------------------------------------------------------------
    y_full = (
        train[
            TARGET
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    print()
    print(
        "=" * 112
    )
    print(
        "FINAL L1 CONTROL TRAINING"
    )
    print(
        "=" * 112
    )

    (
        control_model,
        control_cat_cols,
        control_elapsed,
    ) = train_full_model(
        view=final[
            "control_view"
        ],
        y=y_full,
        features=final[
            "control_features"
        ],
        iterations=(
            control_iterations
        ),
    )

    print()
    print(
        "=" * 112
    )
    print(
        "FINAL L1 TREATMENT TRAINING"
    )
    print(
        "=" * 112
    )

    (
        treatment_model,
        treatment_cat_cols,
        treatment_elapsed,
    ) = train_full_model(
        view=final[
            "treatment_view"
        ],
        y=y_full,
        features=final[
            "treatment_features"
        ],
        iterations=(
            treatment_iterations
        ),
    )

    # ---------------------------------------------------------------------
    # Save one self-contained artifact
    # ---------------------------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    artifact = {
        # -------------------------------------------------------------
        # L1 models
        # -------------------------------------------------------------
        "control_model": (
            control_model
        ),
        "treatment_model": (
            treatment_model
        ),

        "routing_policy": {
            "F": (
                "control_model; L2 forbidden"
            ),
            "R": (
                "treatment_model + "
                "0.15 * mean(L2 correction)"
            ),
            "R_definition": (
                "game_type != 'F'"
            ),
        },

        # -------------------------------------------------------------
        # L1 feature schema
        # -------------------------------------------------------------
        "base_features": (
            base_features
        ),
        "control_features": (
            final[
                "control_features"
            ]
        ),
        "treatment_features": (
            final[
                "treatment_features"
            ]
        ),
        "control_cat_cols": (
            control_cat_cols
        ),
        "treatment_cat_cols": (
            treatment_cat_cols
        ),
        "exact_features": (
            EXACT_FEATURES
        ),
        "pc_features": (
            PC_FEATURES
        ),
        "success_features": (
            SUCCESS_FEATURES
        ),

        # -------------------------------------------------------------
        # Frozen L1 preprocessing state for script.py
        # -------------------------------------------------------------
        "pitcher_season_stats": (
            final[
                "pitcher_season_stats"
            ]
        ),
        "hbeb_priors": (
            final[
                "hbeb_priors"
            ]
        ),
        "pc_tables_by_source_year": (
            final[
                "pc_tables_by_source_year"
            ]
        ),
        "success_history": (
            final[
                "success_history"
            ]
        ),

        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "pc_contexts": (
            PC_CONTEXTS
        ),
        "pc_parent_strength": (
            PC_PARENT_STRENGTH
        ),
        "success_spec": (
            SUCCESS_SPEC
        ),

        # -------------------------------------------------------------
        # L2
        # -------------------------------------------------------------
        "l2": (
            l2_state
        ),

        # -------------------------------------------------------------
        # Metadata embedded in artifact
        # -------------------------------------------------------------
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "v42 dual CatBoost L1 "
            "+ R-only ExtraTrees residual L2"
        ),
        "objective": "RMSE",
        "control_iterations": (
            control_iterations
        ),
        "treatment_iterations": (
            treatment_iterations
        ),
        "prediction_policy": (
            "clip L1 predictions to [0,1]; "
            "F=L1 control exactly; "
            "R=clip(L1 treatment + "
            "0.15*mean(3 ExtraTrees corrections), 0,1)"
        ),
        "expected_target_year": (
            EXPECTED_TARGET_YEAR
        ),
        "rule_compliance": {
            "external_model_dependency": False,
            "external_oof_dependency": False,
            "test_rows_used_for_training": False,
            "test_usage": (
                "header/schema only"
            ),
            "test_row_aggregation": False,
            "l2_residual_source": (
                "official 2024 labels "
                "+ standalone v42 2024 OOT predictions"
            ),
            "F_protected_from_L2": True,
        },
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------
    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "standalone_training": True,
        "external_model_reuse": False,
        "external_oof_reuse": False,
        "data_inputs": [
            str(
                TRAIN_PATH
            ),
            (
                f"{TEST_PATH} "
                "(header/schema only)"
            ),
        ],
        "architecture": {
            "L1_F": (
                "Exact-ASOF5 + HBEB control"
            ),
            "L1_R": (
                "Exact-ASOF5 + HBEB "
                "+ PC FULL21 "
                "+ SUCCESS_FULL13"
            ),
            "L2": (
                "ExtraTrees residual adapter; "
                "old exact core44 + Exact5; "
                "ALL_R; shrink 0.15; "
                "F protected"
            ),
        },
        "l1_validation_method": (
            "4-fold expanding window"
        ),
        "l1_folds": (
            EXPANDING_FOLDS
        ),
        "l1_seed": int(
            MODEL_PARAMS[
                "random_seed"
            ]
        ),
        "l1_model_params": (
            MODEL_PARAMS
        ),
        "l1_early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "l1_final_iteration_strategy": (
            "latest validation fold best tree count, "
            "separately for control/treatment"
        ),
        "control_iterations": (
            control_iterations
        ),
        "treatment_iterations": (
            treatment_iterations
        ),
        "l1_pooled_control_bss": float(
            pooled_control_bss
        ),
        "l1_pooled_routed_bss": float(
            pooled_routed_bss
        ),
        "l1_pooled_gain_bss": float(
            pooled_gain_bss
        ),
        "l2_selected_route": (
            SELECTED_L2_ROUTE
        ),
        "l2_selected_shrink": float(
            ADAPTER_SHRINK
        ),
        "l2_adapter_seeds": list(
            ADAPTER_SEEDS
        ),
        "l2_validation_cfg": (
            VALIDATION_ADAPTER_CFG
        ),
        "l2_final_cfg": (
            FINAL_ADAPTER_CFG
        ),
        "l2_latest_transfer_sanity": (
            l2_sanity
        ),
        "l2_source_year": (
            SOURCE_YEAR
        ),
        "l2_expected_target_year": (
            EXPECTED_TARGET_YEAR
        ),
        "l2_n_features": int(
            len(
                l2_state[
                    "adapter_feature_cols"
                ]
            )
        ),
        "n_train": int(
            len(
                train
            )
        ),
        "train_seasons": sorted(
            [
                int(x)
                for x in (
                    train[
                        SEASON
                    ]
                    .unique()
                )
            ]
        ),
        "n_base_features": int(
            len(
                base_features
            )
        ),
        "n_control_features": int(
            len(
                final[
                    "control_features"
                ]
            )
        ),
        "n_treatment_features": int(
            len(
                final[
                    "treatment_features"
                ]
            )
        ),
        "control_elapsed_seconds": float(
            control_elapsed
        ),
        "treatment_elapsed_seconds": float(
            treatment_elapsed
        ),
        "elapsed_seconds": float(
            time.time()
            - total_started
        ),
        "model_path": str(
            MODEL_PATH
        ),
    }

    (
        MODEL_DIR
        / "final_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    validation_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "pooled_control_brier": float(
            brier_score(
                y_oof,
                control_oof,
            )
        ),
        "pooled_routed_brier": float(
            brier_score(
                y_oof,
                routed_oof,
            )
        ),
        "pooled_control_bss": float(
            pooled_control_bss
        ),
        "pooled_routed_bss": float(
            pooled_routed_bss
        ),
        "pooled_gain_bss": float(
            pooled_gain_bss
        ),
        "l2_latest_transfer_sanity": (
            l2_sanity
        ),
    }

    (
        VALIDATION_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            validation_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Reproducibility copy.
    try:
        shutil.copy2(
            Path(
                __file__
            ).resolve(),
            MODEL_DIR
            / "train.py",
        )
    except Exception as exc:
        print(
            f"[WARN] train.py copy failed: {exc}"
        )

    print()
    print(
        "=" * 112
    )
    print(
        "FINAL ARTIFACT SAVED"
    )
    print(
        "=" * 112
    )
    print(
        f"Model    : {MODEL_PATH}"
    )
    print(
        f"Metadata : "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        f"Train.py : "
        f"{MODEL_DIR / 'train.py'}"
    )
    print(
        f"L1 CONTROL features   : "
        f"{len(final['control_features'])}"
    )
    print(
        f"L1 TREATMENT features : "
        f"{len(final['treatment_features'])}"
    )
    print(
        f"L1 CONTROL trees      : "
        f"{control_iterations}"
    )
    print(
        f"L1 TREATMENT trees    : "
        f"{treatment_iterations}"
    )
    print(
        f"L2 features           : "
        f"{len(l2_state['adapter_feature_cols'])}"
    )
    print(
        f"L2 seeds              : "
        f"{ADAPTER_SEEDS}"
    )
    print(
        f"L2 final trees        : "
        f"{FINAL_ADAPTER_CFG['n_estimators']}"
    )
    print(
        f"L2 routing            : "
        f"{SELECTED_L2_ROUTE}"
    )
    print(
        f"L2 shrink             : "
        f"{ADAPTER_SHRINK}"
    )
    print(
        "F policy              : "
        "CONTROL only, L2 exactly protected"
    )
    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()
