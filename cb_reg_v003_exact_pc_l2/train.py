#train.py 

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v003_exact_pc_l2"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = Path("./model") / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
META_PATH = MODEL_DIR / "final_metadata.json"

VALIDATION_DIR = Path("./validation") / EXPERIMENT_NAME
OOF_2024_PATH = VALIDATION_DIR / "l1_2024_oof.csv"
FOLD4_META_PATH = VALIDATION_DIR / "fold4_metadata.json"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

BASE_SEED = 42
SOURCE_YEAR = 2024
EXPECTED_TARGET_YEAR = 2025

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
]

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

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

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": BASE_SEED,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}
EARLY_STOPPING_ROUNDS = 80

EXPECTED_CONTROL_FEATURES = 52
EXPECTED_TREATMENT_FEATURES = 73

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
    for ctx, _ in PC_CONTEXTS
    for field in PC_FIELDS
]

# L2 reference-style adapter settings used in the validated D0 line.
V15_CF_CFG = {
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
ADAPTER_SEEDS = [42, 2026, 314]
ADAPTER_SHRINK = 0.50
N_JOBS = max(1, min(6, os.cpu_count() or 1))

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


def mask_pitcher_conditional_to_r_only(pc_frame, source_rows):
    """
    R(game_type != 'F') 행에서만 PITCHER_CONDITIONAL을 활성화한다.
    F 행은 21개 PC feature를 모두 0으로 둔다.

    주의:
    - 이것은 single global CatBoost의 feature masking이다.
    - shared tree structure 때문에 F prediction이 Control과 완전히 같아지는 것은 보장되지 않는다.
    - 따라서 별도로 routed prediction도 평가한다:
        R -> masked treatment prediction
        F -> control prediction
    """
    out = pc_frame.copy()
    gt = source_rows[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    f_mask = gt == "F"

    out.loc[f_mask, PC_FEATURES] = 0.0

    values = out[PC_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("R-only masked PITCHER_CONDITIONAL contains NaN/Inf")

    if f_mask.any():
        f_values = out.loc[f_mask, PC_FEATURES].to_numpy(dtype=np.float64)
        if not np.all(f_values == 0.0):
            raise RuntimeError("F rows are not fully neutralized in R-only PC mask")

    return out

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

def _ctx_frame_v15(frame):
    x = frame.copy()
    x["v15_count"] = (
        x["balls_before"].astype(str)
        + "_"
        + x["strikes_before"].astype(str)
    )
    x["v15_inning_phase"] = pd.cut(
        pd.to_numeric(x["inning"], errors="coerce"),
        [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype(str)
    x["v15_pressure"] = np.where(
        (x["balls_before"] == 3) & (x["strikes_before"] == 2),
        "full",
        np.where(
            x["balls_before"] == 3,
            "three_ball",
            np.where(x["strikes_before"] == 2, "two_strike", "normal"),
        ),
    )
    x["v15_runners"] = np.where(
        pd.to_numeric(x["num_runners_on"], errors="coerce").fillna(0) > 0,
        "on",
        "empty",
    )
    x["v15_li_bucket"] = pd.cut(
        pd.to_numeric(x["li"], errors="coerce"),
        [-np.inf, 0.7, 1.5, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)
    return x

def _pc_eb_table(history, keys, parent_keys=None, strength=40.0, parent_strength=80.0):
    # Recovered V15 code와 동일한 empirical-Bayes shrinkage 식.
    h = history
    if len(h) == 0:
        return pd.DataFrame(columns=keys + ["rate", "reliability", "delta", "log_n"])

    league = float(h[TARGET].mean())
    g = h.groupby(keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()

    if parent_keys is None:
        g["parent"] = league
    else:
        pg = h.groupby(parent_keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()
        pg["parent_rate"] = (
            pg["sum"] + parent_strength * league
        ) / (
            pg["size"] + parent_strength
        )
        g = g.merge(pg[parent_keys + ["parent_rate"]], on=parent_keys, how="left")
        g["parent"] = g["parent_rate"].fillna(league)

    g["rate"] = (
        g["sum"] + strength * g["parent"]
    ) / (
        g["size"] + strength
    )
    g["reliability"] = g["size"] / (g["size"] + strength)
    g["delta"] = g["rate"] - g["parent"]
    g["log_n"] = np.log1p(g["size"])
    return g[keys + ["rate", "reliability", "delta", "log_n"]]

def _pc_lookup(table, rows, keys, field, default=0.0):
    # 원본 V15의 unseen default=0.0을 그대로 유지.
    if len(table) == 0:
        return np.full(len(rows), default, dtype=np.float32)

    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)

    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)

def _pc_coverage(table, rows, keys):
    if len(rows) == 0 or len(table) == 0:
        return 0.0
    target_idx = pd.MultiIndex.from_frame(rows[keys].astype(str))
    table_idx = pd.MultiIndex.from_frame(table[keys].astype(str))
    return float(np.mean(target_idx.isin(table_idx)))

def build_pitcher_conditional_for_rows(rows, available_history, scope):
    """
    각 target season y의 PC는 오직 y-1 labeled season으로 생성한다.

    training rows:
      2020 row <- 2019 labels
      2021 row <- 2020 labels ...

    validation rows:
      2024 row <- fold-train의 2023 labels

    같은 target season label은 절대 사용하지 않는다.
    """
    required = {
        SEASON_COL, TARGET, "pitcher_id", "batter_hand", "base_state",
        "balls_before", "strikes_before", "inning", "num_runners_on", "li",
    }
    missing_rows = sorted((required - {TARGET}) - set(rows.columns))
    missing_hist = sorted(required - set(available_history.columns))
    if missing_rows:
        raise KeyError(f"PITCHER_CONDITIONAL row source missing: {missing_rows}")
    if missing_hist:
        raise KeyError(f"PITCHER_CONDITIONAL history source missing: {missing_hist}")

    r_all = _ctx_frame_v15(rows.reset_index(drop=False).rename(columns={"index": "__orig_index"}))
    h_all = _ctx_frame_v15(available_history.reset_index(drop=True))

    out = pd.DataFrame(0.0, index=np.arange(len(r_all)), columns=PC_FEATURES, dtype=np.float32)
    coverage_rows = []

    seasons = pd.to_numeric(r_all[SEASON_COL], errors="raise").astype(int)
    hist_seasons = pd.to_numeric(h_all[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(seasons.unique()):
        target_mask = seasons.eq(int(target_year)).to_numpy()
        target_pos = np.flatnonzero(target_mask)
        rr = r_all.loc[target_mask].reset_index(drop=True)
        source_year = int(target_year) - 1
        hh = h_all.loc[hist_seasons.eq(source_year)].reset_index(drop=True)

        # source season 부재(예: 2019)는 원본 lookup의 neutral 0 vector.
        if len(hh) == 0:
            for ctx, strength in PC_CONTEXTS:
                coverage_rows.append({
                    "scope": scope,
                    "target_season": int(target_year),
                    "source_season": int(source_year),
                    "context": ctx,
                    "strength": float(strength),
                    "n_rows": int(len(rr)),
                    "source_rows": 0,
                    "coverage": 0.0,
                })
            continue

        for ctx, strength in PC_CONTEXTS:
            table = _pc_eb_table(
                hh,
                ["pitcher_id", ctx],
                ["pitcher_id"],
                strength=float(strength),
                parent_strength=PC_PARENT_STRENGTH,
            )
            prefix = f"pc__{ctx}"
            keys = ["pitcher_id", ctx]
            for field in PC_FIELDS:
                out.loc[target_pos, f"{prefix}__{field}"] = _pc_lookup(
                    table, rr, keys, field, 0.0
                )

            coverage_rows.append({
                "scope": scope,
                "target_season": int(target_year),
                "source_season": int(source_year),
                "context": ctx,
                "strength": float(strength),
                "n_rows": int(len(rr)),
                "source_rows": int(len(hh)),
                "coverage": _pc_coverage(table, rr, keys),
            })

    # 원래 rows index로 복원.
    out.index = rows.index
    if list(out.columns) != PC_FEATURES:
        raise RuntimeError("PITCHER_CONDITIONAL feature order mismatch")
    if out.columns.duplicated().any():
        raise RuntimeError("PITCHER_CONDITIONAL duplicate columns")
    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("PITCHER_CONDITIONAL contains NaN/Inf")

    return out, pd.DataFrame(coverage_rows)

# =============================================================================
# L2 adapter core
# =============================================================================

def numeric_series(frame: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(
            np.full(len(frame), default, dtype=np.float64),
            index=frame.index,
        )
    return pd.to_numeric(frame[col], errors="coerce")


def add_eng(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    career = numeric_series(x, "asof_pitcher_success_rate", 0.5)
    p1 = numeric_series(x, "asof_pitcher_prev1_game_success_rate", np.nan)
    p3 = numeric_series(x, "asof_pitcher_prev3_game_success_rate", np.nan)
    p5 = numeric_series(x, "asof_pitcher_prev5_game_success_rate", np.nan)
    p1 = p1.fillna(career)
    p3 = p3.fillna(p1)
    p5 = p5.fillna(p3)
    x["eng_recent35"] = 0.5 * p1 + 0.3 * p3 + 0.2 * p5
    x["eng_recent_delta"] = x["eng_recent35"] - career
    x["eng_sb"] = (
        numeric_series(x, "asof_pitcher_strike_rate", np.nan)
        - numeric_series(x, "asof_pitcher_ball_rate", np.nan)
    )
    if "pitcher_hand" in x.columns and "batter_hand" in x.columns:
        x["eng_same"] = (
            x["pitcher_hand"].astype(str)
            .eq(x["batter_hand"].astype(str))
            .astype(np.float32)
        )
    else:
        x["eng_same"] = 0.0
    x["eng_close"] = (
        numeric_series(x, "score_diff_pitcher_team", np.nan)
        .abs().le(1).astype(np.float32)
    )
    x["eng_late"] = (
        numeric_series(x, "inning", np.nan)
        .ge(7).astype(np.float32)
    )
    return x


def fit_adapter_prep(map_rows: pd.DataFrame) -> dict:
    m = add_eng(map_rows.reset_index(drop=True))
    numeric_cols = [c for c in REFERENCE_RAWNUM if c in m.columns] + ENG_COLS
    med = (
        m[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0.0)
        .to_dict()
    )
    cat_cols = [c for c in REFERENCE_CAT_SMALL if c in m.columns]
    maps = {}
    for col in cat_cols:
        s = m[col].astype("string").fillna("<NA>").astype(str)
        maps[col] = {value: idx for idx, value in enumerate(pd.Index(s.unique()))}
    return {"numeric_cols": numeric_cols, "med": med, "maps": maps}


def transform_adapter_core(prep: dict, rows: pd.DataFrame, base_pred) -> pd.DataFrame:
    r = add_eng(rows.reset_index(drop=True))
    A = (
        r[prep["numeric_cols"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    for col in prep["numeric_cols"]:
        A[col] = A[col].fillna(float(prep["med"].get(col, 0.0))).astype(np.float32)
    for col, mapping in prep["maps"].items():
        A["code_" + col] = (
            r[col].astype("string").fillna("<NA>").astype(str)
            .map(mapping).fillna(-1).astype(np.float32)
        )
    A["base_pred"] = np.asarray(base_pred, dtype=np.float32)
    if not np.isfinite(A.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("adapter core contains NaN/Inf")
    return A.reset_index(drop=True)


def exact5_matrix(rows: pd.DataFrame, pitcher_season_stats: pd.DataFrame) -> pd.DataFrame:
    transformed, _ = add_pitcher_exact_asof_features(rows, pitcher_season_stats)
    exact = (
        transformed[EXACT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )
    if not np.isfinite(exact.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Exact-ASOF5 matrix contains NaN/Inf")
    return exact


def fit_final_l2_models(X24, raw24, residual24):
    gt = raw24[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    r_mask = gt != "F"
    f_mask = gt == "F"
    n_r = int(r_mask.sum())
    minimum = int(FINAL_ADAPTER_CFG["min_samples_leaf"]) * 2
    if n_r <= minimum:
        raise RuntimeError(f"Final L2 R rows insufficient: {n_r} <= {minimum}")

    models = {}
    diagnostics = []
    for seed in ADAPTER_SEEDS:
        model = ExtraTreesRegressor(
            n_jobs=N_JOBS,
            random_state=16200 + int(seed),
            **FINAL_ADAPTER_CFG,
        )
        t0 = time.time()
        model.fit(X24.loc[r_mask], residual24[r_mask])
        elapsed = time.time() - t0
        corr = np.asarray(model.predict(X24.loc[r_mask]), dtype=np.float64)
        models[int(seed)] = model
        diagnostics.append({
            "adapter_seed": int(seed),
            "n_train_R": n_r,
            "n_protected_F": int(f_mask.sum()),
            "fit_elapsed_seconds": float(elapsed),
            "train_correction_mean_R": float(corr.mean()),
            "train_correction_std_R": float(corr.std()),
            "train_correction_abs_mean_R": float(np.mean(np.abs(corr))),
        })
        print(
            f"[L2 seed={seed}] R rows={n_r:,} | fit={elapsed:.1f}s | "
            f"corr mean={corr.mean():+.6f} | corr abs={np.mean(np.abs(corr)):.6f}"
        )
    return models, pd.DataFrame(diagnostics), r_mask, f_mask


def build_pc_tables_by_source_season(train: pd.DataFrame):
    train_ctx = _ctx_frame_v15(train.reset_index(drop=True))
    seasons = sorted(
        pd.to_numeric(train_ctx[SEASON_COL], errors="raise").astype(int).unique().tolist()
    )
    result = {}
    for source_year in seasons:
        hh = train_ctx.loc[
            pd.to_numeric(train_ctx[SEASON_COL], errors="raise").astype(int).eq(int(source_year))
        ].copy().reset_index(drop=True)
        season_tables = {}
        for context, strength in PC_CONTEXTS:
            table = _pc_eb_table(
                hh,
                ["pitcher_id", context],
                ["pitcher_id"],
                strength=float(strength),
                parent_strength=float(PC_PARENT_STRENGTH),
            )
            season_tables[context] = table
        result[int(source_year)] = season_tables
    return result


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


# =============================================================================
# L1 view construction / training
# =============================================================================

def make_l1_views(df_train_raw, df_apply_raw, base_features, scope):
    pitcher_stats = build_pitcher_season_stats(df_train_raw)
    tr_exact, tr_exact_diag = add_pitcher_exact_asof_features(df_train_raw, pitcher_stats)
    ap_exact, ap_exact_diag = add_pitcher_exact_asof_features(df_apply_raw, pitcher_stats)

    exact_features = list(base_features)
    for col in EXACT_FEATURES:
        if col not in exact_features:
            exact_features.append(col)

    tr_control, control_features, hbeb_priors = apply_hbeb_pair(
        df_train=tr_exact,
        df_apply=tr_exact,
        base_features=exact_features,
    )
    ap_control, apply_features, _ = apply_hbeb_pair(
        df_train=tr_exact,
        df_apply=ap_exact,
        base_features=exact_features,
    )
    if list(apply_features) != list(control_features):
        raise RuntimeError("HB-EB feature order mismatch")

    pc_tr, pc_cov_tr = build_pitcher_conditional_for_rows(
        rows=df_train_raw,
        available_history=df_train_raw,
        scope=f"{scope}_train",
    )
    pc_ap, pc_cov_ap = build_pitcher_conditional_for_rows(
        rows=df_apply_raw,
        available_history=df_train_raw,
        scope=f"{scope}_apply",
    )
    pc_tr = mask_pitcher_conditional_to_r_only(pc_tr, df_train_raw)
    pc_ap = mask_pitcher_conditional_to_r_only(pc_ap, df_apply_raw)

    tr_treat = tr_control.copy()
    ap_treat = ap_control.copy()
    for col in PC_FEATURES:
        if col in tr_treat.columns or col in ap_treat.columns:
            raise RuntimeError(f"Duplicate PC feature: {col}")
        tr_treat[col] = pc_tr[col]
        ap_treat[col] = pc_ap[col]

    treatment_features = list(control_features) + list(PC_FEATURES)
    control_cat_cols = [c for c in CAT_COLS if c in control_features]
    treatment_cat_cols = [c for c in CAT_COLS if c in treatment_features]

    if len(control_features) != EXPECTED_CONTROL_FEATURES:
        raise RuntimeError(
            f"control feature count {len(control_features)} != {EXPECTED_CONTROL_FEATURES}"
        )
    if len(treatment_features) != EXPECTED_TREATMENT_FEATURES:
        raise RuntimeError(
            f"treatment feature count {len(treatment_features)} != {EXPECTED_TREATMENT_FEATURES}"
        )

    return {
        "train_control": tr_control,
        "apply_control": ap_control,
        "train_treatment": tr_treat,
        "apply_treatment": ap_treat,
        "control_features": control_features,
        "treatment_features": treatment_features,
        "control_cat_cols": control_cat_cols,
        "treatment_cat_cols": treatment_cat_cols,
        "pitcher_season_stats": pitcher_stats,
        "hbeb_priors": hbeb_priors,
        "exact_diag_train": tr_exact_diag,
        "exact_diag_apply": ap_exact_diag,
        "pc_coverage_train": pc_cov_tr,
        "pc_coverage_apply": pc_cov_ap,
    }


def fit_fold_model(train_view, val_view, features, cat_cols, y_train, y_val, label):
    X_tr = prepare_features(train_view[features], cat_cols)
    X_va = prepare_features(val_view[features], cat_cols)
    model = CatBoostRegressor(**MODEL_PARAMS)
    print(f"\n[{label}] fold4 training | rows={len(X_tr):,} -> {len(X_va):,} | features={len(features)}")
    t0 = time.time()
    model.fit(
        X_tr,
        y_train,
        eval_set=(X_va, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )
    elapsed = time.time() - t0
    pred = np.clip(np.asarray(model.predict(X_va), dtype=np.float64), 0.0, 1.0)
    best_zero = int(model.get_best_iteration())
    if best_zero < 0:
        raise RuntimeError(f"{label} best iteration invalid: {best_zero}")
    return model, pred, best_zero, elapsed


def fit_full_model(view, features, cat_cols, y, iterations, label):
    X = prepare_features(view[features], cat_cols)
    params = {**MODEL_PARAMS, "iterations": int(iterations), "random_seed": BASE_SEED}
    model = CatBoostRegressor(**params)
    print(f"\n[{label}] final full training | rows={len(X):,} | features={len(features)} | trees={iterations}")
    t0 = time.time()
    model.fit(X, y, cat_features=cat_cols, verbose=100)
    elapsed = time.time() - t0
    if hasattr(model, "feature_names_") and list(model.feature_names_) != list(features):
        raise RuntimeError(f"{label} final feature order mismatch")
    return model, params, elapsed


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()

    print("=" * 112)
    print(EXPERIMENT_NAME)
    print("=" * 112)
    print("Standalone    : YES (no existing model/validation artifact)")
    print("L1            : CatBoost Exact-ASOF5 + HB-EB + R-only PC FULL")
    print("L1 routing    : R=treatment / F=control")
    print("L2            : ExtraTrees residual adapter")
    print("L2 features   : adapter core + Exact-ASOF5")
    print("PC in L2      : NO")
    print("Exact in L2   : YES")
    print("L2 routing    : R=L1+L2 / F=L1 protected")
    print(f"Adapter seeds : {ADAPTER_SEEDS}")
    print(f"FINAL ET cfg  : {FINAL_ADAPTER_CFG}")
    print(f"Shrink        : {ADAPTER_SHRINK}")
    print("L3            : OFF")
    print("Test usage    : HEADER ONLY")
    print("=" * 112)

    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"기존 model 보호를 위해 중단: {MODEL_DIR}\n"
            "기존 폴더를 이동/삭제하거나 EXPERIMENT_NAME을 변경하세요."
        )
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보호를 위해 중단: {VALIDATION_DIR}\n"
            "기존 폴더를 이동/삭제하거나 EXPERIMENT_NAME을 변경하세요."
        )

    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns.tolist()
    base_features = [c for c in test_cols if c != ID]
    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError("canonical feature asof_pitcher_pitchmix_n missing from test schema")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    train[SEASON_COL] = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)
    if int(train[SEASON_COL].max()) != SOURCE_YEAR:
        raise RuntimeError(
            f"latest train season={int(train[SEASON_COL].max())}; expected {SOURCE_YEAR}"
        )

    required = {
        ID, TARGET, SEASON_COL, GAME_TYPE_COL, "pitcher_id", "batter_hand",
        "base_state", "balls_before", "strikes_before", "inning",
        "num_runners_on", "li", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_batter_n", "asof_batter_success_rate",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise KeyError(f"train required columns missing: {missing}")

    # -----------------------------------------------------------------
    # 1) Latest temporal fold: 2019~2023 -> 2024
    #    This creates the L1 OOT residual source and final tree counts.
    # -----------------------------------------------------------------
    fold_train = train.loc[train[SEASON_COL] < SOURCE_YEAR].copy().reset_index(drop=True)
    fold_val = train.loc[train[SEASON_COL] == SOURCE_YEAR].copy().reset_index(drop=True)
    if len(fold_train) == 0 or len(fold_val) == 0:
        raise RuntimeError("fold4 source/validation rows missing")

    fold_views = make_l1_views(
        df_train_raw=fold_train,
        df_apply_raw=fold_val,
        base_features=base_features,
        scope="fold4",
    )

    y_tr = fold_train[TARGET].to_numpy(dtype=np.float64)
    y24 = fold_val[TARGET].to_numpy(dtype=np.float64)

    _, pred_control24, control_best_zero, control_fold_time = fit_fold_model(
        fold_views["train_control"],
        fold_views["apply_control"],
        fold_views["control_features"],
        fold_views["control_cat_cols"],
        y_tr,
        y24,
        "CONTROL",
    )

    _, pred_treat24, treatment_best_zero, treatment_fold_time = fit_fold_model(
        fold_views["train_treatment"],
        fold_views["apply_treatment"],
        fold_views["treatment_features"],
        fold_views["treatment_cat_cols"],
        y_tr,
        y24,
        "TREATMENT",
    )

    gt24 = fold_val[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    r24 = gt24 != "F"
    f24 = gt24 == "F"
    l1_oof24 = np.where(r24, pred_treat24, pred_control24)
    l1_oof24 = np.clip(l1_oof24, 0.0, 1.0)

    control_trees = int(control_best_zero + 1)
    treatment_trees = int(treatment_best_zero + 1)

    print("\n" + "=" * 112)
    print("LATEST FOLD RESULT (2019~2023 -> 2024)")
    print("=" * 112)
    print(f"CONTROL best tree count  : {control_trees}")
    print(f"TREATMENT best tree count: {treatment_trees}")
    print(f"CONTROL 2024 Brier       : {brier_score(y24, pred_control24):.10f}")
    print(f"TREATMENT 2024 Brier     : {brier_score(y24, pred_treat24):.10f}")
    print(f"ROUTED L1 2024 Brier     : {brier_score(y24, l1_oof24):.10f}")
    print(f"ROUTED L1 2024 BSS       : {calculate_bss(y24, l1_oof24):+.4f}")
    print(f"2024 R/F rows             : {int(r24.sum()):,} / {int(f24.sum()):,}")

    # Save generated OOT source before final model training.
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    oof_df = pd.DataFrame({
        ID: fold_val[ID].to_numpy(),
        SEASON_COL: fold_val[SEASON_COL].to_numpy(),
        GAME_TYPE_COL: fold_val[GAME_TYPE_COL].to_numpy(),
        "y_true": y24,
        "CONTROL": pred_control24,
        "TREATMENT": pred_treat24,
        "L1_ROUTED": l1_oof24,
    })
    oof_df.to_csv(OOF_2024_PATH, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # 2) Final full L1 from raw official train
    # -----------------------------------------------------------------
    final_views = make_l1_views(
        df_train_raw=train,
        df_apply_raw=train,
        base_features=base_features,
        scope="final",
    )
    y_full = train[TARGET].to_numpy(dtype=np.float64)

    control_model, control_final_params, control_final_time = fit_full_model(
        final_views["train_control"],
        final_views["control_features"],
        final_views["control_cat_cols"],
        y_full,
        control_trees,
        "CONTROL",
    )

    treatment_model, treatment_final_params, treatment_final_time = fit_full_model(
        final_views["train_treatment"],
        final_views["treatment_features"],
        final_views["treatment_cat_cols"],
        y_full,
        treatment_trees,
        "TREATMENT",
    )

    pc_tables_by_source_season = build_pc_tables_by_source_season(train)

    # -----------------------------------------------------------------
    # 3) Final L2 on 2024 OOT residuals
    # -----------------------------------------------------------------
    raw24 = fold_val.copy().reset_index(drop=True)
    residual24 = y24 - l1_oof24

    adapter_prep = fit_adapter_prep(raw24)
    core24 = transform_adapter_core(adapter_prep, raw24, l1_oof24)

    # Full train stats are safe here: exact transform itself only subtracts
    # completed seasons strictly earlier than each row's season.
    full_pitcher_stats = final_views["pitcher_season_stats"]
    exact24 = exact5_matrix(raw24, full_pitcher_stats)
    X24 = pd.concat([core24, exact24], axis=1)

    if X24.columns.duplicated().any():
        dup = X24.columns[X24.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate L2 columns: {dup}")
    if not np.isfinite(X24.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("L2 X24 contains NaN/Inf")

    print("\n" + "=" * 112)
    print("FINAL L2 TRAINING")
    print("=" * 112)
    print(f"2024 rows      : {len(raw24):,}")
    print(f"R residual rows: {int(r24.sum()):,}")
    print(f"L2 features    : {X24.shape[1]} ({core24.shape[1]} core + {exact24.shape[1]} Exact)")
    print(f"Residual mean  : {residual24.mean():+.8f}")
    print(f"Residual std   : {residual24.std():.8f}")

    l2_models, l2_fit_diag, l2_r_mask, l2_f_mask = fit_final_l2_models(
        X24=X24,
        raw24=raw24,
        residual24=residual24,
    )

    # -----------------------------------------------------------------
    # 4) Save one self-contained artifact
    # -----------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "Standalone routed CatBoost L1 + R-only ExtraTrees L2",
        "random_seed": BASE_SEED,
        "expected_target_year": EXPECTED_TARGET_YEAR,

        "control": {
            "model": control_model,
            "features": list(final_views["control_features"]),
            "cat_cols": list(final_views["control_cat_cols"]),
            "final_iterations": control_trees,
            "model_params": control_final_params,
        },

        "treatment": {
            "model": treatment_model,
            "features": list(final_views["treatment_features"]),
            "cat_cols": list(final_views["treatment_cat_cols"]),
            "final_iterations": treatment_trees,
            "model_params": treatment_final_params,
            "pc_features": list(PC_FEATURES),
        },

        "shared_l1_state": {
            "base_features": list(base_features),
            "exact_features": list(EXACT_FEATURES),
            "pitcher_season_stats": full_pitcher_stats,
            "hbeb_priors": final_views["hbeb_priors"],
            "hbeb_configs": HBEB_CONFIGS,
            "pc_contexts": [
                {"context": c, "strength": float(s)} for c, s in PC_CONTEXTS
            ],
            "pc_parent_strength": float(PC_PARENT_STRENGTH),
            "pc_fields": list(PC_FIELDS),
            "pc_tables_by_source_season": pc_tables_by_source_season,
            "pc_unseen_default": 0.0,
        },

        "l1_routing": {
            "column": GAME_TYPE_COL,
            "R_rule": "game_type != 'F'",
            "F_rule": "game_type == 'F'",
            "R_model": "treatment",
            "F_model": "control",
        },

        "l2": {
            "source_year": SOURCE_YEAR,
            "residual_definition": "official_y_2024 - standalone_L1_2019_2023_to_2024_routed_prediction",
            "adapter_models": l2_models,
            "adapter_seeds": list(ADAPTER_SEEDS),
            "validation_crossfit_cfg": dict(V15_CF_CFG),
            "final_adapter_cfg": dict(FINAL_ADAPTER_CFG),
            "adapter_shrink": float(ADAPTER_SHRINK),
            "adapter_prep": adapter_prep,
            "adapter_feature_cols": list(X24.columns),
            "adapter_core_feature_cols": list(core24.columns),
            "exact_cols": list(EXACT_FEATURES),
            "pitcher_season_stats": full_pitcher_stats,
            "reference_rawnum": list(REFERENCE_RAWNUM),
            "reference_cat_small": list(REFERENCE_CAT_SMALL),
            "eng_cols": list(ENG_COLS),
            "pc_in_l2": False,
            "exact_in_l2": True,
            "fit_scope": "all official 2024 R rows",
        },

        "l2_routing": {
            "column": GAME_TYPE_COL,
            "L2_rule": "game_type != 'F'",
            "protected_rule": "game_type == 'F'",
            "R_prediction": "clip(L1 + 0.50 * mean(3 ExtraTrees corrections), 0, 1)",
            "F_prediction": "L1 exactly",
        },

        "prediction_policy": "np.clip(prediction, 0.0, 1.0)",
        "rule_compliance": {
            "test_rows_used_during_training": False,
            "test_usage_during_training": "header_only",
            "test_row_aggregation": False,
            "l2_residual_source": "official 2024 labels + standalone 2024 L1 out-of-time prediction",
            "exact_asof": "row-local official ASOF minus completed prior-season train labels",
            "pc_source": "exactly previous labeled season only",
        },
    }

    joblib.dump(artifact, MODEL_PATH, compress=3)

    # Copy the actual standalone train.py next to the model.
    try:
        shutil.copy2(Path(__file__), MODEL_DIR / "train.py")
    except Exception:
        pass

    fold_meta = {
        "train_seasons": sorted(fold_train[SEASON_COL].unique().tolist()),
        "validation_season": SOURCE_YEAR,
        "n_train": int(len(fold_train)),
        "n_validation": int(len(fold_val)),
        "control_best_iteration_zero_based": control_best_zero,
        "control_final_tree_count": control_trees,
        "treatment_best_iteration_zero_based": treatment_best_zero,
        "treatment_final_tree_count": treatment_trees,
        "control_brier_2024": brier_score(y24, pred_control24),
        "treatment_brier_2024": brier_score(y24, pred_treat24),
        "routed_l1_brier_2024": brier_score(y24, l1_oof24),
        "routed_l1_bss_2024": calculate_bss(y24, l1_oof24),
        "control_fold_seconds": control_fold_time,
        "treatment_fold_seconds": treatment_fold_time,
    }
    FOLD4_META_PATH.write_text(
        json.dumps(to_jsonable(fold_meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "standalone_training": True,
        "external_model_dependency": False,
        "external_validation_dependency": False,
        "architecture": {
            "L1": "CatBoost Exact-ASOF5 + HB-EB + R-only PC FULL routed",
            "L2": "ExtraTrees residual adapter; core + Exact-ASOF5; no PC; R-only correction",
            "L3": False,
        },
        "latest_fold": fold_meta,
        "final_l1": {
            "control_tree_count": control_trees,
            "treatment_tree_count": treatment_trees,
            "control_fit_seconds": control_final_time,
            "treatment_fit_seconds": treatment_final_time,
            "control_features": len(final_views["control_features"]),
            "treatment_features": len(final_views["treatment_features"]),
        },
        "l2": {
            "source_year": SOURCE_YEAR,
            "n_source_rows": int(len(raw24)),
            "n_R": int(l2_r_mask.sum()),
            "n_F_protected": int(l2_f_mask.sum()),
            "n_features": int(X24.shape[1]),
            "n_core_features": int(core24.shape[1]),
            "n_exact_features": int(exact24.shape[1]),
            "residual_mean": float(residual24.mean()),
            "residual_std": float(residual24.std()),
            "adapter_seeds": list(ADAPTER_SEEDS),
            "final_cfg": dict(FINAL_ADAPTER_CFG),
            "shrink": ADAPTER_SHRINK,
            "fit_diagnostics": l2_fit_diag.to_dict(orient="records"),
        },
        "elapsed_seconds": float(time.time() - total_start),
    }
    META_PATH.write_text(
        json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 112)
    print("DONE")
    print("=" * 112)
    print(f"Model       : {MODEL_PATH}")
    print(f"Metadata    : {META_PATH}")
    print(f"2024 L1 OOF : {OOF_2024_PATH}")
    print(f"Fold4 meta  : {FOLD4_META_PATH}")
    print(f"CONTROL trees  : {control_trees}")
    print(f"TREATMENT trees: {treatment_trees}")
    print(f"Final ET cfg   : {FINAL_ADAPTER_CFG}")
    print(f"Total elapsed  : {time.time() - total_start:.1f}s")
    print("=" * 112)


if __name__ == "__main__":
    main()
