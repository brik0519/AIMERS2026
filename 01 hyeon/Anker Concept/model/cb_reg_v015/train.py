# train.py: cb_reg_v015
# v016 causal ablation:
# R1 Residual-Control vs RP Placebo-Platoon vs R2 Real-Platoon
#
# 목적
# ----
# v015에서 관측된 R2-R1 개선이
# (A) 실제 pitcher × batter_hand conditional information 때문인지,
# (B) feature 하나를 추가하면서 CatBoost tree path가 달라진 효과인지
# 분리한다.
#
# Arm
# ----
# R0_ANCHOR_ONLY
#     pred = dynamic_anchor
#
# R1_RESIDUAL_CONTROL
#     target   = y - dynamic_anchor
#     features = v013 residual base features
#
# RP_RESIDUAL_PLACEBO
#     target   = y - dynamic_anchor
#     features = v013 residual base features + cf_platoon_placebo_delta
#
# R2_RESIDUAL_REAL
#     target   = y - dynamic_anchor
#     features = v013 residual base features + cf_platoon_delta
#
# Placebo 설계
# ------------
# - strictly-past bank에서 REAL cell delta를 먼저 계산한다.
# - 같은 batter_hand 안에서 cell delta 값만 deterministic permutation한다.
# - pitcher_id ↔ delta 의미 연결은 파괴한다.
# - batter_hand별 marginal distribution, K, bank history, unseen=0 구조는 유지한다.
# - validation/test 행끼리 셔플하지 않는다. frozen bank lookup만 수행한다.
# - 따라서 row independence를 유지한다.
#
# 하드 통제
# ---------
# - Dynamic Anchor 계산 로직은 R1/R2에서 동일
# - fold / seed / HB-EB / CatBoost params / early stopping 동일
# - conditional bank는 row season보다 strictly-past label만 사용
# - validation/test 내부 집계 금지
#
# 출력
# ----
# validation/cb_reg_v015/
#   summary.csv
#   oof_pred.csv
#   conditional_leakage_audit.csv
#   residual_seen_unseen.csv
#   fold_YYYY/
#       pred.csv
#       metadata.json
#
# model/cb_reg_v015/
#   cb_reg_v015.pkl
#   final_metadata.json
#   train.py
#
# NOTE
# ----
# 이 파일은 v013/v014의 핵심 구조를 하나의 실험 파일로 합친 최소 ablation이다.
# Dynamic Anchor 수식은 row-local ASOF current-season pitcher state +
# strictly-past frozen bank를 사용한다.

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v015"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
METADATA_PATH = MODEL_DIR / "final_metadata.json"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# 기본 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
DOMAIN_COL = "game_type"

BASE_FEATURES = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
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
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
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

EXPANDING_FOLDS = [
    ([2019, 2020], 2021),
    ([2019, 2020, 2021], 2022),
    ([2019, 2020, 2021, 2022], 2023),
    ([2019, 2020, 2021, 2022, 2023], 2024),
]

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80
FINAL_ITERATION_STRATEGY = "latest_fold"

# v013 Dynamic Anchor
DYNAMIC_ANCHOR_STRENGTH = 80.0

# v014 A0_ZERO platoon
CF_KEY_COLS = ["pitcher_id", "batter_hand"]
CF_PARENT_COLS = ["pitcher_id"]
CF_STRENGTH = 80.0
CF_FEATURE = "cf_platoon_delta"
PLACEBO_FEATURE = "cf_platoon_placebo_delta"
PLACEBO_SEED = 20260826

AUDIT_SEASONS = [2021, 2022, 2023, 2024]
AUDIT_TOLERANCE = 1e-12


# =========================================================
# HB-EB
# cb_reg_v003 호환
# =========================================================

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

HBEB_MIN_GROUP_ROWS = 30


# =========================================================
# 공통 유틸
# =========================================================

def brier_score(y_true, pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def baseline_brier(y_true):
    y = np.asarray(y_true, dtype=np.float64)
    r = float(np.mean(y))
    return float(np.mean((y - r) ** 2))


def bss_raw(y_true, pred):
    bs = brier_score(y_true, pred)
    base = baseline_brier(y_true)
    if base <= 0:
        return 0.0
    return 1e5 * (base - bs) / base


def ensure_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns: {missing}")


def sanitize_categoricals(df, cat_cols):
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
    return out


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


# =========================================================
# HB-EB
# =========================================================

def _safe_mom_beta(mu, var):
    mu = float(np.clip(mu, 1e-6, 1 - 1e-6))
    k = mu * (1.0 - mu)

    if (not np.isfinite(var)) or var <= 0.0 or var >= k:
        var = 0.5 * k

    m = k / var - 1.0
    if (not np.isfinite(m)) or m <= 0.0:
        m = 1.0

    return mu * m, (1.0 - mu) * m


def compute_hbeb_priors(df_train, group_cols, rate_col):
    rate = pd.to_numeric(df_train[rate_col], errors="coerce")
    valid = rate.notna()

    global_mu = float(rate[valid].mean())
    global_var = float(rate[valid].var(ddof=1))
    global_alpha, global_beta = _safe_mom_beta(global_mu, global_var)

    work = df_train.loc[valid, list(group_cols)].copy()
    work[rate_col] = rate.loc[valid].to_numpy()

    stats = (
        work.groupby(group_cols, dropna=False, observed=True)[rate_col]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    alphas = []
    betas = []

    for row in stats.itertuples(index=False):
        mu = float(getattr(row, "mean"))
        var = float(getattr(row, "var"))
        count = int(getattr(row, "count"))

        ok = (
            count >= HBEB_MIN_GROUP_ROWS
            and np.isfinite(var)
            and var > 0.0
            and np.isfinite(mu)
            and 0.0 < mu < 1.0
        )

        if ok:
            a, b = _safe_mom_beta(mu, var)
        else:
            a, b = global_alpha, global_beta

        alphas.append(a)
        betas.append(b)

    prior_df = stats[list(group_cols)].copy()
    prior_df["alpha"] = alphas
    prior_df["beta"] = betas

    return prior_df, {
        "alpha": float(global_alpha),
        "beta": float(global_beta),
        "mean": float(global_alpha / (global_alpha + global_beta)),
        "strength": float(global_alpha + global_beta),
    }


def apply_hbeb(df, group_cols, n_col, rate_col, prior_df, global_prior):
    lookup = (
        df[list(group_cols)]
        .reset_index(drop=True)
        .merge(prior_df, on=list(group_cols), how="left", sort=False)
    )

    alpha = (
        pd.to_numeric(lookup["alpha"], errors="coerce")
        .fillna(global_prior["alpha"])
        .to_numpy(np.float64)
    )
    beta = (
        pd.to_numeric(lookup["beta"], errors="coerce")
        .fillna(global_prior["beta"])
        .to_numpy(np.float64)
    )

    n = pd.to_numeric(df[n_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).to_numpy(np.float64)

    n = np.clip(n, 0.0, None)
    rate = np.clip(rate, 0.0, 1.0)

    denom = n + alpha + beta
    if np.any(denom <= 0.0):
        raise ValueError(f"HB-EB denominator <= 0: {rate_col}")

    eb = (n * rate + alpha) / denom
    return pd.Series(eb, index=df.index, dtype=np.float64)


def apply_hbeb_configs(df_train, df_apply, base_features, priors=None):
    out = df_apply.copy()
    features = list(base_features)
    out_priors = {}

    for cfg in HBEB_CONFIGS:
        new_col = cfg["new_col"]

        if priors is None:
            prior_df, global_prior = compute_hbeb_priors(
                df_train, cfg["group_cols"], cfg["rate_col"]
            )
        else:
            saved = priors[new_col]
            prior_df = saved["prior_df"].copy()
            global_prior = dict(saved["global_prior"])

        out[new_col] = apply_hbeb(
            out,
            cfg["group_cols"],
            cfg["n_col"],
            cfg["rate_col"],
            prior_df,
            global_prior,
        )

        if cfg["rate_col"] in features:
            features.remove(cfg["rate_col"])
        if new_col not in features:
            features.append(new_col)

        out_priors[new_col] = {
            "config": dict(cfg),
            "prior_df": prior_df,
            "global_prior": global_prior,
        }

    return out, features, out_priors


# =========================================================
# v014 A0_ZERO strictly-past platoon bank
# =========================================================

def _attach_real_and_placebo_delta(cell, parent, strength, seed):
    """
    cell table에 real delta와 placebo delta를 모두 저장한다.

    placebo:
      1) real delta를 계산
      2) batter_hand별로 delta 값만 deterministic permutation
      3) key 자체는 유지

    따라서 같은 hand 안의 delta marginal distribution은 보존되지만
    pitcher_id와 delta의 의미 연결은 제거된다.
    """
    if len(cell) == 0:
        out = cell.copy()
        out["cell_delta_real"] = np.array([], dtype=np.float64)
        out["cell_delta_placebo"] = np.array([], dtype=np.float64)
        return out

    work = cell.merge(
        parent[CF_PARENT_COLS + ["parent_rate"]],
        on=CF_PARENT_COLS,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    parent_rate = pd.to_numeric(
        work["parent_rate"], errors="coerce"
    ).to_numpy(np.float64)
    cell_n = pd.to_numeric(
        work["cell_n"], errors="coerce"
    ).to_numpy(np.float64)
    cell_s = pd.to_numeric(
        work["cell_success"], errors="coerce"
    ).to_numpy(np.float64)

    posterior = (
        cell_s + strength * parent_rate
    ) / (
        cell_n + strength
    )
    real_delta = posterior - parent_rate

    work["cell_delta_real"] = real_delta.astype(np.float64)
    work["cell_delta_placebo"] = np.nan

    rng = np.random.default_rng(int(seed))

    for _, idx in work.groupby(
        "batter_hand",
        dropna=False,
        observed=True,
    ).groups.items():
        idx = list(idx)
        values = work.loc[
            idx, "cell_delta_real"
        ].to_numpy(np.float64, copy=True)

        if len(values) > 1:
            values = values[rng.permutation(len(values))]

        work.loc[
            idx, "cell_delta_placebo"
        ] = values

    return work[
        CF_KEY_COLS
        + [
            "cell_n",
            "cell_success",
            "cell_delta_real",
            "cell_delta_placebo",
        ]
    ].copy()


def build_platoon_bank(history_df, placebo_seed=PLACEBO_SEED):
    ensure_columns(
        history_df,
        CF_KEY_COLS + [TARGET],
        "build_platoon_bank",
    )

    if len(history_df) == 0:
        return {
            "cell_table": pd.DataFrame(
                columns=CF_KEY_COLS
                + [
                    "cell_n",
                    "cell_success",
                    "cell_delta_real",
                    "cell_delta_placebo",
                ]
            ),
            "parent_table": pd.DataFrame(
                columns=CF_PARENT_COLS
                + [
                    "parent_n",
                    "parent_success",
                    "parent_rate",
                ]
            ),
            "global_rate": 0.5,
            "n_history": 0,
            "history_seasons": [],
            "strength": float(CF_STRENGTH),
            "placebo_seed": int(placebo_seed),
        }

    y = pd.to_numeric(
        history_df[TARGET],
        errors="raise",
    ).astype(np.float64)

    cell_tmp = history_df[CF_KEY_COLS].copy()
    cell_tmp[TARGET] = y.to_numpy()

    cell = (
        cell_tmp.groupby(
            CF_KEY_COLS,
            dropna=False,
            observed=True,
        )[TARGET]
        .agg(
            cell_n="size",
            cell_success="sum",
        )
        .reset_index()
    )

    parent = (
        history_df[CF_PARENT_COLS]
        .assign(__y=y.to_numpy())
        .groupby(
            CF_PARENT_COLS,
            dropna=False,
            observed=True,
        )["__y"]
        .agg(
            parent_n="size",
            parent_success="sum",
        )
        .reset_index()
    )
    parent["parent_rate"] = (
        parent["parent_success"]
        / parent["parent_n"].clip(lower=1)
    )

    history_seasons = sorted(
        int(x)
        for x in pd.unique(
            history_df[SEASON_COL]
        )
    )

    bank_seed = int(
        placebo_seed
        + 1009 * len(history_seasons)
        + 9176
        * (
            max(history_seasons)
            if history_seasons
            else 0
        )
    )

    cell = _attach_real_and_placebo_delta(
        cell=cell,
        parent=parent,
        strength=float(CF_STRENGTH),
        seed=bank_seed,
    )

    return {
        "cell_table": cell,
        "parent_table": parent,
        "global_rate": float(y.mean()),
        "n_history": int(len(history_df)),
        "history_seasons": history_seasons,
        "strength": float(CF_STRENGTH),
        "placebo_seed": bank_seed,
    }


def apply_platoon_bank_a0(df_apply, bank):
    """
    REAL과 PLACEBO를 같은 frozen bank lookup으로 동시에 생성한다.

    unseen:
        real delta    = 0
        placebo delta = 0
    """
    out = df_apply.copy()
    n_rows = len(out)

    cell_lookup = (
        out[CF_KEY_COLS]
        .reset_index(drop=True)
        .merge(
            bank["cell_table"],
            on=CF_KEY_COLS,
            how="left",
            sort=False,
        )
    )

    cell_n = (
        pd.to_numeric(
            cell_lookup["cell_n"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    real_delta = (
        pd.to_numeric(
            cell_lookup["cell_delta_real"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    placebo_delta = (
        pd.to_numeric(
            cell_lookup["cell_delta_placebo"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(np.float64)
    )

    cell_n = np.clip(
        cell_n,
        0.0,
        None,
    )
    seen = cell_n > 0.0

    real_delta = np.where(
        seen,
        real_delta,
        0.0,
    )
    placebo_delta = np.where(
        seen,
        placebo_delta,
        0.0,
    )

    out[CF_FEATURE] = real_delta.astype(np.float64)
    out[PLACEBO_FEATURE] = placebo_delta.astype(np.float64)
    out["cf_platoon_seen"] = seen.astype(np.int8)
    out["cf_platoon_n"] = cell_n.astype(np.float64)

    return out, {
        "n_rows": int(n_rows),
        "unseen_rows": int(np.sum(~seen)),
        "unseen_rate": (
            float(np.mean(~seen))
            if n_rows
            else 0.0
        ),
        "mean_abs_real_delta": (
            float(np.mean(np.abs(real_delta)))
            if n_rows
            else 0.0
        ),
        "mean_abs_placebo_delta": (
            float(np.mean(np.abs(placebo_delta)))
            if n_rows
            else 0.0
        ),
        "mean_n": (
            float(np.mean(cell_n))
            if n_rows
            else 0.0
        ),
        "history_seasons": list(
            bank["history_seasons"]
        ),
        "placebo_seed": int(
            bank.get(
                "placebo_seed",
                PLACEBO_SEED,
            )
        ),
    }


def build_strictly_past_platoon_features(df):
    ensure_columns(
        df,
        [SEASON_COL, TARGET] + CF_KEY_COLS,
        "strictly_past",
    )

    pieces = []
    diags = []
    season_int = df[SEASON_COL].astype(int)

    for season in sorted(
        int(x)
        for x in pd.unique(
            df[SEASON_COL]
        )
    ):
        cur = df.loc[
            season_int == season
        ].copy()
        hist = df.loc[
            season_int < season
        ]

        if len(hist) == 0:
            cur[CF_FEATURE] = 0.0
            cur[PLACEBO_FEATURE] = 0.0
            cur["cf_platoon_seen"] = 0
            cur["cf_platoon_n"] = 0.0
            diag = {
                "target_season": int(season),
                "n_rows": int(len(cur)),
                "unseen_rows": int(len(cur)),
                "unseen_rate": (
                    1.0
                    if len(cur)
                    else 0.0
                ),
                "mean_abs_real_delta": 0.0,
                "mean_abs_placebo_delta": 0.0,
                "mean_n": 0.0,
                "history_seasons": [],
                "placebo_seed": int(
                    PLACEBO_SEED
                ),
            }
        else:
            bank = build_platoon_bank(hist)
            cur, diag = apply_platoon_bank_a0(
                cur,
                bank,
            )
            diag["target_season"] = int(
                season
            )

        pieces.append(cur)
        diags.append(diag)

    result = pd.concat(
        pieces,
        axis=0,
    ).sort_index()

    if not result.index.equals(df.index):
        result = result.reindex(df.index)

    return result, diags


def build_validation_platoon_features(
    history_df,
    validation_df,
):
    valid_seasons = sorted(
        int(x)
        for x in pd.unique(
            validation_df[SEASON_COL]
        )
    )
    if len(valid_seasons) != 1:
        raise ValueError(valid_seasons)

    valid_season = valid_seasons[0]

    if bool(
        (
            history_df[SEASON_COL]
            .astype(int)
            >= valid_season
        ).any()
    ):
        raise RuntimeError(
            "Conditional leakage guard failed."
        )

    bank = build_platoon_bank(history_df)
    out, diag = apply_platoon_bank_a0(
        validation_df,
        bank,
    )
    diag["target_season"] = valid_season

    return out, diag, bank


# =========================================================
# v013-style Dynamic Anchor
# =========================================================

def build_pitcher_history_bank(history_df):
    """
    strictly-past official train labels를 pitcher별 누적 history로 저장한다.
    row-local ASOF에서 current-season 누적 상태를 역산하기 위한 frozen bank.
    """
    ensure_columns(
        history_df,
        ["pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", TARGET],
        "build_pitcher_history_bank",
    )

    if len(history_df) == 0:
        return {
            "table": pd.DataFrame(
                columns=["pitcher_id", "hist_n", "hist_success", "hist_rate"]
            ),
            "global_rate": 0.5,
            "history_seasons": [],
        }

    # row의 terminal asof를 쓰지 않고 official history label 자체를 집계한다.
    tmp = history_df[["pitcher_id"]].copy()
    tmp["__y"] = pd.to_numeric(history_df[TARGET], errors="raise").to_numpy(np.float64)

    table = (
        tmp.groupby(["pitcher_id"], dropna=False, observed=True)["__y"]
        .agg(hist_n="size", hist_success="sum")
        .reset_index()
    )
    table["hist_rate"] = table["hist_success"] / table["hist_n"].clip(lower=1)

    return {
        "table": table,
        "global_rate": float(tmp["__y"].mean()),
        "history_seasons": sorted(int(x) for x in pd.unique(history_df[SEASON_COL])),
    }


def apply_dynamic_anchor(df_apply, history_bank):
    """
    row-local ASOF current-season pitcher state를 복원한다.

    total asof count/success에서 strictly-past frozen history를 빼고,
    current-season count/success를 얻는다.

      total_success ~= asof_pitcher_n * asof_pitcher_success_rate
      current_n      = total_n - hist_n
      current_s      = total_s - hist_s

    anchor =
      (current_s + K * hist_rate) / (current_n + K)

    current_n <= 0 또는 복원 invalid 시 hist_rate/global_rate fallback.

    test 내부 다른 행을 사용하지 않는다.
    """
    ensure_columns(
        df_apply,
        ["pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"],
        "apply_dynamic_anchor",
    )

    out = df_apply.copy()
    n_rows = len(out)

    lookup = (
        out[["pitcher_id"]]
        .reset_index(drop=True)
        .merge(history_bank["table"], on=["pitcher_id"], how="left", sort=False)
    )

    hist_n = (
        pd.to_numeric(lookup["hist_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    hist_success = (
        pd.to_numeric(lookup["hist_success"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    hist_rate = (
        pd.to_numeric(lookup["hist_rate"], errors="coerce")
        .fillna(history_bank["global_rate"])
        .to_numpy(np.float64)
    )

    total_n = (
        pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    total_rate = (
        pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
        .fillna(history_bank["global_rate"])
        .to_numpy(np.float64)
    )
    total_n = np.clip(total_n, 0.0, None)
    total_rate = np.clip(total_rate, 0.0, 1.0)

    total_success = np.rint(total_n * total_rate)

    current_n = total_n - hist_n
    current_success = total_success - hist_success

    invalid = (
        (~np.isfinite(current_n))
        | (~np.isfinite(current_success))
        | (current_n < 0.0)
        | (current_success < 0.0)
        | (current_success > current_n)
    )

    current_n_safe = np.where(invalid, 0.0, current_n)
    current_s_safe = np.where(invalid, 0.0, current_success)

    K = float(DYNAMIC_ANCHOR_STRENGTH)
    anchor = (
        current_s_safe + K * hist_rate
    ) / (
        current_n_safe + K
    )

    anchor = np.where(np.isfinite(anchor), anchor, history_bank["global_rate"])
    anchor = np.clip(anchor, 0.0, 1.0)

    return anchor.astype(np.float64), {
        "n_rows": int(n_rows),
        "anchor_mean": float(np.mean(anchor)) if n_rows else 0.0,
        "anchor_std": float(np.std(anchor)) if n_rows else 0.0,
        "current_n_mean": float(np.mean(current_n_safe)) if n_rows else 0.0,
        "current_n_median": float(np.median(current_n_safe)) if n_rows else 0.0,
        "invalid_rows": int(np.sum(invalid)),
        "invalid_rate": float(np.mean(invalid)) if n_rows else 0.0,
        "history_seasons": list(history_bank["history_seasons"]),
    }


def build_strictly_past_dynamic_anchors(df):
    """
    training season s row anchor는 season < s의 frozen pitcher history + row ASOF만 사용.
    """
    pieces = []
    diagnostics = []
    season_int = df[SEASON_COL].astype(int)

    anchor_series = pd.Series(index=df.index, dtype=np.float64)

    for season in sorted(int(x) for x in pd.unique(df[SEASON_COL])):
        idx = df.index[season_int == season]
        cur = df.loc[idx]
        hist = df.loc[season_int < season]

        bank = build_pitcher_history_bank(hist)
        anchor, diag = apply_dynamic_anchor(cur, bank)
        anchor_series.loc[idx] = anchor

        diag["target_season"] = int(season)
        diagnostics.append(diag)

    return anchor_series.reindex(df.index).to_numpy(np.float64), diagnostics


def build_validation_dynamic_anchor(history_df, validation_df):
    valid_seasons = sorted(int(x) for x in pd.unique(validation_df[SEASON_COL]))
    if len(valid_seasons) != 1:
        raise ValueError(valid_seasons)

    valid_season = valid_seasons[0]
    if bool((history_df[SEASON_COL].astype(int) >= valid_season).any()):
        raise RuntimeError("Dynamic anchor leakage guard failed.")

    bank = build_pitcher_history_bank(history_df)
    anchor, diag = apply_dynamic_anchor(validation_df, bank)
    diag["target_season"] = int(valid_season)

    return anchor, diag, bank


# =========================================================
# Conditional leakage audit
# =========================================================

def feature_max_abs_diff(a_df, b_df, idx, cols):
    max_diff = 0.0

    for col in cols:
        a = a_df.loc[idx, col].to_numpy(np.float64)
        b = b_df.loc[idx, col].to_numpy(np.float64)

        if a.shape != b.shape:
            return float("inf")

        if len(a):
            max_diff = max(
                max_diff,
                float(
                    np.max(
                        np.abs(a - b)
                    )
                ),
            )

    return max_diff


def run_conditional_leakage_audit(df):
    audit_cols = list(dict.fromkeys([SEASON_COL, TARGET] + CF_KEY_COLS))
    audit_df = df[audit_cols].copy()

    full, _ = build_strictly_past_platoon_features(audit_df)

    rows = []
    for s in AUDIT_SEASONS:
        season_int = audit_df[SEASON_COL].astype(int)
        idx = audit_df.index[season_int == s]

        trunc = audit_df.loc[season_int <= s].copy()
        trunc_feat, _ = build_strictly_past_platoon_features(trunc)
        d_future = feature_max_abs_diff(full, trunc_feat, idx, [CF_FEATURE, PLACEBO_FEATURE])

        shuffled = audit_df.copy()
        mask = shuffled[SEASON_COL].astype(int) == s
        y = shuffled.loc[mask, TARGET].to_numpy(copy=True)
        np.random.default_rng(1000 + s).shuffle(y)
        shuffled.loc[mask, TARGET] = y
        sh_feat, _ = build_strictly_past_platoon_features(shuffled)
        d_perm = feature_max_abs_diff(full, sh_feat, idx, [CF_FEATURE, PLACEBO_FEATURE])

        row = {
            "season": int(s),
            "n_rows": int(len(idx)),
            "future_removal_max_abs_diff": d_future,
            "current_label_permutation_max_abs_diff": d_perm,
            "passed": bool(
                d_future <= AUDIT_TOLERANCE and d_perm <= AUDIT_TOLERANCE
            ),
        }
        rows.append(row)

        if not row["passed"]:
            raise RuntimeError(f"Conditional leakage audit failed: {row}")

    return rows


# =========================================================
# Fold 학습
# =========================================================

def train_residual_model(
    train_df,
    valid_df,
    features,
    train_target,
    valid_target,
):
    tr = sanitize_categoricals(train_df[features], CAT_COLS)
    va = sanitize_categoricals(valid_df[features], CAT_COLS)

    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(
        tr,
        train_target,
        cat_features=CAT_COLS,
        eval_set=(va, valid_target),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )
    return model


def domain_metrics(df_valid, y, predictions):
    rows = []
    domains = pd.unique(df_valid[DOMAIN_COL])

    for d in domains:
        mask = df_valid[DOMAIN_COL].to_numpy() == d
        rows.append(
            {
                "domain": str(d),
                "n": int(mask.sum()),
                "actual_mean": float(np.mean(y[mask])),
                **{
                    f"{name}_brier": brier_score(y[mask], pred[mask])
                    for name, pred in predictions.items()
                },
            }
        )
    return rows


def train_fold(
    df_all,
    train_seasons,
    validation_season,
):
    t0 = time.time()

    season_int = df_all[
        SEASON_COL
    ].astype(int)

    df_train_raw = df_all.loc[
        season_int.isin(train_seasons)
    ].copy()
    df_valid_raw = df_all.loc[
        season_int == validation_season
    ].copy()

    # HB-EB
    (
        df_train_hbeb,
        hbeb_features,
        fold_priors,
    ) = apply_hbeb_configs(
        df_train_raw,
        df_train_raw,
        BASE_FEATURES,
        priors=None,
    )
    (
        df_valid_hbeb,
        valid_hbeb_features,
        _,
    ) = apply_hbeb_configs(
        df_train_raw,
        df_valid_raw,
        BASE_FEATURES,
        priors=fold_priors,
    )

    if hbeb_features != valid_hbeb_features:
        raise RuntimeError(
            "HB-EB feature mismatch"
        )

    # Dynamic Anchor: identical for all arms
    (
        anchor_train,
        anchor_train_diag,
    ) = build_strictly_past_dynamic_anchors(
        df_train_raw
    )
    (
        anchor_valid,
        anchor_valid_diag,
        dynamic_bank,
    ) = build_validation_dynamic_anchor(
        df_train_raw,
        df_valid_raw,
    )

    y_train = df_train_raw[
        TARGET
    ].to_numpy(np.float64)
    y_valid = df_valid_raw[
        TARGET
    ].to_numpy(np.float64)

    residual_train = (
        y_train - anchor_train
    )
    residual_valid = (
        y_valid - anchor_valid
    )

    # REAL + PLACEBO
    (
        train_cf,
        train_cf_diag,
    ) = build_strictly_past_platoon_features(
        df_train_hbeb
    )
    (
        valid_cf,
        valid_cf_diag,
        platoon_bank,
    ) = build_validation_platoon_features(
        df_train_raw,
        df_valid_hbeb,
    )

    features_r1 = list(
        hbeb_features
    )
    features_rp = (
        list(hbeb_features)
        + [PLACEBO_FEATURE]
    )
    features_r2 = (
        list(hbeb_features)
        + [CF_FEATURE]
    )

    # R0
    pred_r0 = np.clip(
        anchor_valid,
        0.0,
        1.0,
    )
    bs_r0 = brier_score(
        y_valid,
        pred_r0,
    )
    bss_r0 = bss_raw(
        y_valid,
        pred_r0,
    )

    # R1
    print()
    print("[R1_RESIDUAL_CONTROL]")

    model_r1 = train_residual_model(
        train_cf,
        valid_cf,
        features_r1,
        residual_train,
        residual_valid,
    )
    pred_resid_r1 = model_r1.predict(
        sanitize_categoricals(
            valid_cf[features_r1],
            CAT_COLS,
        )
    )
    pred_r1 = np.clip(
        anchor_valid + pred_resid_r1,
        0.0,
        1.0,
    )
    bs_r1 = brier_score(
        y_valid,
        pred_r1,
    )
    bss_r1 = bss_raw(
        y_valid,
        pred_r1,
    )
    iter_r1 = (
        int(
            model_r1.get_best_iteration()
        )
        + 1
    )

    # RP
    print()
    print("[RP_RESIDUAL_PLACEBO]")

    model_rp = train_residual_model(
        train_cf,
        valid_cf,
        features_rp,
        residual_train,
        residual_valid,
    )
    pred_resid_rp = model_rp.predict(
        sanitize_categoricals(
            valid_cf[features_rp],
            CAT_COLS,
        )
    )
    pred_rp = np.clip(
        anchor_valid + pred_resid_rp,
        0.0,
        1.0,
    )
    bs_rp = brier_score(
        y_valid,
        pred_rp,
    )
    bss_rp = bss_raw(
        y_valid,
        pred_rp,
    )
    iter_rp = (
        int(
            model_rp.get_best_iteration()
        )
        + 1
    )

    # R2
    print()
    print("[R2_RESIDUAL_REAL]")

    model_r2 = train_residual_model(
        train_cf,
        valid_cf,
        features_r2,
        residual_train,
        residual_valid,
    )
    pred_resid_r2 = model_r2.predict(
        sanitize_categoricals(
            valid_cf[features_r2],
            CAT_COLS,
        )
    )
    pred_r2 = np.clip(
        anchor_valid + pred_resid_r2,
        0.0,
        1.0,
    )
    bs_r2 = brier_score(
        y_valid,
        pred_r2,
    )
    bss_r2 = bss_raw(
        y_valid,
        pred_r2,
    )
    iter_r2 = (
        int(
            model_r2.get_best_iteration()
        )
        + 1
    )

    causal = {
        "placebo_vs_control_bss": (
            bss_rp - bss_r1
        ),
        "real_vs_placebo_bss": (
            bss_r2 - bss_rp
        ),
        "real_vs_control_bss": (
            bss_r2 - bss_r1
        ),
        "placebo_vs_control_brier": (
            bs_r1 - bs_rp
        ),
        "real_vs_placebo_brier": (
            bs_rp - bs_r2
        ),
        "real_vs_control_brier": (
            bs_r1 - bs_r2
        ),
    }

    # seen / unseen
    seen = (
        valid_cf[
            "cf_platoon_seen"
        ].to_numpy(np.int8)
        == 1
    )
    breakdown_rows = []

    for label, mask in [
        ("seen", seen),
        ("unseen", ~seen),
    ]:
        if not np.any(mask):
            continue

        r1_bs = brier_score(
            y_valid[mask],
            pred_r1[mask],
        )
        rp_bs = brier_score(
            y_valid[mask],
            pred_rp[mask],
        )
        r2_bs = brier_score(
            y_valid[mask],
            pred_r2[mask],
        )

        r1_mse = float(
            np.mean(
                (
                    pred_resid_r1[mask]
                    - residual_valid[mask]
                ) ** 2
            )
        )
        rp_mse = float(
            np.mean(
                (
                    pred_resid_rp[mask]
                    - residual_valid[mask]
                ) ** 2
            )
        )
        r2_mse = float(
            np.mean(
                (
                    pred_resid_r2[mask]
                    - residual_valid[mask]
                ) ** 2
            )
        )

        breakdown_rows.append(
            {
                "validation_season": int(
                    validation_season
                ),
                "group": label,
                "n_rows": int(
                    mask.sum()
                ),
                "share": float(
                    mask.mean()
                ),
                "r0_brier": brier_score(
                    y_valid[mask],
                    pred_r0[mask],
                ),
                "r1_brier": r1_bs,
                "rp_brier": rp_bs,
                "r2_brier": r2_bs,
                "delta_brier_placebo_vs_control": (
                    r1_bs - rp_bs
                ),
                "delta_brier_real_vs_placebo": (
                    rp_bs - r2_bs
                ),
                "delta_brier_real_vs_control": (
                    r1_bs - r2_bs
                ),
                "residual_mse_r1": r1_mse,
                "residual_mse_rp": rp_mse,
                "residual_mse_r2": r2_mse,
                "delta_residual_mse_placebo_vs_control": (
                    r1_mse - rp_mse
                ),
                "delta_residual_mse_real_vs_placebo": (
                    rp_mse - r2_mse
                ),
            }
        )

    domains = domain_metrics(
        df_valid_raw,
        y_valid,
        {
            "r0": pred_r0,
            "r1": pred_r1,
            "rp": pred_rp,
            "r2": pred_r2,
        },
    )

    real_v = valid_cf[
        CF_FEATURE
    ].to_numpy(np.float64)
    placebo_v = valid_cf[
        PLACEBO_FEATURE
    ].to_numpy(np.float64)

    if np.sum(seen) >= 2:
        corr = float(
            np.corrcoef(
                real_v[seen],
                placebo_v[seen],
            )[0, 1]
        )
    else:
        corr = float("nan")

    placebo_sanity = {
        "real_mean": float(
            np.mean(real_v)
        ),
        "placebo_mean": float(
            np.mean(placebo_v)
        ),
        "real_std": float(
            np.std(real_v)
        ),
        "placebo_std": float(
            np.std(placebo_v)
        ),
        "seen_real_placebo_corr": corr,
        "seen_rate": float(
            np.mean(seen)
        ),
    }

    fold_dir = (
        VALIDATION_DIR
        / f"fold_{validation_season}"
    )
    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_dict = {
        "season": int(
            validation_season
        ),
        "target": y_valid,
        "anchor": anchor_valid,
        "residual_target": residual_valid,
        "pred_r0_anchor_only": pred_r0,
        "pred_residual_r1": pred_resid_r1,
        "pred_r1": pred_r1,
        "pred_residual_rp": pred_resid_rp,
        "pred_rp": pred_rp,
        "pred_residual_r2": pred_resid_r2,
        "pred_r2": pred_r2,
        CF_FEATURE: real_v,
        PLACEBO_FEATURE: placebo_v,
        "cf_platoon_seen": valid_cf[
            "cf_platoon_seen"
        ].to_numpy(np.int8),
    }

    if ID in df_valid_raw.columns:
        pred_dict = {
            ID: df_valid_raw[
                ID
            ].to_numpy(),
            **pred_dict,
        }
    else:
        pred_dict = {
            "row_index": (
                df_valid_raw.index.to_numpy()
            ),
            **pred_dict,
        }

    pred_df = pd.DataFrame(
        pred_dict
    )
    pred_df.to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    meta = {
        "experiment_name": EXPERIMENT_NAME,
        "train_seasons": [
            int(x)
            for x in train_seasons
        ],
        "validation_season": int(
            validation_season
        ),
        "n_train": int(
            len(df_train_raw)
        ),
        "n_validation": int(
            len(df_valid_raw)
        ),
        "dynamic_anchor": {
            "strength": (
                DYNAMIC_ANCHOR_STRENGTH
            ),
            "train_diagnostics": (
                anchor_train_diag
            ),
            "validation_diagnostics": (
                anchor_valid_diag
            ),
        },
        "conditional": {
            "real_feature": CF_FEATURE,
            "placebo_feature": (
                PLACEBO_FEATURE
            ),
            "strength": CF_STRENGTH,
            "unseen_encoding": "zero",
            "placebo_policy": (
                "deterministic "
                "within-batter_hand "
                "permutation of "
                "strictly-past cell delta"
            ),
            "train_diagnostics": (
                train_cf_diag
            ),
            "validation_diagnostics": (
                valid_cf_diag
            ),
            "placebo_sanity": (
                placebo_sanity
            ),
        },
        "r0_anchor_only": {
            "brier": bs_r0,
            "bss": bss_r0,
        },
        "r1_residual_control": {
            "brier": bs_r1,
            "bss": bss_r1,
            "iterations": iter_r1,
        },
        "rp_residual_placebo": {
            "brier": bs_rp,
            "bss": bss_rp,
            "iterations": iter_rp,
        },
        "r2_residual_real": {
            "brier": bs_r2,
            "bss": bss_r2,
            "iterations": iter_r2,
        },
        "causal_contrasts": causal,
        "domain_metrics": domains,
        "seen_unseen": breakdown_rows,
        "elapsed_seconds": float(
            time.time() - t0
        ),
    }

    write_json(
        fold_dir / "metadata.json",
        meta,
    )

    return {
        "metadata": meta,
        "pred_df": pred_df,
        "breakdown_rows": (
            breakdown_rows
        ),
    }


# =========================================================
# Main
# =========================================================

def main():
    t0 = time.time()

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. "
            "Change EXPERIMENT_NAME."
        )
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. "
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

    df = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
    )
    ensure_columns(
        df,
        BASE_FEATURES + [TARGET],
        "train.csv",
    )

    print("=" * 108)
    print(
        f"Experiment           : "
        f"{EXPERIMENT_NAME}"
    )
    print(
        "Ablation             : "
        "R1 Control vs RP Placebo "
        "vs R2 Real Platoon"
    )
    print(
        "Dynamic Anchor       : "
        "v013 row-local ASOF "
        "current-season pitcher state"
    )
    print(
        "Real Feature         : "
        "v014 A0_ZERO "
        "cf_platoon_delta"
    )
    print(
        "Placebo              : "
        "within-batter_hand "
        "deterministic bank permutation"
    )
    print(
        f"Placebo seed         : "
        f"{PLACEBO_SEED}"
    )
    print(
        f"Anchor K             : "
        f"{DYNAMIC_ANCHOR_STRENGTH}"
    )
    print(
        f"Platoon K            : "
        f"{CF_STRENGTH}"
    )
    print(
        f"Train shape          : "
        f"{df.shape}"
    )
    print("=" * 108)

    audit_rows = (
        run_conditional_leakage_audit(
            df
        )
    )
    pd.DataFrame(
        audit_rows
    ).to_csv(
        VALIDATION_DIR
        / "conditional_leakage_audit.csv",
        index=False,
    )

    print(
        "[Conditional leakage audit] "
        "PASS (real + placebo)"
    )

    fold_metas = []
    oof_parts = []
    breakdown_all = []

    latest_iters = {
        "r1": None,
        "rp": None,
        "r2": None,
    }

    for (
        fold_idx,
        (
            train_seasons,
            valid_season,
        ),
    ) in enumerate(
        EXPANDING_FOLDS,
        1,
    ):
        print()
        print("=" * 108)
        print(
            f"Fold {fold_idx}: "
            f"{min(train_seasons)}~"
            f"{max(train_seasons)} "
            f"-> {valid_season}"
        )
        print("=" * 108)

        result = train_fold(
            df,
            train_seasons,
            valid_season,
        )
        meta = result[
            "metadata"
        ]

        fold_metas.append(
            meta
        )
        oof_parts.append(
            result["pred_df"]
        )
        breakdown_all.extend(
            result[
                "breakdown_rows"
            ]
        )

        latest_iters["r1"] = (
            meta[
                "r1_residual_control"
            ]["iterations"]
        )
        latest_iters["rp"] = (
            meta[
                "rp_residual_placebo"
            ]["iterations"]
        )
        latest_iters["r2"] = (
            meta[
                "r2_residual_real"
            ]["iterations"]
        )

        c = meta[
            "causal_contrasts"
        ]

        print()
        print(
            "[Fold causal comparison]"
        )
        print(
            "R1 CONTROL  "
            f"BSS="
            f"{meta['r1_residual_control']['bss']:.2f} "
            f"iter="
            f"{meta['r1_residual_control']['iterations']}"
        )
        print(
            "RP PLACEBO  "
            f"BSS="
            f"{meta['rp_residual_placebo']['bss']:.2f} "
            f"iter="
            f"{meta['rp_residual_placebo']['iterations']} "
            f"ΔvsR1="
            f"{c['placebo_vs_control_bss']:+.2f}"
        )
        print(
            "R2 REAL     "
            f"BSS="
            f"{meta['r2_residual_real']['bss']:.2f} "
            f"iter="
            f"{meta['r2_residual_real']['iterations']} "
            f"ΔvsRP="
            f"{c['real_vs_placebo_bss']:+.2f} "
            f"ΔvsR1="
            f"{c['real_vs_control_bss']:+.2f}"
        )
        print(
            "Placebo sanity | "
            "corr(real,placebo|seen)="
            f"{meta['conditional']['placebo_sanity']['seen_real_placebo_corr']:.4f}"
        )

    oof = pd.concat(
        oof_parts,
        axis=0,
        ignore_index=True,
    )
    oof.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        breakdown_all
    ).to_csv(
        VALIDATION_DIR
        / "causal_seen_unseen.csv",
        index=False,
    )

    summary_rows = []

    for m in fold_metas:
        c = m[
            "causal_contrasts"
        ]

        summary_rows.append(
            {
                "validation_season": (
                    m[
                        "validation_season"
                    ]
                ),
                "r1_bss": (
                    m[
                        "r1_residual_control"
                    ]["bss"]
                ),
                "r1_iter": (
                    m[
                        "r1_residual_control"
                    ]["iterations"]
                ),
                "rp_bss": (
                    m[
                        "rp_residual_placebo"
                    ]["bss"]
                ),
                "rp_iter": (
                    m[
                        "rp_residual_placebo"
                    ]["iterations"]
                ),
                "r2_bss": (
                    m[
                        "r2_residual_real"
                    ]["bss"]
                ),
                "r2_iter": (
                    m[
                        "r2_residual_real"
                    ]["iterations"]
                ),
                "placebo_vs_control_bss": (
                    c[
                        "placebo_vs_control_bss"
                    ]
                ),
                "real_vs_placebo_bss": (
                    c[
                        "real_vs_placebo_bss"
                    ]
                ),
                "real_vs_control_bss": (
                    c[
                        "real_vs_control_bss"
                    ]
                ),
                "placebo_vs_control_brier": (
                    c[
                        "placebo_vs_control_brier"
                    ]
                ),
                "real_vs_placebo_brier": (
                    c[
                        "real_vs_placebo_brier"
                    ]
                ),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        VALIDATION_DIR
        / "summary.csv",
        index=False,
    )

    pooled = {}

    for name, col in [
        (
            "R1_CONTROL",
            "pred_r1",
        ),
        (
            "RP_PLACEBO",
            "pred_rp",
        ),
        (
            "R2_REAL",
            "pred_r2",
        ),
    ]:
        pooled[name] = {
            "brier": brier_score(
                oof["target"],
                oof[col],
            ),
            "bss": bss_raw(
                oof["target"],
                oof[col],
            ),
        }

    pooled_placebo = (
        pooled[
            "RP_PLACEBO"
        ]["bss"]
        - pooled[
            "R1_CONTROL"
        ]["bss"]
    )
    pooled_semantic = (
        pooled[
            "R2_REAL"
        ]["bss"]
        - pooled[
            "RP_PLACEBO"
        ]["bss"]
    )
    pooled_total = (
        pooled[
            "R2_REAL"
        ]["bss"]
        - pooled[
            "R1_CONTROL"
        ]["bss"]
    )

    print()
    print("=" * 108)
    print(
        "Pooled OOF causal decomposition"
    )
    print("=" * 108)

    for name, d in pooled.items():
        print(
            f"{name:12s} "
            f"Brier={d['brier']:.8f} "
            f"BSS={d['bss']:.2f}"
        )

    print()
    print(
        "RP - R1  feature/path effect : "
        f"{pooled_placebo:+.2f} BSS"
    )
    print(
        "R2 - RP  semantic effect     : "
        f"{pooled_semantic:+.2f} BSS"
    )
    print(
        "R2 - R1  total effect        : "
        f"{pooled_total:+.2f} BSS"
    )

    # -----------------------------------------------------
    # Final models: reproducibility only
    # -----------------------------------------------------
    (
        df_hbeb,
        hbeb_features,
        final_hbeb_priors,
    ) = apply_hbeb_configs(
        df,
        df,
        BASE_FEATURES,
        priors=None,
    )

    (
        anchor_train,
        final_anchor_train_diag,
    ) = (
        build_strictly_past_dynamic_anchors(
            df
        )
    )

    residual_target = (
        df[TARGET].to_numpy(
            np.float64
        )
        - anchor_train
    )

    (
        df_cf,
        final_cf_train_diag,
    ) = (
        build_strictly_past_platoon_features(
            df_hbeb
        )
    )

    feature_manifests = {
        "R1_CONTROL": list(
            hbeb_features
        ),
        "RP_PLACEBO": (
            list(hbeb_features)
            + [PLACEBO_FEATURE]
        ),
        "R2_REAL": (
            list(hbeb_features)
            + [CF_FEATURE]
        ),
    }

    final_iter_map = {
        "R1_CONTROL": int(
            latest_iters["r1"]
        ),
        "RP_PLACEBO": int(
            latest_iters["rp"]
        ),
        "R2_REAL": int(
            latest_iters["r2"]
        ),
    }

    models = {}

    for name in [
        "R1_CONTROL",
        "RP_PLACEBO",
        "R2_REAL",
    ]:
        print()
        print(
            f"[Final {name}] "
            f"iterations="
            f"{final_iter_map[name]}"
        )

        params = dict(
            MODEL_PARAMS
        )
        params[
            "iterations"
        ] = final_iter_map[
            name
        ]

        model = CatBoostRegressor(
            **params
        )
        model.fit(
            sanitize_categoricals(
                df_cf[
                    feature_manifests[
                        name
                    ]
                ],
                CAT_COLS,
            ),
            residual_target,
            cat_features=CAT_COLS,
            verbose=100,
        )

        models[name] = model

    final_dynamic_bank = (
        build_pitcher_history_bank(
            df
        )
    )
    final_platoon_bank = (
        build_platoon_bank(
            df
        )
    )

    artifact = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressorResidual"
            "CausalAblation"
        ),
        "objective": "RMSE",
        "models": models,
        "features": (
            feature_manifests
        ),
        "cat_cols": CAT_COLS,
        "base_features": (
            BASE_FEATURES
        ),
        "hbeb_priors": (
            final_hbeb_priors
        ),
        "hbeb_configs": (
            HBEB_CONFIGS
        ),
        "dynamic_anchor": {
            "strength": (
                DYNAMIC_ANCHOR_STRENGTH
            ),
            "final_2025_bank": (
                final_dynamic_bank
            ),
            "training_diagnostics": (
                final_anchor_train_diag
            ),
        },
        "conditional": {
            "real_feature": (
                CF_FEATURE
            ),
            "placebo_feature": (
                PLACEBO_FEATURE
            ),
            "placebo_seed": (
                PLACEBO_SEED
            ),
            "placebo_policy": (
                "within-batter_hand "
                "deterministic permutation "
                "of strictly-past cell deltas"
            ),
            "strength": (
                CF_STRENGTH
            ),
            "unseen_encoding": (
                "zero"
            ),
            "final_2025_bank": (
                final_platoon_bank
            ),
            "training_diagnostics": (
                final_cf_train_diag
            ),
        },
        "final_iterations": (
            final_iter_map
        ),
        "prediction_policy": (
            "anchor_plus_residual_"
            "then_clip_0_1"
        ),
        "deployment_note": (
            "RP_PLACEBO is "
            "diagnostic only."
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    final_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "validation_method": (
            "expanding_window"
        ),
        "ablation": (
            "R1_control_vs_RP_placebo"
            "_vs_R2_real"
        ),
        "placebo_policy": {
            "seed": (
                PLACEBO_SEED
            ),
            "within": (
                "batter_hand"
            ),
            "source": (
                "strictly-past "
                "frozen-bank "
                "cell delta"
            ),
            "unseen_encoding": (
                "zero"
            ),
            "deployment_candidate": (
                False
            ),
        },
        "folds": fold_metas,
        "pooled": pooled,
        "causal_decomposition": {
            "placebo_vs_control_bss": (
                pooled_placebo
            ),
            "real_vs_placebo_bss": (
                pooled_semantic
            ),
            "real_vs_control_bss": (
                pooled_total
            ),
        },
        "final_iterations": (
            final_iter_map
        ),
        "elapsed_seconds": float(
            time.time() - t0
        ),
    }

    write_json(
        METADATA_PATH,
        final_metadata,
    )

    try:
        shutil.copy2(
            Path(
                __file__
            ).resolve(),
            MODEL_DIR
            / "train.py",
        )
    except Exception:
        pass

    print()
    print("=" * 108)
    print("완료")
    print(
        f"Model      : "
        f"{MODEL_PATH}"
    )
    print(
        f"Validation : "
        f"{VALIDATION_DIR}"
    )
    print("=" * 108)

    print("[판정]")
    print(
        "1) RP-R1 ≈ feature 추가/"
        "tree-path 효과"
    )
    print(
        "2) R2-RP = 실제 conditional "
        "semantics 핵심 추정치"
    )
    print(
        "3) 성공: R2-RP가 2023/2024 "
        "및 pooled에서 양수"
    )
    print(
        "4) causal_seen_unseen.csv에서 "
        "seen의 R2-RP 개선 확인"
    )
    print(
        "5) RP_PLACEBO는 진단 전용이며 "
        "제출 후보가 아님"
    )


if __name__ == "__main__":
    main()
