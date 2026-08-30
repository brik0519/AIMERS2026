# train.py: cb_reg_v018
# V18-lite architecture
#
# Layer 0 : Row-local Dynamic Anchor
# Layer 1 : General Residual CatBoost
# Layer 2 : Conditional Residual CatBoost
#
# 핵심 구조
# =========
#
#   anchor_i
#       ↓
#   r_general_target = y - anchor
#       ↓
#   General Residual CatBoost
#       ↓
#   OOF general residual prediction
#       ↓
#   r_conditional_target = y - anchor - general_oof_pred
#       ↓
#   Conditional Residual CatBoost
#       ↓
#   final = clip(anchor + gamma_g * general_pred + gamma_c * conditional_pred, 0, 1)
#
# 중요
# ----
# 1) Layer 2 target 생성 시 Layer 1의 in-sample prediction을 절대 사용하지 않는다.
#    반드시 season-expanding OOF prediction을 사용한다.
# 2) Conditional feature는 strictly-past pitcher_id × batter_hand delta(A0_ZERO)를 사용한다.
# 3) 첫 실험에서는 gamma_g = gamma_c = 1.0으로 고정한다.
# 4) Dynamic anchor / conditional bank는 모두 train-only frozen state + row-local ASOF만 사용한다.
# 5) test 내부 groupby/mean/rank/rolling/transform을 사용하지 않는다.
#
# 출력
# ----
# model/cb_reg_v018/cb_reg_v018.pkl
# model/cb_reg_v018/final_metadata.json
# validation/cb_reg_v018/oof_pred.csv
# validation/cb_reg_v018/layer_comparison.csv
# validation/cb_reg_v018/fold_YYYY/metadata.json
#
# 이 파일은 구조 실험용이다.
# 우선 Layer 2의 순수 증분:
#
#   BSS(anchor + general + conditional)
# - BSS(anchor + general)
#
# 이 2021/2022/2023/2024에서 어떻게 움직이는지가 핵심 판정값이다.

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =============================================================================
# 경로
# =============================================================================

EXPERIMENT_NAME = "cb_reg_v018"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
METADATA_PATH = MODEL_DIR / "final_metadata.json"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =============================================================================
# 기본 설정
# =============================================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

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

# 처음에는 구조 자체만 본다.
GAMMA_GENERAL = 1.0
GAMMA_CONDITIONAL = 1.0

# Dynamic Anchor
ANCHOR_STRENGTH = 80.0

# Platoon conditional
PLATOON_STRENGTH = 80.0
CF_KEY_COLS = ["pitcher_id", "batter_hand"]
CF_PARENT_COLS = ["pitcher_id"]
CF_COL = "cf_platoon_delta"

# Layer 2는 "conditional residual" 전용이지만,
# conditional만 주면 capacity가 너무 작을 수 있으므로
# 1차 최소형은 General context + conditional feature를 같이 준다.
# 순수 conditional-only arm은 후속 ablation으로 분리한다.
LAYER2_USE_BASE_FEATURES = True

FINAL_ITERATION_STRATEGY = "latest_fold"


# =============================================================================
# 공통 유틸
# =============================================================================

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
    if base <= 0.0:
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


# =============================================================================
# HB-EB — cb_reg_v003-compatible
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

HBEB_MIN_GROUP_ROWS = 30


def _safe_mom_beta(mu, var):
    mu = float(np.clip(mu, 1e-6, 1.0 - 1e-6))
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
    valid_prior = []

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
        valid_prior.append(bool(ok))

    prior_df = stats[list(group_cols)].copy()
    prior_df["alpha"] = alphas
    prior_df["beta"] = betas
    prior_df["valid_group_prior"] = valid_prior

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

    success_count = n * rate
    denom = n + alpha + beta

    eb = (success_count + alpha) / denom
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


# =============================================================================
# Layer 0 — Dynamic Anchor
# =============================================================================

def build_pitcher_history_bank(history_df):
    ensure_columns(
        history_df,
        ["pitcher_id", TARGET],
        "build_pitcher_history_bank",
    )

    if len(history_df) == 0:
        return {
            "pitcher_table": pd.DataFrame(
                columns=["pitcher_id", "hist_n", "hist_success"]
            ),
            "global_rate": 0.5,
            "history_seasons": [],
        }

    y = pd.to_numeric(history_df[TARGET], errors="raise").to_numpy(np.float64)

    temp = history_df[["pitcher_id"]].copy()
    temp["__y"] = y

    table = (
        temp.groupby("pitcher_id", dropna=False, observed=True)["__y"]
        .agg(hist_n="size", hist_success="sum")
        .reset_index()
    )

    return {
        "pitcher_table": table,
        "global_rate": float(np.mean(y)),
        "history_seasons": sorted(
            int(x) for x in pd.unique(history_df[SEASON_COL])
        ),
    }


def reconstruct_current_season_pitcher_state(df_apply, bank):
    """
    현재 행의 asof_pitcher_n / asof_pitcher_success_rate 에서
    frozen history cumulative count를 빼 current-season pitcher state를 복원한다.

    current_n = max(asof_n - history_n, 0)
    current_success = clip(asof_n*asof_rate - history_success, 0, current_n)

    anchor =
        (current_success + K * history_parent_rate)
        / (current_n + K)

    unseen / reconstruction anomaly는 history parent/global fallback으로 처리한다.
    """
    ensure_columns(
        df_apply,
        [
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
        ],
        "reconstruct_current_season_pitcher_state",
    )

    n_rows = len(df_apply)

    lookup = (
        df_apply[["pitcher_id"]]
        .reset_index(drop=True)
        .merge(
            bank["pitcher_table"],
            on="pitcher_id",
            how="left",
            sort=False,
        )
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

    asof_n = (
        pd.to_numeric(df_apply["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    asof_rate = (
        pd.to_numeric(df_apply["asof_pitcher_success_rate"], errors="coerce")
        .fillna(bank["global_rate"])
        .to_numpy(np.float64)
    )

    asof_n = np.clip(asof_n, 0.0, None)
    asof_rate = np.clip(asof_rate, 0.0, 1.0)

    asof_success = asof_n * asof_rate

    current_n_raw = asof_n - hist_n
    current_success_raw = asof_success - hist_success

    invalid = (
        (current_n_raw < -1e-8)
        | (current_success_raw < -1e-8)
        | (current_success_raw - current_n_raw > 1e-8)
    )

    current_n = np.clip(current_n_raw, 0.0, None)
    current_success = np.clip(current_success_raw, 0.0, current_n)

    hist_rate = np.divide(
        hist_success,
        np.maximum(hist_n, 1.0),
        out=np.full(n_rows, bank["global_rate"], dtype=np.float64),
        where=hist_n > 0.0,
    )

    anchor = (
        current_success + ANCHOR_STRENGTH * hist_rate
    ) / (
        current_n + ANCHOR_STRENGTH
    )

    return anchor.astype(np.float64), {
        "n_rows": int(n_rows),
        "anchor_mean": float(np.mean(anchor)) if n_rows else 0.0,
        "anchor_std": float(np.std(anchor)) if n_rows else 0.0,
        "current_n_mean": float(np.mean(current_n)) if n_rows else 0.0,
        "current_n_median": float(np.median(current_n)) if n_rows else 0.0,
        "invalid_reconstruction_rows": int(np.sum(invalid)),
        "invalid_reconstruction_rate": float(np.mean(invalid)) if n_rows else 0.0,
        "history_seasons": list(bank["history_seasons"]),
        "history_global_rate": float(bank["global_rate"]),
    }


def build_strictly_past_dynamic_anchor(df):
    pieces = []
    diagnostics = []

    seasons = sorted(int(x) for x in pd.unique(df[SEASON_COL]))
    season_int = df[SEASON_COL].astype(int)

    for season in seasons:
        current = df.loc[season_int == season]
        history = df.loc[season_int < season]

        if len(history) == 0:
            anchor = np.full(len(current), 0.5, dtype=np.float64)
            diag = {
                "n_rows": int(len(current)),
                "anchor_mean": 0.5,
                "anchor_std": 0.0,
                "current_n_mean": 0.0,
                "current_n_median": 0.0,
                "invalid_reconstruction_rows": 0,
                "invalid_reconstruction_rate": 0.0,
                "history_seasons": [],
                "history_global_rate": 0.5,
            }
        else:
            bank = build_pitcher_history_bank(history)
            anchor, diag = reconstruct_current_season_pitcher_state(current, bank)

        part = pd.Series(anchor, index=current.index, name="dynamic_anchor")
        pieces.append(part)
        diag["target_season"] = int(season)
        diagnostics.append(diag)

    result = pd.concat(pieces).sort_index().reindex(df.index)
    return result, diagnostics


def build_validation_dynamic_anchor(history_df, validation_df):
    valid_seasons = sorted(
        int(x) for x in pd.unique(validation_df[SEASON_COL])
    )
    if len(valid_seasons) != 1:
        raise ValueError("validation_df must contain exactly one season")

    validation_season = valid_seasons[0]

    if bool((history_df[SEASON_COL].astype(int) >= validation_season).any()):
        raise RuntimeError(
            "Dynamic anchor leakage guard failed: validation/future season in history."
        )

    bank = build_pitcher_history_bank(history_df)
    anchor, diag = reconstruct_current_season_pitcher_state(validation_df, bank)
    diag["target_season"] = validation_season

    return pd.Series(anchor, index=validation_df.index), diag, bank


# =============================================================================
# Layer 2 feature — Strictly-Past Platoon A0_ZERO
# =============================================================================

def build_platoon_bank(history_df):
    ensure_columns(
        history_df,
        CF_KEY_COLS + [TARGET],
        "build_platoon_bank",
    )

    if len(history_df) == 0:
        return {
            "cell_table": pd.DataFrame(
                columns=CF_KEY_COLS + ["cell_n", "cell_success"]
            ),
            "parent_table": pd.DataFrame(
                columns=CF_PARENT_COLS
                + ["parent_n", "parent_success", "parent_rate"]
            ),
            "global_rate": 0.5,
            "history_seasons": [],
        }

    y = pd.to_numeric(history_df[TARGET], errors="raise").to_numpy(np.float64)

    tmp = history_df[CF_KEY_COLS].copy()
    tmp[TARGET] = y

    cell = (
        tmp.groupby(CF_KEY_COLS, dropna=False, observed=True)[TARGET]
        .agg(cell_n="size", cell_success="sum")
        .reset_index()
    )

    parent = (
        history_df[CF_PARENT_COLS]
        .assign(__y=y)
        .groupby(CF_PARENT_COLS, dropna=False, observed=True)["__y"]
        .agg(parent_n="size", parent_success="sum")
        .reset_index()
    )
    parent["parent_rate"] = (
        parent["parent_success"] / parent["parent_n"].clip(lower=1)
    )

    return {
        "cell_table": cell,
        "parent_table": parent,
        "global_rate": float(np.mean(y)),
        "history_seasons": sorted(
            int(x) for x in pd.unique(history_df[SEASON_COL])
        ),
    }


def apply_platoon_bank(df_apply, bank):
    n_rows = len(df_apply)

    parent_lookup = (
        df_apply[CF_PARENT_COLS]
        .reset_index(drop=True)
        .merge(
            bank["parent_table"],
            on=CF_PARENT_COLS,
            how="left",
            sort=False,
        )
    )
    parent_rate = (
        pd.to_numeric(parent_lookup["parent_rate"], errors="coerce")
        .fillna(bank["global_rate"])
        .to_numpy(np.float64)
    )

    cell_lookup = (
        df_apply[CF_KEY_COLS]
        .reset_index(drop=True)
        .merge(
            bank["cell_table"],
            on=CF_KEY_COLS,
            how="left",
            sort=False,
        )
    )

    cell_n = (
        pd.to_numeric(cell_lookup["cell_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    cell_success = (
        pd.to_numeric(cell_lookup["cell_success"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )

    cell_n = np.clip(cell_n, 0.0, None)
    cell_success = np.clip(cell_success, 0.0, None)

    posterior_rate = (
        cell_success + PLATOON_STRENGTH * parent_rate
    ) / (
        cell_n + PLATOON_STRENGTH
    )

    delta = posterior_rate - parent_rate

    out = df_apply.copy()
    # v014 A0_ZERO: unseen cell == delta 0
    out[CF_COL] = np.where(cell_n > 0.0, delta, 0.0).astype(np.float64)

    return out, {
        "n_rows": int(n_rows),
        "unseen_cell_rows": int(np.sum(cell_n <= 0.0)),
        "unseen_cell_rate": float(np.mean(cell_n <= 0.0)) if n_rows else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(out[CF_COL]))) if n_rows else 0.0,
        "history_seasons": list(bank["history_seasons"]),
    }


def build_strictly_past_platoon_features(df):
    pieces = []
    diagnostics = []

    seasons = sorted(int(x) for x in pd.unique(df[SEASON_COL]))
    season_int = df[SEASON_COL].astype(int)

    for season in seasons:
        current = df.loc[season_int == season]
        history = df.loc[season_int < season]

        if len(history) == 0:
            transformed = current.copy()
            transformed[CF_COL] = 0.0
            diag = {
                "n_rows": int(len(current)),
                "unseen_cell_rows": int(len(current)),
                "unseen_cell_rate": 1.0 if len(current) else 0.0,
                "mean_abs_delta": 0.0,
                "history_seasons": [],
            }
        else:
            bank = build_platoon_bank(history)
            transformed, diag = apply_platoon_bank(current, bank)

        diag["target_season"] = int(season)
        diagnostics.append(diag)
        pieces.append(transformed)

    result = pd.concat(pieces).sort_index().reindex(df.index)
    return result, diagnostics


def build_validation_platoon_features(history_df, validation_df):
    validation_season = int(
        pd.unique(validation_df[SEASON_COL])[0]
    )

    if bool((history_df[SEASON_COL].astype(int) >= validation_season).any()):
        raise RuntimeError(
            "Conditional leakage guard failed: validation/future season in history."
        )

    bank = build_platoon_bank(history_df)
    transformed, diag = apply_platoon_bank(validation_df, bank)
    diag["target_season"] = validation_season

    return transformed, diag, bank


# =============================================================================
# CatBoost helper
# =============================================================================

def fit_model(X_train, y_train, X_valid, y_valid, cat_cols, label):
    print()
    print(f"[{label}]")

    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )

    best_iteration = int(model.get_best_iteration())
    return model, best_iteration, best_iteration + 1


# =============================================================================
# Layer 1 OOF builder
# =============================================================================

def build_general_oof_for_training(df_train_raw, hbeb_features, hbeb_priors):
    """
    Layer 2 training target용 General Residual OOF를 만든다.

    train 구간 안에서 다시 expanding:
      earliest season은 OOF 생성 불가 -> 0 residual prediction
      이후 season s는 season < s 로 General Residual model 학습 후 s 예측

    이 prediction만 Layer 2 target 생성에 사용한다.
    """
    season_int = df_train_raw[SEASON_COL].astype(int)
    seasons = sorted(int(x) for x in pd.unique(df_train_raw[SEASON_COL]))

    general_oof = pd.Series(0.0, index=df_train_raw.index, dtype=np.float64)
    records = []

    for season in seasons:
        valid_mask = season_int == season
        hist_mask = season_int < season

        df_valid = df_train_raw.loc[valid_mask].copy()
        df_hist = df_train_raw.loc[hist_mask].copy()

        if len(df_hist) == 0:
            records.append({
                "target_season": int(season),
                "status": "cold_start_zero",
                "n_train": 0,
                "n_valid": int(len(df_valid)),
                "best_iteration": None,
            })
            continue

        # HBEB for inner OOF
        hist_hbeb, hist_features, inner_priors = apply_hbeb_configs(
            df_hist, df_hist, BASE_FEATURES, priors=None
        )
        valid_hbeb, valid_features, _ = apply_hbeb_configs(
            df_hist, df_valid, BASE_FEATURES, priors=inner_priors
        )

        if hist_features != valid_features:
            raise RuntimeError("Inner OOF HBEB manifest mismatch.")

        # Dynamic anchor
        hist_anchor, _ = build_strictly_past_dynamic_anchor(df_hist)
        valid_anchor, _, _ = build_validation_dynamic_anchor(df_hist, df_valid)

        hist_hbeb["dynamic_anchor"] = hist_anchor
        valid_hbeb["dynamic_anchor"] = valid_anchor

        y_hist = hist_hbeb[TARGET].to_numpy(np.float64)
        y_valid = valid_hbeb[TARGET].to_numpy(np.float64)

        target_hist = y_hist - hist_hbeb["dynamic_anchor"].to_numpy(np.float64)
        target_valid = y_valid - valid_hbeb["dynamic_anchor"].to_numpy(np.float64)

        hist_hbeb = sanitize_categoricals(hist_hbeb, CAT_COLS)
        valid_hbeb = sanitize_categoricals(valid_hbeb, CAT_COLS)

        model, best_it, final_it = fit_model(
            hist_hbeb[hist_features],
            target_hist,
            valid_hbeb[valid_features],
            target_valid,
            CAT_COLS,
            f"Layer1 inner OOF -> {season}",
        )

        pred = model.predict(valid_hbeb[valid_features])
        general_oof.loc[df_valid.index] = pred

        records.append({
            "target_season": int(season),
            "status": "trained",
            "n_train": int(len(df_hist)),
            "n_valid": int(len(df_valid)),
            "best_iteration": int(best_it),
            "final_iterations": int(final_it),
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred)),
        })

    return general_oof, records


# =============================================================================
# Fold
# =============================================================================

def train_fold(df_all, train_seasons, validation_season):
    start = time.time()

    season_int = df_all[SEASON_COL].astype(int)
    df_train_raw = df_all.loc[season_int.isin(train_seasons)].copy()
    df_valid_raw = df_all.loc[season_int == validation_season].copy()

    print()
    print("=" * 96)
    print(
        f"Fold: {min(train_seasons)}~{max(train_seasons)} -> {validation_season}"
    )
    print("=" * 96)

    # -----------------------------------------------------------------
    # HB-EB
    # -----------------------------------------------------------------
    df_train_hbeb, hbeb_features, hbeb_priors = apply_hbeb_configs(
        df_train_raw,
        df_train_raw,
        BASE_FEATURES,
        priors=None,
    )
    df_valid_hbeb, valid_hbeb_features, _ = apply_hbeb_configs(
        df_train_raw,
        df_valid_raw,
        BASE_FEATURES,
        priors=hbeb_priors,
    )

    if hbeb_features != valid_hbeb_features:
        raise RuntimeError("HBEB feature mismatch.")

    # -----------------------------------------------------------------
    # Layer 0: Dynamic Anchor
    # -----------------------------------------------------------------
    train_anchor, train_anchor_diag = build_strictly_past_dynamic_anchor(
        df_train_raw
    )
    valid_anchor, valid_anchor_diag, anchor_bank = build_validation_dynamic_anchor(
        df_train_raw,
        df_valid_raw,
    )

    df_train_hbeb["dynamic_anchor"] = train_anchor
    df_valid_hbeb["dynamic_anchor"] = valid_anchor

    y_train = df_train_hbeb[TARGET].to_numpy(np.float64)
    y_valid = df_valid_hbeb[TARGET].to_numpy(np.float64)

    anchor_train = train_anchor.to_numpy(np.float64)
    anchor_valid = valid_anchor.to_numpy(np.float64)

    target_general_train = y_train - anchor_train
    target_general_valid = y_valid - anchor_valid

    # -----------------------------------------------------------------
    # Layer 1: General Residual
    # -----------------------------------------------------------------
    train_l1 = sanitize_categoricals(df_train_hbeb, CAT_COLS)
    valid_l1 = sanitize_categoricals(df_valid_hbeb, CAT_COLS)

    general_model, general_best_it, general_final_it = fit_model(
        train_l1[hbeb_features],
        target_general_train,
        valid_l1[hbeb_features],
        target_general_valid,
        CAT_COLS,
        "Layer1 General Residual",
    )

    general_valid_pred = general_model.predict(valid_l1[hbeb_features])

    pred_anchor = np.clip(anchor_valid, 0.0, 1.0)
    pred_general_raw = anchor_valid + GAMMA_GENERAL * general_valid_pred
    pred_general = np.clip(pred_general_raw, 0.0, 1.0)

    anchor_brier = brier_score(y_valid, pred_anchor)
    anchor_bss = bss_raw(y_valid, pred_anchor)

    general_brier = brier_score(y_valid, pred_general)
    general_bss = bss_raw(y_valid, pred_general)

    # -----------------------------------------------------------------
    # Layer 1 OOF predictions for Layer 2 training target
    # -----------------------------------------------------------------
    print()
    print("[Layer1 OOF generation for Layer2 target]")
    general_train_oof, inner_oof_meta = build_general_oof_for_training(
        df_train_raw,
        hbeb_features,
        hbeb_priors,
    )

    # -----------------------------------------------------------------
    # Layer 2 conditional feature
    # -----------------------------------------------------------------
    train_cf, train_cf_diag = build_strictly_past_platoon_features(
        df_train_hbeb
    )
    valid_cf, valid_cf_diag, conditional_bank = (
        build_validation_platoon_features(
            df_train_raw,
            df_valid_hbeb,
        )
    )

    # Layer 2 target:
    # y - anchor - Layer1 OOF prediction
    target_conditional_train = (
        y_train
        - anchor_train
        - GAMMA_GENERAL * general_train_oof.to_numpy(np.float64)
    )

    # validation target은 현재 fold의 honest Layer1 valid prediction을 사용
    target_conditional_valid = (
        y_valid
        - anchor_valid
        - GAMMA_GENERAL * general_valid_pred
    )

    if LAYER2_USE_BASE_FEATURES:
        layer2_features = list(hbeb_features) + [CF_COL]
        layer2_cat_cols = list(CAT_COLS)
    else:
        layer2_features = [CF_COL]
        layer2_cat_cols = []

    train_cf = sanitize_categoricals(train_cf, CAT_COLS)
    valid_cf = sanitize_categoricals(valid_cf, CAT_COLS)

    conditional_model, cond_best_it, cond_final_it = fit_model(
        train_cf[layer2_features],
        target_conditional_train,
        valid_cf[layer2_features],
        target_conditional_valid,
        layer2_cat_cols,
        "Layer2 Conditional Residual",
    )

    conditional_valid_pred = conditional_model.predict(
        valid_cf[layer2_features]
    )

    pred_final_raw = (
        anchor_valid
        + GAMMA_GENERAL * general_valid_pred
        + GAMMA_CONDITIONAL * conditional_valid_pred
    )
    pred_final = np.clip(pred_final_raw, 0.0, 1.0)

    final_brier = brier_score(y_valid, pred_final)
    final_bss = bss_raw(y_valid, pred_final)

    layer2_delta_brier = general_brier - final_brier
    layer2_delta_bss = final_bss - general_bss

    print()
    print("[Fold comparison]")
    print(
        f"L0 Anchor-only       Brier={anchor_brier:.8f} "
        f"BSS={anchor_bss:.2f}"
    )
    print(
        f"L1 General Residual  Brier={general_brier:.8f} "
        f"BSS={general_bss:.2f} iter={general_final_it}"
    )
    print(
        f"L2 + Conditional     Brier={final_brier:.8f} "
        f"BSS={final_bss:.2f} iter={cond_final_it} "
        f"ΔvsL1={layer2_delta_bss:+.2f}"
    )

    # -----------------------------------------------------------------
    # 저장
    # -----------------------------------------------------------------
    fold_dir = VALIDATION_DIR / f"fold_{validation_season}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        ID: (
            df_valid_raw[ID].to_numpy()
            if ID in df_valid_raw.columns
            else df_valid_raw.index.to_numpy()
        ),
        "season": int(validation_season),
        "target": y_valid,
        "anchor": anchor_valid,
        "general_residual_pred": general_valid_pred,
        "conditional_residual_pred": conditional_valid_pred,
        "pred_anchor": pred_anchor,
        "pred_general": pred_general,
        "pred_final": pred_final,
        CF_COL: valid_cf[CF_COL].to_numpy(np.float64),
    })
    pred_df.to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "train_seasons": [int(x) for x in train_seasons],
        "validation_season": int(validation_season),
        "n_train": int(len(df_train_raw)),
        "n_validation": int(len(df_valid_raw)),
        "gamma_general": GAMMA_GENERAL,
        "gamma_conditional": GAMMA_CONDITIONAL,
        "layer0": {
            "type": "row_local_dynamic_anchor",
            "strength": ANCHOR_STRENGTH,
            "train_diagnostics": train_anchor_diag,
            "valid_diagnostics": valid_anchor_diag,
            "brier": anchor_brier,
            "bss": anchor_bss,
        },
        "layer1": {
            "type": "general_residual_catboost",
            "features": hbeb_features,
            "best_iteration": general_best_it,
            "final_iterations": general_final_it,
            "brier": general_brier,
            "bss": general_bss,
            "train_residual_target_mean": float(
                np.mean(target_general_train)
            ),
            "valid_residual_target_mean": float(
                np.mean(target_general_valid)
            ),
            "inner_oof": inner_oof_meta,
        },
        "layer2": {
            "type": "conditional_residual_catboost",
            "features": layer2_features,
            "conditional_feature": CF_COL,
            "best_iteration": cond_best_it,
            "final_iterations": cond_final_it,
            "train_target_mean": float(
                np.mean(target_conditional_train)
            ),
            "valid_target_mean": float(
                np.mean(target_conditional_valid)
            ),
            "train_conditional_diagnostics": train_cf_diag,
            "valid_conditional_diagnostics": valid_cf_diag,
            "brier": final_brier,
            "bss": final_bss,
            "delta_brier_vs_layer1": layer2_delta_brier,
            "delta_bss_vs_layer1": layer2_delta_bss,
        },
        "prediction": {
            "outside_0_1_raw_rate": float(
                np.mean(
                    (pred_final_raw < 0.0)
                    | (pred_final_raw > 1.0)
                )
            ),
            "mean": float(np.mean(pred_final)),
            "std": float(np.std(pred_final)),
        },
        "elapsed_seconds": float(time.time() - start),
    }

    write_json(fold_dir / "metadata.json", metadata)

    return {
        "metadata": metadata,
        "pred_df": pred_df,
        "general_model": general_model,
        "conditional_model": conditional_model,
        "hbeb_priors": hbeb_priors,
        "anchor_bank": anchor_bank,
        "conditional_bank": conditional_bank,
        "hbeb_features": hbeb_features,
        "layer2_features": layer2_features,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    global_start = time.time()

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. Change EXPERIMENT_NAME."
        )
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} exists. Change EXPERIMENT_NAME."
        )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    ensure_columns(
        df,
        BASE_FEATURES + [TARGET],
        "train.csv",
    )

    print("=" * 96)
    print(f"Experiment           : {EXPERIMENT_NAME}")
    print("Architecture         : V18-lite")
    print("Layer 0              : Row-local Dynamic Anchor")
    print("Layer 1              : General Residual CatBoost")
    print("Layer 2              : Conditional Residual CatBoost")
    print(f"Conditional feature  : {CF_COL} (A0_ZERO)")
    print(f"Gamma general        : {GAMMA_GENERAL}")
    print(f"Gamma conditional    : {GAMMA_CONDITIONAL}")
    print(f"Train shape          : {df.shape}")
    print("=" * 96)

    fold_results = []
    oof_parts = []

    latest = None

    for train_seasons, validation_season in EXPANDING_FOLDS:
        result = train_fold(
            df,
            train_seasons,
            validation_season,
        )
        fold_results.append(result["metadata"])
        oof_parts.append(result["pred_df"])
        latest = result

    # -----------------------------------------------------------------
    # OOF
    # -----------------------------------------------------------------
    oof = pd.concat(
        oof_parts,
        axis=0,
        ignore_index=True,
    )
    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison_rows = []

    for meta in fold_results:
        season = meta["validation_season"]

        comparison_rows.extend([
            {
                "season": season,
                "layer": "L0_ANCHOR",
                "brier": meta["layer0"]["brier"],
                "bss": meta["layer0"]["bss"],
                "delta_bss_vs_previous": 0.0,
            },
            {
                "season": season,
                "layer": "L1_GENERAL",
                "brier": meta["layer1"]["brier"],
                "bss": meta["layer1"]["bss"],
                "delta_bss_vs_previous": (
                    meta["layer1"]["bss"]
                    - meta["layer0"]["bss"]
                ),
            },
            {
                "season": season,
                "layer": "L2_CONDITIONAL",
                "brier": meta["layer2"]["brier"],
                "bss": meta["layer2"]["bss"],
                "delta_bss_vs_previous": (
                    meta["layer2"]["delta_bss_vs_layer1"]
                ),
            },
        ])

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        VALIDATION_DIR / "layer_comparison.csv",
        index=False,
    )

    pooled_rows = []

    for name, col in [
        ("L0_ANCHOR", "pred_anchor"),
        ("L1_GENERAL", "pred_general"),
        ("L2_CONDITIONAL", "pred_final"),
    ]:
        pooled_rows.append({
            "layer": name,
            "brier": brier_score(oof["target"], oof[col]),
            "bss": bss_raw(oof["target"], oof[col]),
        })

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(
        VALIDATION_DIR / "pooled_layer_comparison.csv",
        index=False,
    )

    print()
    print("=" * 96)
    print("V18-lite Layer Summary")
    print("=" * 96)

    for season in sorted(comparison_df["season"].unique()):
        part = comparison_df[
            comparison_df["season"] == season
        ]
        vals = {
            row.layer: row
            for row in part.itertuples(index=False)
        }

        print(
            f"Valid {season} | "
            f"L0={vals['L0_ANCHOR'].bss:.2f} | "
            f"L1={vals['L1_GENERAL'].bss:.2f} "
            f"(Δ={vals['L1_GENERAL'].delta_bss_vs_previous:+.2f}) | "
            f"L2={vals['L2_CONDITIONAL'].bss:.2f} "
            f"(Δ={vals['L2_CONDITIONAL'].delta_bss_vs_previous:+.2f})"
        )

    print()
    print("[Pooled]")
    for row in pooled_df.itertuples(index=False):
        print(
            f"{row.layer:<16} "
            f"Brier={row.brier:.8f} BSS={row.bss:.2f}"
        )

    # -----------------------------------------------------------------
    # 최종 학습
    # -----------------------------------------------------------------
    #
    # General layer:
    #   latest fold iteration을 사용
    #
    # Conditional layer:
    #   latest fold iteration을 사용
    #
    # Layer2 target은 전체 train 내부 General OOF로 생성.
    # -----------------------------------------------------------------
    final_general_iterations = int(
        fold_results[-1]["layer1"]["final_iterations"]
    )
    final_cond_iterations = int(
        fold_results[-1]["layer2"]["final_iterations"]
    )

    print()
    print("=" * 96)
    print("Final V18-lite Training")
    print("=" * 96)
    print(f"General iterations     : {final_general_iterations}")
    print(f"Conditional iterations : {final_cond_iterations}")

    # HBEB
    df_hbeb, hbeb_features, final_hbeb_priors = apply_hbeb_configs(
        df,
        df,
        BASE_FEATURES,
        priors=None,
    )

    # Anchor
    final_anchor, final_anchor_diag = (
        build_strictly_past_dynamic_anchor(df)
    )
    df_hbeb["dynamic_anchor"] = final_anchor

    y = df[TARGET].to_numpy(np.float64)
    anchor_arr = final_anchor.to_numpy(np.float64)

    # General target
    general_target = y - anchor_arr

    # General final model
    final_general_params = dict(MODEL_PARAMS)
    final_general_params["iterations"] = final_general_iterations

    df_hbeb_s = sanitize_categoricals(
        df_hbeb,
        CAT_COLS,
    )

    final_general_model = CatBoostRegressor(
        **final_general_params
    )
    final_general_model.fit(
        df_hbeb_s[hbeb_features],
        general_target,
        cat_features=CAT_COLS,
        verbose=100,
    )

    # General OOF for full-data Layer2 target
    print()
    print("[Final Layer1 OOF generation]")
    final_general_oof, final_inner_oof_meta = (
        build_general_oof_for_training(
            df,
            hbeb_features,
            final_hbeb_priors,
        )
    )

    # Conditional feature
    df_cf, final_cf_diag = (
        build_strictly_past_platoon_features(df_hbeb)
    )

    conditional_target = (
        y
        - anchor_arr
        - GAMMA_GENERAL
        * final_general_oof.to_numpy(np.float64)
    )

    if LAYER2_USE_BASE_FEATURES:
        layer2_features = list(hbeb_features) + [CF_COL]
        layer2_cat_cols = list(CAT_COLS)
    else:
        layer2_features = [CF_COL]
        layer2_cat_cols = []

    df_cf_s = sanitize_categoricals(
        df_cf,
        CAT_COLS,
    )

    final_cond_params = dict(MODEL_PARAMS)
    final_cond_params["iterations"] = final_cond_iterations

    final_conditional_model = CatBoostRegressor(
        **final_cond_params
    )
    final_conditional_model.fit(
        df_cf_s[layer2_features],
        conditional_target,
        cat_features=layer2_cat_cols,
        verbose=100,
    )

    # 2025 frozen banks
    final_anchor_bank = build_pitcher_history_bank(df)
    final_conditional_bank = build_platoon_bank(df)

    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "architecture": "v18_lite_dynamic_anchor_general_conditional",
        "gamma_general": GAMMA_GENERAL,
        "gamma_conditional": GAMMA_CONDITIONAL,
        "base_features": BASE_FEATURES,
        "cat_cols": CAT_COLS,
        "hbeb_features": hbeb_features,
        "hbeb_priors": final_hbeb_priors,
        "layer0": {
            "type": "dynamic_anchor",
            "strength": ANCHOR_STRENGTH,
            "final_bank": final_anchor_bank,
        },
        "layer1": {
            "type": "general_residual_catboost",
            "model": final_general_model,
            "features": hbeb_features,
            "iterations": final_general_iterations,
        },
        "layer2": {
            "type": "conditional_residual_catboost",
            "model": final_conditional_model,
            "features": layer2_features,
            "conditional_feature": CF_COL,
            "strength": PLATOON_STRENGTH,
            "final_bank": final_conditional_bank,
            "iterations": final_cond_iterations,
        },
        "prediction_formula": (
            "clip(anchor + gamma_general*general_residual "
            "+ gamma_conditional*conditional_residual, 0, 1)"
        ),
        "training_seasons": sorted(
            int(x) for x in pd.unique(df[SEASON_COL])
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    final_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "architecture": artifact["architecture"],
        "gamma_general": GAMMA_GENERAL,
        "gamma_conditional": GAMMA_CONDITIONAL,
        "anchor_strength": ANCHOR_STRENGTH,
        "platoon_strength": PLATOON_STRENGTH,
        "layer2_use_base_features": LAYER2_USE_BASE_FEATURES,
        "folds": fold_results,
        "pooled": pooled_rows,
        "final_general_iterations": final_general_iterations,
        "final_conditional_iterations": final_cond_iterations,
        "final_anchor_diagnostics": final_anchor_diag,
        "final_conditional_diagnostics": final_cf_diag,
        "final_inner_oof": final_inner_oof_meta,
        "elapsed_seconds": float(
            time.time() - global_start
        ),
    }

    write_json(
        METADATA_PATH,
        final_metadata,
    )

    try:
        shutil.copy2(
            Path(__file__).resolve(),
            MODEL_DIR / "train.py",
        )
    except Exception:
        pass

    print()
    print("=" * 96)
    print("완료")
    print(f"Model      : {MODEL_PATH}")
    print(f"Validation : {VALIDATION_DIR}")
    print("=" * 96)
    print("[핵심 판정]")
    print("1) Layer2 ΔBSS = L2_CONDITIONAL - L1_GENERAL 이 핵심.")
    print("2) 2021/2022/2024에서 Layer2 ΔBSS > 0인지 본다.")
    print("3) 2023에서 Layer2가 L1을 더 악화시키는지 확인한다.")
    print("4) Layer1 train/valid residual mean gap으로 zero-point drift를 함께 본다.")
    print("5) Layer2가 반복 양수면 다음 단계에서 Real vs Placebo를 Layer2 단독으로 수행한다.")


if __name__ == "__main__":
    main()
