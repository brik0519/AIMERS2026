# train.py: cb_reg_v011
# cb_reg_v003 baseline + strictly-past pitcher_id × batter_hand conditional EB feature
#
# ── 이번 arm (A1) ────────────────────────────────────────────────────────────
#   CF_FEATURES  = ["cf_platoon_delta"]              (delta only)
#   unseen cell  = NaN                               (0.0 이 아님)
#
#   cb_reg_v010 계보에서 바뀐 것은 이 두 가지뿐이다.
#   모델 파라미터 / fold / HB-EB / K=80 / seed 는 전부 고정한다.
#
# ── 왜 delta only 인가 ──────────────────────────────────────────────────────
#   delta = reliability × (r̂ − parent)  라는 항등식이 성립한다.
#   두 변수를 함께 주면 트리가 delta / reliability 로 축소되지 않은 원시 편차를
#   복원할 수 있고, 저표본 영역에서 분산이 8배 이상 커진다.
#   실측: delta+reliability 는 2024 −41.75, 2021 −32.87 로 기각되었다.
#
# ── 왜 unseen = NaN 인가 ────────────────────────────────────────────────────
#   0.0 으로 채우면 "편차가 없다"와 "정보가 없다"가 같은 값이 된다.
#   CatBoost 는 결측을 별도 분기로 처리하므로 NaN 이 두 상태를 구조적으로 분리한다.
#   최종 학습 행의 약 33%, fold_1 은 약 65% 가 "정보 없음"이므로 영향이 크다.
#
# ── 규칙 준수 ───────────────────────────────────────────────────────────────
#   - training row 의 conditional feature 도 자기 season label 을 절대 쓰지 않는다.
#   - 2025 inference bank 는 공식 train 2019~2024 label 로만 만들어 artifact 에 넣는다.
#   - 미래 누수 / 당해 라벨 누수 / 행 독립성을 실행 시점에 강제 검사한다.
#   - F correction 없음. MLE / partial pooling 없음. max_ctr_complexity 변경 없음.
#
# 실행:
#   python train.py
#
# 출력:
#   model/cb_reg_v011/{cb_reg_v011.pkl, final_metadata.json, train.py}
#   validation/cb_reg_v011/oof_pred.csv
#   validation/cb_reg_v011/fold_YYYY/{pred.csv, metadata.json, conditional_reliability_bins.csv}
#   validation/cb_reg_v011/conditional_diagnostics.csv
#   validation/cb_reg_v011/conditional_leakage_audit.csv
#   validation/cb_reg_v011/parity_fixture_test.csv        ← script.py 정합성 검사용
#   validation/cb_reg_v011/parity_fixture_expected.csv

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

EXPERIMENT_NAME = "cb_reg_v010"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
METADATA_PATH = MODEL_DIR / "final_metadata.json"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================

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
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# HB-EB (cb_reg_v003 호환)
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
# Strictly-Past Platoon 설정
# =========================================================

CF_KEY_COLS = ["pitcher_id", "batter_hand"]
CF_PARENT_COLS = ["pitcher_id"]

CF_PLATOON_STRENGTH = 80.0

# bank 가 생성하는 전체 컬럼. 모델 입력은 CF_FEATURES 가 고른다.
CF_ALL_COLUMNS = [
    "cf_platoon_rate",
    "cf_platoon_delta",
    "cf_platoon_reliability",
    "cf_platoon_n",
]

# ★ arm A1: delta only
CF_FEATURES = [
    "cf_platoon_delta",
]

# script.py 정합성 검사용 fixture
PARITY_FIXTURE_ROWS = 20000
PARITY_FIXTURE_SEED = 20260825

AUDIT_SEASONS = [2021, 2022, 2023, 2024]
AUDIT_TOLERANCE = 1e-12


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


def sanitize_categoricals(df, cat_cols):
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
    return out


# =========================================================
# HB-EB — cb_reg_v003 호환 row-level MoM
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

    if np.any(denom <= 0.0):
        raise ValueError(f"HB-EB denominator가 0 이하입니다: {rate_col}")

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

        # cb_reg_v003: raw rate 제거 → HB-EB 값으로 대체
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
# Strictly-Past pitcher × batter_hand Conditional Bank
# =========================================================

def build_platoon_bank(history_df):
    """
    history_df 의 label 만으로 frozen bank 를 만든다.
    train / history 생성 시에만 호출한다.
    """
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
            "n_history": 0,
            "history_seasons": [],
            "strength": float(CF_PLATOON_STRENGTH),
        }

    y = pd.to_numeric(history_df[TARGET], errors="raise").astype(np.float64)

    tmp = history_df[CF_KEY_COLS].copy()
    tmp[TARGET] = y.to_numpy()

    cell = (
        tmp.groupby(CF_KEY_COLS, dropna=False, observed=True)[TARGET]
        .agg(cell_n="size", cell_success="sum")
        .reset_index()
    )

    parent = (
        history_df[CF_PARENT_COLS]
        .assign(__y=y.to_numpy())
        .groupby(CF_PARENT_COLS, dropna=False, observed=True)["__y"]
        .agg(parent_n="size", parent_success="sum")
        .reset_index()
    )
    parent["parent_rate"] = parent["parent_success"] / parent["parent_n"].clip(lower=1)

    return {
        "cell_table": cell,
        "parent_table": parent,
        "global_rate": float(y.mean()),
        "n_history": int(len(history_df)),
        "history_seasons": sorted(int(x) for x in pd.unique(history_df[SEASON_COL])),
        "strength": float(CF_PLATOON_STRENGTH),
    }


def _platoon_diagnostics(cell_n, reliability, delta, bank, n_rows):
    """NaN 전파로 지표가 무력화되지 않도록 유한값만 집계한다."""
    finite = np.isfinite(delta)
    has_finite = bool(finite.any())

    return {
        "n_rows": int(n_rows),
        "history_seasons": list(bank.get("history_seasons", [])),
        "history_n": int(bank.get("n_history", 0)),
        "strength": float(bank["strength"]),
        "global_rate": float(bank["global_rate"]),
        "unseen_cell_rows": int(np.sum(cell_n <= 0.0)),
        "unseen_cell_rate": float(np.mean(cell_n <= 0.0)) if n_rows else 0.0,
        "nan_delta_rows": int(np.sum(~finite)),
        "nan_delta_rate": float(np.mean(~finite)) if n_rows else 0.0,
        "mean_cell_n": float(np.mean(cell_n)) if n_rows else 0.0,
        "median_cell_n": float(np.median(cell_n)) if n_rows else 0.0,
        "p10_cell_n": float(np.quantile(cell_n, 0.10)) if n_rows else 0.0,
        "p90_cell_n": float(np.quantile(cell_n, 0.90)) if n_rows else 0.0,
        "mean_reliability": float(np.mean(reliability)) if n_rows else 0.0,
        "mean_abs_delta": (
            float(np.mean(np.abs(delta[finite]))) if has_finite else 0.0
        ),
        "max_abs_delta": (
            float(np.max(np.abs(delta[finite]))) if has_finite else 0.0
        ),
    }


def apply_platoon_bank(df_apply, bank):
    """
    frozen bank 를 각 행에 독립적으로 lookup 한다.
    df_apply 내부의 label / 다른 행 / 전체 분포를 사용하지 않는다.

        posterior_rate = (cell_success + K·parent_rate) / (cell_n + K)
        reliability    = cell_n / (cell_n + K)
        delta          = posterior_rate − parent_rate      (cell_n = 0 이면 NaN)

    unseen cell    : cell_n = 0        -> delta = NaN
    unseen pitcher : parent_rate = global_rate
    """
    ensure_columns(df_apply, CF_KEY_COLS, "apply_platoon_bank")

    out = df_apply.copy()
    n_rows = len(out)

    parent_table = bank["parent_table"]
    cell_table = bank["cell_table"]
    global_rate = float(bank["global_rate"])
    strength = float(bank["strength"])

    parent_lookup = (
        out[CF_PARENT_COLS]
        .reset_index(drop=True)
        .merge(parent_table, on=CF_PARENT_COLS, how="left", sort=False)
    )
    parent_rate = (
        pd.to_numeric(
            parent_lookup.get(
                "parent_rate", pd.Series(np.nan, index=range(n_rows))
            ),
            errors="coerce",
        )
        .fillna(global_rate)
        .to_numpy(np.float64)
    )

    cell_lookup = (
        out[CF_KEY_COLS]
        .reset_index(drop=True)
        .merge(cell_table, on=CF_KEY_COLS, how="left", sort=False)
    )
    cell_n = (
        pd.to_numeric(
            cell_lookup.get("cell_n", pd.Series(np.nan, index=range(n_rows))),
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    cell_success = (
        pd.to_numeric(
            cell_lookup.get("cell_success", pd.Series(np.nan, index=range(n_rows))),
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(np.float64)
    )

    cell_n = np.clip(cell_n, 0.0, None)
    cell_success = np.clip(cell_success, 0.0, None)

    posterior_rate = (cell_success + strength * parent_rate) / (cell_n + strength)
    reliability = cell_n / (cell_n + strength)

    # ★ 미관측 셀은 0 이 아니라 결측으로 표현한다.
    delta = np.where(cell_n > 0.0, posterior_rate - parent_rate, np.nan)

    out["cf_platoon_rate"] = posterior_rate.astype(np.float64)
    out["cf_platoon_delta"] = delta.astype(np.float64)
    out["cf_platoon_reliability"] = reliability.astype(np.float64)
    out["cf_platoon_n"] = cell_n.astype(np.float64)

    return out, _platoon_diagnostics(cell_n, reliability, delta, bank, n_rows)


def cold_start_platoon_features(df_apply):
    """
    history 가 없는 최초 season 용 fallback.

    미관측 셀과 동일한 인코딩을 쓴다.
        rate        = NaN   (parent 도 없다)
        delta       = NaN
        reliability = 0.0
        n           = 0.0
    """
    out = df_apply.copy()
    n_rows = len(out)

    out["cf_platoon_rate"] = np.nan
    out["cf_platoon_delta"] = np.nan
    out["cf_platoon_reliability"] = 0.0
    out["cf_platoon_n"] = 0.0

    return out, {
        "n_rows": int(n_rows),
        "history_seasons": [],
        "history_n": 0,
        "strength": float(CF_PLATOON_STRENGTH),
        "global_rate": 0.5,
        "unseen_cell_rows": int(n_rows),
        "unseen_cell_rate": 1.0 if n_rows else 0.0,
        "nan_delta_rows": int(n_rows),
        "nan_delta_rate": 1.0 if n_rows else 0.0,
        "mean_cell_n": 0.0,
        "median_cell_n": 0.0,
        "p10_cell_n": 0.0,
        "p90_cell_n": 0.0,
        "mean_reliability": 0.0,
        "mean_abs_delta": 0.0,
        "max_abs_delta": 0.0,
    }


def build_strictly_past_platoon_features(df):
    """
    training rows 용. season s 의 feature 는 오직 season < s 의 target 만 사용한다.
    """
    ensure_columns(
        df,
        [SEASON_COL, TARGET] + CF_KEY_COLS,
        "build_strictly_past_platoon_features",
    )

    pieces = []
    diagnostics = []

    seasons = sorted(int(x) for x in pd.unique(df[SEASON_COL]))
    season_int = df[SEASON_COL].astype(int)

    for season in seasons:
        current = df.loc[season_int == season].copy()
        history = df.loc[season_int < season]

        if len(history) == 0:
            transformed, diag = cold_start_platoon_features(current)
        else:
            bank = build_platoon_bank(history)
            transformed, diag = apply_platoon_bank(current, bank)

        diag = dict(diag)
        diag["target_season"] = int(season)
        diagnostics.append(diag)
        pieces.append(transformed)

    result = pd.concat(pieces, axis=0).sort_index()
    if not result.index.equals(df.index):
        result = result.reindex(df.index)

    return result, diagnostics


def build_validation_platoon_features(history_df, validation_df):
    """
    validation season 용. history_df 는 validation season 보다 과거만 포함해야 한다.
    """
    valid_seasons = sorted(int(x) for x in pd.unique(validation_df[SEASON_COL]))
    if len(valid_seasons) != 1:
        raise ValueError(f"validation_df must have one season, got {valid_seasons}")

    validation_season = valid_seasons[0]

    if bool((history_df[SEASON_COL].astype(int) >= validation_season).any()):
        raise RuntimeError(
            "Leakage guard: history_df contains validation/future season rows."
        )

    bank = build_platoon_bank(history_df)
    transformed, diag = apply_platoon_bank(validation_df, bank)
    diag["target_season"] = int(validation_season)

    return transformed, diag, bank


# =========================================================
# Leakage / invariance audit
# =========================================================

def feature_max_abs_diff(a_df, b_df, idx, cols):
    """
    결측 패턴까지 포함해 두 feature 프레임을 비교한다.

    np.abs(nan − nan) = nan 이고 max(0.0, nan) = 0.0 이므로,
    단순 비교는 NaN 구간을 조용히 건너뛴다. 그것을 막는다.
    """
    max_diff = 0.0

    for col in cols:
        a = a_df.loc[idx, col].to_numpy(np.float64)
        b = b_df.loc[idx, col].to_numpy(np.float64)

        a_nan = np.isnan(a)
        b_nan = np.isnan(b)

        if bool(np.any(a_nan ^ b_nan)):
            return float("inf")  # 결측 패턴 불일치 = 즉시 실패

        finite = ~a_nan
        if bool(finite.any()):
            max_diff = max(
                max_diff, float(np.max(np.abs(a[finite] - b[finite])))
            )

    return max_diff


def verify_strictly_past_invariance(audit_df, check_season, full_feat):
    """미래 season 을 제거해도 check_season feature 가 불변인지 확인한다."""
    check_season = int(check_season)
    season_int = audit_df[SEASON_COL].astype(int)

    truncated = audit_df.loc[season_int <= check_season].copy()
    trunc_feat, _ = build_strictly_past_platoon_features(truncated)

    idx = truncated.index[truncated[SEASON_COL].astype(int) == check_season]
    if len(idx) == 0:
        raise ValueError(f"No rows for check_season={check_season}")

    max_diff = feature_max_abs_diff(full_feat, trunc_feat, idx, CF_FEATURES)

    return {
        "check_season": check_season,
        "n_rows": int(len(idx)),
        "max_abs_diff": max_diff,
        "passed": bool(max_diff <= AUDIT_TOLERANCE),
    }


def verify_current_year_label_permutation(audit_df, check_season, full_feat, seed=123):
    """check_season 의 label 을 섞어도 그 season 의 feature 가 불변인지 확인한다."""
    check_season = int(check_season)
    rng = np.random.default_rng(seed)

    shuffled = audit_df.copy()
    mask = shuffled[SEASON_COL].astype(int) == check_season

    y = shuffled.loc[mask, TARGET].to_numpy(copy=True)
    rng.shuffle(y)
    shuffled.loc[mask, TARGET] = y

    shuffled_feat, _ = build_strictly_past_platoon_features(shuffled)

    idx = audit_df.index[mask]
    max_diff = feature_max_abs_diff(full_feat, shuffled_feat, idx, CF_FEATURES)

    return {
        "check_season": check_season,
        "n_rows": int(len(idx)),
        "max_abs_diff": max_diff,
        "passed": bool(max_diff <= AUDIT_TOLERANCE),
    }


def verify_bank_row_independence(frame, bank, n_samples=30, seed=42):
    """
    같은 행을 단독으로 transform 한 값과 batch transform 한 값이 동일한지 확인한다.
    NaN == NaN 을 동일로 보기 위해 equal_nan=True 가 필수다.
    """
    if len(frame) == 0:
        return {"passed": True, "failures": [], "n_samples": 0, "max_abs_diff": 0.0}

    rng = np.random.default_rng(seed)
    k = min(int(n_samples), len(frame))
    positions = rng.choice(len(frame), size=k, replace=False)

    batch, _ = apply_platoon_bank(frame, bank)

    failures = []
    max_abs_diff = 0.0

    for pos in positions:
        one = frame.iloc[[int(pos)]].copy()
        one_out, _ = apply_platoon_bank(one, bank)

        for col in CF_FEATURES:
            a = float(batch.iloc[int(pos)][col])
            b = float(one_out.iloc[0][col])

            if np.isnan(a) and np.isnan(b):
                continue

            if not np.isclose(a, b, atol=AUDIT_TOLERANCE, rtol=0.0, equal_nan=True):
                failures.append(
                    {
                        "position": int(pos),
                        "feature": col,
                        "batch": a,
                        "single": b,
                    }
                )
            elif np.isfinite(a) and np.isfinite(b):
                max_abs_diff = max(max_abs_diff, abs(a - b))

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_samples": int(k),
        "max_abs_diff": float(max_abs_diff),
    }


# =========================================================
# Fold 학습
# =========================================================

def train_fold(df_all, train_seasons, validation_season):
    fold_start = time.time()

    season_int = df_all[SEASON_COL].astype(int)
    df_train_raw = df_all.loc[season_int.isin(train_seasons)].copy()
    df_valid_raw = df_all.loc[season_int == validation_season].copy()

    # ---- 1) HB-EB -----------------------------------------------------
    df_train_hbeb, hbeb_features, fold_hbeb_priors = apply_hbeb_configs(
        df_train_raw, df_train_raw, BASE_FEATURES, priors=None
    )
    df_valid_hbeb, valid_hbeb_features, _ = apply_hbeb_configs(
        df_train_raw, df_valid_raw, BASE_FEATURES, priors=fold_hbeb_priors
    )

    if hbeb_features != valid_hbeb_features:
        raise RuntimeError("HB-EB feature manifest mismatch.")

    # ---- 2) strictly-past conditional ---------------------------------
    df_train_cf, train_cf_diag = build_strictly_past_platoon_features(df_train_hbeb)

    df_valid_cf, valid_cf_diag, validation_bank = build_validation_platoon_features(
        df_train_raw, df_valid_hbeb
    )

    final_features = list(hbeb_features) + list(CF_FEATURES)

    # ---- 3) row-independence audit ------------------------------------
    row_check = verify_bank_row_independence(
        df_valid_hbeb.head(min(5000, len(df_valid_hbeb))),
        validation_bank,
        n_samples=30,
    )
    if not row_check["passed"]:
        raise RuntimeError(
            f"Conditional row independence violated: {row_check['failures'][:3]}"
        )

    # ---- 4) CatBoost --------------------------------------------------
    df_train_cf = sanitize_categoricals(df_train_cf, CAT_COLS)
    df_valid_cf = sanitize_categoricals(df_valid_cf, CAT_COLS)

    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(
        df_train_cf[final_features],
        df_train_cf[TARGET],
        cat_features=CAT_COLS,
        eval_set=(df_valid_cf[final_features], df_valid_cf[TARGET]),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )

    best_iteration = int(model.get_best_iteration())
    final_iterations = best_iteration + 1

    pred_raw = model.predict(df_valid_cf[final_features])
    pred_clip = np.clip(pred_raw, 0.0, 1.0)
    y_valid = df_valid_cf[TARGET].to_numpy(np.float64)

    bs = brier_score(y_valid, pred_clip)
    base = baseline_brier(y_valid)
    bss = bss_raw(y_valid, pred_clip)

    fold_dir = VALIDATION_DIR / f"fold_{validation_season}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ---- 5) OOF 저장 (row_id 포함) -------------------------------------
    pred_cols = {
        "season": int(validation_season),
        "target": y_valid,
        "pred_raw": pred_raw,
        "pred": pred_clip,
    }
    if ID in df_valid_cf.columns:
        pred_cols = {ID: df_valid_cf[ID].to_numpy(), **pred_cols}
    else:
        pred_cols = {"row_index": df_valid_cf.index.to_numpy(), **pred_cols}

    for col in CF_ALL_COLUMNS:
        pred_cols[col] = df_valid_cf[col].to_numpy()

    pred_df = pd.DataFrame(pred_cols)
    pred_df.to_csv(fold_dir / "pred.csv", index=False, encoding="utf-8-sig")

    # ---- 6) reliability bin 진단 ---------------------------------------
    rel = df_valid_cf["cf_platoon_reliability"].to_numpy(np.float64)
    rel_labels = ["r<=0.1", "0.1<r<=0.3", "0.3<r<=0.5", "0.5<r<=0.8", "r>0.8"]
    rel_cat = pd.cut(
        rel, bins=[-np.inf, 0.1, 0.3, 0.5, 0.8, np.inf], labels=rel_labels, right=True
    )

    rel_rows = []
    for label in rel_labels:
        mask = np.asarray(rel_cat == label)
        if not np.any(mask):
            continue
        rel_rows.append(
            {
                "validation_season": int(validation_season),
                "reliability_bin": label,
                "n_rows": int(mask.sum()),
                "share": float(mask.mean()),
                "mean_reliability": float(rel[mask].mean()),
                "mean_cell_n": float(
                    df_valid_cf["cf_platoon_n"].to_numpy(np.float64)[mask].mean()
                ),
                "brier": brier_score(y_valid[mask], pred_clip[mask]),
                "pred_mean": float(pred_clip[mask].mean()),
                "actual_mean": float(y_valid[mask].mean()),
                "bias": float((pred_clip[mask] - y_valid[mask]).mean()),
            }
        )
    pd.DataFrame(rel_rows).to_csv(
        fold_dir / "conditional_reliability_bins.csv", index=False
    )

    feature_importance = dict(
        zip(final_features, model.get_feature_importance().tolist())
    )

    fold_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "train_seasons": [int(s) for s in train_seasons],
        "validation_season": int(validation_season),
        "n_train": int(len(df_train_cf)),
        "n_validation": int(len(df_valid_cf)),
        "model_type": "CatBoostRegressor",
        "objective": "RMSE",
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "final_iterations": final_iterations,
        "features": final_features,
        "categorical_features": CAT_COLS,
        "conditional_features": CF_FEATURES,
        "conditional_key_cols": CF_KEY_COLS,
        "conditional_parent_cols": CF_PARENT_COLS,
        "conditional_strength": CF_PLATOON_STRENGTH,
        "conditional_unseen_encoding": "nan",
        "conditional_train_diagnostics": train_cf_diag,
        "conditional_validation_diagnostics": valid_cf_diag,
        "conditional_row_independence": row_check,
        "brier": bs,
        "baseline_brier": base,
        "bss_raw": bss,
        "prediction_mean": float(np.mean(pred_clip)),
        "prediction_std": float(np.std(pred_clip)),
        "prediction_min": float(np.min(pred_clip)),
        "prediction_max": float(np.max(pred_clip)),
        "feature_importance": feature_importance,
        "elapsed_seconds": float(time.time() - fold_start),
    }

    write_json(fold_dir / "metadata.json", fold_metadata)

    return {"metadata": fold_metadata, "pred_df": pred_df}


# =========================================================
# Main
# =========================================================

def main():
    global_start = time.time()

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"{VALIDATION_DIR} exists. Change EXPERIMENT_NAME."
        )
    if MODEL_DIR.exists():
        raise FileExistsError(f"{MODEL_DIR} exists. Change EXPERIMENT_NAME.")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    ensure_columns(df, BASE_FEATURES + [TARGET], "train.csv")

    # test.csv 스키마 정합성 확인 (헤더만 읽는다. 다른 행/분포 미사용)
    if TEST_PATH.exists():
        test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
        expected = set(test_cols) - {ID}
        if set(BASE_FEATURES) != expected:
            raise RuntimeError(
                "BASE_FEATURES 가 test.csv 스키마와 다릅니다. "
                f"only_train={sorted(set(BASE_FEATURES) - expected)}, "
                f"only_test={sorted(expected - set(BASE_FEATURES))}"
            )

    print("=" * 80)
    print(f"Experiment           : {EXPERIMENT_NAME}")
    print("Model                : CatBoostRegressor")
    print("Objective            : RMSE")
    print("Validation           : Expanding Window")
    print("Baseline             : cb_reg_v003-compatible")
    print("Arm                  : A1 = delta only + unseen NaN")
    print(f"Train shape          : {df.shape}")
    print(f"Base Feature 수      : {len(BASE_FEATURES)}")
    print(f"Conditional Feature  : {CF_FEATURES}")
    print(f"Final Feature 수     : {len(BASE_FEATURES) + len(CF_FEATURES)}")
    print(f"Categorical Feature  : {len(CAT_COLS)}")
    print(f"시즌                 : {int(df[SEASON_COL].min())} ~ {int(df[SEASON_COL].max())}")
    print(f"Target mean          : {df[TARGET].mean():.6f}")
    print(f"Platoon strength K   : {CF_PLATOON_STRENGTH:.2f}")
    print("=" * 80)

    # -----------------------------------------------------
    # Leakage audit
    # full_feat 을 한 번만 계산해 재사용한다 (기존 16회 -> 9회)
    # -----------------------------------------------------
    audit_cols = list(dict.fromkeys([SEASON_COL, TARGET] + CF_KEY_COLS))
    audit_df = df[audit_cols].copy()

    full_feat_all, _ = build_strictly_past_platoon_features(audit_df)
    full_feat = full_feat_all[CF_ALL_COLUMNS].copy()
    del full_feat_all

    audit_records = []
    for check_season in AUDIT_SEASONS:
        inv = verify_strictly_past_invariance(audit_df, check_season, full_feat)
        perm = verify_current_year_label_permutation(audit_df, check_season, full_feat)

        audit_records.append(
            {
                "season": check_season,
                "n_rows": inv["n_rows"],
                "future_removal_max_abs_diff": inv["max_abs_diff"],
                "future_removal_passed": inv["passed"],
                "current_label_permutation_max_abs_diff": perm["max_abs_diff"],
                "current_label_permutation_passed": perm["passed"],
            }
        )

        if not inv["passed"]:
            raise RuntimeError(f"Future leakage audit failed: {inv}")
        if not perm["passed"]:
            raise RuntimeError(f"Current-label leakage audit failed: {perm}")

    pd.DataFrame(audit_records).to_csv(
        VALIDATION_DIR / "conditional_leakage_audit.csv", index=False
    )

    print("[Leakage audit] PASS  (NaN 패턴 포함 비교)")
    for row in audit_records:
        print(
            f"  season={row['season']} | "
            f"future_diff={row['future_removal_max_abs_diff']:.3e} | "
            f"label_perm_diff={row['current_label_permutation_max_abs_diff']:.3e}"
        )

    del audit_df, full_feat

    # -----------------------------------------------------
    # Expanding validation
    # -----------------------------------------------------
    fold_results = []
    oof_parts = []
    latest_fold_iterations = None

    for fold_idx, (train_seasons, validation_season) in enumerate(EXPANDING_FOLDS, 1):
        print()
        print("=" * 80)
        print(
            f"Fold {fold_idx}: "
            f"{min(train_seasons)}~{max(train_seasons)} -> {validation_season}"
        )
        print("=" * 80)

        result = train_fold(df, train_seasons, validation_season)
        meta = result["metadata"]

        fold_results.append(meta)
        oof_parts.append(result["pred_df"])
        latest_fold_iterations = int(meta["final_iterations"])

        vdiag = meta["conditional_validation_diagnostics"]
        tdiag = meta["conditional_train_diagnostics"]
        train_nan = sum(d["nan_delta_rows"] for d in tdiag)
        train_rows = sum(d["n_rows"] for d in tdiag)

        print()
        print(f"Brier Score        : {meta['brier']:.8f}")
        print(f"BSS                : {meta['bss_raw']:.2f}")
        print(f"Best iteration     : {meta['best_iteration']}")
        print(f"Final iterations   : {meta['final_iterations']}")
        print(f"CF valid unseen    : {vdiag['unseen_cell_rate']:.4%}")
        print(f"CF valid nan delta : {vdiag['nan_delta_rate']:.4%}")
        print(f"CF valid mean rel  : {vdiag['mean_reliability']:.6f}")
        print(f"CF valid mean|d|   : {vdiag['mean_abs_delta']:.6f}")
        print(
            "CF train nan delta : "
            f"{train_nan / max(train_rows, 1):.4%}  ({train_nan:,}/{train_rows:,})"
        )
        print(
            "CF importance      : "
            f"{meta['feature_importance'].get('cf_platoon_delta', float('nan')):.4f}"
        )

    oof = pd.concat(oof_parts, axis=0, ignore_index=True)
    oof.to_csv(VALIDATION_DIR / "oof_pred.csv", index=False, encoding="utf-8-sig")

    pooled_brier = brier_score(oof["target"], oof["pred"])
    pooled_bss = bss_raw(oof["target"], oof["pred"])

    cond_rows = []
    for meta in fold_results:
        d = dict(meta["conditional_validation_diagnostics"])
        d["validation_season"] = int(meta["validation_season"])
        d["scope"] = "validation"
        cond_rows.append(d)
        for td in meta["conditional_train_diagnostics"]:
            t = dict(td)
            t["validation_season"] = int(meta["validation_season"])
            t["scope"] = "train"
            cond_rows.append(t)

    pd.DataFrame(cond_rows).to_csv(
        VALIDATION_DIR / "conditional_diagnostics.csv", index=False
    )

    print()
    print("=" * 80)
    print("Expanding Window Summary")
    print("=" * 80)
    for meta in fold_results:
        v = meta["conditional_validation_diagnostics"]
        print(
            f"Valid {meta['validation_season']} | "
            f"Brier={meta['brier']:.8f} | "
            f"BSS={meta['bss_raw']:.2f} | "
            f"iter={meta['final_iterations']} | "
            f"nan_delta={v['nan_delta_rate']:.2%}"
        )

    print()
    print(f"Pooled OOF Brier     : {pooled_brier:.8f}")
    print(f"Pooled OOF BSS       : {pooled_bss:.2f}")
    print(f"Validation 저장      : {VALIDATION_DIR}")
    print("=" * 80)

    # -----------------------------------------------------
    # Final iteration
    # -----------------------------------------------------
    if FINAL_ITERATION_STRATEGY != "latest_fold":
        raise ValueError(
            f"Unsupported FINAL_ITERATION_STRATEGY: {FINAL_ITERATION_STRATEGY}"
        )

    final_iterations = int(latest_fold_iterations)
    print()
    print(f"Final iteration strategy : {FINAL_ITERATION_STRATEGY}")
    print(f"Final iterations         : {final_iterations}")

    # -----------------------------------------------------
    # 최종 학습
    # -----------------------------------------------------
    df_final_hbeb, hbeb_features, final_hbeb_priors = apply_hbeb_configs(
        df, df, BASE_FEATURES, priors=None
    )
    df_final_cf, final_train_cf_diag = build_strictly_past_platoon_features(
        df_final_hbeb
    )

    final_features = list(hbeb_features) + list(CF_FEATURES)
    df_final_cf = sanitize_categoricals(df_final_cf, CAT_COLS)

    final_params = dict(MODEL_PARAMS)
    final_params["iterations"] = final_iterations

    print()
    print("=" * 80)
    print("최종 HB-EB + Strictly-Past Conditional Feature 학습")
    print("=" * 80)

    final_model = CatBoostRegressor(**final_params)
    final_model.fit(
        df_final_cf[final_features],
        df_final_cf[TARGET],
        cat_features=CAT_COLS,
        verbose=100,
    )

    final_nan = sum(d["nan_delta_rows"] for d in final_train_cf_diag)
    final_rows = sum(d["n_rows"] for d in final_train_cf_diag)
    print()
    print(
        "최종 학습 nan delta 비율 : "
        f"{final_nan / max(final_rows, 1):.4%}  ({final_nan:,}/{final_rows:,})"
    )

    # -----------------------------------------------------
    # 2025 inference 용 frozen bank
    # -----------------------------------------------------
    final_2025_bank = build_platoon_bank(df)

    row_check_final = verify_bank_row_independence(
        df.head(min(5000, len(df))), final_2025_bank, n_samples=40
    )
    if not row_check_final["passed"]:
        raise RuntimeError(
            f"Final bank row-independence failed: {row_check_final['failures'][:3]}"
        )

    # -----------------------------------------------------
    # script.py 정합성 fixture
    #   최신 학습 시즌 행을 test.csv 스키마로 내보내고,
    #   train 경로가 계산한 conditional feature 를 함께 저장한다.
    #   script.py 경로의 값과 1e-12 이내로 일치해야 한다.
    # -----------------------------------------------------
    latest_season = int(df[SEASON_COL].max())
    fixture_src = df.loc[df[SEASON_COL].astype(int) == latest_season]
    fixture_src = fixture_src.sample(
        n=min(PARITY_FIXTURE_ROWS, len(fixture_src)),
        random_state=PARITY_FIXTURE_SEED,
    ).sort_index()

    fixture_feat, fixture_diag = apply_platoon_bank(fixture_src, final_2025_bank)

    fixture_key = (
        fixture_src[ID].to_numpy()
        if ID in fixture_src.columns
        else fixture_src.index.to_numpy()
    )
    fixture_key_name = ID if ID in fixture_src.columns else "row_index"

    fixture_test = fixture_src[BASE_FEATURES].copy()
    fixture_test.insert(0, fixture_key_name, fixture_key)
    fixture_test.to_csv(
        VALIDATION_DIR / "parity_fixture_test.csv", index=False, encoding="utf-8-sig"
    )

    fixture_expected = pd.DataFrame({fixture_key_name: fixture_key})
    for col in CF_ALL_COLUMNS:
        fixture_expected[col] = fixture_feat[col].to_numpy()
    fixture_expected.to_csv(
        VALIDATION_DIR / "parity_fixture_expected.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("[Parity fixture]")
    print(f"  rows              : {len(fixture_test):,} (season {latest_season})")
    print(f"  unseen_cell_rate  : {fixture_diag['unseen_cell_rate']:.4%}")
    print(f"  nan_delta_rate    : {fixture_diag['nan_delta_rate']:.4%}")
    print(f"  mean|delta|       : {fixture_diag['mean_abs_delta']:.6f}")
    print(f"  saved             : {VALIDATION_DIR / 'parity_fixture_test.csv'}")

    # -----------------------------------------------------
    # Artifact
    # -----------------------------------------------------
    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "CatBoostRegressor",
        "objective": "RMSE",
        "model": final_model,
        "features": final_features,
        "cat_cols": CAT_COLS,
        "base_features": BASE_FEATURES,
        "hbeb_priors": final_hbeb_priors,
        "hbeb_configs": HBEB_CONFIGS,
        "conditional_features": {
            "type": "strictly_past_pitcher_x_batter_hand",
            "key_cols": CF_KEY_COLS,
            "parent_cols": CF_PARENT_COLS,
            "feature_cols": CF_FEATURES,
            "all_columns": CF_ALL_COLUMNS,
            "strength": CF_PLATOON_STRENGTH,
            "unseen_encoding": "nan",
            "final_2025_bank": final_2025_bank,
            "row_independence": row_check_final,
        },
        "prediction_policy": "clip_to_0_1",
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "final_iterations": final_iterations,
        "training_seasons": sorted(int(x) for x in pd.unique(df[SEASON_COL])),
    }

    joblib.dump(artifact, MODEL_PATH, compress=3)

    final_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "arm": "A1_delta_only_unseen_nan",
        "validation_method": "expanding_window",
        "model_type": "CatBoostRegressor",
        "objective": "RMSE",
        "prediction_for_evaluation": "clipped",
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "final_iterations": final_iterations,
        "features": final_features,
        "categorical_features": CAT_COLS,
        "base_features": BASE_FEATURES,
        "hbeb_configs": HBEB_CONFIGS,
        "conditional_feature_policy": {
            "name": "strictly_past_pitcher_x_batter_hand",
            "key_cols": CF_KEY_COLS,
            "parent_cols": CF_PARENT_COLS,
            "feature_cols": CF_FEATURES,
            "strength": CF_PLATOON_STRENGTH,
            "unseen_encoding": "nan",
            "training_rule": "source_season < row_season",
            "deployment_rule": "2019~2024 official train frozen bank lookup only",
            "mle_enabled": False,
            "partial_pooling_enabled": False,
            "f_correction_enabled": False,
        },
        "folds": fold_results,
        "summary": {
            "pooled_oof_brier": pooled_brier,
            "pooled_oof_bss_raw": pooled_bss,
            "n_oof": int(len(oof)),
            "mean_fold_bss_raw": float(
                np.mean([m["bss_raw"] for m in fold_results])
            ),
            "final_train_nan_delta_rate": float(final_nan / max(final_rows, 1)),
        },
        "final_train_conditional_diagnostics": final_train_cf_diag,
        "final_2025_bank": {
            "n_history": final_2025_bank["n_history"],
            "history_seasons": final_2025_bank["history_seasons"],
            "n_cells": int(len(final_2025_bank["cell_table"])),
            "n_parents": int(len(final_2025_bank["parent_table"])),
            "global_rate": float(final_2025_bank["global_rate"]),
            "strength": float(final_2025_bank["strength"]),
        },
        "final_row_independence": row_check_final,
        "parity_fixture": {
            "season": latest_season,
            "n_rows": int(len(fixture_test)),
            "key": fixture_key_name,
            "diagnostics": fixture_diag,
        },
        "elapsed_seconds": float(time.time() - global_start),
    }

    write_json(METADATA_PATH, final_metadata)

    try:
        shutil.copy2(Path(__file__).resolve(), MODEL_DIR / "train.py")
    except Exception:
        pass

    print()
    print("=" * 80)
    print("최종 Artifact 저장 완료")
    print(f"- Model   : {MODEL_PATH}")
    print(f"- Metadata: {METADATA_PATH}")
    print(f"- Train.py: {MODEL_DIR / 'train.py'}")
    print(f"- elapsed : {time.time() - global_start:.1f}s")
    print("=" * 80)

    print()
    print("[다음 단계 — 제출 전 필수]")
    print("  1) script.py 의 EXPERIMENT_NAME 을 "
          f"'{EXPERIMENT_NAME}' 으로 맞춘다.")
    print("  2) parity 검사:")
    print("     validation/{}/parity_fixture_test.csv 를 script.py 의".format(EXPERIMENT_NAME))
    print("     apply_frozen_platoon_bank 로 변환한 뒤")
    print("     parity_fixture_expected.csv 와 max_abs_diff < 1e-12 인지 확인한다.")
    print("     (dtype 불일치로 bank merge 가 전멸하는 사고를 여기서 잡는다)")
    print("  3) 게이트: 2021 BSS >= 1267, 2024 BSS >= 733.99, seed 3개 부호 일치")


if __name__ == "__main__":
    main()