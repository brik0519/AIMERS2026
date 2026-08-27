# train.py: CatBoost Regressor cb_reg_v002_fe_v001
#
# model/cb_reg_v002/train.py(raw asof_* feature 47개, CatBoostRegressor RMSE,
# Expanding Window Validation)를 기준 모델로 삼고, 다음을 추가/교체한 실험이다.
#
# 1) BBEB(Beta-Binomial Empirical Bayes) smoothing 7개
#    raw rate 7개(asof_pitcher_success/middle/ball/strike/reverse_rate,
#    asof_batter_success/middle_rate)를 제거하고 bbeb_<name>_rate로 교체한다.
#    BBEB 로직은 `Reliability Feature Experiment/cb_reg_v002_eb_experiments/
#    cb_reg_v002_eb_replace_v001_train.py`(pooled_oof_clipped_brier=0.24716861로
#    지금까지 가장 좋은 성과를 낸 변형)의 함수를 그대로 재사용하며, prior는
#    항상 "그 fold의 train 구간 전체(expanding)"로 fit한다(FEATURE_SPECS에서
#    pitch-mix 3개만 제외 — 이 3개는 이번 실험에서는 raw 그대로 유지).
# 2) 신규 상황(situational) feature 5개: count_balance, full_count,
#    scoring_position, pitcher_win_expectancy, platoon
# 3) eb_pitcher_batter_gap = bbeb_pitcher_success_rate - bbeb_batter_success_rate
# 4) 기존 Temporal Feature Experiment EXP-3의 Temporal 10개
#    (`Temporal Feature Experiment/exp_code/exp3_temporal_only.py`와
#    동일한 공식, raw asof_pitcher_success_rate/middle_rate를 계산에만 사용)
#
# 핵심 원칙 (cb_reg_v002와 동일)
# - 평가 데이터(test.csv)의 다른 행/전체 분포를 Feature 생성에 사용하지 않는다.
# - Validation은 train.csv 내부 시즌만으로 Expanding Window 방식으로 수행한다.
# - BBEB prior는 항상 해당 fold의 train 구간에서만 fit하고, 각 행의 EB rate는
#   그 행 자신의 공식 asof_* 값만으로 계산한다(다른 행 참조 없음).
# - 상황/Temporal feature는 순수 row-wise 산술이라 fold와 무관하게 한 번만
#   계산해도 leakage가 없다(다른 행/미래 정보를 참조하지 않음).
#
# 실행 방법 (저장소 루트에서):
#   python "EDA feature experiment/Combined_Feature_Experiment/cb_reg_v002_fe_v001_train.py"

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from scipy.special import betaln


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v002_fe_v001"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

FOLD_MODEL_DIR = MODEL_DIR / "folds"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

BASELINE_METADATA_PATH = VALIDATION_ROOT / "cb_reg_v002" / "metadata.json"

TEST_PATH = DATA_DIR / "test.csv"


# =========================================================
# 데이터 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"

# -----------------------------------------------------------------
# A. 유지 raw feature (34개) — 원본 47개 raw feature 중 남기기로 한 것
# -----------------------------------------------------------------
RAW_KEPT_FEATURES = [
    # 경기 정보 (6)
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    # Count / 경기 진행 (3)
    "balls_before", "strikes_before", "outs_before",
    # 점수 (3)
    "run_top_before", "run_bot_before", "score_diff_pitcher_team",
    # 주자 (3)
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    # 경기 중요도 (1)
    "li",
    # 선수 / 팀 (6)
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand",
    # Reliability n (3)
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
    # 최근 기록 (6)
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    # Pitch Mix (3, raw 유지 — 이번 실험은 BBEB 대상에서 제외)
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

# 원본 47개 raw feature 중 최종 모델 입력에서 제외한 13개
# (7개는 BBEB로 교체, 6개는 이번 baseline feature set에서 단순 제외 —
#  base_state는 컬럼 자체는 df에 그대로 두되 feature 목록에서만 뺀다)
RAW_FEATURES_REMOVED = [
    "run_total_before", "score_diff_home", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy",
    "asof_pitcher_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_reverse_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]

SITUATIONAL_FEATURES = [
    "count_balance", "full_count", "scoring_position", "pitcher_win_expectancy", "platoon",
]

TEMPORAL_FEATURES = [
    "recent1_success_vs_career", "recent3_success_vs_career", "recent5_success_vs_career",
    "recent1_middle_vs_career", "recent3_middle_vs_career", "recent5_middle_vs_career",
    "success_trend_1_3", "success_trend_3_5",
    "middle_trend_1_3", "middle_trend_3_5",
]

GAP_FEATURE = "eb_pitcher_batter_gap"

CAT_COLS = [
    "top_bottom", "game_type",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand",
    "count_balance", "platoon",
]

EXPANDING_FOLDS = [
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {"fold": 4, "train_seasons": [2019, 2020, 2021, 2022, 2023], "validation_season": 2024},
]


# =========================================================
# CatBoost 설정 (model/cb_reg_v002/train.py와 동일)
# =========================================================

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
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# BBEB (Beta-Binomial Empirical Bayes) — cb_reg_v002_eb_replace_v001_train.py와
# 동일한 로직을 그대로 재사용한다 (pitch-mix 3개 spec만 제외).
# =========================================================

EB_PRIOR_MIN_GROUP_TRIALS = 1
EB_INITIAL_STRENGTHS = [20.0, 100.0, 500.0]
EB_LOG_PARAM_LOWER = -12.0
EB_LOG_PARAM_UPPER = 14.0
EB_MAX_INTEGER_DISTANCE = 0.05

# method:
#   "target"        -> (id_col, TARGET)을 그대로 집계해 prior fit (정확한 관측치)
#   "asof_snapshot" -> id_col별 n이 가장 큰 행에서 (n, rate)로 count 복원 후 prior fit (근사)
BBEB_FEATURE_SPECS = [
    {"name": "pitcher_success", "rate_col": "asof_pitcher_success_rate", "n_col": "asof_pitcher_n", "id_col": "pitcher_id", "method": "target"},
    {"name": "batter_success", "rate_col": "asof_batter_success_rate", "n_col": "asof_batter_n", "id_col": "batter_id", "method": "target"},
    {"name": "pitcher_middle", "rate_col": "asof_pitcher_middle_rate", "n_col": "asof_pitcher_n", "id_col": "pitcher_id", "method": "asof_snapshot"},
    {"name": "pitcher_reverse", "rate_col": "asof_pitcher_reverse_rate", "n_col": "asof_pitcher_n", "id_col": "pitcher_id", "method": "asof_snapshot"},
    {"name": "batter_middle", "rate_col": "asof_batter_middle_rate", "n_col": "asof_batter_n", "id_col": "batter_id", "method": "asof_snapshot"},
    {"name": "pitcher_ball", "rate_col": "asof_pitcher_ball_rate", "n_col": "asof_pitcher_n", "id_col": "pitcher_id", "method": "asof_snapshot"},
    {"name": "pitcher_strike", "rate_col": "asof_pitcher_strike_rate", "n_col": "asof_pitcher_n", "id_col": "pitcher_id", "method": "asof_snapshot"},
]

BBEB_FEATURES = [f"bbeb_{spec['name']}_rate" for spec in BBEB_FEATURE_SPECS]

FINAL_FEATURES = (
    RAW_KEPT_FEATURES + SITUATIONAL_FEATURES + BBEB_FEATURES + [GAP_FEATURE] + TEMPORAL_FEATURES
)


def audit_denominator(df, n_col, rate_col):
    n = pd.to_numeric(df[n_col], errors="coerce")
    rate = pd.to_numeric(df[rate_col], errors="coerce")

    valid = (
        n.notna() & rate.notna()
        & np.isfinite(n.to_numpy(dtype=float))
        & np.isfinite(rate.to_numpy(dtype=float))
        & (n >= 0) & (rate >= 0) & (rate <= 1)
    )

    n_valid = n.loc[valid].to_numpy(dtype=float)
    rate_valid = rate.loc[valid].to_numpy(dtype=float)
    product = n_valid * rate_valid
    integer_distance = np.abs(product - np.rint(product))

    return {
        "n_valid_rows": int(len(integer_distance)),
        "n_total_rows": int(len(df)),
        "max_integer_distance": float(np.max(integer_distance)) if len(integer_distance) > 0 else float("nan"),
    }


def validate_exact_count_reconstruction(audit, feature_name):
    max_distance = float(audit["max_integer_distance"])
    if not np.isfinite(max_distance):
        raise ValueError(f"[{feature_name}] denominator audit 결과가 유효하지 않습니다.")
    if max_distance > EB_MAX_INTEGER_DISTANCE:
        raise ValueError(
            f"[{feature_name}] Exact EB count reconstruction을 중단합니다. "
            f"max |n*p-round(n*p)|={max_distance:.8f} > {EB_MAX_INTEGER_DISTANCE:.8f}"
        )


def reconstruct_counts(df, n_col, rate_col):
    n_raw = pd.to_numeric(df[n_col], errors="coerce").to_numpy(dtype=float)
    rate = pd.to_numeric(df[rate_col], errors="coerce").to_numpy(dtype=float)

    valid = (
        np.isfinite(n_raw) & np.isfinite(rate)
        & (n_raw >= 0.0) & (rate >= 0.0) & (rate <= 1.0)
    )

    n = np.full(len(df), np.nan, dtype=float)
    success = np.full(len(df), np.nan, dtype=float)
    failure = np.full(len(df), np.nan, dtype=float)

    n_valid = np.rint(n_raw[valid])
    success_valid = np.clip(np.rint(n_valid * rate[valid]), 0.0, n_valid)
    failure_valid = n_valid - success_valid

    n[valid] = n_valid
    success[valid] = success_valid
    failure[valid] = failure_valid

    return {"valid": valid, "n": n, "success": success, "failure": failure}


def build_binomial_summary_from_target(train_df, id_col, target_col):
    summary = (
        train_df[[id_col, target_col]]
        .dropna(subset=[id_col, target_col])
        .groupby(id_col, observed=True)[target_col]
        .agg(successes="sum", trials="count")
        .reset_index()
    )

    summary["successes"] = pd.to_numeric(summary["successes"], errors="raise").astype(float)
    summary["trials"] = pd.to_numeric(summary["trials"], errors="raise").astype(float)
    summary["failures"] = summary["trials"] - summary["successes"]

    summary = summary.loc[summary["trials"] >= EB_PRIOR_MIN_GROUP_TRIALS].copy()
    if len(summary) < 2:
        raise ValueError(f"[{id_col}/{target_col}] Beta prior를 추정할 그룹 수가 부족합니다.")

    return summary[[id_col, "trials", "successes", "failures"]].reset_index(drop=True)


def build_binomial_summary_from_asof_snapshot(train_df, id_col, n_col, rate_col):
    counts = reconstruct_counts(train_df, n_col, rate_col)

    work = pd.DataFrame({
        id_col: train_df[id_col].to_numpy(),
        "n": counts["n"],
        "success": counts["success"],
        "failure": counts["failure"],
    })
    work = work.loc[counts["valid"] & work[id_col].notna()].copy()

    if work.empty:
        raise ValueError(f"[{id_col}/{rate_col}] prior fit에 사용할 유효한 행이 없습니다.")

    best_idx = work.groupby(id_col)["n"].idxmax()
    summary = work.loc[best_idx].rename(columns={"n": "trials", "success": "successes", "failure": "failures"})

    summary = summary.loc[summary["trials"] >= EB_PRIOR_MIN_GROUP_TRIALS].copy()
    if len(summary) < 2:
        raise ValueError(f"[{id_col}/{rate_col}] Beta prior를 추정할 그룹 수가 부족합니다.")

    return summary[[id_col, "trials", "successes", "failures"]].reset_index(drop=True)


def beta_binomial_negative_log_likelihood(log_params, successes, failures):
    log_alpha, log_beta = float(log_params[0]), float(log_params[1])
    alpha, beta = float(np.exp(log_alpha)), float(np.exp(log_beta))

    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        return float("inf")

    ll = betaln(successes + alpha, failures + beta) - betaln(alpha, beta)
    if not np.all(np.isfinite(ll)):
        return float("inf")

    return -float(np.sum(ll))


def fit_beta_binomial_prior_from_summary(summary):
    successes = summary["successes"].to_numpy(dtype=float)
    failures = summary["failures"].to_numpy(dtype=float)

    total_successes = float(np.sum(successes))
    total_trials = float(np.sum(successes) + np.sum(failures))
    if total_trials <= 0.0:
        raise ValueError("Beta prior fit용 trial 수가 0입니다.")

    empirical_mean = float(np.clip(total_successes / total_trials, 1e-6, 1.0 - 1e-6))

    bounds = [(EB_LOG_PARAM_LOWER, EB_LOG_PARAM_UPPER), (EB_LOG_PARAM_LOWER, EB_LOG_PARAM_UPPER)]

    best_result = None
    for initial_strength in EB_INITIAL_STRENGTHS:
        alpha0 = empirical_mean * float(initial_strength)
        beta0 = (1.0 - empirical_mean) * float(initial_strength)
        x0 = np.log([max(alpha0, 1e-6), max(beta0, 1e-6)])

        result = minimize(
            beta_binomial_negative_log_likelihood, x0=x0,
            args=(successes, failures), method="L-BFGS-B", bounds=bounds,
        )
        if result.success and np.isfinite(result.fun):
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    if best_result is None:
        raise RuntimeError("Beta-Binomial prior MLE 최적화에 실패했습니다.")

    alpha = float(np.exp(best_result.x[0]))
    beta = float(np.exp(best_result.x[1]))
    prior_strength = float(alpha + beta)
    prior_mean = float(alpha / prior_strength)
    prior_variance = float(alpha * beta / (prior_strength**2 * (prior_strength + 1.0)))
    prior_std = float(np.sqrt(prior_variance))

    return {
        "alpha": alpha, "beta": beta, "prior_mean": prior_mean, "prior_strength": prior_strength,
        "prior_std": prior_std, "n_groups": int(len(summary)), "n_trials": int(total_trials),
        "n_successes": int(total_successes), "optimizer_success": bool(best_result.success),
        "negative_log_likelihood": float(best_result.fun),
    }


def fit_feature_prior(train_df, spec):
    if spec["method"] == "target":
        summary = build_binomial_summary_from_target(train_df, spec["id_col"], TARGET)
    elif spec["method"] == "asof_snapshot":
        summary = build_binomial_summary_from_asof_snapshot(train_df, spec["id_col"], spec["n_col"], spec["rate_col"])
    else:
        raise ValueError(f"알 수 없는 method: {spec['method']}")

    return fit_beta_binomial_prior_from_summary(summary)


def compute_eb_rate(df, n_col, rate_col, prior_state):
    alpha = float(prior_state["alpha"])
    beta = float(prior_state["beta"])
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Reliability prior alpha/beta가 유효하지 않습니다.")

    counts = reconstruct_counts(df, n_col, rate_col)
    valid = counts["valid"]
    success = counts["success"]
    failure = counts["failure"]

    prior_total = alpha + beta
    prior_mean = alpha / prior_total
    prior_var = alpha * beta / (prior_total**2 * (prior_total + 1.0))
    prior_std = float(np.sqrt(prior_var))

    eb_rate = np.full(len(df), prior_mean, dtype=float)
    posterior_std = np.full(len(df), prior_std, dtype=float)

    if np.any(valid):
        posterior_alpha = alpha + success[valid]
        posterior_beta = beta + failure[valid]
        posterior_total = posterior_alpha + posterior_beta
        eb_rate[valid] = posterior_alpha / posterior_total
        posterior_var = np.maximum(
            posterior_alpha * posterior_beta / (posterior_total**2 * (posterior_total + 1.0)), 0.0,
        )
        posterior_std[valid] = np.sqrt(posterior_var)

    if not np.all(np.isfinite(eb_rate)) or not np.all(np.isfinite(posterior_std)):
        raise ValueError("EB posterior 값에 비정상 값이 있습니다.")
    if np.any(eb_rate <= 0.0) or np.any(eb_rate >= 1.0) or np.any(posterior_std < 0.0):
        raise ValueError("EB posterior Feature 범위가 유효하지 않습니다.")

    return eb_rate, posterior_std


def fit_all_bbeb_priors(train_df):
    return {spec["name"]: fit_feature_prior(train_df, spec) for spec in BBEB_FEATURE_SPECS}


def apply_bbeb_features(df, priors_by_name):
    """
    df에 bbeb_<name>_rate 7개 + eb_pitcher_batter_gap 1개를 추가한다.
    df 자신의 asof_* 값만 사용하며 다른 행을 참조하지 않는다(row-wise).
    """
    df = df.copy()

    for spec in BBEB_FEATURE_SPECS:
        prior_state = priors_by_name[spec["name"]]
        eb_rate, _posterior_std = compute_eb_rate(df, spec["n_col"], spec["rate_col"], prior_state)
        df[f"bbeb_{spec['name']}_rate"] = eb_rate

    df[GAP_FEATURE] = df["bbeb_pitcher_success_rate"] - df["bbeb_batter_success_rate"]

    gap_recheck = np.allclose(
        df[GAP_FEATURE].to_numpy(),
        (df["bbeb_pitcher_success_rate"] - df["bbeb_batter_success_rate"]).to_numpy(),
    )
    if not gap_recheck:
        raise ValueError("[QA 실패] eb_pitcher_batter_gap 재계산 불일치")

    return df


# =========================================================
# 신규 상황(situational) / Temporal feature
# =========================================================

def add_situational_features(df):
    df = df.copy()

    balls = pd.to_numeric(df["balls_before"], errors="coerce")
    strikes = pd.to_numeric(df["strikes_before"], errors="coerce")

    df["count_balance"] = np.where(
        balls > strikes, "batter_ahead",
        np.where(balls < strikes, "pitcher_ahead", "even"),
    )

    df["full_count"] = ((balls == 3) & (strikes == 2)).astype(int)

    df["scoring_position"] = (
        (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)
    ).astype(int)

    is_pitcher_home = df["top_bottom"] == "T"
    df["pitcher_win_expectancy"] = np.where(
        is_pitcher_home, df["home_win_expectancy"], df["away_win_expectancy"]
    )

    df["platoon"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)

    return df


def add_temporal_features(df):
    """EXP-3 공식(Temporal Feature Experiment/exp_code/exp3_temporal_only.py:194-263)과 동일."""
    df = df.copy()

    df["recent1_success_vs_career"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["recent3_success_vs_career"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["recent5_success_vs_career"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]

    df["recent1_middle_vs_career"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["recent3_middle_vs_career"] = df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["recent5_middle_vs_career"] = df["asof_pitcher_prev5_game_middle_rate"] - df["asof_pitcher_middle_rate"]

    df["success_trend_1_3"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev3_game_success_rate"]
    df["success_trend_3_5"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]

    df["middle_trend_1_3"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_prev3_game_middle_rate"]
    df["middle_trend_3_5"] = df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_prev5_game_middle_rate"]

    return df


# =========================================================
# Data QA (situational / temporal — fold와 무관한 부분)
# =========================================================

def _numeric_stats(series):
    s = pd.to_numeric(series, errors="coerce")
    has_valid = bool(s.notna().any())
    return {
        "n_nan": int(s.isna().sum()),
        "n_inf": int(np.isinf(s.to_numpy(dtype=float)).sum()) if has_valid else 0,
        "min": float(np.nanmin(s)) if has_valid else None,
        "max": float(np.nanmax(s)) if has_valid else None,
        "mean": float(np.nanmean(s)) if has_valid else None,
    }


def run_data_qa(raw_train, engineered_train):
    qa = {}

    # --------------------------------------------------- Row QA
    row_id_order_match = bool((raw_train[ID].to_numpy() == engineered_train[ID].to_numpy()).all())
    row_id_unique = bool(engineered_train[ID].is_unique)
    target_unchanged = bool(np.array_equal(
        raw_train[TARGET].to_numpy(), engineered_train[TARGET].to_numpy()
    ))
    row_count_match = bool(len(raw_train) == len(engineered_train))

    qa["row_qa"] = {
        "n_rows_before": int(len(raw_train)),
        "n_rows_after": int(len(engineered_train)),
        "row_count_match": row_count_match,
        "row_id_order_match": row_id_order_match,
        "row_id_unique": row_id_unique,
        "target_unchanged": target_unchanged,
    }

    for key in ("row_count_match", "row_id_order_match", "row_id_unique", "target_unchanged"):
        if not qa["row_qa"][key]:
            raise ValueError(f"[Row QA 실패] {key}")

    # --------------------------------------------------- Feature QA: situational
    feature_qa = {}

    situational_recompute = add_situational_features(raw_train[[
        "balls_before", "strikes_before", "runner_on_2b", "runner_on_3b",
        "top_bottom", "home_win_expectancy", "away_win_expectancy",
        "pitcher_hand", "batter_hand",
    ]])

    for col in ["full_count", "scoring_position", "pitcher_win_expectancy"]:
        a = pd.to_numeric(engineered_train[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(situational_recompute[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        match = bool(np.allclose(a[~both_nan], b[~both_nan]) and np.array_equal(np.isnan(a), np.isnan(b)))
        feature_qa[col] = {"recompute_match": match, **_numeric_stats(engineered_train[col])}
        if not match:
            raise ValueError(f"[Feature QA 실패] {col} 재계산 불일치")

    for col in ["count_balance", "platoon"]:
        exact_match = bool((engineered_train[col].to_numpy() == situational_recompute[col].to_numpy()).all())
        feature_qa[col] = {
            "recompute_match": exact_match,
            "unique_values": sorted(str(v) for v in engineered_train[col].unique()),
            "n_nan": int(engineered_train[col].isna().sum()),
        }
        if not exact_match:
            raise ValueError(f"[Feature QA 실패] {col} 재계산 불일치")

    # --------------------------------------------------- Feature QA: temporal
    temporal_recompute = add_temporal_features(raw_train[[
        "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]])

    for col in TEMPORAL_FEATURES:
        a = pd.to_numeric(engineered_train[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(temporal_recompute[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        match = bool(np.allclose(a[~both_nan], b[~both_nan]) and np.array_equal(np.isnan(a), np.isnan(b)))
        feature_qa[col] = {"recompute_match": match, **_numeric_stats(engineered_train[col])}
        if not match:
            raise ValueError(f"[Feature QA 실패] {col} 재계산 불일치")

    qa["feature_qa"] = feature_qa

    # --------------------------------------------------- 특수 검증
    special = {}

    strikes_over2 = int((pd.to_numeric(engineered_train["strikes_before"], errors="coerce") > 2).sum())
    special["strikes_before_gt_2_count"] = strikes_over2
    if strikes_over2 > 0:
        print(f"[QA 경고] strikes_before > 2인 행이 {strikes_over2:,}개 존재합니다.")

    full_count_rows = engineered_train["full_count"] == 1
    full_count_valid = bool((
        (engineered_train.loc[full_count_rows, "balls_before"] == 3)
        & (engineered_train.loc[full_count_rows, "strikes_before"] == 2)
    ).all())
    special["full_count_definition_valid"] = full_count_valid
    if not full_count_valid:
        raise ValueError("[QA 실패] full_count==1인데 balls==3,strikes==2가 아닌 행이 있습니다.")

    scoring_rows = engineered_train["scoring_position"] == 1
    scoring_valid = bool((
        (engineered_train.loc[scoring_rows, "runner_on_2b"] == 1)
        | (engineered_train.loc[scoring_rows, "runner_on_3b"] == 1)
    ).all())
    special["scoring_position_definition_valid"] = scoring_valid
    if not scoring_valid:
        raise ValueError("[QA 실패] scoring_position==1인데 runner_on_2b/3b 조건을 만족 못하는 행이 있습니다.")

    qa["special_checks"] = special

    return qa


# =========================================================
# Feature 전처리 / Metric (model/cb_reg_v002/train.py와 동일)
# =========================================================

def prepare_features(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna("__MISSING__").astype(str)
    return df


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(100000.0 * (1.0 - brier / baseline_brier))
        bss = max(0.0, bss_raw)

    return {
        "positive_rate": positive_rate, "brier": brier,
        "baseline_brier": baseline_brier, "bss_raw": bss_raw, "bss": bss,
    }


def calculate_range_diagnostics(y_pred_raw):
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.float64)
    below_zero = y_pred_raw < 0.0
    above_one = y_pred_raw > 1.0
    outside_range = below_zero | above_one

    return {
        "n_prediction_below_0": int(np.sum(below_zero)),
        "ratio_prediction_below_0": float(np.mean(below_zero)),
        "n_prediction_above_1": int(np.sum(above_one)),
        "ratio_prediction_above_1": float(np.mean(above_one)),
        "n_prediction_outside_0_1": int(np.sum(outside_range)),
        "ratio_prediction_outside_0_1": float(np.mean(outside_range)),
    }


def save_fold_validation(
    experiment_name, fold, train_seasons, validation_season, estimator, model_params,
    row_ids, y_true, y_pred_raw, y_pred_clip, raw_metrics, clipped_metrics, output_root,
    model_path, n_train, elapsed_seconds, best_iteration=None, final_iterations=None,
    reliability_priors=None,
):
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred_raw = np.asarray(y_pred_raw)
    y_pred_clip = np.asarray(y_pred_clip)

    assert len(row_ids) == len(y_true) == len(y_pred_raw) == len(y_pred_clip)

    output_dir = Path(output_root) / experiment_name / f"fold_{validation_season}"
    output_dir.mkdir(parents=True, exist_ok=False)

    range_diag = calculate_range_diagnostics(y_pred_raw)

    pred_df = pd.DataFrame({
        "row_id": row_ids, "target": y_true, "prediction_raw": y_pred_raw,
        "prediction_clip": y_pred_clip, "prediction": y_pred_clip,
        "validation_season": validation_season,
    })
    pred_df.to_csv(output_dir / "pred.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        output_dir / "validation.npz", row_id=row_ids, target=y_true,
        prediction_raw=y_pred_raw, prediction_clip=y_pred_clip, prediction=y_pred_clip,
        validation_season=np.full(len(y_true), validation_season),
    )

    metadata = {
        "experiment_name": experiment_name, "fold": int(fold),
        "train_seasons": [int(s) for s in train_seasons],
        "validation_season": int(validation_season),
        "n_train": int(n_train), "n_validation": int(len(y_true)),
        "model_type": type(estimator).__name__, "model_params": model_params,
        "positive_rate": float(raw_metrics["positive_rate"]),
        "raw_brier": float(raw_metrics["brier"]), "raw_bss_raw": float(raw_metrics["bss_raw"]),
        "raw_bss": float(raw_metrics["bss"]),
        "clipped_brier": float(clipped_metrics["brier"]), "clipped_bss_raw": float(clipped_metrics["bss_raw"]),
        "clipped_bss": float(clipped_metrics["bss"]),
        "brier": float(clipped_metrics["brier"]), "baseline_brier": float(clipped_metrics["baseline_brier"]),
        "bss_raw": float(clipped_metrics["bss_raw"]), "bss": float(clipped_metrics["bss"]),
        "prediction_raw_mean": float(np.mean(y_pred_raw)), "prediction_raw_std": float(np.std(y_pred_raw)),
        "prediction_raw_min": float(np.min(y_pred_raw)), "prediction_raw_max": float(np.max(y_pred_raw)),
        "prediction_clip_mean": float(np.mean(y_pred_clip)), "prediction_clip_std": float(np.std(y_pred_clip)),
        "prediction_clip_min": float(np.min(y_pred_clip)), "prediction_clip_max": float(np.max(y_pred_clip)),
        "elapsed_seconds": float(elapsed_seconds), "model_path": str(model_path),
        **range_diag,
    }

    if best_iteration is not None:
        metadata["best_iteration"] = int(best_iteration)
    if final_iterations is not None:
        metadata["final_iterations"] = int(final_iterations)
    if reliability_priors is not None:
        metadata["bbeb_priors"] = reliability_priors

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return pred_df, metadata


# =========================================================
# Expanding Window Validation
# =========================================================

def run_expanding_validation(train, features, cat_cols):
    fold_results = []
    oof_frames = []

    for fold_config in EXPANDING_FOLDS:
        fold = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        validation_season = fold_config["validation_season"]

        train_mask = train["season"].isin(train_seasons)
        val_mask = train["season"] == validation_season

        assert not (train_mask & val_mask).any(), f"Fold {fold}: train/val row overlap 발견"

        fold_train_raw = train.loc[train_mask].copy()
        fold_val_raw = train.loc[val_mask].copy()

        # BBEB prior는 그 fold의 train 구간 전체(expanding)로만 fit한다.
        fold_bbeb_priors = fit_all_bbeb_priors(fold_train_raw)

        fold_train_df = apply_bbeb_features(fold_train_raw, fold_bbeb_priors)
        fold_val_df = apply_bbeb_features(fold_val_raw, fold_bbeb_priors)

        assert set(fold_train_df[ID]).isdisjoint(set(fold_val_df[ID])), (
            f"Fold {fold}: train/val row_id overlap 발견"
        )

        X_train = prepare_features(fold_train_df[features], cat_cols)
        y_train = fold_train_df[TARGET].astype(np.float64)

        X_val = prepare_features(fold_val_df[features], cat_cols)
        y_val = fold_val_df[TARGET].astype(np.float64)

        val_row_id = fold_val_df[ID]

        print()
        print("=" * 70)
        print(f"Fold {fold}: {train_seasons[0]}~{train_seasons[-1]} -> {validation_season}")
        print("=" * 70)
        print("학습 데이터:", X_train.shape)
        print("검증 데이터:", X_val.shape)

        print()
        print("[BBEB prior — fold train 구간 전체(expanding)에서만 fit]")
        for spec in BBEB_FEATURE_SPECS:
            prior = fold_bbeb_priors[spec["name"]]
            print(
                f"  - {spec['name']:<18s} prior_mean={prior['prior_mean']:.4f}  "
                f"prior_strength={prior['prior_strength']:.1f}  n_groups={prior['n_groups']:,}"
            )

        estimator = CatBoostRegressor(**MODEL_PARAMS)

        start_time = time.time()
        estimator.fit(
            X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True,
        )
        elapsed = time.time() - start_time

        val_pred_raw = estimator.predict(X_val)
        val_pred_clip = np.clip(val_pred_raw, 0.0, 1.0)

        raw_metrics = calculate_metrics(y_val, val_pred_raw)
        clipped_metrics = calculate_metrics(y_val, val_pred_clip)

        best_iteration = estimator.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            final_iterations = MODEL_PARAMS["iterations"]
        else:
            final_iterations = best_iteration + 1

        fold_model_dir = FOLD_MODEL_DIR / f"fold_{validation_season}"
        fold_model_dir.mkdir(parents=True, exist_ok=False)
        fold_model_path = fold_model_dir / f"{EXPERIMENT_NAME}_fold_{validation_season}.pkl"
        joblib.dump(estimator, fold_model_path, compress=3)

        pred_df, metadata = save_fold_validation(
            experiment_name=EXPERIMENT_NAME, fold=fold, train_seasons=train_seasons,
            validation_season=validation_season, estimator=estimator, model_params=MODEL_PARAMS,
            row_ids=val_row_id, y_true=y_val, y_pred_raw=val_pred_raw, y_pred_clip=val_pred_clip,
            raw_metrics=raw_metrics, clipped_metrics=clipped_metrics, output_root=VALIDATION_ROOT,
            model_path=fold_model_path, n_train=len(y_train), elapsed_seconds=elapsed,
            best_iteration=best_iteration, final_iterations=final_iterations,
            reliability_priors=fold_bbeb_priors,
        )

        fold_results.append(metadata)
        oof_frames.append(pred_df)

        print()
        print("[Clipped prediction]")
        print(f"Brier Score      : {clipped_metrics['brier']:.8f}")
        print(f"BSS              : {clipped_metrics['bss']:.2f}")
        print()
        print(f"Best iteration   : {best_iteration}")
        print(f"Final iterations : {final_iterations}")
        print(f"학습 시간         : {elapsed:.1f}초")

    oof = pd.concat(oof_frames, ignore_index=True)
    return fold_results, oof


def save_validation_summary(features, cat_cols, fold_results, oof):
    briers = np.asarray([fold["brier"] for fold in fold_results], dtype=float)
    bss_values = np.asarray([fold["bss"] for fold in fold_results], dtype=float)
    fold_sizes = np.asarray([fold["n_validation"] for fold in fold_results], dtype=float)

    weighted_fold_brier = float(np.average(briers, weights=fold_sizes))
    weighted_fold_bss = float(np.average(bss_values, weights=fold_sizes))
    recent_briers = [fold["brier"] for fold in fold_results if fold["validation_season"] in (2023, 2024)]

    raw_oof_metrics = calculate_metrics(oof["target"], oof["prediction_raw"])
    clipped_oof_metrics = calculate_metrics(oof["target"], oof["prediction_clip"])
    raw_range_diag = calculate_range_diagnostics(oof["prediction_raw"])

    summary = {
        "n_oof": int(len(oof)),
        "mean_fold_brier": float(np.mean(briers)), "std_fold_brier": float(np.std(briers)),
        "weighted_fold_brier": weighted_fold_brier, "best_fold_brier": float(np.min(briers)),
        "worst_fold_brier": float(np.max(briers)), "fold_brier_range": float(np.max(briers) - np.min(briers)),
        "recent_2023_2024_mean_brier": float(np.mean(recent_briers)),
        "mean_fold_bss": float(np.mean(bss_values)), "weighted_fold_bss": weighted_fold_bss,
        "pooled_oof_raw_brier": float(raw_oof_metrics["brier"]),
        "pooled_oof_raw_bss_raw": float(raw_oof_metrics["bss_raw"]), "pooled_oof_raw_bss": float(raw_oof_metrics["bss"]),
        "pooled_oof_clipped_brier": float(clipped_oof_metrics["brier"]),
        "pooled_oof_clipped_baseline_brier": float(clipped_oof_metrics["baseline_brier"]),
        "pooled_oof_clipped_bss_raw": float(clipped_oof_metrics["bss_raw"]),
        "pooled_oof_clipped_bss": float(clipped_oof_metrics["bss"]),
        "pooled_oof_brier": float(clipped_oof_metrics["brier"]),
        "pooled_oof_baseline_brier": float(clipped_oof_metrics["baseline_brier"]),
        "pooled_oof_bss_raw": float(clipped_oof_metrics["bss_raw"]), "pooled_oof_bss": float(clipped_oof_metrics["bss"]),
        **raw_range_diag,
    }

    oof.to_csv(VALIDATION_DIR / "oof_pred.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        VALIDATION_DIR / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(), target=oof["target"].to_numpy(),
        prediction_raw=oof["prediction_raw"].to_numpy(), prediction_clip=oof["prediction_clip"].to_numpy(),
        prediction=oof["prediction"].to_numpy(), validation_season=oof["validation_season"].to_numpy(),
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME, "baseline_experiment": "cb_reg_v002",
        "validation_method": "expanding_window", "model_type": "CatBoostRegressor", "objective": "RMSE",
        "prediction_for_evaluation": "clipped", "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "features": features, "categorical_features": cat_cols,
        "bbeb_feature_specs": BBEB_FEATURE_SPECS,
        "folds": fold_results, "summary": summary,
    }

    with open(VALIDATION_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return summary


def select_final_iterations(fold_results):
    if FINAL_ITERATION_STRATEGY == "latest_fold":
        latest_fold = max(fold_results, key=lambda x: x["validation_season"])
        return int(latest_fold["final_iterations"])
    raise ValueError(f"Unsupported FINAL_ITERATION_STRATEGY: {FINAL_ITERATION_STRATEGY}")


def train_final_model(train, features, cat_cols, final_iterations):
    final_bbeb_priors = fit_all_bbeb_priors(train)
    train_with_bbeb = apply_bbeb_features(train, final_bbeb_priors)

    X_full = prepare_features(train_with_bbeb[features], cat_cols)
    y_full = train_with_bbeb[TARGET].astype(np.float64)

    final_params = {**MODEL_PARAMS, "iterations": int(final_iterations)}
    final_model = CatBoostRegressor(**final_params)

    print()
    print("=" * 70)
    print("전체 데이터 CatBoostRegressor 재학습 시작")
    print(f"Final iterations: {final_iterations}")
    print("=" * 70)

    print()
    print("[BBEB prior — 전체 train에서 fit, test에 얼려서 적용]")
    for spec in BBEB_FEATURE_SPECS:
        prior = final_bbeb_priors[spec["name"]]
        print(
            f"  - {spec['name']:<18s} prior_mean={prior['prior_mean']:.4f}  "
            f"prior_strength={prior['prior_strength']:.1f}  n_groups={prior['n_groups']:,}"
        )

    start_time = time.time()
    final_model.fit(X_full, y_full, cat_features=cat_cols)
    elapsed = time.time() - start_time

    print(f"전체 데이터 재학습 완료: {elapsed:.1f}초")

    return final_model, final_params, elapsed, final_bbeb_priors


def save_final_metadata(train, features, cat_cols, final_params, final_iterations, elapsed_seconds, final_bbeb_priors):
    priors_path = MODEL_DIR / "bbeb_priors.json"
    with open(priors_path, "w", encoding="utf-8") as f:
        json.dump(final_bbeb_priors, f, indent=2, ensure_ascii=False)

    metadata = {
        "experiment_name": EXPERIMENT_NAME, "baseline_experiment": "cb_reg_v002",
        "model_type": "CatBoostRegressor", "objective": "RMSE",
        "training_seasons": sorted(int(x) for x in train["season"].unique()),
        "n_train": int(len(train)), "features": features, "categorical_features": cat_cols,
        "bbeb_feature_specs": BBEB_FEATURE_SPECS, "bbeb_priors_path": str(priors_path),
        "model_params": final_params, "final_iterations": int(final_iterations),
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY, "elapsed_seconds": float(elapsed_seconds),
        "model_path": str(MODEL_PATH),
        "inference_note": (
            "CatBoostRegressor predict() output may fall outside [0, 1]. "
            "Use np.clip(pred, 0.0, 1.0) for submission. "
            "test.csv 추론 시 bbeb_priors.json으로 bbeb_<name>_rate를 다시 계산해야 한다."
        ),
    }

    with open(MODEL_DIR / "final_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_feature_manifest():
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "raw_features_used": RAW_KEPT_FEATURES,
        "raw_features_removed": RAW_FEATURES_REMOVED,
        "bbeb_features": BBEB_FEATURES,
        "situation_features": SITUATIONAL_FEATURES + [GAP_FEATURE],
        "temporal_features": TEMPORAL_FEATURES,
        "categorical_features": CAT_COLS,
        "total_feature_count": len(FINAL_FEATURES),
        "validation_method": "expanding_window",
    }
    with open(MODEL_DIR / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def build_test_features(test, final_bbeb_priors):
    """test.csv에 동일한 상황/Temporal/BBEB 파생 로직 + 얼린 prior를 적용한다."""
    test = add_situational_features(test)
    test = add_temporal_features(test)
    test = apply_bbeb_features(test, final_bbeb_priors)
    return test


# =========================================================
# Main
# =========================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(f"Validation experiment already exists: {VALIDATION_DIR}\n새 EXPERIMENT_NAME을 사용하세요.")
    if MODEL_DIR.exists():
        raise FileExistsError(f"Model experiment already exists: {MODEL_DIR}\n새 EXPERIMENT_NAME을 사용하세요.")

    assert len(FINAL_FEATURES) == len(set(FINAL_FEATURES)), "FINAL_FEATURES에 중복이 있습니다."
    assert set(CAT_COLS) <= set(FINAL_FEATURES), "CAT_COLS가 FINAL_FEATURES에 전부 포함되어야 합니다."

    needed_cols = (
        set(RAW_KEPT_FEATURES)
        | {ID, TARGET, "season"}
        | {"home_win_expectancy", "away_win_expectancy"}
        | {spec["rate_col"] for spec in BBEB_FEATURE_SPECS}
    )

    raw_train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=sorted(needed_cols))

    print("=" * 70)
    print("데이터 로드 완료")
    print("train shape:", raw_train.shape)
    print("최종 Feature 수:", len(FINAL_FEATURES))
    print("범주형 Feature 수:", len(CAT_COLS))
    print("범주형 Feature:", CAT_COLS)
    print("시즌:", raw_train["season"].min(), "~", raw_train["season"].max())
    print(f"제구 성공률: {raw_train[TARGET].mean():.6f}")
    print("=" * 70)

    # -----------------------------------------------------
    # denominator audit (BBEB 대상 7개 (n, rate) 쌍)
    # -----------------------------------------------------
    print()
    print("[denominator audit] n * rate가 정수에 가까운지 확인")
    for spec in BBEB_FEATURE_SPECS:
        audit = audit_denominator(raw_train, spec["n_col"], spec["rate_col"])
        print(f"  - {spec['name']:<18s} valid={audit['n_valid_rows']:>9,}/{audit['n_total_rows']:>9,}  max|n*p-round|={audit['max_integer_distance']:.3e}")
        validate_exact_count_reconstruction(audit, spec["name"])

    # -----------------------------------------------------
    # 상황(situational) + Temporal feature (fold와 무관, 1회 계산)
    # -----------------------------------------------------
    train = add_situational_features(raw_train)
    train = add_temporal_features(train)

    print()
    print("[Data QA] situational / temporal feature 검증 중...")
    qa_report = run_data_qa(raw_train, train)
    print("[Data QA] 통과")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    FOLD_MODEL_DIR.mkdir(parents=True, exist_ok=False)

    with open(VALIDATION_DIR / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------
    # Expanding Window Validation (BBEB는 fold-safe로 fold 루프 안에서 적용)
    # -----------------------------------------------------
    fold_results, oof = run_expanding_validation(train=train, features=FINAL_FEATURES, cat_cols=CAT_COLS)
    summary = save_validation_summary(features=FINAL_FEATURES, cat_cols=CAT_COLS, fold_results=fold_results, oof=oof)

    print()
    print("=" * 70)
    print("Expanding Window 결과 요약")
    print("=" * 70)
    for fold in fold_results:
        print(
            f"Valid {fold['validation_season']}: Brier={fold['brier']:.8f}, "
            f"BSS={fold['bss']:.2f}, iter={fold['final_iterations']}, "
            f"outside={fold['ratio_prediction_outside_0_1']:.6%}"
        )
    print(f"Fold 단순 평균 Brier : {summary['mean_fold_brier']:.8f}")
    print(f"Fold 가중 평균 Brier : {summary['weighted_fold_brier']:.8f}")
    print(f"OOF clipped Brier    : {summary['pooled_oof_clipped_brier']:.8f}")
    print(f"OOF clipped BSS      : {summary['pooled_oof_clipped_bss']:.2f}")
    print("=" * 70)

    final_iterations = select_final_iterations(fold_results)
    print()
    print(f"Final iteration strategy : {FINAL_ITERATION_STRATEGY}")
    print(f"Final iterations         : {final_iterations}")

    final_model, final_params, final_elapsed, final_bbeb_priors = train_final_model(
        train=train, features=FINAL_FEATURES, cat_cols=CAT_COLS, final_iterations=final_iterations,
    )

    # script.py/package.py는 model/<name>/<name>.pkl 파일 하나만 제출 zip에
    # 담으므로(final_metadata.json, bbeb_priors.json은 zip에 포함되지 않음),
    # 추론에 필요한 모든 정보를 pkl 하나에 자체 포함(self-contained)시킨다.
    final_artifact = {
        "model": final_model,
        "experiment_name": EXPERIMENT_NAME,
        "objective": MODEL_PARAMS["loss_function"],
        "features": FINAL_FEATURES,
        "cat_cols": CAT_COLS,
        "bbeb_feature_specs": BBEB_FEATURE_SPECS,
        "bbeb_priors": final_bbeb_priors,
        "final_iterations": int(final_iterations),
    }
    joblib.dump(final_artifact, MODEL_PATH, compress=3)
    save_final_metadata(
        train=train, features=FINAL_FEATURES, cat_cols=CAT_COLS, final_params=final_params,
        final_iterations=final_iterations, elapsed_seconds=final_elapsed, final_bbeb_priors=final_bbeb_priors,
    )
    manifest = save_feature_manifest()

    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, MODEL_DIR / "train.py")

    # -----------------------------------------------------
    # test.csv에도 동일 파생 로직 + 얼린 prior 적용 (제출용 feature 생성까지)
    # -----------------------------------------------------
    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
        test_with_features = build_test_features(test, final_bbeb_priors)
        missing_test_features = [c for c in FINAL_FEATURES if c not in test_with_features.columns]
        if missing_test_features:
            raise ValueError(f"test.csv에서 생성하지 못한 feature: {missing_test_features}")
        test_with_features[[ID] + FINAL_FEATURES].to_csv(
            MODEL_DIR / "test_features_preview.csv", index=False, encoding="utf-8-sig"
        )
        print()
        print(f"test.csv feature 생성 확인 완료 (rows={len(test_with_features)}) -> {MODEL_DIR / 'test_features_preview.csv'}")

    print()
    print("=" * 70)
    print("Final Model 저장 완료")
    print(f"- {MODEL_PATH}")
    print(f"- {MODEL_DIR / 'final_metadata.json'}")
    print(f"- {MODEL_DIR / 'feature_manifest.json'}")
    print(f"- {MODEL_DIR / 'bbeb_priors.json'}")
    print(f"- {VALIDATION_DIR / 'qa_report.json'}")
    print("=" * 70)

    # -----------------------------------------------------
    # Baseline(cb_reg_v002) 비교 — 재학습 없이 기존 결과 재사용
    # -----------------------------------------------------
    print()
    print("=" * 70)
    print("Baseline(cb_reg_v002) 대비 비교 (재학습 없이 기존 결과 재사용)")
    print("=" * 70)
    if BASELINE_METADATA_PATH.exists():
        with open(BASELINE_METADATA_PATH, "r", encoding="utf-8") as f:
            baseline_metadata = json.load(f)
        baseline_brier = baseline_metadata["summary"]["pooled_oof_clipped_brier"]
        new_brier = summary["pooled_oof_clipped_brier"]
        diff = new_brier - baseline_brier
        print(f"baseline (cb_reg_v002)      pooled_oof_clipped_brier = {baseline_brier:.8f}")
        print(f"new ({EXPERIMENT_NAME}) pooled_oof_clipped_brier = {new_brier:.8f}")
        print(f"diff (new - baseline)                              = {diff:+.8f} ({'개선' if diff < 0 else '악화'})")
    else:
        print(f"[경고] baseline metadata를 찾을 수 없습니다: {BASELINE_METADATA_PATH}")

    print()
    print(f"total_feature_count = {manifest['total_feature_count']}")


if __name__ == "__main__":
    main()
