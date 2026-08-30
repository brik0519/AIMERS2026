# train.py: XGBoost xgb_v002_fe_v001_exact4_success_only
#
# xgb_v002_fe_v001_exact4(61개)에서 reliability smoothing 정책을 바꾼 실험이다.
# 기존에는 7개 rate 컬럼(pitcher_success/batter_success/pitcher_middle/
# pitcher_reverse/batter_middle/pitcher_ball/pitcher_strike) 전부를 BBEB로
# 스무딩했는데, 이번엔 asof_pitcher_success_rate/asof_batter_success_rate
# 2개만 BBEB로 보정하고 나머지 5개(asof_pitcher_middle/reverse/ball/strike_rate,
# asof_batter_middle_rate)는 raw 그대로 둔다 — xgb_v002_exact_asof_hbeb_platoon/
# Asof_Residual_base와 동일한 원칙("success_rate만 보정, 나머지 asof는 raw").
# feature 개수는 61개로 동일(스무딩 대상 5개가 빠지고 raw 5개가 그 자리를
# 대신 채움). gated residual correction은 이번 실험에 포함하지 않는다.
#
#
# xgb_v002_fe_v001(57개: raw34+BBEB7+situational5+gap+temporal10)에 Exact-ASOF
# 4개(exact_pitcher_season_n, exact_pitcher_season_success_rate,
# exact_pitcher_past_success_rate, exact_pitcher_season_minus_past)를 추가한
# 실험이다(exact_pitcher_season_logn은 이번엔 포함하지 않음 — 사용자가 명시한
# 4개만 추가). Exact-ASOF 로직은 Asof_Residual_base/train.py의
# build_pitcher_season_stats/add_pitcher_exact_asof_features를 그대로
# 재사용한다(fold-safe: 그 fold의 train 구간에서 완료된 과거 시즌 통계만
# 사용해 "이번 시즌 지금까지" 값을 복원).
#
# model/xgb_v002/train.py(raw47, XGBClassifier, 4-Fold Expanding Window
# Validation)를 기준 모델로 삼고, EDA feature experiment/
# Combined_Feature_Experiment/cb_reg_v002_fe_v001_train.py에서 검증한
# feature set(57개: raw34 + BBEB7 + situational5 + eb_pitcher_batter_gap +
# Temporal10)을 그대로 이식한 실험이다. xgb_v001(단일 holdout)에는 적용하지
# 않는다 — xgb_v002의 4-fold Expanding Window가 cb_reg_v002_fe_v001과 같은
# validation 방식이라 더 신뢰도 높은 비교이기 때문.
#
# 1) BBEB(Beta-Binomial Empirical Bayes) smoothing 7개
#    cb_reg_v002_fe_v001_train.py의 함수를 그대로 재사용하며, prior는 항상
#    "그 fold의 train 구간 전체(expanding)"로 fit한다.
# 2) 신규 상황(situational) feature 5개: count_balance, full_count,
#    scoring_position, pitcher_win_expectancy, platoon
# 3) eb_pitcher_batter_gap = bbeb_pitcher_success_rate - bbeb_batter_success_rate
# 4) 기존 Temporal Feature Experiment EXP-3의 Temporal 10개
#
# Categorical/ID 인코딩 정책: xgb_v002의 기존 방식(저카디널리티만
# OrdinalEncoder, 고카디널리티 ID는 numeric passthrough)을 그대로 유지하고
# feature 목록만 fe_v001로 교체한다 — "feature set 변경의 효과"만 순수하게
# 비교하기 위해 인코딩 전략 자체는 xgb_v002와 동일하게 둔다. CatBoost
# fe_v001처럼 pitcher_id 등을 native categorical로 다루지 않는다.
#
# 핵심 원칙 (xgb_v002/cb_reg_v002_fe_v001과 동일)
# - 평가 데이터(test.csv)의 다른 행/전체 분포를 Feature 생성에 사용하지 않는다.
# - Validation은 train.csv 내부 시즌만으로 Expanding Window 방식으로 수행한다.
# - BBEB prior는 항상 해당 fold의 train 구간에서만 fit하고, 각 행의 EB rate는
#   그 행 자신의 공식 asof_* 값만으로 계산한다(다른 행 참조 없음).
# - 상황/Temporal feature는 순수 row-wise 산술이라 fold와 무관하게 한 번만
#   계산해도 leakage가 없다(다른 행/미래 정보를 참조하지 않음).
#
# 실행 방법 (저장소 루트에서):
#   python "EDA feature experiment/Combined_Feature_Experiment/xgb_v002_fe_v001_train.py"

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import betaln
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "xgb_v002_fe_v001_exact4_success_only"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

BASELINE_METADATA_PATH = VALIDATION_ROOT / "xgb_v002_fe_v001_exact4" / "metadata.json"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


# =========================================================
# 데이터 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"

# -----------------------------------------------------------------
# A. 유지 raw feature (34개) — cb_reg_v002_fe_v001_train.py와 동일
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
    # Pitch Mix (3, raw 유지 — BBEB 대상에서 제외)
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    # success_rate 외 reliability rate 5개 — 이번 실험에서는 BBEB 대상에서 빼고 raw 유지
    "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_batter_middle_rate",
]

# 원본 47개 raw feature 중 최종 모델 입력에서 제외한 13개
# 이번 실험은 asof_pitcher_middle/reverse/ball/strike_rate, asof_batter_middle_rate를
# raw로 되돌려서 RAW_KEPT_FEATURES에 다시 포함시켰으므로, 실제로 제외되는 건
# success_rate 2개(BBEB로 대체됨)와 그 외 raw47 curation에서 원래 뺐던 6개뿐이다.
RAW_FEATURES_REMOVED = [
    "run_total_before", "score_diff_home", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy",
    "asof_pitcher_success_rate", "asof_batter_success_rate",
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

# Exact-ASOF 4개 (요청된 4개만 — exact_pitcher_season_logn은 제외)
EXACT_COLS = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

# 저카디널리티 명목형만 categorical encoding (xgb_v002 정책 유지, base_state는
# fe_v001에 없으므로 빠지고 situational categorical 2개가 추가된다)
CAT_COLS = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "count_balance", "platoon",
]

# 고카디널리티 ID는 categorical encoding하지 않고 numeric 값으로 유지
# (xgb_v001/xgb_v002와 동일한 정책)
ID_LIKE_COLS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
]

EXPANDING_FOLDS = [
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {"fold": 4, "train_seasons": [2019, 2020, 2021, 2022, 2023], "validation_season": 2024},
]


# =========================================================
# XGBoost 학습 설정 (model/xgb_v002/train.py와 동일)
# =========================================================

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42

# Hidden future와 가장 가까운 2019~2023 -> 2024 Fold의 best iteration을
# Final model iteration으로 사용한다.
FINAL_ITERATION_STRATEGY = "latest_fold"

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 10.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# BBEB (Beta-Binomial Empirical Bayes) — cb_reg_v002_fe_v001_train.py와
# 완전히 동일한 로직을 그대로 재사용한다.
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
]

BBEB_FEATURES = [f"bbeb_{spec['name']}_rate" for spec in BBEB_FEATURE_SPECS]

FINAL_FEATURES = (
    RAW_KEPT_FEATURES + SITUATIONAL_FEATURES + BBEB_FEATURES + [GAP_FEATURE] + TEMPORAL_FEATURES
    + EXACT_COLS
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
# (cb_reg_v002_fe_v001_train.py와 동일)
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
# Exact-ASOF (Asof_Residual_base/train.py의 build_pitcher_season_stats /
# add_pitcher_exact_asof_features와 동일한 로직 — fold-safe: 그 fold의
# train 구간에서 "완료된 과거 시즌"만으로 만든 통계를 공식 누적치에서 빼서
# "이번 시즌 지금까지" 값을 복원한다. 다른 행(같은 시즌 미래 행 포함)은
# 절대 참조하지 않는다.
# =========================================================

SEASON_COL = "season"


def build_pitcher_season_stats(history):
    required = {"pitcher_id", SEASON_COL, TARGET}
    missing = sorted(required - set(history.columns))
    if missing:
        raise KeyError(f"pitcher season stats 필수 컬럼 없음: {missing}")

    stats = (
        history.groupby(["pitcher_id", SEASON_COL], sort=False, dropna=False)[TARGET]
        .agg(["size", "sum"]).reset_index().rename(columns={"size": "n", "sum": "success"})
    )
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)
    stats["n"] = stats["n"].astype(np.float64)
    stats["success"] = stats["success"].astype(np.float64)
    return stats


def add_pitcher_exact_asof_features(rows, pitcher_season_stats):
    required = {SEASON_COL, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise KeyError(f"Exact-ASOF 필수 컬럼 없음: {missing}")

    out = rows.copy()
    out["exact_pitcher_season_n"] = np.nan
    out["exact_pitcher_season_success_rate"] = np.nan
    out["exact_pitcher_past_success_rate"] = np.nan
    out["exact_pitcher_season_minus_past"] = np.nan

    row_season = pd.to_numeric(out[SEASON_COL], errors="raise").astype(int)
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(row_season.unique()):
        mask = row_season == int(target_year)
        part = out.loc[mask]
        hist = stats.loc[stats[SEASON_COL] < int(target_year)]

        if len(hist):
            hist_agg = hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]].sum()
            hist_n = part["pitcher_id"].map(hist_agg["n"]).fillna(0.0).to_numpy(dtype=np.float64)
            hist_success = part["pitcher_id"].map(hist_agg["success"]).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            hist_n = np.zeros(len(part), dtype=np.float64)
            hist_success = np.zeros(len(part), dtype=np.float64)

        asof_n = pd.to_numeric(part["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
        asof_rate = pd.to_numeric(part["asof_pitcher_success_rate"], errors="coerce").to_numpy(dtype=np.float64)
        asof_success = asof_n * asof_rate

        current_n = np.maximum(asof_n - hist_n, 0.0)
        current_success = asof_success - hist_success
        current_success = np.where(np.isfinite(current_success), np.clip(current_success, 0.0, current_n), np.nan)

        current_rate = np.full(len(part), np.nan, dtype=np.float64)
        valid_current = (current_n > 0.0) & np.isfinite(current_success)
        current_rate[valid_current] = current_success[valid_current] / current_n[valid_current]

        past_rate = np.full(len(part), np.nan, dtype=np.float64)
        valid_past = hist_n > 0.0
        past_rate[valid_past] = hist_success[valid_past] / hist_n[valid_past]

        idx = part.index
        out.loc[idx, "exact_pitcher_season_n"] = current_n
        out.loc[idx, "exact_pitcher_season_success_rate"] = current_rate
        out.loc[idx, "exact_pitcher_past_success_rate"] = past_rate
        out.loc[idx, "exact_pitcher_season_minus_past"] = current_rate - past_rate

    return out


# =========================================================
# Data QA (situational / temporal — fold와 무관한 부분)
# (cb_reg_v002_fe_v001_train.py와 동일)
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
# Brier Score (XGBoost eval_metric callable)
# =========================================================

def brier_metric(y_true, y_pred):
    """XGBoost validation용 Brier Score. 값이 작을수록 좋다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


# =========================================================
# Raw Feature 준비 (model/xgb_v002/train.py와 동일)
# =========================================================

def prepare_raw_features(df, features, id_like_cols):
    """
    XGBoost 전처리 직전 DataFrame을 생성한다.

    1. 학습 Feature 순서 고정
    2. ID 계열은 numeric으로 강제 변환
    3. 나머지 numeric NaN은 그대로 유지 -> XGBoost가 직접 처리
    """
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError("입력 데이터에 학습 Feature가 없습니다: " + ", ".join(missing_features))

    X = df[features].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def create_preprocessor(cat_cols, num_cols):
    """
    새로운 OrdinalEncoder를 포함하는 XGBoost 전처리 객체를 매번 생성한다.
    각 Expanding Fold와 Final model이 서로 독립된 fitted encoder를 갖도록 한다.

    categorical: unseen category -> -1, missing category -> -1
    numeric: 그대로 통과, NaN도 그대로 XGBoost에 전달
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )


def create_validation_model():
    """Brier Score 기반 early stopping을 사용하는 Validation용 XGBClassifier."""
    return XGBClassifier(
        n_estimators=MAX_ITERATIONS,
        eval_metric=brier_metric,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **MODEL_PARAMS,
    )


def create_final_model(n_estimators):
    """
    Validation에서 결정한 iteration만큼 전체 데이터로 다시 학습할 Final 모델.
    eval_metric callable/early stopping을 넣지 않아 저장된 model.pkl이
    이 파일의 함수 정의에 의존하지 않게 한다.
    """
    return XGBClassifier(n_estimators=n_estimators, **MODEL_PARAMS)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0.0:
        raise RuntimeError("Baseline Brier Score가 0 이하입니다.")

    bss_raw = float(100000.0 * (1.0 - brier / baseline_brier))
    bss = max(0.0, bss_raw)

    return {
        "positive_rate": positive_rate, "brier": brier,
        "baseline_brier": baseline_brier, "bss_raw": bss_raw, "bss": bss,
    }


def validate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_pred) != len(y_true):
        raise RuntimeError("Validation 예측 수와 정답 수가 다릅니다.")
    if not np.isfinite(y_pred).all():
        raise RuntimeError("Validation 예측에 NaN 또는 무한대가 있습니다.")
    if not ((y_pred >= 0.0) & (y_pred <= 1.0)).all():
        raise RuntimeError("Validation 예측이 0~1 범위를 벗어났습니다.")


# =========================================================
# Fold Validation 결과 저장
# =========================================================

def save_fold_validation(
    experiment_name, fold, train_seasons, validation_season, estimator, preprocessor,
    model_params, features, cat_cols, id_like_cols, num_cols, row_ids, y_true, y_pred,
    metrics, output_root, model_path, n_train, elapsed_seconds, best_iteration,
    final_iterations, bbeb_priors,
):
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert len(row_ids) == len(y_true) == len(y_pred)

    output_dir = Path(output_root) / experiment_name / f"fold_{validation_season}"
    output_dir.mkdir(parents=True, exist_ok=False)

    pred_df = pd.DataFrame({
        "row_id": row_ids, "target": y_true, "prediction": y_pred,
        "validation_season": validation_season,
    })
    pred_df.to_csv(output_dir / "pred.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        output_dir / "validation.npz", row_id=row_ids, target=y_true, prediction=y_pred,
        validation_season=np.full(len(y_true), validation_season),
    )

    metadata = {
        "experiment_name": experiment_name,
        "fold": int(fold),
        "train_seasons": [int(season) for season in train_seasons],
        "validation_season": int(validation_season),
        "n_train": int(n_train),
        "n_validation": int(len(y_true)),
        "model_type": type(estimator).__name__,
        "model_params": model_params,
        "max_iterations": MAX_ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(best_iteration),
        "final_iterations": int(final_iterations),
        "positive_rate": float(metrics["positive_rate"]),
        "brier": float(metrics["brier"]),
        "baseline_brier": float(metrics["baseline_brier"]),
        "bss_raw": float(metrics["bss_raw"]),
        "bss": float(metrics["bss"]),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "elapsed_seconds": float(elapsed_seconds),
        "model_path": str(model_path),
        "features": features,
        "categorical_features": cat_cols,
        "id_like_features": id_like_cols,
        "numeric_features": num_cols,
        "bbeb_priors": bbeb_priors,
        "xgboost_version": xgb.__version__,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Fold checkpoint는 연구/재현용이다. Final submission에서는 사용하지 않는다.
    fold_artifact = {
        "model": estimator,
        "preprocessor": preprocessor,
        "features": features,
        "cat_cols": cat_cols,
        "id_like_cols": id_like_cols,
        "num_cols": num_cols,
        "best_iteration": int(best_iteration),
        "final_iterations": int(final_iterations),
        "validation_brier": float(metrics["brier"]),
        "validation_score": float(metrics["bss"]),
        "validation_bss_raw": float(metrics["bss_raw"]),
        "validation_season": int(validation_season),
        "model_name": f"{experiment_name}_fold_{validation_season}",
        "xgboost_version": xgb.__version__,
    }

    joblib.dump(fold_artifact, model_path, compress=3)

    return pred_df, metadata


# =========================================================
# Expanding Window Validation (BBEB는 fold-safe로 fold 루프 안에서 적용)
# =========================================================

def run_expanding_validation(train, features, cat_cols, id_like_cols, num_cols):
    fold_results = []
    oof_frames = []

    for fold_config in EXPANDING_FOLDS:
        fold = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        validation_season = fold_config["validation_season"]

        train_mask = train["season"].isin(train_seasons)
        val_mask = train["season"] == validation_season

        if not train_mask.any():
            raise ValueError(f"Fold {fold}의 학습 데이터가 없습니다.")
        if not val_mask.any():
            raise ValueError(f"Fold {fold}의 Validation 데이터가 없습니다.")

        assert not (train_mask & val_mask).any(), f"Fold {fold}: train/val row overlap 발견"

        fold_train_raw = train.loc[train_mask].copy()
        fold_val_raw = train.loc[val_mask].copy()

        # -------------------------------------------------
        # Exact-ASOF: 그 fold의 train 구간에서 "완료된 과거 시즌" 통계만 사용
        # (fold-safe, 다른 행/미래 시즌 참조 없음)
        # -------------------------------------------------

        fold_pitcher_season_stats = build_pitcher_season_stats(fold_train_raw)
        fold_train_exact = add_pitcher_exact_asof_features(fold_train_raw, fold_pitcher_season_stats)
        fold_val_exact = add_pitcher_exact_asof_features(fold_val_raw, fold_pitcher_season_stats)

        # -------------------------------------------------
        # BBEB prior는 그 fold의 train 구간 전체(expanding)로만 fit한다.
        # -------------------------------------------------

        fold_bbeb_priors = fit_all_bbeb_priors(fold_train_exact)

        fold_train_df = apply_bbeb_features(fold_train_exact, fold_bbeb_priors)
        fold_val_df = apply_bbeb_features(fold_val_exact, fold_bbeb_priors)

        assert set(fold_train_df[ID]).isdisjoint(set(fold_val_df[ID])), (
            f"Fold {fold}: train/val row_id overlap 발견"
        )

        y_train = fold_train_df[TARGET].copy()
        y_val = fold_val_df[TARGET].copy()
        val_row_id = fold_val_df[ID].copy()

        # -------------------------------------------------
        # Fold별 Raw Feature + Fold별 독립 Encoder
        # (미래 Validation 시즌의 범주 정보를 fit에 사용하지 않는다)
        # -------------------------------------------------

        X_train_raw = prepare_raw_features(df=fold_train_df, features=features, id_like_cols=id_like_cols)
        X_val_raw = prepare_raw_features(df=fold_val_df, features=features, id_like_cols=id_like_cols)

        validation_preprocessor = create_preprocessor(cat_cols=cat_cols, num_cols=num_cols)
        X_train = validation_preprocessor.fit_transform(X_train_raw)
        X_val = validation_preprocessor.transform(X_val_raw)

        print()
        print("=" * 70)
        print(f"Fold {fold}: {train_seasons[0]}~{train_seasons[-1]} -> {validation_season}")
        print("=" * 70)
        print("학습 데이터 shape     :", X_train.shape)
        print("검증 데이터 shape     :", X_val.shape)
        print(f"학습 Target 평균      : {y_train.mean():.6f}")
        print(f"검증 Target 평균      : {y_val.mean():.6f}")

        print()
        print("[BBEB prior — fold train 구간 전체(expanding)에서만 fit]")
        for spec in BBEB_FEATURE_SPECS:
            prior = fold_bbeb_priors[spec["name"]]
            print(
                f"  - {spec['name']:<18s} prior_mean={prior['prior_mean']:.4f}  "
                f"prior_strength={prior['prior_strength']:.1f}  n_groups={prior['n_groups']:,}"
            )

        # -------------------------------------------------
        # Fold Validation 모델
        # -------------------------------------------------

        validation_model = create_validation_model()

        validation_start = time.time()
        validation_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
        validation_elapsed = time.time() - validation_start

        best_iteration = getattr(validation_model, "best_iteration", None)
        if best_iteration is None or best_iteration < 0:
            best_iteration = MAX_ITERATIONS - 1

        # XGBoost best_iteration은 0-based
        final_iterations = best_iteration + 1

        val_pred = validation_model.predict_proba(X_val)[:, 1]

        validate_predictions(y_true=y_val, y_pred=val_pred)
        metrics = calculate_metrics(y_true=y_val, y_pred=val_pred)

        fold_model_dir = MODEL_DIR / f"fold_{validation_season}"
        fold_model_dir.mkdir(parents=True, exist_ok=False)
        fold_model_path = fold_model_dir / f"{EXPERIMENT_NAME}_fold_{validation_season}.pkl"

        fold_pred_df, fold_metadata = save_fold_validation(
            experiment_name=EXPERIMENT_NAME, fold=fold, train_seasons=train_seasons,
            validation_season=validation_season, estimator=validation_model,
            preprocessor=validation_preprocessor, model_params=MODEL_PARAMS, features=features,
            cat_cols=cat_cols, id_like_cols=id_like_cols, num_cols=num_cols, row_ids=val_row_id,
            y_true=y_val, y_pred=val_pred, metrics=metrics, output_root=VALIDATION_ROOT,
            model_path=fold_model_path, n_train=len(fold_train_df), elapsed_seconds=validation_elapsed,
            best_iteration=best_iteration, final_iterations=final_iterations,
            bbeb_priors=fold_bbeb_priors,
        )

        fold_results.append(fold_metadata)
        oof_frames.append(fold_pred_df)

        print()
        print(f"Brier Score          : {metrics['brier']:.10f}")
        print(f"Baseline Brier       : {metrics['baseline_brier']:.10f}")
        print(f"BSS Raw              : {metrics['bss_raw']:.4f}")
        print(f"Validation Score     : {metrics['bss']:.4f}")
        print(f"Prediction mean      : {val_pred.mean():.6f}")
        print(f"Prediction std       : {val_pred.std():.6f}")
        print(f"Prediction min       : {val_pred.min():.6f}")
        print(f"Prediction max       : {val_pred.max():.6f}")
        print(f"Best iteration       : {best_iteration}")
        print(f"Final iterations     : {final_iterations}")
        print(f"학습 시간            : {validation_elapsed:.1f}초")
        print(f"Fold model           : {fold_model_path}")

    oof = pd.concat(oof_frames, ignore_index=True)
    return fold_results, oof


# =========================================================
# Validation Summary 저장
# =========================================================

def save_validation_summary(features, cat_cols, id_like_cols, num_cols, fold_results, oof):
    briers = np.asarray([fold["brier"] for fold in fold_results], dtype=float)
    bss_values = np.asarray([fold["bss"] for fold in fold_results], dtype=float)
    bss_raw_values = np.asarray([fold["bss_raw"] for fold in fold_results], dtype=float)
    fold_sizes = np.asarray([fold["n_validation"] for fold in fold_results], dtype=float)

    recent_briers = [fold["brier"] for fold in fold_results if fold["validation_season"] in (2023, 2024)]

    oof_metrics = calculate_metrics(y_true=oof["target"], y_pred=oof["prediction"])

    summary = {
        "n_oof": int(len(oof)),
        "mean_fold_brier": float(np.mean(briers)),
        "std_fold_brier": float(np.std(briers)),
        "weighted_fold_brier": float(np.average(briers, weights=fold_sizes)),
        "best_fold_brier": float(np.min(briers)),
        "worst_fold_brier": float(np.max(briers)),
        "fold_brier_range": float(np.max(briers) - np.min(briers)),
        "recent_2023_2024_mean_brier": float(np.mean(recent_briers)),
        "mean_fold_bss_raw": float(np.mean(bss_raw_values)),
        "weighted_fold_bss_raw": float(np.average(bss_raw_values, weights=fold_sizes)),
        "mean_fold_bss": float(np.mean(bss_values)),
        "weighted_fold_bss": float(np.average(bss_values, weights=fold_sizes)),
        "pooled_oof_brier": float(oof_metrics["brier"]),
        "pooled_oof_baseline_brier": float(oof_metrics["baseline_brier"]),
        "pooled_oof_bss_raw": float(oof_metrics["bss_raw"]),
        "pooled_oof_bss": float(oof_metrics["bss"]),
    }

    oof.to_csv(VALIDATION_DIR / "oof_pred.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        VALIDATION_DIR / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(), target=oof["target"].to_numpy(),
        prediction=oof["prediction"].to_numpy(), validation_season=oof["validation_season"].to_numpy(),
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "baseline_experiment": "xgb_v002",
        "validation_method": "expanding_window",
        "model_params": MODEL_PARAMS,
        "max_iterations": MAX_ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "features": features,
        "categorical_features": cat_cols,
        "id_like_features": id_like_cols,
        "numeric_features": num_cols,
        "bbeb_feature_specs": BBEB_FEATURE_SPECS,
        "folds": fold_results,
        "summary": summary,
        "xgboost_version": xgb.__version__,
    }

    with open(VALIDATION_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return summary


def select_final_iterations(fold_results):
    if FINAL_ITERATION_STRATEGY == "latest_fold":
        latest_fold = max(fold_results, key=lambda fold: fold["validation_season"])
        return int(latest_fold["final_iterations"])
    raise ValueError(f"지원하지 않는 FINAL_ITERATION_STRATEGY입니다: {FINAL_ITERATION_STRATEGY}")


def save_final_metadata(
    train, features, cat_cols, id_like_cols, num_cols, final_params, final_iterations,
    elapsed_seconds, final_bbeb_priors, latest_fold,
):
    priors_path = MODEL_DIR / "bbeb_priors.json"
    with open(priors_path, "w", encoding="utf-8") as f:
        json.dump(final_bbeb_priors, f, indent=2, ensure_ascii=False)

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "baseline_experiment": "xgb_v002",
        "model_type": "XGBClassifier",
        "training_seasons": sorted(int(season) for season in train["season"].unique()),
        "n_train": int(len(train)),
        "features": features,
        "categorical_features": cat_cols,
        "id_like_features": id_like_cols,
        "numeric_features": num_cols,
        "bbeb_feature_specs": BBEB_FEATURE_SPECS,
        "bbeb_priors_path": str(priors_path),
        "model_params": final_params,
        "final_iterations": int(final_iterations),
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "reference_validation_season": int(latest_fold["validation_season"]),
        "reference_validation_brier": float(latest_fold["brier"]),
        "reference_validation_bss": float(latest_fold["bss"]),
        "elapsed_seconds": float(elapsed_seconds),
        "model_path": str(MODEL_PATH),
        "xgboost_version": xgb.__version__,
    }

    with open(MODEL_DIR / "final_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_feature_manifest(cat_cols, id_like_cols, num_cols):
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "raw_features_used": RAW_KEPT_FEATURES,
        "raw_features_removed": RAW_FEATURES_REMOVED,
        "bbeb_features": BBEB_FEATURES,
        "situation_features": SITUATIONAL_FEATURES + [GAP_FEATURE],
        "temporal_features": TEMPORAL_FEATURES,
        "exact_features": EXACT_COLS,
        "categorical_features": cat_cols,
        "id_like_features": id_like_cols,
        "numeric_features": num_cols,
        "total_feature_count": len(FINAL_FEATURES),
        "validation_method": "expanding_window",
        "encoding_note": (
            "xgb_v002의 기존 인코딩 정책(저카디널리티 categorical만 "
            "OrdinalEncoder, 고카디널리티 ID는 numeric passthrough)을 그대로 "
            "유지했다 — CatBoost fe_v001처럼 ID를 native categorical로 다루지 "
            "않는다. feature set 변경의 효과만 순수하게 비교하기 위함."
        ),
    }
    with open(MODEL_DIR / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataframe_to_portable(df):
    """pandas 버전 간 pickle 호환성 문제(StringDtype 내부 포맷 변경 등) 재발
    방지 — cb_reg_v019_asof_residual_platoon_only 제출 때 겪은 실제 사고
    (TypeError: StringDtype.__init__()) 이후 확립된 패턴."""
    return {
        "columns": list(df.columns),
        "data": {col: [_to_native(v) for v in df[col].tolist()] for col in df.columns},
    }


def build_test_features(test, final_bbeb_priors, final_pitcher_season_stats):
    """test.csv에 동일한 상황/Temporal/Exact-ASOF/BBEB 파생 로직 + 얼린 prior/통계를 적용한다."""
    test = add_situational_features(test)
    test = add_temporal_features(test)
    test = add_pitcher_exact_asof_features(test, final_pitcher_season_stats)
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

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"train.csv를 찾을 수 없습니다: {TRAIN_PATH}")

    assert len(FINAL_FEATURES) == len(set(FINAL_FEATURES)), "FINAL_FEATURES에 중복이 있습니다."
    assert set(CAT_COLS) <= set(FINAL_FEATURES), "CAT_COLS가 FINAL_FEATURES에 전부 포함되어야 합니다."
    assert set(ID_LIKE_COLS) <= set(FINAL_FEATURES), "ID_LIKE_COLS가 FINAL_FEATURES에 전부 포함되어야 합니다."

    cat_cols = [col for col in CAT_COLS if col in FINAL_FEATURES]
    id_like_cols = [col for col in ID_LIKE_COLS if col in FINAL_FEATURES]
    num_cols = [col for col in FINAL_FEATURES if col not in cat_cols]

    needed_cols = (
        set(RAW_KEPT_FEATURES)
        | {ID, TARGET, "season"}
        | {"home_win_expectancy", "away_win_expectancy"}
        | {spec["rate_col"] for spec in BBEB_FEATURE_SPECS}
    )

    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=sorted(needed_cols))

    print("=" * 70)
    print("XGBoost xgb_v002_fe_v001 데이터 로드 완료")
    print("=" * 70)
    print("Experiment           :", EXPERIMENT_NAME)
    print("XGBoost version      :", xgb.__version__)
    print("train shape          :", raw_train.shape)
    print("최종 Feature 수      :", len(FINAL_FEATURES))
    print("Categorical 수       :", len(cat_cols))
    print("Numeric 수           :", len(num_cols))
    print("Categorical Feature  :", cat_cols)
    print("Numeric ID Feature   :", id_like_cols)
    print("시즌 범위            :", raw_train["season"].min(), "~", raw_train["season"].max())
    print(f"전체 제구 성공률      : {raw_train[TARGET].mean():.6f}")
    print("Early stopping metric: Brier Score")
    print("Device               : cpu")
    print("=" * 70)

    if raw_train[TARGET].isna().any():
        raise ValueError(f"타깃 컬럼 '{TARGET}'에 결측값이 있습니다.")

    # -----------------------------------------------------
    # denominator audit (BBEB 대상 7개 (n, rate) 쌍)
    # -----------------------------------------------------

    print()
    print("[denominator audit] n * rate가 정수에 가까운지 확인")
    for spec in BBEB_FEATURE_SPECS:
        audit = audit_denominator(raw_train, spec["n_col"], spec["rate_col"])
        print(
            f"  - {spec['name']:<18s} valid={audit['n_valid_rows']:>9,}/{audit['n_total_rows']:>9,}  "
            f"max|n*p-round|={audit['max_integer_distance']:.3e}"
        )
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

    with open(VALIDATION_DIR / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------
    # Expanding Window Validation (BBEB는 fold-safe로 fold 루프 안에서 적용)
    # -----------------------------------------------------

    fold_results, oof = run_expanding_validation(
        train=train, features=FINAL_FEATURES, cat_cols=cat_cols,
        id_like_cols=id_like_cols, num_cols=num_cols,
    )
    summary = save_validation_summary(
        features=FINAL_FEATURES, cat_cols=cat_cols, id_like_cols=id_like_cols,
        num_cols=num_cols, fold_results=fold_results, oof=oof,
    )

    print()
    print("=" * 70)
    print("XGBoost Expanding Window 결과 요약")
    print("=" * 70)
    for fold in fold_results:
        print(
            f"Valid {fold['validation_season']}: Brier={fold['brier']:.10f}, "
            f"BSS={fold['bss']:.4f}, iter={fold['final_iterations']}"
        )
    print(f"Fold 단순 평균 Brier : {summary['mean_fold_brier']:.10f}")
    print(f"Fold 가중 평균 Brier : {summary['weighted_fold_brier']:.10f}")
    print(f"Fold Brier 표준편차  : {summary['std_fold_brier']:.10f}")
    print(f"2023~2024 평균 Brier : {summary['recent_2023_2024_mean_brier']:.10f}")
    print(f"OOF pooled Brier     : {summary['pooled_oof_brier']:.10f}")
    print(f"OOF pooled BSS       : {summary['pooled_oof_bss']:.4f}")
    print("=" * 70)

    # -----------------------------------------------------
    # Final iteration 선택 + 전체 데이터 재학습
    # -----------------------------------------------------

    final_iterations = select_final_iterations(fold_results)
    latest_fold = max(fold_results, key=lambda fold: fold["validation_season"])

    print()
    print(f"Final iteration strategy : {FINAL_ITERATION_STRATEGY}")
    print(f"기준 Validation          : {latest_fold['validation_season']}")
    print(f"Final iterations         : {final_iterations}")

    final_pitcher_season_stats = build_pitcher_season_stats(train)
    train_with_exact = add_pitcher_exact_asof_features(train, final_pitcher_season_stats)

    final_bbeb_priors = fit_all_bbeb_priors(train_with_exact)
    train_with_bbeb = apply_bbeb_features(train_with_exact, final_bbeb_priors)

    X_full_raw = prepare_raw_features(df=train_with_bbeb, features=FINAL_FEATURES, id_like_cols=id_like_cols)
    y_full = train_with_bbeb[TARGET].copy()

    final_preprocessor = create_preprocessor(cat_cols=cat_cols, num_cols=num_cols)
    X_full = final_preprocessor.fit_transform(X_full_raw)

    final_model = create_final_model(n_estimators=final_iterations)

    print()
    print("=" * 70)
    print("XGBoost 전체 데이터 재학습 시작")
    print(f"사용 iteration       : {final_iterations}")
    print(f"학습 시즌            : {int(train['season'].min())}~{int(train['season'].max())}")
    print("Device               : CPU")
    print("=" * 70)

    print()
    print("[BBEB prior — 전체 train에서 fit, test에 얼려서 적용]")
    for spec in BBEB_FEATURE_SPECS:
        prior = final_bbeb_priors[spec["name"]]
        print(
            f"  - {spec['name']:<18s} prior_mean={prior['prior_mean']:.4f}  "
            f"prior_strength={prior['prior_strength']:.1f}  n_groups={prior['n_groups']:,}"
        )

    final_start = time.time()
    final_model.fit(X_full, y_full, verbose=False)
    final_elapsed = time.time() - final_start

    print(f"전체 데이터 재학습 완료: {final_elapsed:.1f}초")

    # -----------------------------------------------------
    # 최종 Model artifact — script/package 호환을 위해 xgb_v002와 동일한
    # 핵심 key를 유지하고, BBEB 관련 정보를 추가한다(self-contained).
    # -----------------------------------------------------

    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,
        "features": FINAL_FEATURES,
        "cat_cols": cat_cols,
        "id_like_cols": id_like_cols,
        "num_cols": num_cols,
        "best_iteration": int(latest_fold["best_iteration"]),
        "final_iterations": int(final_iterations),
        "validation_brier": float(latest_fold["brier"]),
        "validation_score": float(latest_fold["bss"]),
        "model_name": EXPERIMENT_NAME,
        "xgboost_version": xgb.__version__,
        "validation_method": "expanding_window",
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "expanding_summary": summary,
        "bbeb_feature_specs": BBEB_FEATURE_SPECS,
        "bbeb_priors": final_bbeb_priors,
        "exact_cols": EXACT_COLS,
        "pitcher_season_stats": dataframe_to_portable(final_pitcher_season_stats),
        "artifact_format": "portable_v1",
    }

    joblib.dump(artifact, MODEL_PATH, compress=3)

    save_final_metadata(
        train=train, features=FINAL_FEATURES, cat_cols=cat_cols, id_like_cols=id_like_cols,
        num_cols=num_cols, final_params=MODEL_PARAMS, final_iterations=final_iterations,
        elapsed_seconds=final_elapsed, final_bbeb_priors=final_bbeb_priors, latest_fold=latest_fold,
    )
    manifest = save_feature_manifest(cat_cols=cat_cols, id_like_cols=id_like_cols, num_cols=num_cols)

    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, MODEL_DIR / "train.py")

    # -----------------------------------------------------
    # test.csv에도 동일 파생 로직 + 얼린 prior 적용 (제출용 feature 생성까지)
    # -----------------------------------------------------

    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
        test_with_features = build_test_features(test, final_bbeb_priors, final_pitcher_season_stats)
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
    print("XGBoost xgb_v002_fe_v001 모델 저장 완료")
    print(f"- {MODEL_PATH}")
    print(f"- {MODEL_DIR / 'final_metadata.json'}")
    print(f"- {MODEL_DIR / 'feature_manifest.json'}")
    print(f"- {MODEL_DIR / 'bbeb_priors.json'}")
    print(f"- {VALIDATION_DIR / 'qa_report.json'}")
    print("=" * 70)

    # -----------------------------------------------------
    # Baseline(xgb_v002, raw47) 비교 — 재학습 없이 기존 결과 재사용
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Baseline(xgb_v002, raw47) 대비 비교 (재학습 없이 기존 결과 재사용)")
    print("=" * 70)
    if BASELINE_METADATA_PATH.exists():
        with open(BASELINE_METADATA_PATH, "r", encoding="utf-8") as f:
            baseline_metadata = json.load(f)

        # 2024 fold(가장 최근 validation season) 기준 비교 — 이 프로젝트의
        # 다른 실험(cb_reg_v002_fe_v001 등)과 동일하게 fold_2024 brier/BSS를
        # 1차 비교 지표로 사용한다. pooled OOF는 참고용으로 함께 출력한다.
        baseline_fold_2024 = next(
            (fold for fold in baseline_metadata["folds"] if fold["validation_season"] == 2024), None,
        )
        new_fold_2024 = next(
            (fold for fold in fold_results if fold["validation_season"] == 2024), None,
        )

        if baseline_fold_2024 is not None and new_fold_2024 is not None:
            baseline_brier_2024 = baseline_fold_2024["brier"]
            new_brier_2024 = new_fold_2024["brier"]
            diff_2024 = new_brier_2024 - baseline_brier_2024
            print(f"[2024 fold 기준]")
            print(f"baseline (xgb_v002)         fold_2024 brier = {baseline_brier_2024:.10f}  bss = {baseline_fold_2024['bss']:.4f}")
            print(f"new ({EXPERIMENT_NAME}) fold_2024 brier = {new_brier_2024:.10f}  bss = {new_fold_2024['bss']:.4f}")
            print(f"diff (new - baseline)                          = {diff_2024:+.10f} ({'개선' if diff_2024 < 0 else '악화'})")
        else:
            print("[경고] fold_2024 결과를 찾을 수 없어 2024 fold 비교를 건너뜁니다.")

        print()
        baseline_brier = baseline_metadata["summary"]["pooled_oof_brier"]
        new_brier = summary["pooled_oof_brier"]
        diff = new_brier - baseline_brier
        print(f"[참고: pooled OOF 기준]")
        print(f"baseline (xgb_v002)         pooled_oof_brier = {baseline_brier:.10f}")
        print(f"new ({EXPERIMENT_NAME}) pooled_oof_brier = {new_brier:.10f}")
        print(f"diff (new - baseline)                          = {diff:+.10f} ({'개선' if diff < 0 else '악화'})")
    else:
        print(f"[경고] baseline metadata를 찾을 수 없습니다: {BASELINE_METADATA_PATH}")

    print()
    print(f"total_feature_count = {manifest['total_feature_count']}")


if __name__ == "__main__":
    main()
