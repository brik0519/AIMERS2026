# script.py — cb_reg_shilft_v001 추론 전용 (standalone, train.py import 없음)
#
# 평가 데이터 독립 추론:
#   - 각 test row는 해당 row의 입력값 + train에서 freeze한 artifact
#     (모델, pitcher_season_stats, gseason bank, HB-EB prior)만 사용한다.
#   - 다른 test 행/전체 분포 집계는 어디에도 없다. unique season 순회는
#     조회용일 뿐 다른 row 값을 집계하지 않는다.
#   - 조건 3: 제출 전 selfcheck — 단일행 입력 vs 전체 입력의 예측 동일성을
#     기계적으로 검증하고 불일치 시 즉시 실패한다.

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPERIMENT_NAME = "cb_reg_shilft_v001"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

MODEL_ROOT = Path("./model")
MODEL_PATH = MODEL_ROOT / f"{EXPERIMENT_NAME}.pkl"
MODEL_PATH_NESTED = MODEL_ROOT / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

EXACT_COLS = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

P_COMPONENT_RATES = [
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
B_COMPONENT_RATES = ["asof_batter_middle_rate"]

SELFCHECK_N_SINGLE = 5
SELFCHECK_ATOL = 1e-9


# ============================================================
# 공통 유틸 (train과 동일 로직, 코드 사본)
# ============================================================

def prepare_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def _n(frame: pd.DataFrame, c: str, default=np.nan) -> np.ndarray:
    if c in frame:
        return pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    return np.full(len(frame), default, np.float64)


def _rounded_cumulative_count(n, rate) -> np.ndarray:
    n = np.asarray(n, np.float64)
    rate = np.asarray(rate, np.float64)
    out = np.zeros(len(n), np.float64)
    ok = (n > 0) & np.isfinite(rate)
    out[ok] = np.rint(n[ok] * rate[ok])
    return out


# ============================================================
# Exact-ASOF (frozen pitcher_season_stats 사용, row 독립)
# ============================================================

def add_pitcher_exact_asof_features_test(
    test: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    required = {SEASON_COL, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise KeyError(f"Exact-ASOF 필수 test 컬럼 없음: {missing}")

    out = test.copy()
    for col in EXACT_COLS:
        out[col] = np.nan

    test_season = pd.to_numeric(out[SEASON_COL], errors="raise").astype(int)
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(test_season.unique()):
        mask = test_season == int(target_year)
        part = out.loc[mask]
        hist = stats.loc[stats[SEASON_COL] < int(target_year)]

        if len(hist):
            hist_agg = (
                hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]]
                .sum()
            )
            hist_n = (
                part["pitcher_id"].map(hist_agg["n"]).fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            hist_success = (
                part["pitcher_id"].map(hist_agg["success"]).fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            hist_n = np.zeros(len(part), dtype=np.float64)
            hist_success = np.zeros(len(part), dtype=np.float64)

        asof_n = (
            pd.to_numeric(part["asof_pitcher_n"], errors="coerce")
            .fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
        )
        asof_rate = (
            pd.to_numeric(part["asof_pitcher_success_rate"], errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        asof_success = asof_n * asof_rate

        current_n = np.maximum(asof_n - hist_n, 0.0)
        current_success = asof_success - hist_success
        current_success = np.where(
            np.isfinite(current_success),
            np.clip(current_success, 0.0, current_n),
            np.nan,
        )

        current_rate = np.full(len(part), np.nan, dtype=np.float64)
        valid_current = (current_n > 0.0) & np.isfinite(current_success)
        current_rate[valid_current] = (
            current_success[valid_current] / current_n[valid_current]
        )

        past_rate = np.full(len(part), np.nan, dtype=np.float64)
        valid_past = hist_n > 0.0
        past_rate[valid_past] = hist_success[valid_past] / hist_n[valid_past]

        idx = part.index
        out.loc[idx, "exact_pitcher_season_n"] = current_n
        out.loc[idx, "exact_pitcher_season_logn"] = np.log1p(current_n)
        out.loc[idx, "exact_pitcher_season_success_rate"] = current_rate
        out.loc[idx, "exact_pitcher_past_success_rate"] = past_rate
        out.loc[idx, "exact_pitcher_season_minus_past"] = current_rate - past_rate

    return out


# ============================================================
# shift 블록 ①: self_state (완전 row-local)
# ============================================================

def build_self_state(frame: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=frame.index)
    pn = np.maximum(np.nan_to_num(_n(frame, "asof_pitcher_n", 0), nan=0.0), 0)
    bn = np.maximum(np.nan_to_num(_n(frame, "asof_batter_n", 0), nan=0.0), 0)
    career = _n(frame, "asof_pitcher_success_rate", 0.5)
    career = np.where(np.isfinite(career), career, 0.5)
    p1 = _n(frame, "asof_pitcher_prev1_game_success_rate")
    p3 = _n(frame, "asof_pitcher_prev3_game_success_rate")
    p5 = _n(frame, "asof_pitcher_prev5_game_success_rate")
    p1 = np.where(np.isfinite(p1), p1, career)
    p3 = np.where(np.isfinite(p3), p3, p1)
    p5 = np.where(np.isfinite(p5), p5, p3)
    bsucc = _n(frame, "asof_batter_success_rate", 0.5)
    bsucc = np.where(np.isfinite(bsucc), bsucc, 0.5)
    middle = _n(frame, "asof_pitcher_middle_rate", 0)
    reverse = _n(frame, "asof_pitcher_reverse_rate", 0)
    ball = _n(frame, "asof_pitcher_ball_rate", 0)
    strike = _n(frame, "asof_pitcher_strike_rate", 0)
    fast = _n(frame, "asof_pitcher_fastball_rate", 1 / 3)
    brk = _n(frame, "asof_pitcher_breaking_rate", 1 / 3)
    off = _n(frame, "asof_pitcher_offspeed_rate", 1 / 3)
    fast = np.where(np.isfinite(fast), fast, 1 / 3)
    brk = np.where(np.isfinite(brk), brk, 1 / 3)
    off = np.where(np.isfinite(off), off, 1 / 3)
    sm = np.maximum(fast + brk + off, 1e-6)
    fast /= sm
    brk /= sm
    off /= sm
    balls = np.nan_to_num(_n(frame, "balls_before", 0))
    strikes = np.nan_to_num(_n(frame, "strikes_before", 0))
    inning = np.nan_to_num(_n(frame, "inning", 1))
    score = np.nan_to_num(_n(frame, "score_diff_pitcher_team", 0))
    li = np.maximum(np.nan_to_num(_n(frame, "li", 0)), 0)
    top = frame["top_bottom"].astype(str).to_numpy()
    hwe = np.nan_to_num(_n(frame, "home_win_expectancy", 50), nan=50) / 100.0
    awe = np.nan_to_num(_n(frame, "away_win_expectancy", 50), nan=50) / 100.0
    pwe = np.where(top == "T", hwe, awe)
    ph = frame["pitcher_hand"].astype(str).to_numpy()
    bh = frame["batter_hand"].astype(str).to_numpy()

    z["self__log_pitcher_n"] = np.log1p(pn).astype(np.float32)
    z["self__pitcher_reliability"] = (pn / (pn + 120)).astype(np.float32)
    z["self__career_success"] = career.astype(np.float32)
    z["self__prev1_success"] = p1.astype(np.float32)
    z["self__prev3_success"] = p3.astype(np.float32)
    z["self__prev5_success"] = p5.astype(np.float32)
    z["self__recent_blend"] = (0.50 * p1 + 0.30 * p3 + 0.20 * p5).astype(np.float32)
    z["self__prev1_minus_prev5"] = (p1 - p5).astype(np.float32)
    z["self__recent_minus_career"] = (
        0.5 * p1 + 0.3 * p3 + 0.2 * p5 - career
    ).astype(np.float32)

    middle_safe = np.nan_to_num(middle, nan=0.0)
    m1 = _n(frame, "asof_pitcher_prev1_game_middle_rate")
    m3 = _n(frame, "asof_pitcher_prev3_game_middle_rate")
    m5 = _n(frame, "asof_pitcher_prev5_game_middle_rate")
    m1 = np.where(np.isfinite(m1), m1, middle_safe)
    m3 = np.where(np.isfinite(m3), m3, m1)
    m5 = np.where(np.isfinite(m5), m5, m3)
    z["self__recent_middle_blend"] = (
        0.50 * m1 + 0.30 * m3 + 0.20 * m5
    ).astype(np.float32)
    z["self__recent_middle_delta"] = (m1 - m5).astype(np.float32)

    z["self__log_batter_n"] = np.log1p(bn).astype(np.float32)
    z["self__batter_reliability"] = (bn / (bn + 120)).astype(np.float32)
    z["self__batter_success"] = bsucc.astype(np.float32)
    z["self__pitcher_minus_batter"] = (career - bsucc).astype(np.float32)

    z["self__middle"] = np.nan_to_num(middle, nan=0).astype(np.float32)
    z["self__reverse"] = np.nan_to_num(reverse, nan=0).astype(np.float32)
    z["self__strike_ball_margin"] = np.nan_to_num(strike - ball, nan=0).astype(np.float32)

    entropy = -(
        fast * np.log(np.clip(fast, 1e-6, 1))
        + brk * np.log(np.clip(brk, 1e-6, 1))
        + off * np.log(np.clip(off, 1e-6, 1))
    )
    z["self__pitchmix_entropy"] = entropy.astype(np.float32)
    z["self__offspeed"] = off.astype(np.float32)
    z["self__fast_off_logratio"] = np.log(
        np.clip(fast, 1e-6, 1) / np.clip(off, 1e-6, 1)
    ).astype(np.float32)
    z["self__break_off_logratio"] = np.log(
        np.clip(brk, 1e-6, 1) / np.clip(off, 1e-6, 1)
    ).astype(np.float32)

    z["self__count_pressure"] = (
        0.7 * (balls == 3)
        + 1.1 * ((balls == 3) & (strikes == 2))
        + 0.25 * (strikes == 2)
    ).astype(np.float32)
    z["self__full_count"] = ((balls == 3) & (strikes == 2)).astype(np.float32)
    z["self__inning_f"] = inning.astype(np.float32)
    z["self__late"] = (inning >= 7).astype(np.float32)
    z["self__score_abs"] = np.abs(score).astype(np.float32)
    z["self__li_log1p"] = np.log1p(li).astype(np.float32)
    z["self__pitcher_we"] = np.clip(pwe, 0, 1).astype(np.float32)
    z["self__win_uncertainty"] = (
        1 - 2 * np.abs(np.clip(pwe, 0, 1) - 0.5)
    ).astype(np.float32)
    z["self__same_hand"] = (ph == bh).astype(np.float32)

    pressure = z["self__count_pressure"].to_numpy(np.float64)
    z["self__career_x_pressure"] = (career * pressure).astype(np.float32)
    z["self__recent_x_pressure"] = (
        (0.5 * p1 + 0.3 * p3 + 0.2 * p5) * pressure
    ).astype(np.float32)
    z["self__offspeed_x_same_hand"] = (off * (ph == bh)).astype(np.float32)
    return z


# ============================================================
# shift 블록 ②: eng_state (완전 row-local)
# ============================================================

def build_eng_state(frame: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=frame.index)
    p1 = pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"], errors="coerce")
    p3 = pd.to_numeric(frame["asof_pitcher_prev3_game_success_rate"], errors="coerce")
    p5 = pd.to_numeric(frame["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    career = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
    z["eng_recent35"] = (0.5 * p1 + 0.3 * p3 + 0.2 * p5).astype(np.float32)
    z["eng_recent_delta"] = (z["eng_recent35"] - career).astype(np.float32)
    z["eng_sb"] = (
        pd.to_numeric(frame["asof_pitcher_strike_rate"], errors="coerce")
        - pd.to_numeric(frame["asof_pitcher_ball_rate"], errors="coerce")
    ).astype(np.float32)
    z["eng_same"] = (
        frame["pitcher_hand"].astype(str) == frame["batter_hand"].astype(str)
    ).astype(np.float32)
    z["eng_close"] = (
        pd.to_numeric(frame["score_diff_pitcher_team"], errors="coerce").abs() <= 1
    ).astype(np.float32)
    z["eng_late"] = (
        pd.to_numeric(frame["inning"], errors="coerce") >= 7
    ).astype(np.float32)
    return z


# ============================================================
# shift 블록 ③④: gseason (row-local + frozen bank lookup)
# ============================================================

def transform_global_season(bank: dict, rows: pd.DataFrame) -> pd.DataFrame:
    r = rows
    z = pd.DataFrame(index=r.index)
    prior = float(bank["prior"])

    def one(ent, ncol, rcol, prefix, kn=80.0):
        ids = r[ent]
        hn = ids.map(bank[prefix + "_n"]).fillna(0).to_numpy(np.float64)
        hs = ids.map(bank[prefix + "_s"]).fillna(0).to_numpy(np.float64)
        an = np.maximum(np.nan_to_num(_n(r, ncol, 0), nan=0), 0)
        ar = _n(r, rcol, prior)
        ar = np.where(np.isfinite(ar), ar, prior)
        cumulative_success = _rounded_cumulative_count(an, ar)
        sn = np.maximum(an - hn, 0.0)
        ss = np.clip(cumulative_success - hs, 0.0, sn)
        raw = np.divide(
            ss, sn, out=np.full(len(r), prior, np.float64), where=sn > 0
        )
        for k in [40.0, 80.0, 160.0]:
            z[f"gseason__{prefix}_rate_k{int(k)}"] = (
                (ss + k * prior) / (sn + k)
            ).astype(np.float32)
        z[f"gseason__{prefix}_raw"] = raw.astype(np.float32)
        z[f"gseason__{prefix}_log_n"] = np.log1p(sn).astype(np.float32)
        z[f"gseason__{prefix}_reliability"] = (sn / (sn + kn)).astype(np.float32)
        z[f"gseason__{prefix}_minus_career"] = (raw - ar).astype(np.float32)

    one("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "pitcher", 80.0)
    one("batter_id", "asof_batter_n", "asof_batter_success_rate", "batter", 100.0)
    z["gseason__pitcher_minus_batter_k80"] = (
        z["gseason__pitcher_rate_k80"] - z["gseason__batter_rate_k80"]
    ).astype(np.float32)

    ids = r["pitcher_id"]
    an = np.maximum(np.nan_to_num(_n(r, "asof_pitcher_n", 0), nan=0), 0)
    base_n = ids.map(bank["pitcher_component_base_n"]).fillna(0).to_numpy(np.float64)
    win_n = np.maximum(an - base_n, 0.0)
    comp = {}
    for c in P_COMPONENT_RATES:
        pr = float(bank["component_priors"].get(c, 0.5))
        rr = _n(r, c, pr)
        rr = np.where(np.isfinite(rr), rr, pr)
        cum = _rounded_cumulative_count(an, rr)
        base = (
            ids.map(bank["pitcher_component_base"].get(c, {}))
            .fillna(0)
            .to_numpy(np.float64)
        )
        wc = np.clip(cum - base, 0.0, win_n)
        comp[c] = ((wc + 60.0 * pr) / (win_n + 60.0)).astype(np.float32)
    z["gseason__pitcher_component_log_n"] = np.log1p(win_n).astype(np.float32)
    z["gseason__pitcher_component_reliability"] = (
        win_n / (win_n + 60.0)
    ).astype(np.float32)
    z["gseason__pitcher_strike_rate"] = comp["asof_pitcher_strike_rate"]
    z["gseason__pitcher_ball_rate"] = comp["asof_pitcher_ball_rate"]
    z["gseason__pitcher_strike_ball_margin"] = (
        comp["asof_pitcher_strike_rate"] - comp["asof_pitcher_ball_rate"]
    ).astype(np.float32)
    z["gseason__pitcher_middle_rate"] = comp["asof_pitcher_middle_rate"]
    z["gseason__pitcher_reverse_rate"] = comp["asof_pitcher_reverse_rate"]
    fast = np.clip(comp["asof_pitcher_fastball_rate"], 1e-6, 1)
    brk = np.clip(comp["asof_pitcher_breaking_rate"], 1e-6, 1)
    off = np.clip(comp["asof_pitcher_offspeed_rate"], 1e-6, 1)
    sm = np.maximum(fast + brk + off, 1e-6)
    fast = fast / sm
    brk = brk / sm
    off = off / sm
    z["gseason__pitcher_fastball_rate"] = fast.astype(np.float32)
    z["gseason__pitcher_breaking_rate"] = brk.astype(np.float32)
    z["gseason__pitcher_offspeed_rate"] = off.astype(np.float32)
    z["gseason__pitcher_pitchmix_entropy"] = (
        -(fast * np.log(fast) + brk * np.log(brk) + off * np.log(off))
    ).astype(np.float32)

    bids = r["batter_id"]
    ban = np.maximum(np.nan_to_num(_n(r, "asof_batter_n", 0), nan=0), 0)
    bbase_n = (
        bids.map(bank["batter_component_base_n"]).fillna(0).to_numpy(np.float64)
    )
    bwin = np.maximum(ban - bbase_n, 0.0)
    c = "asof_batter_middle_rate"
    pr = float(bank["component_priors"].get(c, 0.5))
    rr = _n(r, c, pr)
    rr = np.where(np.isfinite(rr), rr, pr)
    cum = _rounded_cumulative_count(ban, rr)
    base = (
        bids.map(bank["batter_component_base"].get(c, {}))
        .fillna(0)
        .to_numpy(np.float64)
    )
    wc = np.clip(cum - base, 0.0, bwin)
    bmid = ((wc + 80.0 * pr) / (bwin + 80.0)).astype(np.float32)
    z["gseason__batter_middle_rate"] = bmid
    z["gseason__batter_middle_minus_pitcher_middle"] = (
        bmid - z["gseason__pitcher_middle_rate"]
    ).astype(np.float32)
    return z


# ============================================================
# HB-EB (frozen prior 적용, row 독립)
# ============================================================

def apply_hbeb(df, group_cols, n_col, rate_col, prior_df, global_prior):
    lookup = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(prior_df, on=group_cols, how="left", sort=False)
    )
    alpha = lookup["alpha"].fillna(global_prior[0]).to_numpy(dtype=np.float64)
    beta = lookup["beta"].fillna(global_prior[1]).to_numpy(dtype=np.float64)
    n = (
        pd.to_numeric(df[n_col], errors="coerce")
        .fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    )
    rate = (
        pd.to_numeric(df[rate_col], errors="coerce")
        .fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=np.float64)
    )
    success_count = n * rate
    denominator = n + alpha + beta
    return np.divide(
        success_count + alpha,
        denominator,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


# ============================================================
# Feature 파이프라인 (row 독립 — selfcheck로 기계 검증)
# ============================================================

def build_feature_frame(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    out = add_pitcher_exact_asof_features_test(
        test, art["pitcher_season_stats"]
    )

    z_self = build_self_state(out)
    z_eng = build_eng_state(out)
    z_gs = transform_global_season(art["gseason_bank_2025"], out)
    out = pd.concat([out, z_self, z_eng, z_gs], axis=1)

    for new_col, info in art["hbeb_priors"].items():
        out[new_col] = apply_hbeb(
            df=out,
            group_cols=info["group_cols"],
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=info["global_prior"],
        )
    return out


def predict_frame(test: pd.DataFrame, art: dict) -> np.ndarray:
    df = build_feature_frame(test, art)
    missing = [c for c in art["features"] if c not in df.columns]
    if missing:
        raise KeyError(f"feature 누락: {missing[:10]}")
    X = prepare_features(df[art["features"]], art["cat_cols"])
    pred_raw = np.asarray(art["model"].predict(X), dtype=np.float64)
    return np.clip(pred_raw, 0.0, 1.0)


# ============================================================
# 조건 3: selfcheck — 단일행 vs 전체행 예측 동일성
# ============================================================

def run_selfcheck(test: pd.DataFrame, art: dict, full_pred: np.ndarray) -> dict:
    n = min(SELFCHECK_N_SINGLE, len(test))
    max_diff = 0.0
    for i in range(n):
        single = test.iloc[[i]].copy()
        p_one = predict_frame(single, art)
        diff = abs(float(p_one[0]) - float(full_pred[i]))
        max_diff = max(max_diff, diff)
    result = {
        "n_single_rows": int(n),
        "single_vs_full_max_abs_diff": float(max_diff),
        "atol": SELFCHECK_ATOL,
        "pass": bool(max_diff <= SELFCHECK_ATOL),
    }
    if not result["pass"]:
        raise RuntimeError(
            f"selfcheck 실패: 단일행 vs 전체행 예측 불일치 "
            f"max_abs_diff={max_diff:.3e} > atol={SELFCHECK_ATOL}"
        )
    return result


# ============================================================
# Main
# ============================================================

def load_artifact() -> dict:
    for path in (MODEL_PATH, MODEL_PATH_NESTED):
        if path.exists():
            art = joblib.load(path)
            break
    else:
        raise FileNotFoundError(
            f"모델 없음: {MODEL_PATH} / {MODEL_PATH_NESTED}"
        )
    required = {
        "model", "features", "cat_cols", "hbeb_priors",
        "pitcher_season_stats", "gseason_bank_2025",
    }
    missing = sorted(required - set(art.keys()))
    if missing:
        raise KeyError(f"artifact 필수 키 없음: {missing}")
    if art.get("experiment_name") != EXPERIMENT_NAME:
        raise RuntimeError(
            f"artifact experiment 불일치: {art.get('experiment_name')}"
        )
    return art


def main():
    started = time.time()

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(SAMPLE_SUB_PATH)

    art = load_artifact()

    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    if ID not in test.columns:
        raise KeyError(f"test에 {ID} 없음")
    if TARGET in test.columns:
        raise RuntimeError("test.csv에 target이 존재합니다. 입력 확인 필요.")

    # gseason bank 정합성: bank cutoff보다 과거 시즌 행이 들어오면
    # bank가 미래 통계를 포함하게 되므로 차단한다.
    cutoff = int(art["gseason_bank_2025"]["cutoff"])
    seasons = pd.to_numeric(test[SEASON_COL], errors="raise").astype(int)
    if bool((seasons < cutoff).any()):
        raise RuntimeError(
            f"test season < bank cutoff({cutoff}) 행 존재 — "
            "gseason bank 정합성 위반"
        )

    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    if list(sample_sub.columns[:2]) != [ID, TARGET]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID}, {TARGET})이 아님: "
            f"{list(sample_sub.columns)}"
        )

    pred = predict_frame(test, art)

    if len(pred) != len(test):
        raise RuntimeError("예측 길이 불일치")
    if not np.isfinite(pred).all():
        raise RuntimeError("비유한 예측값 존재")
    if (pred < 0).any() or (pred > 1).any():
        raise RuntimeError("예측값 범위 [0,1] 위반")

    check = run_selfcheck(test, art, pred)

    pred_map = pd.Series(pred, index=test[ID].to_numpy())
    sub = sample_sub[[ID]].copy()
    sub[TARGET] = sub[ID].map(pred_map).to_numpy(dtype=np.float64)
    if sub[TARGET].isna().any():
        raise RuntimeError("sample_submission의 row_id 중 예측 누락 존재")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    try:
        with open(OUTPUT_DIR / "script_selfcheck.json", "w", encoding="utf-8") as f:
            json.dump(check, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    elapsed = time.time() - started
    print("=" * 72)
    print(f"experiment : {EXPERIMENT_NAME}")
    print(f"rows       : {len(sub):,}")
    print(
        f"pred       : min={sub[TARGET].min():.6f} "
        f"max={sub[TARGET].max():.6f} mean={sub[TARGET].mean():.6f}"
    )
    print(
        f"selfcheck  : PASS "
        f"(n={check['n_single_rows']}, "
        f"max_diff={check['single_vs_full_max_abs_diff']:.3e})"
    )
    print(f"output     : {OUTPUT_PATH}")
    print(f"elapsed    : {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()