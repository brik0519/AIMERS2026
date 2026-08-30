# train_cb_reg_shilft_v001.py
# Baseline: cb_reg_v003_exact (canonical v003 + Exact-ASOF5 + HB-EB)
# 추가: shift_v002 이식 feature 4블록
#   ① self_state  : 완전 row-local (라벨 미사용)                    -> 무조건 안전
#   ② eng_state   : 완전 row-local (라벨 미사용)                    -> 무조건 안전
#   ③ gseason 선수 성공률 재구성 : row + fold-train 라벨 bank        -> 조건 1 적용
#   ④ gseason component rate 재구성 : row + fold-train feature base -> 조건 2 적용
#
# 규칙 준수 (평가 데이터 독립 추론):
#   - 모든 bank/통계는 fold-train(최종은 공식 train 전체)에서만 구축하고,
#     각 row에는 해당 row의 season보다 과거 시즌 통계만 적용한다.
#   - 다른 test 행 또는 test 분포는 어디에도 사용하지 않는다.
#   - 조건 2: rint(n*rate) 정수성 가정을 train에서 재검증해 기록한다.
#   - 조건 3(단일행 selfcheck)은 script.py에서 수행한다.

from __future__ import annotations

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
EXPERIMENT_NAME = "cb_reg_shilft_v001"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"

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

EXACT_COLS = [
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

EXPANDING_FOLDS = [
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {"fold": 4, "train_seasons": [2019, 2020, 2021, 2022, 2023], "validation_season": 2024},
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
FINAL_ITERATION_STRATEGY = "latest_fold"

# shift_v002 component rate 컬럼 (④)
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

# 조건 2: 반올림 복원 실패로 간주하는 임계 (0.5에 근접하면 복원 불능)
ROUNDING_WARN_MAX = 0.35


# ============================================================
# Common
# ============================================================

def prepare_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    brier = float(np.mean((y_pred - y_true) ** 2))
    positive_rate = float(np.mean(y_true))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0:
        bss_raw = 0.0
    else:
        bss_raw = float(100000.0 * (1.0 - brier / baseline_brier))

    return {
        "brier": brier,
        "positive_rate": positive_rate,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": max(0.0, bss_raw),
    }


def _n(frame: pd.DataFrame, c: str, default=np.nan) -> np.ndarray:
    """해당 행의 컬럼만 numeric으로 추출. row-local."""
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
# 조건 2: rint(n*rate) 정수성 감사 (공식 train에서 검증·기록)
# ============================================================

def audit_rounding_integrality(train: pd.DataFrame) -> dict:
    checks = (
        [("asof_pitcher_n", "asof_pitcher_success_rate")]
        + [("asof_pitcher_n", c) for c in P_COMPONENT_RATES]
        + [("asof_batter_n", "asof_batter_success_rate")]
        + [("asof_batter_n", c) for c in B_COMPONENT_RATES]
    )
    report = {}
    worst = 0.0
    for ncol, rcol in checks:
        if ncol not in train.columns or rcol not in train.columns:
            report[f"{ncol}*{rcol}"] = {"status": "missing_column"}
            continue
        n = pd.to_numeric(train[ncol], errors="coerce").to_numpy(np.float64)
        r = pd.to_numeric(train[rcol], errors="coerce").to_numpy(np.float64)
        mask = np.isfinite(n) & np.isfinite(r) & (n > 0)
        x = n[mask] * r[mask]
        dev = np.abs(x - np.rint(x))
        stats = {
            "n_rows": int(mask.sum()),
            "dev_mean": float(dev.mean()) if len(dev) else None,
            "dev_p99": float(np.quantile(dev, 0.99)) if len(dev) else None,
            "dev_max": float(dev.max()) if len(dev) else None,
        }
        report[f"{ncol}*{rcol}"] = stats
        if len(dev):
            worst = max(worst, float(dev.max()))
    report["_worst_dev_max"] = worst
    report["_warn_threshold"] = ROUNDING_WARN_MAX
    report["_pass"] = bool(worst <= ROUNDING_WARN_MAX)
    return report


# ============================================================
# shift_v002 블록 ①: self_state (완전 row-local, 라벨 미사용)
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
# shift_v002 블록 ②: eng_state (완전 row-local, 라벨 미사용)
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
# shift_v002 블록 ③+④: gseason bank / transform
#
# 조건 1: bank는 반드시 fold-train(최종은 공식 train 전체)에서만
#         구축하고, row에는 row season보다 과거 시즌 통계만 쓴다.
# ============================================================

def build_gseason_bank(history: pd.DataFrame, cutoff: int) -> dict:
    """
    history: 라벨이 있는 공식 train 부분집합 (fold-train 또는 전체 train).
    cutoff 미만 시즌만 사용한다. 라벨(TARGET) 집계 + 마지막 행 feature 기반
    component base를 만든다. 산출물은 frozen dict — 추론 시 lookup만 한다.
    """
    h = history.loc[
        pd.to_numeric(history[SEASON_COL], errors="raise").astype(int) < int(cutoff)
    ].copy()

    if len(h):
        h_season = pd.to_numeric(h[SEASON_COL], errors="raise").astype(int)
        last_season = int(h_season.max())
        prior = float(h.loc[h_season.eq(last_season), TARGET].mean())
        pg = h.groupby("pitcher_id", sort=False)[TARGET].agg(["size", "sum"])
        bg = h.groupby("batter_id", sort=False)[TARGET].agg(["size", "sum"])
    else:
        last_season = None
        prior = 0.5
        pg = pd.DataFrame(columns=["size", "sum"])
        bg = pd.DataFrame(columns=["size", "sum"])

    bank = {
        "cutoff": int(cutoff),
        "prior": prior,
        "source_max_season": last_season,
        "pitcher_n": pg["size"].to_dict(),
        "pitcher_s": pg["sum"].to_dict(),
        "batter_n": bg["size"].to_dict(),
        "batter_s": bg["sum"].to_dict(),
    }

    prev = h.loc[h_season.eq(last_season)] if len(h) else h
    bank["component_priors"] = {}
    for c in P_COMPONENT_RATES + B_COMPONENT_RATES:
        if c in prev.columns:
            v = pd.to_numeric(prev[c], errors="coerce")
            bank["component_priors"][c] = (
                float(v.mean()) if v.notna().any() else 0.5
            )
        else:
            bank["component_priors"][c] = 0.5

    if len(h):
        pidx = h.groupby("pitcher_id", sort=False)["asof_pitcher_n"].idxmax()
        plast = h.loc[
            pidx, ["pitcher_id", "asof_pitcher_n"] + P_COMPONENT_RATES
        ].set_index("pitcher_id")
        pbase_n = (
            pd.to_numeric(plast["asof_pitcher_n"], errors="coerce")
            .fillna(0)
            .astype(np.float64)
        )
        bank["pitcher_component_base_n"] = pbase_n.to_dict()
        bank["pitcher_component_base"] = {}
        for c in P_COMPONENT_RATES:
            rr = pd.to_numeric(plast[c], errors="coerce")
            cnt = np.rint(pbase_n * rr)
            bank["pitcher_component_base"][c] = cnt.to_dict()

        bidx = h.groupby("batter_id", sort=False)["asof_batter_n"].idxmax()
        blast = h.loc[
            bidx, ["batter_id", "asof_batter_n"] + B_COMPONENT_RATES
        ].set_index("batter_id")
        bbase_n = (
            pd.to_numeric(blast["asof_batter_n"], errors="coerce")
            .fillna(0)
            .astype(np.float64)
        )
        bank["batter_component_base_n"] = bbase_n.to_dict()
        bank["batter_component_base"] = {}
        for c in B_COMPONENT_RATES:
            rr = pd.to_numeric(blast[c], errors="coerce")
            cnt = np.rint(bbase_n * rr)
            bank["batter_component_base"][c] = cnt.to_dict()
    else:
        bank["pitcher_component_base_n"] = {}
        bank["pitcher_component_base"] = {c: {} for c in P_COMPONENT_RATES}
        bank["batter_component_base_n"] = {}
        bank["batter_component_base"] = {c: {} for c in B_COMPONENT_RATES}

    return bank


def transform_global_season(bank: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """row-local + frozen bank lookup. 다른 행을 참조하지 않는다."""
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


def add_gseason_features(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    조건 1 구현.
    각 row의 season y에 대해 history(fold-train)의 y 미만 시즌으로만
    bank를 만들어 적용한다. 다른 평가 행은 어디에도 쓰지 않는다.
    """
    seasons = pd.to_numeric(rows[SEASON_COL], errors="raise").astype(int)
    parts = []
    for y in sorted(seasons.unique()):
        idx = rows.index[seasons == y]
        bank = build_gseason_bank(history, int(y))
        parts.append(transform_global_season(bank, rows.loc[idx]))
    z = pd.concat(parts).reindex(rows.index)
    return z


def add_shift_blocks(rows: pd.DataFrame, history: pd.DataFrame):
    """①②③④를 한 번에 붙인다. 반환: (확장 frame, 추가 컬럼 목록)"""
    z_self = build_self_state(rows)
    z_eng = build_eng_state(rows)
    z_gs = add_gseason_features(rows, history)
    shift_cols = list(z_self.columns) + list(z_eng.columns) + list(z_gs.columns)
    dup = set(shift_cols) & set(rows.columns)
    if dup:
        raise RuntimeError(f"shift feature 이름 충돌: {sorted(dup)}")
    out = pd.concat([rows, z_self, z_eng, z_gs], axis=1)
    return out, shift_cols


# ============================================================
# Exact-ASOF (v003 원본 유지)
# ============================================================

def build_pitcher_season_stats(history: pd.DataFrame) -> pd.DataFrame:
    required = {"pitcher_id", SEASON_COL, TARGET}
    missing = sorted(required - set(history.columns))
    if missing:
        raise KeyError(f"pitcher season stats 필수 컬럼 없음: {missing}")

    stats = (
        history.groupby(["pitcher_id", SEASON_COL], sort=False, dropna=False)[TARGET]
        .agg(["size", "sum"])
        .reset_index()
        .rename(columns={"size": "n", "sum": "success"})
    )
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)
    stats["n"] = stats["n"].astype(np.float64)
    stats["success"] = stats["success"].astype(np.float64)
    return stats


def add_pitcher_exact_asof_features(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    required = {SEASON_COL, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise KeyError(f"Exact-ASOF 필수 컬럼 없음: {missing}")

    out = rows.copy()
    for col in EXACT_COLS:
        out[col] = np.nan

    row_season = pd.to_numeric(out[SEASON_COL], errors="raise").astype(int)
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(row_season.unique()):
        mask = row_season == int(target_year)
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
# HB-EB (v003 원본 유지)
# ============================================================

def compute_hbeb_priors(df_train, group_cols, rate_col):
    rate = pd.to_numeric(df_train[rate_col], errors="coerce")
    g_mu = float(rate.mean())
    g_var = float(rate.var())
    max_bernoulli_var = g_mu * (1.0 - g_mu)

    if not np.isfinite(g_var) or g_var <= 0.0 or g_var >= max_bernoulli_var:
        g_var = max_bernoulli_var * 0.5

    g_m = (max_bernoulli_var / g_var) - 1.0
    g_alpha = g_mu * g_m
    g_beta = (1.0 - g_mu) * g_m

    tmp = df_train[group_cols].copy()
    tmp["_rate_"] = rate
    stats = (
        tmp.groupby(group_cols, dropna=False)["_rate_"]
        .agg(["mean", "var", "count"])
        .reset_index()
    )
    group_max_var = stats["mean"] * (1.0 - stats["mean"])
    valid = (
        stats["var"].notna()
        & np.isfinite(stats["var"])
        & (stats["var"] > 0.0)
        & (stats["var"] < group_max_var)
        & (stats["count"] >= 30)
    )
    stats["alpha"] = g_alpha
    stats["beta"] = g_beta
    if valid.any():
        mu = stats.loc[valid, "mean"]
        var = stats.loc[valid, "var"]
        m = (mu * (1.0 - mu) / var) - 1.0
        stats.loc[valid, "alpha"] = mu * m
        stats.loc[valid, "beta"] = (1.0 - mu) * m

    prior_df = stats[group_cols + ["alpha", "beta"]].copy()
    return prior_df, (float(g_alpha), float(g_beta))


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


def add_hbeb_features(df_train_for_prior, df_apply, features):
    out = df_apply.copy()
    final_features = list(features)
    priors = {}
    for cfg in HBEB_CONFIGS:
        prior_df, global_prior = compute_hbeb_priors(
            df_train=df_train_for_prior,
            group_cols=cfg["group_cols"],
            rate_col=cfg["rate_col"],
        )
        out[cfg["new_col"]] = apply_hbeb(
            df=out,
            group_cols=cfg["group_cols"],
            n_col=cfg["n_col"],
            rate_col=cfg["rate_col"],
            prior_df=prior_df,
            global_prior=global_prior,
        )
        if cfg["rate_col"] in final_features:
            final_features.remove(cfg["rate_col"])
        if cfg["new_col"] not in final_features:
            final_features.append(cfg["new_col"])
        priors[cfg["new_col"]] = {
            "prior_df": prior_df,
            "global_prior": global_prior,
            "group_cols": list(cfg["group_cols"]),
            "n_col": cfg["n_col"],
            "rate_col": cfg["rate_col"],
        }
    return out, final_features, priors


def apply_existing_hbeb_priors(df_apply, features, priors):
    out = df_apply.copy()
    final_features = list(features)
    for cfg in HBEB_CONFIGS:
        info = priors[cfg["new_col"]]
        out[cfg["new_col"]] = apply_hbeb(
            df=out,
            group_cols=info["group_cols"],
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=info["global_prior"],
        )
        if cfg["rate_col"] in final_features:
            final_features.remove(cfg["rate_col"])
        if cfg["new_col"] not in final_features:
            final_features.append(cfg["new_col"])
    return out, final_features


# ============================================================
# Validation
# ============================================================

def run_expanding_validation(train, base_features, cat_cols):
    fold_rows = []
    oof_frames = []
    latest_final_features = None
    latest_shift_cols = None

    base_plus_exact = list(base_features) + [
        c for c in EXACT_COLS if c not in base_features
    ]

    for cfg in EXPANDING_FOLDS:
        fold = cfg["fold"]
        train_seasons = cfg["train_seasons"]
        val_year = cfg["validation_season"]

        df_tr_raw = train.loc[train[SEASON_COL].isin(train_seasons)].copy()
        df_va_raw = train.loc[train[SEASON_COL] == val_year].copy()

        # Exact-ASOF: fold-train 라벨 통계만 사용
        pitcher_stats = build_pitcher_season_stats(df_tr_raw)
        df_tr = add_pitcher_exact_asof_features(df_tr_raw, pitcher_stats)
        df_va = add_pitcher_exact_asof_features(df_va_raw, pitcher_stats)

        # shift 블록 ①②③④: bank/history는 fold-train만 사용 (조건 1)
        df_tr, shift_cols_tr = add_shift_blocks(df_tr, history=df_tr_raw)
        df_va, shift_cols_va = add_shift_blocks(df_va, history=df_tr_raw)
        if shift_cols_tr != shift_cols_va:
            raise RuntimeError("train/validation shift feature 목록 불일치")

        feats = base_plus_exact + shift_cols_tr

        # HB-EB: prior는 fold-train에서만 fit
        df_tr, final_features, hbeb_priors = add_hbeb_features(
            df_train_for_prior=df_tr_raw,
            df_apply=df_tr,
            features=feats,
        )
        df_va, final_features_check = apply_existing_hbeb_priors(
            df_apply=df_va,
            features=feats,
            priors=hbeb_priors,
        )
        if final_features_check != final_features:
            raise RuntimeError("train/validation feature 목록 불일치")
        if "asof_pitcher_pitchmix_n" not in final_features:
            raise RuntimeError(
                "canonical v003 feature 'asof_pitcher_pitchmix_n'이 사라졌습니다."
            )

        fold_cat_cols = [c for c in cat_cols if c in final_features]

        X_tr = prepare_features(df_tr[final_features], fold_cat_cols)
        X_va = prepare_features(df_va[final_features], fold_cat_cols)
        y_tr = df_tr[TARGET].astype(np.float64)
        y_va = df_va[TARGET].astype(np.float64)

        model = CatBoostRegressor(**MODEL_PARAMS)
        started = time.time()
        model.fit(
            X_tr,
            y_tr,
            cat_features=fold_cat_cols,
            eval_set=(X_va, y_va),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        elapsed = time.time() - started

        pred_raw = np.asarray(model.predict(X_va), dtype=np.float64)
        pred = np.clip(pred_raw, 0.0, 1.0)
        metrics = calculate_metrics(y_va, pred)

        best_iter_zero = int(model.get_best_iteration())
        best_tree_count = (
            best_iter_zero + 1 if best_iter_zero >= 0 else int(model.tree_count_)
        )

        fold_rows.append({
            "fold": int(fold),
            "train_seasons": [int(x) for x in train_seasons],
            "validation_season": int(val_year),
            "n_train": int(len(df_tr)),
            "n_validation": int(len(df_va)),
            "brier": float(metrics["brier"]),
            "bss_raw": float(metrics["bss_raw"]),
            "bss": float(metrics["bss"]),
            "best_iteration_zero_based": best_iter_zero,
            "best_tree_count": int(best_tree_count),
            "elapsed_seconds": float(elapsed),
        })

        oof_frames.append(
            pd.DataFrame({
                ID: df_va[ID].to_numpy(),
                "target": y_va.to_numpy(dtype=np.float64),
                "prediction_raw": pred_raw,
                "prediction": pred,
                "validation_season": int(val_year),
            })
        )

        latest_final_features = list(final_features)
        latest_shift_cols = list(shift_cols_tr)

        print(
            f"[Fold {fold}] {train_seasons} -> {val_year} | "
            f"Brier={metrics['brier']:.10f} | "
            f"BSS={metrics['bss_raw']:.2f} | "
            f"trees={best_tree_count}"
        )

    oof = pd.concat(oof_frames, ignore_index=True)
    pooled = calculate_metrics(oof["target"], oof["prediction"])
    return fold_rows, oof, latest_final_features, latest_shift_cols, pooled


# ============================================================
# Final training
# ============================================================

def train_final(train, base_features, cat_cols, final_iterations):
    base_plus_exact = list(base_features) + [
        c for c in EXACT_COLS if c not in base_features
    ]

    # Exact-ASOF: 전체 train 라벨 통계 (각 row는 자기 과거 시즌만 사용)
    pitcher_stats = build_pitcher_season_stats(train)
    df_full = add_pitcher_exact_asof_features(train, pitcher_stats)

    # shift 블록: history = 전체 train (각 row는 자기 과거 시즌만 사용)
    df_full, shift_cols = add_shift_blocks(df_full, history=train)

    feats = base_plus_exact + shift_cols

    df_full, final_features, hbeb_priors = add_hbeb_features(
        df_train_for_prior=train,
        df_apply=df_full,
        features=feats,
    )
    if "asof_pitcher_pitchmix_n" not in final_features:
        raise RuntimeError(
            "canonical v003 feature 'asof_pitcher_pitchmix_n'이 사라졌습니다."
        )

    final_cat_cols = [c for c in cat_cols if c in final_features]
    X_full = prepare_features(df_full[final_features], final_cat_cols)
    y_full = df_full[TARGET].astype(np.float64)

    final_params = {**MODEL_PARAMS, "iterations": int(final_iterations)}
    model = CatBoostRegressor(**final_params)

    started = time.time()
    model.fit(X_full, y_full, cat_features=final_cat_cols)
    elapsed = time.time() - started

    # 2025 추론용 frozen bank (공식 train 전체, cutoff=2025)
    gseason_bank_2025 = build_gseason_bank(train, 2025)

    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model": model,
        "features": final_features,
        "cat_cols": final_cat_cols,
        "hbeb_priors": hbeb_priors,
        "pitcher_season_stats": pitcher_stats,
        "exact_cols": list(EXACT_COLS),
        "shift_cols": list(shift_cols),
        "gseason_bank_2025": gseason_bank_2025,
        "p_component_rates": list(P_COMPONENT_RATES),
        "b_component_rates": list(B_COMPONENT_RATES),
        "final_iterations": int(final_iterations),
        "model_params": final_params,
        "prediction_policy": "np.clip(raw_prediction, 0.0, 1.0)",
        "baseline_definition": (
            "cb_reg_v003_exact + shift_v002 blocks "
            "(self_state, eng_state, gseason player, gseason component); "
            "no residual, no conditional, no routing"
        ),
    }
    return artifact, elapsed


# ============================================================
# Main
# ============================================================

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)
    if VALIDATION_DIR.exists():
        raise FileExistsError(f"기존 validation 보존을 위해 중단: {VALIDATION_DIR}")
    if MODEL_DIR.exists():
        raise FileExistsError(f"기존 model 보존을 위해 중단: {MODEL_DIR}")

    # test는 schema(header)만 읽는다.
    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns.tolist()
    base_features = [c for c in test_cols if c != ID]

    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError(
            "test schema에 asof_pitcher_pitchmix_n이 없습니다. "
            "canonical v003 기준선과 다른 schema입니다."
        )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[ID] + base_features + [TARGET],
    )

    required = {
        ID, TARGET, SEASON_COL,
        "pitcher_id", "batter_id",
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_batter_n", "asof_batter_success_rate",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise KeyError(f"train 필수 컬럼 없음: {missing}")

    cat_cols = [c for c in CAT_COLS if c in base_features]

    print("=" * 88)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print("Baseline   : cb_reg_v003_exact + shift_v002 blocks(1,2,3,4)")
    print("Residual   : NO")
    print("Conditional: NO")
    print("Routing    : NO")
    print("pitchmix_n : KEEP")
    print(f"Train rows : {len(train):,}")
    print(f"Base feats : {len(base_features)}")
    print("=" * 88)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    # ----------------------------------------------------
    # 조건 2: rint(n*rate) 정수성 감사 (train에서 검증·기록)
    # ----------------------------------------------------
    rounding_report = audit_rounding_integrality(train)
    with open(VALIDATION_DIR / "rounding_audit.json", "w", encoding="utf-8") as f:
        json.dump(rounding_report, f, ensure_ascii=False, indent=2)
    print(
        f"Rounding audit: worst dev_max={rounding_report['_worst_dev_max']:.6f} "
        f"(warn>{ROUNDING_WARN_MAX}) -> "
        f"{'PASS' if rounding_report['_pass'] else 'WARN'}"
    )
    if not rounding_report["_pass"]:
        print(
            "경고: n*rate 반올림 복원 가정이 흔들립니다. "
            "gseason 재구성 feature(③④)의 신뢰도를 재검토하십시오. "
            "(학습은 계속 진행)"
        )

    (
        fold_rows,
        oof,
        final_features,
        shift_cols,
        pooled,
    ) = run_expanding_validation(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
    )

    oof.to_csv(VALIDATION_DIR / "oof_pred.csv", index=False, encoding="utf-8-sig")

    validation_meta = {
        "experiment_name": EXPERIMENT_NAME,
        "baseline": "cb_reg_v003_exact",
        "added_blocks": {
            "1_self_state": "row-local, no label",
            "2_eng_state": "row-local, no label",
            "3_gseason_player": "row + fold-train label bank (cond.1)",
            "4_gseason_component": "row + fold-train feature base (cond.2 audited)",
        },
        "validation_method": "expanding_window",
        "folds": fold_rows,
        "pooled": pooled,
        "final_features": final_features,
        "n_final_features": len(final_features),
        "shift_cols": shift_cols,
        "n_shift_cols": len(shift_cols),
        "exact_cols": EXACT_COLS,
        "hbeb_configs": HBEB_CONFIGS,
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "final_iteration_strategy": FINAL_ITERATION_STRATEGY,
        "rounding_audit_pass": bool(rounding_report["_pass"]),
    }
    with open(VALIDATION_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(validation_meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 88)
    print("Expanding Window Summary")
    print("=" * 88)
    for row in fold_rows:
        print(
            f"{row['validation_season']} | "
            f"Brier={row['brier']:.10f} | "
            f"BSS={row['bss_raw']:.2f} | "
            f"trees={row['best_tree_count']}"
        )
    print("-" * 88)
    print(f"Pooled OOF Brier = {pooled['brier']:.10f}")
    print(f"Pooled OOF BSS   = {pooled['bss_raw']:.2f}")
    print("=" * 88)

    final_iterations = int(fold_rows[-1]["best_tree_count"])
    print(f"Final iterations ({FINAL_ITERATION_STRATEGY}) = {final_iterations}")

    artifact, elapsed = train_final(
        train=train,
        base_features=base_features,
        cat_cols=cat_cols,
        final_iterations=final_iterations,
    )

    joblib.dump(artifact, MODEL_PATH, compress=3)

    final_meta = {
        "experiment_name": EXPERIMENT_NAME,
        "baseline": "cb_reg_v003_exact + shift_v002 blocks(1,2,3,4)",
        "residual": False,
        "pitcher_conditional": False,
        "routing": False,
        "asof_pitcher_pitchmix_n_kept": True,
        "n_features": len(artifact["features"]),
        "features": artifact["features"],
        "n_shift_cols": len(artifact["shift_cols"]),
        "shift_cols": artifact["shift_cols"],
        "exact_cols": EXACT_COLS,
        "final_iterations": final_iterations,
        "elapsed_seconds": float(elapsed),
        "pooled_oof_brier": float(pooled["brier"]),
        "pooled_oof_bss_raw": float(pooled["bss_raw"]),
        "fold_2024_bss_raw": float(fold_rows[-1]["bss_raw"]),
        "fold_2023_bss_raw": float(fold_rows[-2]["bss_raw"]),
    }
    with open(MODEL_DIR / "final_metadata.json", "w", encoding="utf-8") as f:
        json.dump(final_meta, f, ensure_ascii=False, indent=2)

    try:
        shutil.copy2(Path(__file__).resolve(), MODEL_DIR / "train.py")
    except Exception:
        pass

    print()
    print("=" * 88)
    print("Final model saved")
    print(f"- {MODEL_PATH}")
    print(f"- features: {len(artifact['features'])} "
          f"(shift blocks: {len(artifact['shift_cols'])})")
    print(
        "- asof_pitcher_pitchmix_n: "
        f"{'YES' if 'asof_pitcher_pitchmix_n' in artifact['features'] else 'NO'}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()