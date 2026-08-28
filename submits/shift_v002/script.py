"""script.py — shift_v002 추론 (EXPERIMENT_NAME 만 바꾸면 anchor_v001 도 동일).

MODEL_PATH(단일 pkl) 를 읽어 test.csv 각 행을 독립적으로 예측하고 submission.csv 를 만든다.

예측 경로 (train.py Phase 6 의 test 측 경로와 동일):
  Layer 0  anchor      : p0 = clip( (exact_lgb + exact_ridge + trend_lgb) / 3 , 0, 1 )
  Layer 1  residual    : p1 = clip( p0 + residual_shrink · mean_seed(ExtraTrees[segment](x)) , 0, 1 )
                         (shift_v002: residual_shrink=0.5 / anchor_v001: 0, residual_seeds=[] → p1 = p0)
  Layer 2  shift       : p  = clip( p1 + SHIFT_SCALE · Ridge(sens(x)) , 0, 1 )   (CORE 행만)

평가 데이터 독립 추론:
  모든 lookup 은 artifact 안의 frozen table(공식 train 만으로 생성)이며, 다른 test 행·test 집계를
  사용하지 않는다. `--selfcheck` 로 predict(test.iloc[[i]])[0] == predict(test)[i] 를 실측한다.

사용:
  python script.py                 # ./data/test.csv -> ./output/submission.csv
  python script.py --selfcheck     # 행 독립성 자가검사 (A 단일행 vs 전체 · B 셔플 · C 분할결합)
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 독립 실행: train.py 를 import 하지 않는다. 필요한 row-local feature 함수는 train.py 에서
# 그대로 복사했다 (동일 버전 유지 필수 — train.py 변경 시 아래 블록도 갱신할 것).

# =========================================================
# 경로
# =========================================================
EXPERIMENT_NAME = "shift_v002"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
SEGMENTS = ["CORE", "ANCHOR_TEAM", "FUTURES"]



# =========================================================
# train.py 와 동일해야 하는 상수 (row-local feature 계산에 필요한 것만)
# =========================================================
P_COMPONENT_RATES = [
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
B_COMPONENT_RATES = ["asof_batter_middle_rate"]
EXPECTED_DIMS = {
    "self_state": 34, "gseason": 28, "asof_state": 62, "pitcher_context": 21,
    "residual_base": 45, "residual_matrix": 128, "sensitivity": 25,
}

# =========================================================
# row-local feature 함수 — train.py 에서 verbatim 복사
# =========================================================

def _n(frame, c, default=np.nan):
    """notebook cell 2 원본."""
    if c in frame:
        return pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    return np.full(len(frame), default, np.float64)


def _rounded_cumulative_count(n, rate):
    n = np.asarray(n, np.float64)
    rate = np.asarray(rate, np.float64)
    out = np.zeros(len(n), np.float64)
    ok = (n > 0) & np.isfinite(rate)
    out[ok] = np.rint(n[ok] * rate[ok])
    return out


def build_self_state(frame: pd.DataFrame) -> pd.DataFrame:
    """완전 row-local. 다른 행을 전혀 참조하지 않는다."""
    n = len(frame)
    z = pd.DataFrame(index=np.arange(n))
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
    z["self__recent_middle_blend"] = (0.50 * m1 + 0.30 * m3 + 0.20 * m5).astype(np.float32)
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
        0.7 * (balls == 3) + 1.1 * ((balls == 3) & (strikes == 2)) + 0.25 * (strikes == 2)
    ).astype(np.float32)
    z["self__full_count"] = ((balls == 3) & (strikes == 2)).astype(np.float32)
    z["self__inning"] = inning.astype(np.float32)
    z["self__late"] = (inning >= 7).astype(np.float32)
    z["self__score_abs"] = np.abs(score).astype(np.float32)
    z["self__li_log1p"] = np.log1p(li).astype(np.float32)
    z["self__pitcher_we"] = np.clip(pwe, 0, 1).astype(np.float32)
    z["self__win_uncertainty"] = (1 - 2 * np.abs(np.clip(pwe, 0, 1) - 0.5)).astype(np.float32)
    z["self__same_hand"] = (ph == bh).astype(np.float32)

    pressure = z["self__count_pressure"].to_numpy(np.float64)
    z["self__career_x_pressure"] = (career * pressure).astype(np.float32)
    z["self__recent_x_pressure"] = (
        (0.5 * p1 + 0.3 * p3 + 0.2 * p5) * pressure
    ).astype(np.float32)
    z["self__offspeed_x_same_hand"] = (off * (ph == bh)).astype(np.float32)

    assert z.shape[1] == EXPECTED_DIMS["self_state"], z.shape
    return z


def transform_global_season(bank: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """row-local + frozen bank. 다른 행을 참조하지 않는다."""
    r = rows.reset_index(drop=True)
    z = pd.DataFrame(index=np.arange(len(r)))
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
        raw = np.divide(ss, sn, out=np.full(len(r), prior, np.float64), where=sn > 0)
        for k in [40.0, 80.0, 160.0]:
            z[f"gseason__{prefix}_rate_k{int(k)}"] = ((ss + k * prior) / (sn + k)).astype(np.float32)
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
        base = ids.map(bank["pitcher_component_base"].get(c, {})).fillna(0).to_numpy(np.float64)
        wc = np.clip(cum - base, 0.0, win_n)
        comp[c] = ((wc + 60.0 * pr) / (win_n + 60.0)).astype(np.float32)
    z["gseason__pitcher_component_log_n"] = np.log1p(win_n).astype(np.float32)
    z["gseason__pitcher_component_reliability"] = (win_n / (win_n + 60.0)).astype(np.float32)
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
    bbase_n = bids.map(bank["batter_component_base_n"]).fillna(0).to_numpy(np.float64)
    bwin = np.maximum(ban - bbase_n, 0.0)
    c = "asof_batter_middle_rate"
    pr = float(bank["component_priors"].get(c, 0.5))
    rr = _n(r, c, pr)
    rr = np.where(np.isfinite(rr), rr, pr)
    cum = _rounded_cumulative_count(ban, rr)
    base = bids.map(bank["batter_component_base"].get(c, {})).fillna(0).to_numpy(np.float64)
    wc = np.clip(cum - base, 0.0, bwin)
    bmid = ((wc + 80.0 * pr) / (bwin + 80.0)).astype(np.float32)
    z["gseason__batter_middle_rate"] = bmid
    z["gseason__batter_middle_minus_pitcher_middle"] = (
        bmid - z["gseason__pitcher_middle_rate"]
    ).astype(np.float32)

    assert z.shape[1] == EXPECTED_DIMS["gseason"], z.shape
    return z


def add_eng(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["eng_recent35"] = (
        0.5 * pd.to_numeric(x.asof_pitcher_prev1_game_success_rate, errors="coerce")
        + 0.3 * pd.to_numeric(x.asof_pitcher_prev3_game_success_rate, errors="coerce")
        + 0.2 * pd.to_numeric(x.asof_pitcher_prev5_game_success_rate, errors="coerce")
    )
    x["eng_recent_delta"] = x.eng_recent35 - pd.to_numeric(
        x.asof_pitcher_success_rate, errors="coerce"
    )
    x["eng_sb"] = pd.to_numeric(x.asof_pitcher_strike_rate, errors="coerce") - pd.to_numeric(
        x.asof_pitcher_ball_rate, errors="coerce"
    )
    x["eng_same"] = (x.pitcher_hand.astype(str) == x.batter_hand.astype(str)).astype(np.float32)
    x["eng_close"] = (
        pd.to_numeric(x.score_diff_pitcher_team, errors="coerce").abs() <= 1
    ).astype(np.float32)
    x["eng_late"] = (pd.to_numeric(x.inning, errors="coerce") >= 7).astype(np.float32)
    return x


def transform_residual_features(prep: dict, rows: pd.DataFrame, base_pred) -> pd.DataFrame:
    r = add_eng(rows.reset_index(drop=True))
    A = r[prep["cols"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in prep["cols"]:
        A[c] = A[c].fillna(float(prep["med"].get(c, 0.0))).astype(np.float32)
    for c, mp in prep["maps"].items():
        A["code_" + c] = (
            r[c].astype("string").fillna("<NA>").astype(str).map(mp).fillna(-1).astype(np.float32)
        )
    A["base_pred"] = np.asarray(base_pred, dtype=np.float32)
    assert len(A) == len(rows) and np.isfinite(A.to_numpy(float)).all()
    assert A.shape[1] == EXPECTED_DIMS["residual_base"], A.shape
    del r
    return A


def _lookup(table, rows, keys, field, default=0.0):
    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)
    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)


def _table_features(table, rows, keys, prefix, fields=("rate", "reliability", "delta", "log_n")):
    z = pd.DataFrame(index=np.arange(len(rows)))
    for f in fields:
        z[prefix + "__" + f] = _lookup(table, rows, keys, f, 0.0)
    return z


def _context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["ctx_count"] = x["balls_before"].astype(str) + "_" + x["strikes_before"].astype(str)
    x["ctx_inning_phase"] = (
        pd.cut(
            pd.to_numeric(x["inning"], errors="coerce"),
            [-np.inf, 3, 6, 9, np.inf],
            labels=["early", "middle", "late", "extra"],
        ).astype(str)
    )
    x["ctx_pressure"] = np.where(
        (x["balls_before"] == 3) & (x["strikes_before"] == 2),
        "full",
        np.where(
            x["balls_before"] == 3,
            "three_ball",
            np.where(x["strikes_before"] == 2, "two_strike", "normal"),
        ),
    )
    x["ctx_runners"] = np.where(
        pd.to_numeric(x["num_runners_on"], errors="coerce").fillna(0) > 0, "on", "empty"
    )
    x["ctx_li_bucket"] = (
        pd.cut(
            pd.to_numeric(x["li"], errors="coerce"),
            [-np.inf, 0.7, 1.5, np.inf],
            labels=["low", "mid", "high"],
        ).astype(str)
    )
    return x


def transform_context_specs(specs: list, rows: pd.DataFrame) -> pd.DataFrame:
    r = _context_frame(rows.reset_index(drop=True))
    parts = []
    for spec in specs:
        z = _table_features(
            spec["table"], r, ["pitcher_id", spec["ctx"]], spec["prefix"],
            ("rate", "reliability", "delta"),
        )
        parts.append(z)
    out = pd.concat(parts, axis=1).astype(np.float32)
    del r
    assert out.shape[1] == EXPECTED_DIMS["pitcher_context"], out.shape
    return out


def _pressure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame.copy()
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int)
    r["sens_pressure"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(balls == 3, "3ball", np.where(strikes == 2, "2strike", "normal")),
    )
    r["sens_pressure_bhand"] = r["sens_pressure"].astype(str) + "|" + r["batter_hand"].astype(str)
    return r


def transform_sensitivity_specs(specs: list, rows: pd.DataFrame) -> pd.DataFrame:
    r = _pressure_frame(rows.reset_index(drop=True))
    z = pd.DataFrame(index=np.arange(len(r)))
    for spec in specs:
        de = _lookup(spec["table"], r, ["pitcher_id", spec["ctx"]], "delta", 0.0).astype(np.float32)
        re = _lookup(spec["table"], r, ["pitcher_id", spec["ctx"]], "reliability", 0.0).astype(
            np.float32
        )
        z[spec["prefix"] + "__delta"] = de
        z[spec["prefix"] + "__weighted_delta"] = (de * re).astype(np.float32)
        z[spec["prefix"] + "__reliability"] = re
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    for b in range(4):
        z[f"ctx__ball_{b}"] = (balls == b).astype(np.float32)
    for s in range(3):
        z[f"ctx__strike_{s}"] = (strikes == s).astype(np.float32)
    for b in range(4):
        for s in range(3):
            z[f"ctx__count_{b}_{s}"] = ((balls == b) & (strikes == s)).astype(np.float32)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    assert len(z) == len(rows) and not z.columns.duplicated().any()
    assert np.isfinite(z.to_numpy(float)).all()
    assert z.shape[1] == EXPECTED_DIMS["sensitivity"], z.shape
    del r
    return z


def attach_route_segment(frame: pd.DataFrame, anchor_team_id: int) -> pd.DataFrame:
    """row-local 3-way router. train / test 공통."""
    frame["route_segment"] = np.where(
        frame.game_type.eq("F"),
        "FUTURES",
        np.where(
            (frame.pitcher_team_id == anchor_team_id)
            | (frame.batter_team_id == anchor_team_id),
            "ANCHOR_TEAM",
            "CORE",
        ),
    )
    return frame


def forest_predict_rowwise(model, X) -> np.ndarray:
    """행 독립·결정론이 보장되는 sklearn forest 예측.

    n_jobs>1 이면 트리 출력을 여러 스레드가 공유 배열에 더하므로 누적 순서가 실행마다 달라져
    2^-53 수준의 잡음이 생긴다(재현 감사에서 실측). 단일 스레드로 예측하면 트리 순서가 고정되어
    결과가 결정론적이고 배치 크기와도 무관하다. train.py 의 동일 함수를 복사했다.
    """
    saved = getattr(model, "n_jobs", None)
    try:
        model.n_jobs = 1
        return np.asarray(model.predict(X), dtype=np.float64)
    finally:
        model.n_jobs = saved


def lgb_predict_batchpath(model, X, proba: bool = False) -> np.ndarray:
    """행 독립이 보장되는 LightGBM 예측.

    LightGBM 은 입력이 1행이면 별도의 single-row 경로를 타고, 그 결과가 배치 경로와 마지막 비트
    (2^-53 수준)에서 어긋난다(행 독립성 검사에서 실측: 셔플/분할결합은 0, 단일행만 1e-16).
    배치 경로는 행별로 독립이므로, 1행이면 같은 행을 복제해 2행으로 예측하고 첫 값만 취한다.
    """
    n = len(X)
    if n == 1:
        X2 = pd.concat([X, X], axis=0, ignore_index=True) if isinstance(X, pd.DataFrame) \
            else np.concatenate([X, X], axis=0)
        out = model.predict_proba(X2)[:, 1] if proba else model.predict(X2)
        return np.asarray(out[:1], dtype=np.float64)
    out = model.predict_proba(X)[:, 1] if proba else model.predict(X)
    return np.asarray(out, dtype=np.float64)


def ridge_predict_rowwise(model, X) -> np.ndarray:
    """행 독립이 보장되는 Ridge 예측.

    sklearn `Ridge.predict` 는 float32 입력이면 float32 BLAS 로 X@coef 를 계산하고, BLAS/numpy 의
    합산 순서는 행 수에 따라 달라질 수 있다(대회 행 독립성 검사에서 1e-10 실측). 여기서는 float64 로
    올린 뒤 **열 순서를 고정한 순차 누적**(elementwise 연산만 사용)으로 계산하므로 다른 행의 존재나
    배치 크기가 결과에 영향을 주지 않는다. 배포(Phase 6)·script.py·deploy_validation.py 가 같은 함수를 쓴다.
    """
    X64 = np.asarray(X, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    icpt = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    acc = np.full(X64.shape[0], float(icpt[0]) if icpt.size else 0.0, dtype=np.float64)
    for j in range(X64.shape[1]):
        acc = acc + X64[:, j] * coef[j]
    return acc


# =========================================================
# 적재
# =========================================================

def load_artifact(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        art = pickle.load(f)
    # train.py 의 런타임 컬럼 목록을 artifact 값으로 고정한다.
    return art


def load_test(path: Path = TEST_PATH) -> pd.DataFrame:
    # base_state 는 frozen table 의 문자열 키다. '123' 만 있는 파일(예: 단일 행)을 읽으면
    # pandas 가 int 로 추론해 조회가 조용히 실패하므로 항상 str 로 읽는다.
    test = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"base_state": str})
    if ID not in test.columns:
        raise ValueError(f"test 에 {ID} 가 없다")
    return test


def validate_test(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    test = test.copy()
    test[SEASON_COL] = pd.to_numeric(test[SEASON_COL], errors="raise").astype(int)
    test["game_type"] = test["game_type"].astype("string").fillna("<NA>").astype(str)
    assert TARGET not in test.columns, "test 에 target 이 있다"
    if bool((~test.game_type.isin(["R", "F"])).any()):
        raise RuntimeError("unexpected game_type")
    if bool((test[SEASON_COL] != int(art["expected_year"])).any()):
        raise RuntimeError(f"evaluation season differs from expected {art['expected_year']}")
    assert str(test["base_state"].dtype) == "object", "base_state 는 object 여야 한다"
    need = set(art["rawnum"]) | set(c for c in art["cat_small"] if c != "route_segment") | \
        {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "pitcher_hand",
         "batter_hand", "away_win_expectancy", "top_bottom"}
    missing = sorted(need - set(test.columns))
    assert not missing, f"test 에 없는 컬럼: {missing}"
    return attach_route_segment(test, int(art["anchor_team_id"]))


# =========================================================
# 예측
# =========================================================

def _apply_maps(frame: pd.DataFrame, maps: dict, prefix: str = "code_") -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(frame)))
    for c, mp in maps.items():
        out[prefix + c] = (
            frame[c].astype("string").fillna("<NA>").astype(str).map(mp).fillna(-1).astype(np.float32)
        )
    return out


def predict(test: pd.DataFrame, art: dict, verbose: bool = False) -> dict:
    """행 독립 예측. 반환: dict(p_base, adjusted, final, segment)."""
    n = len(test)
    test = test.reset_index(drop=True)
    seg = test.route_segment.to_numpy()

    # ---- Layer 0 : anchor ----
    st = build_self_state(test)
    gt = transform_global_season(art["target_global_bank"], test)
    B0 = pd.concat(
        [test[art["rawnum"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
         st.reset_index(drop=True), gt.reset_index(drop=True)], axis=1)

    B_exact = pd.concat([B0, _apply_maps(test, art["exact_cat_maps"])], axis=1)
    assert list(B_exact.columns) == list(art["exact_feature_cols"])
    p1 = lgb_predict_batchpath(art["base_exact_lgb"], B_exact, proba=True)

    Br = B0.replace([np.inf, -np.inf], np.nan)
    assert list(Br.columns) == list(art["ridge_feature_cols"])
    Br = Br.fillna(pd.Series(art["ridge_med"])).to_numpy(np.float32)
    Br = np.clip((Br - art["ridge_mu"]) / art["ridge_sd"], -8, 8)
    p2 = np.clip(ridge_predict_rowwise(art["base_ridge"], Br), 0.001, 0.999)

    t = add_eng(test)
    Bn = t[art["trend_num_cols"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    Bt = pd.concat([Bn, _apply_maps(t, art["trend_cat_maps"])], axis=1).astype(np.float32)
    assert list(Bt.columns) == list(art["trend_feature_cols"])
    bt = np.array([art["trend_prior"][d] for d in seg], dtype=np.float64)
    p3 = np.clip(bt + lgb_predict_batchpath(art["base_trend_lgb"], Bt), 0.001, 0.999)

    p_base = np.clip((np.asarray(p1, float) + np.asarray(p2, float) + np.asarray(p3, float)) / 3.0, 0, 1)

    # ---- Layer 1 : residual adjuster ----
    X_base = transform_residual_features(art["residual_prep"], test, p_base)
    assert list(X_base.columns) == list(art["residual_base_feature_cols"])
    pc = transform_context_specs(art["pitcher_context_specs"], test)
    assert list(pc.columns) == list(art["pitcher_context_cols"])
    ex = pd.concat([st.add_prefix("exact__"), gt.add_prefix("exact__")], axis=1) \
        .replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    assert list(ex.columns) == list(art["asof_residual_cols"])
    X = pd.concat([X_base.reset_index(drop=True), pc.reset_index(drop=True),
                   ex.reset_index(drop=True)], axis=1)
    assert list(X.columns) == list(art["residual_feature_cols"])
    assert np.isfinite(X.to_numpy(float)).all()

    corrections = []
    for seed in art["residual_seeds"]:            # 비어 있으면(배포 shrink 0) Layer 1 은 적용되지 않는다
        c = np.zeros(n, dtype=np.float64)
        for d in SEGMENTS:
            te = seg == d
            if te.any():
                c[te] = forest_predict_rowwise(art["residual_models"][(int(seed), d)], X.loc[te])
        corrections.append(c)
    correction = (np.mean(np.column_stack(corrections), axis=1) if corrections
                  else np.zeros(n, dtype=np.float64))
    adjusted = np.clip(p_base + float(art["residual_shrink"]) * correction, 0, 1)

    # ---- Layer 2 : sensitivity shift (CORE 만) ----
    sens = transform_sensitivity_specs(art["sensitivity_specs"], test)
    assert list(sens.columns) == list(art["sensitivity_cols"])
    Bz = np.clip((sens.to_numpy(np.float32) - art["sens_mu"]) / art["sens_sd"], -8, 8)
    shift = ridge_predict_rowwise(art["shift_model"], Bz)
    final = adjusted.copy()
    core = seg == art["shift_segment"]
    final[core] = np.clip(final[core] + float(art["shift_scale"]) * shift[core], 0, 1)
    assert np.isfinite(final).all() and ((final >= 0) & (final <= 1)).all()

    if verbose:
        print(f"  p_base mean={p_base.mean():.6f}  adjusted mean={adjusted.mean():.6f}  "
              f"final mean={final.mean():.6f}  core rows={int(core.sum())}  "
              f"layer2_mode={art.get('layer2_mode')}")
    return {"p_base": p_base, "adjusted": adjusted, "final": final, "segment": seg,
            # 진단용 성분 (selfcheck 상세 보고)
            "_p1_lgb": np.asarray(p1, np.float64), "_p2_ridge": np.asarray(p2, np.float64),
            "_p3_trend": np.asarray(p3, np.float64), "_shift": shift, "_residual": correction,
            "_B_exact": B_exact.to_numpy(np.float64), "_Br": np.asarray(Br, np.float64),
            "_Bt": Bt.to_numpy(np.float64), "_Bz": np.asarray(Bz, np.float64)}


# =========================================================
# 제출
# =========================================================

def merge_predictions(sub: pd.DataFrame, ids, preds) -> pd.DataFrame:
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID], sub[TARGET]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder 를 유지한 row_id {n_missing}건")
    sub[TARGET] = values
    return sub


# =========================================================
# 행 독립성 자가검사
# =========================================================

def selfcheck(test: pd.DataFrame, art: dict, n_single: int = 30, seed: int = 0) -> dict:
    """A 단일행 vs 전체 · B 셔플 · C 분할결합. 모두 max|Δ| == 0 이어야 한다.

    최종 예측뿐 아니라 단계별(p_base, adjusted, final)로 보고해 어긋나는 층을 바로 찾는다.
    """
    keys = ["p_base", "adjusted", "final"]
    detail = ["_p1_lgb", "_p2_ridge", "_p3_trend", "_residual", "_shift", "_B_exact", "_Br", "_Bt", "_Bz"]
    full = predict(test, art)
    rng = np.random.default_rng(seed)
    idx = np.unique(np.concatenate([
        rng.choice(len(test), size=min(n_single, len(test)), replace=False),
        [int(np.flatnonzero(test.route_segment.to_numpy() == d)[0])
         for d in SEGMENTS if (test.route_segment == d).any()],
    ]))
    singles = {i: predict(test.iloc[[i]], art) for i in idx}
    perm = rng.permutation(len(test))
    shuf = predict(test.iloc[perm], art)
    cut = len(test) // 2
    left, right = predict(test.iloc[:cut], art), predict(test.iloc[cut:], art)
    out = {"n_single_rows": int(len(idx))}
    for k in keys:
        out[f"A_single_vs_full_{k}"] = max(float(abs(singles[i][k][0] - full[k][i])) for i in idx)
        out[f"B_shuffle_{k}"] = float(np.abs(shuf[k] - full[k][perm]).max())
        out[f"C_split_concat_{k}"] = float(np.abs(np.concatenate([left[k], right[k]]) - full[k]).max())
    # 성분별 단일행 진단: 어느 층/입력에서 어긋나는지
    for k in detail:
        out[f"A_detail_{k}"] = max(float(np.abs(np.asarray(singles[i][k])[0] - np.asarray(full[k])[i]).max())
                                   for i in idx)
    print("  selfcheck detail (A 단일행 vs 전체, max|Δ|): " +
          "  ".join(f"{k.lstrip('_')}={out[f'A_detail_{k}']:.3e}" for k in detail))
    out["pass"] = bool(all(v == 0.0 for kk, v in out.items() if kk.startswith(("A_", "B_", "C_"))))
    for k in keys:
        print(f"  selfcheck[{k:<8}] A={out[f'A_single_vs_full_{k}']:.3e}  "
              f"B={out[f'B_shuffle_{k}']:.3e}  C={out[f'C_split_concat_{k}']:.3e}")
    print(f"  selfcheck pass={out['pass']}  (n_single_rows={out['n_single_rows']})")
    return out


# =========================================================
# main
# =========================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--test", default=str(TEST_PATH))
    args = ap.parse_args(argv)

    t0 = time.time()
    print("Load model...")
    art = load_artifact(Path(args.model))
    print(f" OK. experiment={art['experiment_name']}  expected_year={art['expected_year']}  "
          f"layer2_mode={art.get('layer2_mode')}")

    print("Load test data...")
    test = validate_test(load_test(Path(args.test)), art)
    sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID, TARGET]:
        raise ValueError(f"sample_submission 컬럼이 ({ID}, {TARGET}) 이 아님: {list(sub.columns)}")
    print(f" test={len(test)}  submission={len(sub)}  "
          f"segments={test.route_segment.value_counts().to_dict()}")

    print("Inference...")
    out = predict(test, art, verbose=True)

    if args.selfcheck:
        print("Row-independence selfcheck...")
        res = selfcheck(test, art)
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        import json
        (VALIDATION_DIR / "script_selfcheck.json").write_text(json.dumps(res, indent=2))
        if not res["pass"]:
            print("!! 행 독립성 검사 실패 — 제출 금지")
            sys.exit(2)

    print("Build submission...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub = merge_predictions(sub, test[ID].tolist(), out["final"])
    sub.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH} (rows={len(sub)})  elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()