"""V18 Reproduction — 평가 서버 추론 코드 (Phase 7).

./data/test.csv 를 읽어 ./output/submission.csv 를 만든다.

원본        : kaggle_kbo_v18_RECOVERED_FROM_V20_exact.ipynb 의 cell 16 이 생성한 script.py
              (= validation/v18_repro_v001/reference/script_reference.py, Phase 7 ground truth)
학습 코드   : train.py  (EXPERIMENT_NAME = "v18_repro_v001")

zip 루트에는 script.py · requirements.txt · model/ 만 두어야 하므로 자기완결형으로 작성한다.
train.py 를 import 하지 않는다.

────────────────────────────────────────────────────────────────────────
최우선 규칙 — 평가 데이터 독립 추론
────────────────────────────────────────────────────────────────────────
한 test row 의 예측에는 다음만 쓴다.
  · 해당 row 의 입력값
  · 그 row 만으로 계산 가능한 deterministic feature
  · artifact 에 담긴 **공식 train.csv 만으로 만든** frozen 통계 / lookup / model

test 전체 groupby · frequency · 평균/분포/rank/quantile · rolling/lag/cumulative ·
pitcher/team/month/game 단위 집계 · train+test concat 통계 · test 분포 기반 normalization
을 **어디에서도 하지 않는다.**

이 파일 안에서 `test` 전체를 건드리는 연산은 다음뿐이며 전부 row-local 이다.
  · domain3 라우팅        : 그 행의 game_type / pitcher_team_id / batter_team_id
  · domain3 마스크        : 어떤 모델을 적용할지 고르는 것뿐
  · 모델 .predict         : 행별 독립

invariant:  predict(test.iloc[[i]])[0] == predict(test)[i]

────────────────────────────────────────────────────────────────────────
reference 대비 의도적 차이 3가지 (그 외는 verbatim)
────────────────────────────────────────────────────────────────────────
 D1) artifact 를 **단일 pkl** 로 읽는다. reference 는 preprocess.joblib + 모델 12개
     파일을 개별 로드한다. 프로젝트 표준(MODEL_PATH)을 따른 것이며 계산은 동일하다.
 D2) 프로젝트 표준 경로 상수를 쓴다 (DATA_DIR / MODEL_ROOT / MODEL_DIR / MODEL_PATH /
     OUTPUT_DIR / TEST_PATH / SAMPLE_SUB_PATH / OUTPUT_PATH).
 D3) `_lookup` 의 MultiIndex 키를 str 로 강제한다. 근거는 아래 참조.
 D4) 선형 모델(Ridge 2개)의 계수를 float64 로 보장한다. **규칙 준수 조치**이며
     Phase 8 진단에서 batch 크기 의존성(2^-25)이 실측되어 적용했다. 아래 `_load_artifact` 참조.

D3 상세 — 유일한 로직 hardening:
    base_state 는 값이 '___','1__',…,'123' 인데 **'123' 만 순수 숫자**다.
    그 값만 담긴 파일을 읽으면 pandas 가 int64 로 추론하고, object "123" 으로 만들어진
    EB 표와 MultiIndex 키가 어긋나 pc__base_state__* 3열이 **조용히 0.0** 이 된다.
    (T2 검사 D 에서 실측: 0.48936915 -> 0.0)

    V18 원본은 `transform_adapter` 에서 **같은 컬럼에 이미 astype(str) 을 쓰고 있다.**
    따라서 `_lookup` 에 같은 처리를 넣는 것은 이질적 개입이 아니라 내부 일관성 회복이며,
    정상 입력(object dtype)에서는 **값이 전혀 바뀌지 않는다** (str -> str 은 항등).

    조용히 틀린 답을 제출하는 것보다 낫다고 판단해 채택했다.
    dtype 이 예상과 다르면 경고를 남기되 **중단하지는 않는다** (제출 0점 방지).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "v18_repro_v001"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"

# _lookup 키로 쓰이면서 숫자로 오인될 수 있는 컬럼 (D3)
STR_KEY_COLS = ("base_state",)


def _load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"artifact 없음: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        meta = pickle.load(f)
    # D4 (규칙 준수) — 선형 모델 계수를 float64 로 보장한다.
    #
    #   V18 은 Ridge 를 float32 행렬로 적합해 coef_ 가 float32 가 된다.
    #   그러면 sklearn predict 가 float32 BLAS 로 돌고, **batch 크기에 따라**
    #   같은 행의 결과가 1 float32 ulp 흔들린다.
    #     실측: p2 차이 2.980e-08 = 2^-25,  base=(p1+p2+p3)/3 -> 9.934e-09
    #           batch 1·8 에서만 어긋나고 64 이상은 전체 배치와 동일
    #   대회 공지 §3("행 1개일 때와 전체일 때 예측값이 동일해야 한다")에 저촉된다.
    #
    #   계수를 float64 로 올리면 numpy 가 float32 입력을 **정확히** 승격한 뒤
    #   float64 BLAS 로 계산하므로 batch 의존성이 ~1e-16 으로 떨어진다.
    #   train.py 가 이미 float64 로 저장하면 여기서는 아무 일도 하지 않는다(멱등).
    for key in ("base_ridge", "v18_stable_model"):
        m = meta.get(key)
        if m is None:
            continue
        for attr in ("coef_", "intercept_"):
            v = getattr(m, attr, None)
            if v is not None and np.asarray(v).dtype != np.float64:
                setattr(m, attr, np.asarray(v).astype(np.float64))
    return meta


META: dict = _load_artifact()


# =========================================================
# Row-wise features — notebook cell 2 원본
# =========================================================


def _n(frame, c, default=np.nan):
    if c in frame:
        return pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    return np.full(len(frame), default, np.float64)


def build_self_state(frame):
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
    z["self__recent_minus_career"] = (0.5 * p1 + 0.3 * p3 + 0.2 * p5 - career).astype(np.float32)

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
    z["self__recent_x_pressure"] = ((0.5 * p1 + 0.3 * p3 + 0.2 * p5) * pressure).astype(np.float32)
    z["self__offspeed_x_same_hand"] = (off * (ph == bh)).astype(np.float32)
    return z


P_COMPONENT_RATES = [
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
B_COMPONENT_RATES = ["asof_batter_middle_rate"]


def _rounded_cumulative_count(n, rate):
    n = np.asarray(n, np.float64)
    rate = np.asarray(rate, np.float64)
    out = np.zeros(len(n), np.float64)
    ok = (n > 0) & np.isfinite(rate)
    out[ok] = np.rint(n[ok] * rate[ok])
    return out


def transform_global_season(bank, rows):
    """bank 는 artifact 에 담긴 frozen 값이다 (공식 train 만으로 생성).
    나머지는 전부 그 행 자신의 asof_* 컬럼이다. 다른 test 행을 보지 않는다."""
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
    return z


def add_eng(frame):
    x = frame.copy()
    x["eng_recent35"] = (
        0.5 * pd.to_numeric(x.asof_pitcher_prev1_game_success_rate, errors="coerce")
        + 0.3 * pd.to_numeric(x.asof_pitcher_prev3_game_success_rate, errors="coerce")
        + 0.2 * pd.to_numeric(x.asof_pitcher_prev5_game_success_rate, errors="coerce")
    )
    x["eng_recent_delta"] = x.eng_recent35 - pd.to_numeric(
        x.asof_pitcher_success_rate, errors="coerce")
    x["eng_sb"] = pd.to_numeric(x.asof_pitcher_strike_rate, errors="coerce") - pd.to_numeric(
        x.asof_pitcher_ball_rate, errors="coerce")
    x["eng_same"] = (x.pitcher_hand.astype(str) == x.batter_hand.astype(str)).astype(np.float32)
    x["eng_close"] = (
        pd.to_numeric(x.score_diff_pitcher_team, errors="coerce").abs() <= 1).astype(np.float32)
    x["eng_late"] = (pd.to_numeric(x.inning, errors="coerce") >= 7).astype(np.float32)
    return x


def encode_with_maps(frame, maps):
    out = pd.DataFrame(index=np.arange(len(frame)))
    for c, mp in maps.items():
        out["code_" + c] = (
            frame[c].astype("string").fillna("<NA>").astype(str).map(mp).fillna(-1).astype(np.float32)
        )
    return out


# =========================================================
# Layer 0 — base
# =========================================================


def transform_base(test):
    rawnum = META["rawnum"]
    selfx = build_self_state(test)
    gx = transform_global_season(META["target_global_bank"], test)
    B0 = pd.concat(
        [test[rawnum].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
         selfx.reset_index(drop=True), gx.reset_index(drop=True)], axis=1)

    cb = encode_with_maps(test, META["exact_cat_maps"])
    B_exact = pd.concat([B0, cb], axis=1)
    B_exact = B_exact.reindex(columns=META["exact_feature_cols"])
    p1 = META["base_exact_lgb"].predict_proba(B_exact)[:, 1]

    B = B0.reindex(columns=META["ridge_feature_cols"]).replace([np.inf, -np.inf], np.nan)
    med = pd.Series(META["ridge_med"])
    B = B.fillna(med).to_numpy(np.float32)
    mu = np.asarray(META["ridge_mu"], dtype=np.float32)
    sd = np.asarray(META["ridge_sd"], dtype=np.float32)
    B = np.clip((B - mu) / sd, -8, 8)
    p2 = np.clip(META["base_ridge"].predict(B), 0.001, 0.999)

    t = add_eng(test)
    Bnum = t[META["trend_num_cols"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    cb = encode_with_maps(t, META["trend_cat_maps"])
    Bt = pd.concat([Bnum, cb], axis=1).astype(np.float32).reindex(
        columns=META["trend_feature_cols"])
    prior = META["trend_prior"]
    bt = np.array([float(prior[d]) for d in test.domain3.to_numpy()], dtype=np.float64)
    p3 = np.clip(bt + META["base_trend_lgb"].predict(Bt), 0.001, 0.999)
    return np.clip(
        (np.asarray(p1, float) + np.asarray(p2, float) + np.asarray(p3, float)) / 3.0, 0, 1)


# =========================================================
# Layer 1 — V15 residual adapter
# =========================================================


def transform_adapter(rows, base_pred):
    prep = META["adapter_prep"]
    r = add_eng(rows.reset_index(drop=True))
    A = r[prep["cols"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in prep["cols"]:
        A[c] = A[c].fillna(float(prep["med"].get(c, 0.0))).astype(np.float32)
    for c, mp in prep["maps"].items():
        A["code_" + c] = (
            r[c].astype("string").fillna("<NA>").astype(str).map(mp).fillna(-1).astype(np.float32)
        )
    A["base_pred"] = np.asarray(base_pred, dtype=np.float32)
    A = A.reindex(columns=META["adapter_base_feature_cols"])
    if not np.isfinite(A.to_numpy(float)).all():
        raise RuntimeError("non-finite adapter input")
    return A


def _ctx_frame_v15(frame):
    x = frame.copy()
    x["v15_count"] = x["balls_before"].astype(str) + "_" + x["strikes_before"].astype(str)
    x["v15_inning_phase"] = pd.cut(
        pd.to_numeric(x["inning"], errors="coerce"), [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"]).astype(str)
    x["v15_pressure"] = np.where(
        (x["balls_before"] == 3) & (x["strikes_before"] == 2), "full",
        np.where(x["balls_before"] == 3, "three_ball",
                 np.where(x["strikes_before"] == 2, "two_strike", "normal")))
    x["v15_runners"] = np.where(
        pd.to_numeric(x["num_runners_on"], errors="coerce").fillna(0) > 0, "on", "empty")
    x["v15_li_bucket"] = pd.cut(
        pd.to_numeric(x["li"], errors="coerce"), [-np.inf, 0.7, 1.5, np.inf],
        labels=["low", "mid", "high"]).astype(str)
    return x


def _lookup_v15(table, rows, keys, field, default=0.0):
    """D3: MultiIndex 키의 dtype 불일치를 막기 위해 문자열 키를 str 로 맞춘다.

    정상 입력에서는 이미 object 이므로 값이 바뀌지 않는다 (str -> str 항등).
    """
    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)
    left = rows[keys].copy()
    right = table.copy()
    for k in keys:
        if k in STR_KEY_COLS or right[k].dtype == object or left[k].dtype == object:
            left[k] = left[k].astype(str)
            right[k] = right[k].astype(str)
    idx = pd.MultiIndex.from_frame(left)
    s = right.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)


def transform_pitcher_conditional(rows):
    r = _ctx_frame_v15(rows.reset_index(drop=True))
    parts = []
    for spec in META["pitcher_conditional_specs"]:
        t = spec["table"]
        ctx = spec["ctx"]
        prefix = spec["prefix"]
        z = pd.DataFrame(index=np.arange(len(r)))
        keys = ["pitcher_id", ctx]
        for f in ("rate", "reliability", "delta"):
            z[prefix + "__" + f] = _lookup_v15(t, r, keys, f, 0.0)
        parts.append(z)
    out = pd.concat(parts, axis=1).astype(np.float32)
    return out.reindex(columns=META["pitcher_conditional_cols"])


def transform_exact_adapter_block(rows):
    sx = build_self_state(rows).add_prefix("exact__")
    gx = transform_global_season(META["target_global_bank"], rows).add_prefix("exact__")
    out = pd.concat([sx.reset_index(drop=True), gx.reset_index(drop=True)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    return out.reindex(columns=META["exact_adapter_cols"])


def transform_v15_adapter(rows, base_pred):
    base = transform_adapter(rows, base_pred)
    pc = transform_pitcher_conditional(rows)
    ex = transform_exact_adapter_block(rows)
    X = pd.concat([base.reset_index(drop=True), pc.reset_index(drop=True),
                   ex.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=META["adapter_feature_cols"])
    if X.isna().any().any() or not np.isfinite(X.to_numpy(float)).all():
        raise RuntimeError("non-finite V15 adapter input")
    return X


# =========================================================
# Layer 2 — V18 temporal-stable conditional correction
# =========================================================


def _v18_pressure_frame(rows):
    r = rows.copy()
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int)
    r["v18_pressure"] = np.where(
        (balls == 3) & (strikes == 2), "full",
        np.where(balls == 3, "3ball", np.where(strikes == 2, "2strike", "normal")))
    r["v18_pressure_bhand"] = r["v18_pressure"].astype(str) + "|" + r["batter_hand"].astype(str)
    return r


def transform_v18_stable(rows):
    r = _v18_pressure_frame(rows.reset_index(drop=True))
    parts = []
    for spec in META["v18_stable_specs"]:
        t = spec["table"]
        ctx = spec["ctx"]
        prefix = spec["prefix"]
        z = pd.DataFrame(index=np.arange(len(r)))
        de = _lookup_v15(t, r, ["pitcher_id", ctx], "delta", 0.0).astype(np.float32)
        re = _lookup_v15(t, r, ["pitcher_id", ctx], "reliability", 0.0).astype(np.float32)
        z[prefix + "__delta"] = de
        z[prefix + "__weighted_delta"] = (de * re).astype(np.float32)
        z[prefix + "__reliability"] = re
        parts.append(z)
    z = pd.concat(parts, axis=1)
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    for b in range(4):
        z[f"ctx__ball_{b}"] = (balls == b).astype(np.float32)
    for s in range(3):
        z[f"ctx__strike_{s}"] = (strikes == s).astype(np.float32)
    for b in range(4):
        for s in range(3):
            z[f"ctx__count_{b}_{s}"] = ((balls == b) & (strikes == s)).astype(np.float32)
    z = z.reindex(columns=META["v18_stable_cols"]).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    if not np.isfinite(z.to_numpy(float)).all():
        raise RuntimeError("non-finite V18 stable features")
    return z


# =========================================================
# 입력 검증 / 예측 / 제출
# =========================================================


def validate_input(test: pd.DataFrame) -> pd.DataFrame:
    if TARGET in test.columns:
        raise RuntimeError("test.csv unexpectedly contains target")
    required = ["row_id", "season", "pitcher_id", "batter_id", "game_type",
                "pitcher_team_id", "batter_team_id"]
    missing = [c for c in required if c not in test.columns]
    if missing:
        raise RuntimeError(f"missing test columns: {missing}")
    test["season"] = pd.to_numeric(test["season"], errors="raise").astype(int)
    test["game_type"] = test["game_type"].astype("string").fillna("<NA>").astype(str)
    expected = int(META["expected_year"])
    if bool((test.season != expected).any()):
        raise RuntimeError(f"evaluation season differs from expected {expected}")
    if bool((~test.game_type.isin(["R", "F"])).any()):
        raise RuntimeError("unexpected game_type")
    # D3 경고 — 중단하지 않는다. _lookup_v15 가 str 로 맞춰 계산은 정상 진행된다.
    for c in STR_KEY_COLS:
        if c in test.columns and test[c].dtype != object:
            print(f"  [warn] {c} dtype={test[c].dtype} (object 예상). "
                  f"_lookup 키를 str 로 강제해 계속 진행한다.", file=sys.stderr)
    anchor = int(META["anchor_team_id"])
    test["domain3"] = np.where(
        test.game_type.eq("F"), "F",
        np.where((test.pitcher_team_id == anchor) | (test.batter_team_id == anchor),
                 "R_ANCHOR", "R_CORE"))
    return test


def predict_rows(test: pd.DataFrame) -> np.ndarray:
    """행 독립. domain3 마스크는 '어느 모델을 쓸지' 고르는 용도뿐이다."""
    base = transform_base(test)
    X = transform_v15_adapter(test, base)
    corr = np.zeros(len(test), dtype=np.float64)
    seeds = META["adapter_seeds"]
    for seed in seeds:
        c = np.zeros(len(test), dtype=np.float64)
        for d in ["R_CORE", "R_ANCHOR", "F"]:
            mask = test.domain3.eq(d).to_numpy()
            if not mask.any():
                continue
            c[mask] = META["adapter_models"][(int(seed), d)].predict(X.loc[mask])
        corr += c / len(seeds)
    final = np.clip(base + float(META["adapter_shrink"]) * corr, 0, 1)
    if META.get("v18_stable_enabled", False):
        Xs = transform_v18_stable(test)
        arr = Xs.to_numpy(np.float32)
        mu = np.asarray(META["v18_stable_mu"], dtype=np.float32)
        sd = np.asarray(META["v18_stable_sd"], dtype=np.float32)
        arr = np.clip((arr - mu) / sd, -8, 8)
        sc = np.asarray(META["v18_stable_model"].predict(arr), dtype=np.float64)
        core = test.domain3.eq(META.get("v18_route_domain", "R_CORE")).to_numpy()
        final[core] = np.clip(final[core] + float(META["v18_stable_gamma"]) * sc[core], 0, 1)
    return final


def build_submission(test: pd.DataFrame, final: np.ndarray) -> pd.DataFrame:
    if len(final) != len(test) or not np.isfinite(final).all() \
            or (final < 0).any() or (final > 1).any():
        raise RuntimeError("invalid prediction vector")
    out = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: final})
    if SAMPLE_SUB_PATH.exists():
        sample = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
        if list(sample.columns) != [ID, TARGET]:
            raise RuntimeError(f"sample_submission 컬럼 불일치: {list(sample.columns)}")
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    test = validate_input(test)
    final = predict_rows(test)
    out = build_submission(test, final)
    out.to_csv(OUTPUT_PATH, index=False)
    print("V18_TEMPORAL_STABLE_INFERENCE_OK", len(out),
          float(final.min()), float(final.max()), float(final.mean()))


if __name__ == "__main__":
    main()