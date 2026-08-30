"""wl_v002 — D arm(recency x RMSE) 재평가. LB 0회, train.csv 불필요.

배경       : wl_v001 의 FAIL 은 2023->2024 단일 전이 판정이었다. source 2023 은 병리적
             연도(전 성분 음수 skill, s* 반전)라 '2023의 최근 월을 더 믿기'가 무익했던
             것일 수 있다. 2024-2025 연속성 가설 하에서는 정상 연도 전이가 2025 의
             더 나은 대리다.
설계       : A(균등 x Logloss, 대조군) vs D(recency x RMSE) 를 정상 전이 2개
             (2020->2021, 2021->2022)와 참조 전이(2023->2024)에서 비교.
             입력·프로토콜은 wl_v001 과 동일 (m1 의 98열, pitcher-half 내부 holdout
             트리 수, 가중치는 source 학습에만, anchor 의 m1 자리 대입 평가).
판정 (사전 등록):
             정상 전이 2개 모두에서  [D - A] (anchor 대입) >= +1.0bp
             AND 2023->2024 에서     [D - A] >= -0.5bp  (비열세 가드)
             -> PASS: deploy(2024 source) 에 recency x RMSE m1 을 적용한 full run 을
                Day 2 제출 후보로 승격. FAIL: D arm 최종 폐기.
최우선 규칙: 평가 데이터 독립 추론 — train 내부 전이만 사용, test 무관.

사용:
  python wl_v002_decision.py
  WL2_SMOKE=1 python wl_v002_decision.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

# =========================================================
# 경로
# =========================================================
EXPERIMENT_NAME = "wl_v002"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
CACHE_MATRICES_DIR = VALIDATION_ROOT / "shift_v001" / "matrices"

TRANSITIONS = [(2020, 2021), (2021, 2022), (2023, 2024)]
NORMAL_TRANSITIONS = [(2020, 2021), (2021, 2022)]
REFERENCE_TRANSITION = (2023, 2024)

SMOKE_MODE = os.environ.get("WL2_SMOKE", "0") == "1"
HALF_LIFE_MONTHS = 3.0

GATE_NORMAL_BP = 1.0
GATE_GUARD_BP = -0.5

RAWNUM_27 = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "score_diff_pitcher_team", "num_runners_on", "home_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
CODES_9 = [
    "code_game_dayofweek", "code_top_bottom", "code_game_type", "code_route_segment",
    "code_base_state", "code_pitcher_hand", "code_batter_hand",
    "code_pitcher_team_id", "code_batter_team_id",
]

BASE_PARAMS = dict(
    iterations=800 if not SMOKE_MODE else 30,
    learning_rate=0.035,
    grow_policy="Lossguide",
    max_leaves=7,
    min_data_in_leaf=350 if not SMOKE_MODE else 40,
    bootstrap_type="Bernoulli",
    subsample=0.9,
    rsm=0.9,
    l2_leaf_reg=6.0,
    border_count=127,
    boosting_type="Plain",
    random_seed=918,
    allow_writing_files=False,
    verbose=200,
)
EARLY_STOPPING_ROUNDS = 80
MIN_TREES = 30

ARMS = {
    "A_uniform_logloss": dict(recency=False, loss="Logloss"),
    "D_recency_rmse": dict(recency=True, loss="RMSE"),
}
CONTROL = "A_uniform_logloss"


def skill_bp(y, p):
    base_var = float(y.mean() * (1.0 - y.mean()))
    return float((1.0 - np.mean((np.clip(p, 0, 1) - y) ** 2) / base_var) * 1e4)


def load_frame(name: str, year: int, prefix: str = "") -> pd.DataFrame:
    z = np.load(CACHE_MATRICES_DIR / f"{name}_{year}.npz", allow_pickle=True)
    return pd.DataFrame(
        np.asarray(z["values"], np.float32),
        columns=[prefix + str(c) for c in z["columns"]],
    )


def build_X(year: int) -> pd.DataFrame:
    rm = load_frame("residual_matrix", year)
    X = pd.concat(
        [
            rm[RAWNUM_27 + CODES_9].reset_index(drop=True),
            load_frame("self_state", year, "ss__"),
            load_frame("gseason", year, "gs__"),
        ],
        axis=1,
    )
    assert X.shape[1] == 98, X.shape
    return X


def recency_weights(month: np.ndarray) -> np.ndarray:
    m = np.asarray(month, np.float64)
    w = 0.5 ** ((np.nanmax(m) - m) / HALF_LIFE_MONTHS)
    w = np.where(np.isfinite(w), w, 1.0)
    return w / w.mean()


def make_model(loss: str, iterations: int):
    params = {k: v for k, v in BASE_PARAMS.items() if k != "iterations"}
    if loss == "Logloss":
        return CatBoostClassifier(loss_function="Logloss", iterations=iterations, **params)
    return CatBoostRegressor(loss_function="RMSE", iterations=iterations, **params)


def predict(model, X) -> np.ndarray:
    if isinstance(model, CatBoostClassifier):
        return np.clip(model.predict_proba(X)[:, 1].astype(np.float64), 0, 1)
    return np.clip(np.asarray(model.predict(X), np.float64), 0, 1)


def fit_arm(cfg, X_s, y_s, w_s, half_s):
    w = w_s if cfg["recency"] else np.ones(len(y_s))
    y_fit = y_s.astype(int) if cfg["loss"] == "Logloss" else y_s.astype(np.float64)
    tr, ev = half_s == 0, half_s == 1

    inner = make_model(cfg["loss"], BASE_PARAMS["iterations"])
    inner.fit(
        X_s.loc[tr], y_fit[tr], sample_weight=w[tr],
        eval_set=(X_s.loc[ev], y_fit[ev]),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    bz = int(inner.get_best_iteration())
    raw = bz + 1 if bz >= 0 else int(inner.tree_count_)
    n_trees = max(int(raw), MIN_TREES)

    model = make_model(cfg["loss"], n_trees)
    model.fit(X_s, y_fit, sample_weight=w)
    return model, {"fixed_tree_count": int(n_trees), "inner_best_raw": int(raw)}


def main():
    assert (CACHE_MATRICES_DIR / "residual_matrix_2020.npz").exists()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    out = {"transitions": {}}
    deltas = {}
    for src, tgt in TRANSITIONS:
        print(f"--- {src} -> {tgt}")
        X_s, X_t = build_X(src), build_X(tgt)
        d_s = np.load(CACHE_MATRICES_DIR / f"data_{src}.npz", allow_pickle=True)
        d_t = np.load(CACHE_MATRICES_DIR / f"data_{tgt}.npz", allow_pickle=True)
        y_s = d_s["target"].astype(np.float64)
        y_t = d_t["target"].astype(np.float64)
        half_s = d_s["pitcher_half_42"].astype(int)
        w_s = recency_weights(X_s["game_month"].to_numpy())

        a_t = np.load(CACHE_MATRICES_DIR / f"anchor_{tgt}.npz")
        p2 = a_t["p_exact_ridge"].astype(np.float64)
        p3 = a_t["p_trend_lgb"].astype(np.float64)
        anchor_bp = skill_bp(y_t, np.clip(a_t["p_base"].astype(np.float64), 0, 1))

        row = {"anchor_bp_cached": anchor_bp}
        for name, cfg in ARMS.items():
            t0 = time.time()
            model, info = fit_arm(cfg, X_s, y_s, w_s, half_s)
            p = predict(model, X_t)
            sub = np.clip((p + p2 + p3) / 3.0, 0, 1)
            row[name] = {
                **cfg, **info,
                "standalone_bp": skill_bp(y_t, p),
                "anchor_substituted_bp": skill_bp(y_t, sub),
                "elapsed_seconds": float(time.time() - t0),
            }
            print(f"  [{name}] trees={info['fixed_tree_count']:>4} "
                  f"단독={row[name]['standalone_bp']:+8.2f} "
                  f"anchor대입={row[name]['anchor_substituted_bp']:+8.2f}")

        delta = (row["D_recency_rmse"]["anchor_substituted_bp"]
                 - row[CONTROL]["anchor_substituted_bp"])
        deltas[(src, tgt)] = delta
        row["delta_D_minus_A_bp"] = delta
        out["transitions"][f"{src}->{tgt}"] = row
        print(f"  Δ(D-A) = {delta:+.2f}bp")

    normal_ok = all(deltas[t] >= GATE_NORMAL_BP for t in NORMAL_TRANSITIONS)
    guard_ok = deltas[REFERENCE_TRANSITION] >= GATE_GUARD_BP
    gate_pass = bool(normal_ok and guard_ok)

    out.update({
        "experiment_name": EXPERIMENT_NAME,
        "smoke_mode": SMOKE_MODE,
        "half_life_months": HALF_LIFE_MONTHS,
        "deltas_bp": {f"{s}->{t}": deltas[(s, t)] for s, t in TRANSITIONS},
        "gate": {
            f"정상 전이 2개 모두 Δ >= +{GATE_NORMAL_BP}bp": normal_ok,
            f"2023->2024 Δ >= {GATE_GUARD_BP}bp": guard_ok,
        },
        "decision_pass": gate_pass,
        "next_step_if_pass": (
            "deploy(2024 source) m1 을 recency x RMSE 로 학습하는 full run "
            "-> selfcheck -> Day 2 제출 후보"
        ),
        "next_step_if_fail": "D arm 최종 폐기 — 잔여 슬롯은 shrink 곡선에 전량",
    })
    with open(VALIDATION_DIR / "decision.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 88)
    print(f"wl_v002 decision — D arm 재평가 (정상 전이 기준)")
    print("=" * 88)
    for s, t in TRANSITIONS:
        tag = "정상" if (s, t) in NORMAL_TRANSITIONS else "참조(병리적 source)"
        print(f"{s}->{t} [{tag}]  Δ(D-A) = {deltas[(s, t)]:+.2f}bp")
    print("-" * 88)
    print(f"DECISION: {'PASS — full run 을 Day 2 후보로 승격' if gate_pass else 'FAIL — D arm 최종 폐기'}")
    print("=" * 88)


if __name__ == "__main__":
    main()