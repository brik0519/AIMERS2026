"""script.py — dankerTrack_v001 inference

danker_v005 추론 코드에 E6(TRACK 멤버) 를 반영한 판본이다.
구조 상수는 전부 artifact(pkl) 에서 읽는다. 이 파일에는 튜닝 가능한 값이 없다.

Final architecture
------------------
L1 합성 (라우팅을 곱셈 가중치로 — train.py blend_weights / blend 와 동일 정의):

    R_MEMBER = (1 - track_w) * TREATMENT + track_w * TRACK    # E6, R 전용
    p_main   = where(game_type == "F", CONTROL, R_MEMBER)
    w        = where(game_type == "F", w_f, w_r)        # 전부 artifact 동결 상수
    L1       = clip( (1 - w) * p_main + w * DYNAMIC , 0, 1 )

    track_w = 0 (또는 track 블록 없음) 이면 R_MEMBER = TREATMENT 그대로라
    danker_v005 와 비트 단위 동일하다. w_r = 0 이면 v001/v002 라우팅과도 동일.

E6 TRACK 멤버:
    TREATMENT 피처 + 공식 투구추적 로그의 선수별 프로파일 54열.
    프로파일은 train.py 가 공식 로그(2019~2024)와 공식 두 테이블의 통계 결합
    매핑만으로 만들어 artifact 에 동결한 은행이다. 추론은 각 행이 자기
    pitcher_id 로 은행을 **조회만** 한다 — 다른 test 행 참조가 없다.
    은행에 없는 투수는 전 열 NaN + track_hist_missing=1 (학습과 동일 규약).

L1 시드평균 (E5):
    CONTROL / TREATMENT / DYNAMIC 각각 L1_SEEDS 개의 모델을 산술평균한다.
    artifact 의 각 블록은 "models" 리스트를 갖는다.
    모델이 1개면 평균은 항등이므로 시드평균 이전과 비트 동일하다.

L1 R branch 주 예측기:
    TREATMENT CatBoost
    = Control features + R-only Pitcher Conditional FULL (+ SUCCESS state)

L1 F branch 주 예측기:
    CONTROL CatBoost

Dynamic ASOF anchor:
    A_t = (1 - alpha_t) * A_prev + alpha_t * R_t
    alpha_t = lambda * n_t / (n_t + kappa)
    lambda = 1.0, kappa = 100.0

L2:
    ExtraTrees residual adapter.
    적용 범위도 곱셈 가중치다 (train.py l2_weights 와 동일 정의):
        w_l2 = full(adapter_shrink);  if not l2_apply_f: w_l2[F] = 0
        final = clip( L1 + w_l2 * mean_seed(correction) , 0, 1 )
    l2_apply_f = False 이면 v001/v002 의 `final[f_mask] = pred_l1[f_mask]` 와
    비트 단위 동일하다.

미구현 구조 플래그
------------------
artifact 에는 E3/E4 용 플래그(shift_scale, affine_alpha, affine_offset,
add_tables, l1_target_mode)도 동결되지만 이 판본은 그것들을 **구현하지 않는다**.
중립값이 아니면 `assert_supported_flags()` 가 즉시 예외를 던진다.
조용히 무시되는 일이 없도록 하기 위한 장치다.

Evaluation-row independence
---------------------------
A test row uses only:
  * that row's own input / official ASOF values,
  * frozen statistics/tables/models stored in the train-only artifact.

No test-side groupby, rolling, lag, cumulative statistic, frequency, rank,
mean/distribution, or other-row lookup is used to create predictions.

Usage
-----
    python script.py
    python script.py --selfcheck

--selfcheck verifies that selected rows receive exactly the same predictions
when evaluated alone, in the full test, after shuffling, and after split/concat.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 경로 — 프로젝트 표준
# =============================================================================

EXPERIMENT_NAME = "dankerTrack_v001"

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


# =============================================================================
# 공통 상수
# =============================================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

ENG_COLS = [
    "eng_recent35",
    "eng_recent_delta",
    "eng_sb",
    "eng_same",
    "eng_close",
    "eng_late",
]


# =============================================================================
# 기본 utility
# =============================================================================


def prepare_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def numeric_series(frame: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(
            np.full(len(frame), default, dtype=np.float64),
            index=frame.index,
        )
    return pd.to_numeric(frame[col], errors="coerce")


def add_eng(frame: pd.DataFrame) -> pd.DataFrame:
    """train.py L2 adapter와 동일한 row-local engineered features."""
    x = frame.copy()

    career = numeric_series(x, "asof_pitcher_success_rate", 0.5)
    p1 = numeric_series(x, "asof_pitcher_prev1_game_success_rate", np.nan)
    p3 = numeric_series(x, "asof_pitcher_prev3_game_success_rate", np.nan)
    p5 = numeric_series(x, "asof_pitcher_prev5_game_success_rate", np.nan)

    p1 = p1.fillna(career)
    p3 = p3.fillna(p1)
    p5 = p5.fillna(p3)

    x["eng_recent35"] = 0.5 * p1 + 0.3 * p3 + 0.2 * p5
    x["eng_recent_delta"] = x["eng_recent35"] - career
    x["eng_sb"] = (
        numeric_series(x, "asof_pitcher_strike_rate", np.nan)
        - numeric_series(x, "asof_pitcher_ball_rate", np.nan)
    )

    if "pitcher_hand" in x.columns and "batter_hand" in x.columns:
        x["eng_same"] = (
            x["pitcher_hand"]
            .astype(str)
            .eq(x["batter_hand"].astype(str))
            .astype(np.float32)
        )
    else:
        x["eng_same"] = 0.0

    x["eng_close"] = (
        numeric_series(x, "score_diff_pitcher_team", np.nan)
        .abs()
        .le(1)
        .astype(np.float32)
    )
    x["eng_late"] = (
        numeric_series(x, "inning", np.nan)
        .ge(7)
        .astype(np.float32)
    )
    return x


# =============================================================================
# Artifact load / validation
# =============================================================================


# =============================================================================
# E3c — SHIFT 설계행렬 (train.py build_shift_matrix 와 **동일한 정의**)
# =============================================================================
#
# 30열 전부 현재 행의 공식 컬럼만 쓴다. 다른 test 행을 참조하는 항이 없다.
# median / mu / sd / beta 는 artifact 에서만 온다 — test 분포에서 재추정하지 않는다.

SHIFT_COLS = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "num_runners_on", "score_diff_pitcher_team", "home_win_expectancy", "li",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "log_pn", "log_bn",
    "recent35", "recent_delta", "sb",
    "same_hand", "two_strike", "three_ball", "runners", "late", "close",
]


SHIFT_SOURCE_COLS = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "num_runners_on", "score_diff_pitcher_team", "home_win_expectancy", "li",
    "asof_pitcher_n", "asof_batter_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "pitcher_hand", "batter_hand",
]


def assert_shift_source_columns(frame) -> None:
    """설계행렬의 원천 컬럼이 전부 있는지 확인한다.

    numeric_series 는 컬럼이 없으면 조용히 NaN 을 돌려준다. 그러면 median 으로
    채워져 **에러 없이 틀린** 예측이 나온다 — 가장 추적하기 어려운 실패다.
    """
    missing = [c for c in SHIFT_SOURCE_COLS if c not in frame.columns]
    if missing:
        raise KeyError(f"SHIFT 원천 컬럼 누락: {missing}")


def build_shift_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """E3c 민감도 설계행렬. 결측은 채우지 않고 NaN 으로 돌려준다."""
    assert_shift_source_columns(frame)
    x = pd.DataFrame(index=frame.index)

    def num(col):
        return numeric_series(frame, col)

    for col in [
        "game_month", "inning", "balls_before", "strikes_before", "outs_before",
        "num_runners_on", "score_diff_pitcher_team", "home_win_expectancy", "li",
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_success_rate", "asof_batter_middle_rate",
    ]:
        x[col] = num(col)

    x["log_pn"] = np.log1p(num("asof_pitcher_n").clip(lower=0))
    x["log_bn"] = np.log1p(num("asof_batter_n").clip(lower=0))

    career = num("asof_pitcher_success_rate").fillna(0.5)
    p1 = num("asof_pitcher_prev1_game_success_rate").fillna(career)
    p3 = num("asof_pitcher_prev3_game_success_rate").fillna(p1)
    p5 = num("asof_pitcher_prev5_game_success_rate").fillna(p3)
    x["recent35"] = 0.5 * p1 + 0.3 * p3 + 0.2 * p5
    x["recent_delta"] = x["recent35"] - career
    x["sb"] = num("asof_pitcher_strike_rate") - num("asof_pitcher_ball_rate")

    if "pitcher_hand" not in frame.columns or "batter_hand" not in frame.columns:
        raise KeyError("SHIFT: pitcher_hand / batter_hand 가 test 에 없다")
    x["same_hand"] = (
        frame["pitcher_hand"].astype(str) == frame["batter_hand"].astype(str)
    ).astype(float)

    x["two_strike"] = (num("strikes_before") == 2).astype(float)
    x["three_ball"] = (num("balls_before") == 3).astype(float)
    x["runners"] = (num("num_runners_on") > 0).astype(float)
    x["late"] = (num("inning") >= 7).astype(float)
    x["close"] = (num("score_diff_pitcher_team").abs() <= 1).astype(float)

    x = x.replace([np.inf, -np.inf], np.nan)
    if list(x.columns) != SHIFT_COLS:
        raise RuntimeError(
            f"SHIFT 열 순서 불일치\n  기대: {SHIFT_COLS}\n  실제: {list(x.columns)}"
        )
    return x.astype(np.float64)


def compute_shift_correction(test: pd.DataFrame, art: dict) -> np.ndarray:
    """행별 SHIFT 보정량. artifact 의 median/mu/sd/beta 만 쓴다."""
    cols = list(art["shift_cols"])
    if cols != SHIFT_COLS:
        raise RuntimeError(
            "SHIFT 열 정의가 artifact 와 다르다 — train.py 와 script.py 가 어긋났다\n"
            f"  artifact: {cols}\n  script  : {SHIFT_COLS}"
        )
    mat = build_shift_matrix(test).to_numpy(dtype=np.float64)

    median = np.asarray(art["shift_median"], dtype=np.float64)
    mu = np.asarray(art["sens_mu"], dtype=np.float64)
    sd = np.asarray(art["sens_sd"], dtype=np.float64)
    beta = np.asarray(art["shift_beta"], dtype=np.float64)
    for name, vec in (("shift_median", median), ("sens_mu", mu),
                      ("sens_sd", sd), ("shift_beta", beta)):
        if vec.shape != (len(SHIFT_COLS),):
            raise RuntimeError(f"{name} 길이 {vec.shape} != {len(SHIFT_COLS)}")

    filled = np.where(np.isnan(mat), median, mat)
    if not np.isfinite(filled).all():
        raise RuntimeError("SHIFT 행렬에 median 으로도 못 채운 NaN/Inf 가 있다")

    sd_safe = np.where(np.abs(sd) < 1e-12, 1.0, sd)
    clip = float(art.get("shift_clip", 8.0))
    z = np.clip((filled - mu) / sd_safe, -clip, clip)
    corr = z @ beta
    if not np.isfinite(corr).all():
        raise RuntimeError("SHIFT 보정에 NaN/Inf")
    return corr


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    art = joblib.load(path)
    if not isinstance(art, dict):
        raise TypeError("MODEL_PATH artifact must be dict")

    if art.get("experiment_name") != EXPERIMENT_NAME:
        raise RuntimeError(
            f"artifact experiment={art.get('experiment_name')!r}; "
            f"expected={EXPERIMENT_NAME!r}"
        )

    required_top = {
        "control",
        "treatment",
        "dynamic",
        "shared_l1_state",
        "l1_routing",
        "l2",
        "l2_routing",
        "expected_target_year",
        # --- train.py freeze() 가 동결하는 구조 상수 -------------------------
        # 기본값을 주지 않는다. 구 artifact(v001/v002)를 이 script 로 돌리면
        # 조용히 W_R=0 으로 되돌아가는 대신 여기서 죽어야 한다.
        "w_r",
        "l2_apply_f",
        "shift_scale",
        "track_enabled",
        "track_w",
    }
    if float(art.get("shift_scale", 0.0)) != 0.0:
        required_top |= {
            "shift_beta", "shift_median", "shift_cols",
            "sens_mu", "sens_sd", "shift_apply_f",
        }
    missing = sorted(required_top - set(art))
    if missing:
        raise KeyError(f"artifact required keys missing: {missing}")

    # E6 일관성 — track 플래그와 블록이 서로 맞아야 한다.
    if bool(art["track_enabled"]) and float(art["track_w"]) != 0.0:
        if not isinstance(art.get("track"), dict):
            raise KeyError("track_enabled 인데 artifact['track'] 블록이 없다")
        tw_doc = art.get("l1_routing", {}).get("track_w")
        if tw_doc is not None and float(tw_doc) != float(art["track_w"]):
            raise RuntimeError(
                f"artifact track_w 불일치: top-level={float(art['track_w'])} "
                f"vs l1_routing={float(tw_doc)}"
            )
        need_tk = {"models", "features", "cat_cols", "track_cols", "bank_target"}
        miss_tk = sorted(need_tk - set(art["track"]))
        if miss_tk:
            raise KeyError(f"artifact['track'] 필수 키 누락: {miss_tk}")
    if not (0.0 <= float(art["track_w"]) <= 0.5):
        raise ValueError(f"track_w out of [0, 0.5]: {art['track_w']}")

    assert_supported_flags(art)
    force_deterministic_forests(art)
    return art


def force_deterministic_forests(art: dict) -> None:
    """L2 ExtraTrees 의 예측을 단일 스레드로 고정한다 — 행 독립 결정론.

    sklearn 의 forest predict 는 트리별 예측을 공유 배열에 누산한다.
        Parallel(n_jobs=self.n_jobs)(delayed(_accumulate_prediction)(...) for e in estimators_)
    n_jobs > 1 이면 **누산 순서가 스레드 스케줄링에 따라 달라진다.**
    부동소수 덧셈은 결합법칙을 만족하지 않으므로, 같은 행이라도 호출을 나눠서 하면
    마지막 비트가 흔들린다 (실측 8.674e-19).

    이것은 다른 test 행의 정보가 새는 것이 아니라 **누산 순서 문제**다.
    그러나 selfcheck 의 정확한 0 기준을 지키기 위해, 그리고 코드 검증 시
    오해를 없애기 위해 추론에서는 항상 단일 스레드로 고정한다.

    train.py 는 N_JOBS(최대 6)로 학습하므로 artifact 의 모델이 그 값을 갖고 온다.
    v18 계열도 같은 처리를 한다 (`et.n_jobs = 1  # 행 독립 결정론`).
    """
    fixed = 0
    for model in (art.get("l2", {}).get("adapter_models") or {}).values():
        if getattr(model, "n_jobs", 1) != 1:
            model.n_jobs = 1
            fixed += 1
    if fixed:
        print(f"L2 forests : n_jobs -> 1 ({fixed}개, 행 독립 결정론)")


def assert_supported_flags(art: dict) -> None:
    """이 script 가 구현하지 않은 구조 플래그가 켜져 있으면 즉시 죽는다.

    train.py 의 freeze() 는 E3/E4 용 상수도 artifact 에 담는다. 그것을 여기서
    조용히 무시하면 로컬 검증과 제출 결과가 갈라진다 — 에러 없이 틀리는,
    가장 추적하기 어려운 실패다. 그래서 미구현 축은 전부 예외로 막는다.
    """
    unsupported = []

    # shift_scale (E3c) 는 이 판본에서 구현됐다. 다만 F 적용은 아직 미구현이다 —
    # SHIFT 는 R 도메인에서만 검증됐으므로(46_*) F 로 넓히려면 재검증이 필요하다.
    if float(art.get("shift_scale", 0.0)) != 0.0 and bool(art.get("shift_apply_f", False)):
        unsupported.append("shift_apply_f=True (F 도메인 미검증·미구현)")
    if float(art.get("affine_alpha", 1.0)) != 1.0:
        unsupported.append(f"affine_alpha={art['affine_alpha']} (E3b 미구현)")
    if float(art.get("affine_offset", 0.0)) != 0.0:
        unsupported.append(f"affine_offset={art['affine_offset']} (38_* §5: 영구 0)")
    if len(art.get("add_tables", ()) or ()) > 0:
        unsupported.append(f"add_tables n={len(art['add_tables'])} (E3d 미구현)")
    if str(art.get("l1_target_mode", "direct")) != "direct":
        unsupported.append(f"l1_target_mode={art['l1_target_mode']!r} (E4 미구현)")

    if unsupported:
        raise NotImplementedError(
            "artifact 가 이 script.py 미구현 구조를 요구한다:\n  - "
            + "\n  - ".join(unsupported)
            + "\nscript.py 를 먼저 갱신하라. 무시하고 추론하면 로컬과 제출이 갈라진다."
        )


def validate_test(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    expected_year = int(art["expected_target_year"])

    required = set(art["shared_l1_state"]["base_features"]) | {
        ID,
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }
    missing = sorted(required - set(test.columns))
    if missing:
        raise KeyError(f"test required columns missing: {missing}")

    season = pd.to_numeric(test[SEASON_COL], errors="raise").astype(int)
    invalid = ~season.eq(expected_year)
    if invalid.any():
        bad = sorted(season.loc[invalid].drop_duplicates().tolist())
        raise RuntimeError(
            f"evaluation season differs from expected {expected_year}: {bad}"
        )

    if test[ID].duplicated().any():
        raise RuntimeError("test row_id contains duplicates")

    return test.reset_index(drop=True)


# =============================================================================
# Exact-ASOF5 — frozen official train history + current row only
# =============================================================================


def add_pitcher_exact_asof_features(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    """Reconstruct current-season state for one fixed evaluation year.

    The historical aggregate comes only from artifact-stored official train
    statistics. No other evaluation row is referenced.
    """
    required = {
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise KeyError(f"Exact-ASOF required columns missing: {missing}")

    out = rows.copy()

    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL], errors="raise"
    ).astype(int)

    hist = stats.loc[stats[SEASON_COL] < int(target_year)]
    if len(hist) > 0:
        hist_agg = (
            hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]]
            .sum()
        )
        hist_n = (
            out["pitcher_id"]
            .map(hist_agg["n"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        hist_success = (
            out["pitcher_id"]
            .map(hist_agg["success"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    else:
        hist_n = np.zeros(len(out), dtype=np.float64)
        hist_success = np.zeros(len(out), dtype=np.float64)

    asof_n = (
        pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    asof_rate = pd.to_numeric(
        out["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    asof_success = asof_n * asof_rate
    raw_current_n = asof_n - hist_n
    current_n = np.maximum(raw_current_n, 0.0)
    raw_current_success = asof_success - hist_success

    current_success = np.where(
        np.isfinite(raw_current_success),
        np.clip(raw_current_success, 0.0, current_n),
        np.nan,
    )

    current_rate = np.full(len(out), np.nan, dtype=np.float64)
    valid_current = (current_n > 0.0) & np.isfinite(current_success)
    current_rate[valid_current] = (
        current_success[valid_current] / current_n[valid_current]
    )

    past_rate = np.full(len(out), np.nan, dtype=np.float64)
    valid_past = hist_n > 0.0
    past_rate[valid_past] = hist_success[valid_past] / hist_n[valid_past]

    out["exact_pitcher_season_n"] = current_n
    out["exact_pitcher_season_logn"] = np.log1p(current_n)
    out["exact_pitcher_season_success_rate"] = current_rate
    out["exact_pitcher_past_success_rate"] = past_rate
    out["exact_pitcher_season_minus_past"] = current_rate - past_rate

    return out


def exact5_matrix(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    transformed = add_pitcher_exact_asof_features(
        rows=rows,
        pitcher_season_stats=pitcher_season_stats,
        target_year=target_year,
    )
    exact = (
        transformed[EXACT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )
    if not np.isfinite(exact.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Exact-ASOF5 matrix contains NaN/Inf")
    return exact


# =============================================================================
# Frozen HB-EB transform
# =============================================================================


def apply_frozen_hbeb(
    df: pd.DataFrame,
    state: dict,
) -> np.ndarray:
    group_cols = list(state["group_cols"])
    n_col = state["n_col"]
    rate_col = state["rate_col"]
    prior_df = state["prior_df"]
    global_prior = tuple(state["global_prior"])

    lookup = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )
    if len(lookup) != len(df):
        raise RuntimeError("HB-EB frozen lookup changed row count")

    alpha = (
        lookup["alpha"]
        .fillna(float(global_prior[0]))
        .to_numpy(dtype=np.float64)
    )
    beta = (
        lookup["beta"]
        .fillna(float(global_prior[1]))
        .to_numpy(dtype=np.float64)
    )

    n = (
        pd.to_numeric(df[n_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    rate = (
        pd.to_numeric(df[rate_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.float64)
    )

    success = n * rate
    denominator = n + alpha + beta
    result = np.divide(
        success + alpha,
        denominator,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return result


def build_control_view(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    target_year = int(art["expected_target_year"])
    shared = art["shared_l1_state"]

    exact = add_pitcher_exact_asof_features(
        rows=test,
        pitcher_season_stats=shared["pitcher_season_stats"],
        target_year=target_year,
    )

    out = exact.copy()
    for new_col, state in shared["hbeb_priors"].items():
        out[new_col] = apply_frozen_hbeb(out, state)

    features = list(art["control"]["features"])
    missing = sorted(set(features) - set(out.columns))
    if missing:
        raise KeyError(f"control transformed columns missing: {missing}")

    return out


# =============================================================================
# Pitcher Conditional — frozen previous-season tables only
# =============================================================================


def _ctx_frame_v15(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()

    x["v15_count"] = (
        x["balls_before"].astype(str)
        + "_"
        + x["strikes_before"].astype(str)
    )
    x["v15_inning_phase"] = pd.cut(
        pd.to_numeric(x["inning"], errors="coerce"),
        [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype(str)
    x["v15_pressure"] = np.where(
        (x["balls_before"] == 3) & (x["strikes_before"] == 2),
        "full",
        np.where(
            x["balls_before"] == 3,
            "three_ball",
            np.where(x["strikes_before"] == 2, "two_strike", "normal"),
        ),
    )
    x["v15_runners"] = np.where(
        pd.to_numeric(x["num_runners_on"], errors="coerce").fillna(0) > 0,
        "on",
        "empty",
    )
    x["v15_li_bucket"] = pd.cut(
        pd.to_numeric(x["li"], errors="coerce"),
        [-np.inf, 0.7, 1.5, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)
    return x


def _pc_lookup(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    keys: list[str],
    field: str,
    default: float = 0.0,
) -> np.ndarray:
    if len(table) == 0:
        return np.full(len(rows), default, dtype=np.float32)

    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return (
            rows[keys[0]]
            .map(s)
            .fillna(default)
            .to_numpy(dtype=np.float32)
        )

    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(dtype=np.float32)


def build_frozen_pc(test: pd.DataFrame, art: dict) -> pd.DataFrame:
    shared = art["shared_l1_state"]
    target_year = int(art["expected_target_year"])
    source_year = target_year - 1

    tables_by_year = shared["pc_tables_by_source_season"]
    if source_year not in tables_by_year:
        raise KeyError(
            f"PC source season {source_year} missing from artifact; "
            f"available={sorted(tables_by_year)}"
        )

    rows = _ctx_frame_v15(test.reset_index(drop=True))
    contexts = shared["pc_contexts"]
    fields = list(shared["pc_fields"])
    default = float(shared.get("pc_unseen_default", 0.0))

    pc_features = list(art["treatment"].get("pc_features", []))
    out = pd.DataFrame(
        0.0,
        index=np.arange(len(rows)),
        columns=pc_features,
        dtype=np.float32,
    )

    season_tables = tables_by_year[source_year]
    for spec in contexts:
        context = spec["context"]
        if context not in season_tables:
            raise KeyError(f"PC context {context!r} missing in artifact")
        table = season_tables[context]
        keys = ["pitcher_id", context]
        for field in fields:
            col = f"pc__{context}__{field}"
            if col not in out.columns:
                raise KeyError(f"PC feature {col!r} missing from treatment schema")
            out[col] = _pc_lookup(
                table=table,
                rows=rows,
                keys=keys,
                field=field,
                default=default,
            )

    # train.py treatment uses PC only for R; F is explicitly neutralized.
    f_mask = (
        test[GAME_TYPE_COL]
        .fillna("__MISSING__")
        .astype(str)
        .eq("F")
        .to_numpy()
    )
    if f_mask.any():
        out.loc[f_mask, pc_features] = 0.0

    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("frozen PITCHER_CONDITIONAL contains NaN/Inf")

    return out


# =============================================================================
# danker_sj_v001: SJ v042 SUCCESS_FULL 추론 (원문 이식)
# =============================================================================

SEASON = SEASON_COL          # SJ 코드 호환 별칭
GAME_TYPE = GAME_TYPE_COL

def assert_required_columns(
    df: pd.DataFrame,
    required,
    name: str,
):
    missing = sorted(
        set(required)
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"{name} required columns missing: {missing}"
        )


def rounded_cumulative_count(
    n,
    rate,
):
    """
    Must match train.py SUCCESS_FULL semantics exactly:
        cumulative success ~= round(n * rate)
    """
    n = np.asarray(
        n,
        dtype=np.float64,
    )

    rate = np.asarray(
        rate,
        dtype=np.float64,
    )

    out = np.zeros(
        len(n),
        dtype=np.float64,
    )

    ok = (
        (n > 0.0)
        & np.isfinite(rate)
    )

    out[ok] = np.rint(
        n[ok]
        * rate[ok]
    )

    return out


def _build_frozen_success_bank(
    success_history: dict,
    entity: str,
    target_year: int,
):
    """
    Aggregate only frozen TRAIN season-level statistics
    for seasons < target_year.
    """

    stats = (
        success_history[
            entity
        ]
        .copy()
    )

    id_col = (
        "pitcher_id"
        if entity == "pitcher"
        else "batter_id"
    )

    assert_required_columns(
        stats,
        {
            id_col,
            SEASON,
            "n",
            "success",
        },
        f"{entity} success history artifact",
    )

    stats[SEASON] = (
        pd.to_numeric(
            stats[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    hist = (
        stats.loc[
            stats[SEASON]
            < int(target_year)
        ]
    )

    if len(hist) == 0:
        return {
            "n": {},
            "success": {},
            "prior": 0.5,
        }

    agg = (
        hist
        .groupby(
            id_col,
            sort=False,
            dropna=False,
        )[
            ["n", "success"]
        ]
        .sum()
    )

    max_hist_year = int(
        hist[SEASON]
        .max()
    )

    season_prior = (
        success_history
        .get(
            "season_prior",
            {}
        )
    )

    prior = (
        season_prior
        .get(
            max_hist_year,
            0.5,
        )
    )

    prior = float(
        prior
    )

    if not np.isfinite(
        prior
    ):
        prior = 0.5

    return {
        "n": (
            agg[
                "n"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "success": (
            agg[
                "success"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "prior": prior,
    }


def _transform_success_state(
    rows: pd.DataFrame,
    bank: dict,
    entity: str,
    success_spec: dict,
    out: pd.DataFrame,
):
    spec = (
        success_spec[
            entity
        ]
    )

    id_col = (
        spec[
            "id_col"
        ]
    )

    n_col = (
        spec[
            "n_col"
        ]
    )

    rate_col = (
        spec[
            "rate_col"
        ]
    )

    assert_required_columns(
        rows,
        {
            id_col,
            n_col,
            rate_col,
        },
        f"{entity} SUCCESS state test",
    )

    ids = (
        rows[
            id_col
        ]
    )

    hist_n = (
        ids
        .map(
            bank[
                "n"
            ]
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    hist_success = (
        ids
        .map(
            bank[
                "success"
            ]
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    asof_n = (
        pd.to_numeric(
            rows[
                n_col
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    career = (
        pd.to_numeric(
            rows[
                rate_col
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    prior = float(
        bank[
            "prior"
        ]
    )

    career = np.where(
        np.isfinite(
            career
        ),
        career,
        prior,
    )

    cumulative_success = (
        rounded_cumulative_count(
            asof_n,
            career,
        )
    )

    current_n = np.maximum(
        asof_n
        - hist_n,
        0.0,
    )

    current_success = np.clip(
        cumulative_success
        - hist_success,
        0.0,
        current_n,
    )

    raw = np.divide(
        current_success,
        current_n,
        out=np.full(
            len(rows),
            prior,
            dtype=np.float64,
        ),
        where=(
            current_n
            > 0.0
        ),
    )

    prefix = (
        f"gseason__{entity}"
    )

    for k in (
        spec[
            "strengths"
        ]
    ):
        out[
            f"{prefix}_rate_k{int(k)}"
        ] = (
            (
                current_success
                + float(k)
                * prior
            )
            / (
                current_n
                + float(k)
            )
        ).astype(
            np.float32
        )

    out[
        f"{prefix}_raw"
    ] = (
        raw.astype(
            np.float32
        )
    )

    out[
        f"{prefix}_log_n"
    ] = (
        np.log1p(
            current_n
        )
        .astype(
            np.float32
        )
    )

    reliability_strength = float(
        spec[
            "reliability_strength"
        ]
    )

    out[
        f"{prefix}_reliability"
    ] = (
        current_n
        / (
            current_n
            + reliability_strength
        )
    ).astype(
        np.float32
    )

    out[
        f"{prefix}_minus_career"
    ] = (
        raw
        - career
    ).astype(
        np.float32
    )

    return {
        "k80": (
            out[
                f"{prefix}_rate_k80"
            ]
            .to_numpy(
                dtype=np.float64
            )
        ),
    }


def build_success_test_features(
    test: pd.DataFrame,
    success_history: dict,
    success_spec: dict,
    success_features: list[str],
) -> pd.DataFrame:
    """
    Reconstruct current-season pitcher+batter success state.

    Historical terms are frozen train-derived statistics.
    Current terms use only this row's official cumulative asof_* values.
    """

    assert_required_columns(
        test,
        {
            SEASON,
            "pitcher_id",
            "batter_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_batter_n",
            "asof_batter_success_rate",
        },
        "SUCCESS_FULL test",
    )

    out = pd.DataFrame(
        0.0,
        index=test.index,
        columns=success_features,
        dtype=np.float32,
    )

    seasons = (
        pd.to_numeric(
            test[
                SEASON
            ],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        mask = (
            seasons
            .eq(
                int(target_year)
            )
        )

        rows = (
            test.loc[
                mask
            ]
            .copy()
        )

        z = pd.DataFrame(
            index=rows.index
        )

        pitcher_bank = (
            _build_frozen_success_bank(
                success_history=(
                    success_history
                ),
                entity="pitcher",
                target_year=int(
                    target_year
                ),
            )
        )

        batter_bank = (
            _build_frozen_success_bank(
                success_history=(
                    success_history
                ),
                entity="batter",
                target_year=int(
                    target_year
                ),
            )
        )

        pitcher_state = (
            _transform_success_state(
                rows=rows,
                bank=pitcher_bank,
                entity="pitcher",
                success_spec=(
                    success_spec
                ),
                out=z,
            )
        )

        batter_state = (
            _transform_success_state(
                rows=rows,
                bank=batter_bank,
                entity="batter",
                success_spec=(
                    success_spec
                ),
                out=z,
            )
        )

        z[
            "gseason__pitcher_minus_batter_k80"
        ] = (
            pitcher_state[
                "k80"
            ]
            - batter_state[
                "k80"
            ]
        ).astype(
            np.float32
        )

        missing = sorted(
            set(
                success_features
            )
            - set(
                z.columns
            )
        )

        if missing:
            raise KeyError(
                "SUCCESS_FULL feature generation mismatch: "
                f"{missing}"
            )

        z = (
            z[
                success_features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
            .astype(
                np.float32
            )
        )

        out.loc[
            rows.index,
            success_features,
        ] = (
            z[
                success_features
            ]
        )

    values = (
        out[
            success_features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "SUCCESS_FULL test features contain NaN/Inf."
        )

    return out


def build_treatment_view(
    control_view: pd.DataFrame,
    test: pd.DataFrame,
    art: dict,
) -> pd.DataFrame:
    out = control_view.copy()
    pc = build_frozen_pc(test, art)

    for col in pc.columns:
        if col in out.columns:
            raise RuntimeError(f"duplicate PC feature at inference: {col}")
        out[col] = pc[col].to_numpy()

    # --- danker_sj_v001: SUCCESS_FULL 13 ---
    shared = art["shared_l1_state"]
    success = build_success_test_features(
        test=test,
        success_history=shared["success_history"],
        success_spec=shared["success_spec"],
        success_features=list(shared["success_features"]),
    )
    for col in success.columns:
        if col in out.columns:
            raise RuntimeError(f"duplicate SUCCESS feature at inference: {col}")
        out[col] = success[col].to_numpy()

    features = list(art["treatment"]["features"])
    missing = sorted(set(features) - set(out.columns))
    if missing:
        raise KeyError(f"treatment transformed columns missing: {missing}")

    return out


# =============================================================================
# E6 TRACK — 동결 프로파일 은행 lookup (row-local)
# =============================================================================


def build_track_view(
    treatment_view: pd.DataFrame,
    test: pd.DataFrame,
    art: dict,
) -> pd.DataFrame:
    """TRACK 멤버 입력 = TREATMENT 뷰 + 동결 은행의 프로파일 54열.

    은행은 train.py 가 공식 로그(2019~2024)로 만들어 artifact 에 담은 것이고,
    여기서는 각 행이 자기 pitcher_id 하나로 **조회만** 한다. 다른 test 행을
    보지 않으므로 행 독립이 자명하고, selfcheck 가 이를 실측한다.
    """
    tk = art["track"]
    track_cols = list(tk["track_cols"])
    bank = tk["bank_target"]
    if not isinstance(bank, pd.DataFrame):
        raise TypeError("artifact['track']['bank_target'] must be DataFrame")
    if list(bank.columns) != track_cols:
        raise RuntimeError("TRACK bank 컬럼 순서가 track_cols 와 다르다")

    pids = pd.to_numeric(test["pitcher_id"], errors="raise").astype("int64")
    if len(bank) == 0:
        mat = pd.DataFrame(np.nan, index=test.index, columns=track_cols)
        mat["track_hist_missing"] = 1.0
    else:
        mat = bank.reindex(pids.to_numpy())
        mat.index = test.index
        found = mat["track_hist_n_log"].notna()
        mat["track_hist_missing"] = np.where(found, 0.0, 1.0)

    overlap = set(track_cols) & set(treatment_view.columns)
    if overlap:
        raise RuntimeError(f"TRACK 열이 TREATMENT 뷰와 겹친다: {sorted(overlap)}")

    out = pd.concat(
        [treatment_view.reset_index(drop=True), mat.reset_index(drop=True)],
        axis=1,
    )
    features = list(tk["features"])
    missing = sorted(set(features) - set(out.columns))
    if missing:
        raise KeyError(f"track member columns missing: {missing}")
    return out


# =============================================================================
# Dynamic ASOF anchor — frozen train history + current row only
# =============================================================================


def league_prev_for(
    target_year: int,
    season_league_means: dict,
    fallback_rate: float,
) -> float:
    prev = int(target_year) - 1

    # joblib preserves integer keys, but tolerate stringified keys defensively.
    if prev in season_league_means:
        return float(season_league_means[prev])
    if str(prev) in season_league_means:
        return float(season_league_means[str(prev)])

    older = []
    for year, value in season_league_means.items():
        try:
            y = int(year)
        except Exception:
            continue
        if y < int(target_year):
            older.append(float(value))

    if older:
        return float(np.mean(older))
    return float(fallback_rate)


def compute_prev_anchor(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    season_league_means: dict,
    target_year: int,
    anchor_strength: float,
    fallback_rate: float,
) -> np.ndarray:
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(
        stats[SEASON_COL], errors="raise"
    ).astype(int)

    hist = stats.loc[stats[SEASON_COL] < int(target_year)]
    if len(hist) > 0:
        agg = (
            hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]]
            .sum()
        )
        hist_n = (
            rows["pitcher_id"]
            .map(agg["n"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        hist_success = (
            rows["pitcher_id"]
            .map(agg["success"])
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    else:
        hist_n = np.zeros(len(rows), dtype=np.float64)
        hist_success = np.zeros(len(rows), dtype=np.float64)

    league_prev = league_prev_for(
        target_year=target_year,
        season_league_means=season_league_means,
        fallback_rate=fallback_rate,
    )

    anchor = (
        hist_success + float(anchor_strength) * league_prev
    ) / (
        hist_n + float(anchor_strength)
    )

    if not np.isfinite(anchor).all():
        raise RuntimeError("Previous anchor contains NaN/Inf")
    return np.clip(anchor, 0.0, 1.0)


def compute_dynamic_anchor(
    exact_view: pd.DataFrame,
    prev_anchor: np.ndarray,
    dynamic_lambda: float,
    dynamic_kappa: float,
) -> np.ndarray:
    prev_anchor = np.asarray(prev_anchor, dtype=np.float64)
    if len(prev_anchor) != len(exact_view):
        raise RuntimeError("Dynamic anchor length mismatch")

    n_t = (
        pd.to_numeric(
            exact_view["exact_pitcher_season_n"], errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    r_t = pd.to_numeric(
        exact_view["exact_pitcher_season_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    valid = (n_t > 0.0) & np.isfinite(r_t)
    reliability = np.zeros(len(exact_view), dtype=np.float64)
    reliability[valid] = n_t[valid] / (
        n_t[valid] + float(dynamic_kappa)
    )
    alpha = np.clip(
        float(dynamic_lambda) * reliability,
        0.0,
        1.0,
    )

    current_state = prev_anchor.copy()
    current_state[valid] = np.clip(r_t[valid], 0.0, 1.0)
    dynamic = (1.0 - alpha) * prev_anchor + alpha * current_state

    if not np.isfinite(dynamic).all():
        raise RuntimeError("Dynamic anchor contains NaN/Inf")
    return np.clip(dynamic, 0.0, 1.0)


# =============================================================================
# L2 adapter transform
# =============================================================================


def transform_adapter_core(
    prep: dict,
    rows: pd.DataFrame,
    base_pred: np.ndarray,
) -> pd.DataFrame:
    r = add_eng(rows.reset_index(drop=True))

    A = (
        r[prep["numeric_cols"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    for col in prep["numeric_cols"]:
        A[col] = (
            A[col]
            .fillna(float(prep["med"].get(col, 0.0)))
            .astype(np.float32)
        )

    for col, mapping in prep["maps"].items():
        A["code_" + col] = (
            r[col]
            .astype("string")
            .fillna("<NA>")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.float32)
        )

    A["base_pred"] = np.asarray(base_pred, dtype=np.float32)

    if not np.isfinite(A.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("adapter core contains NaN/Inf")
    return A.reset_index(drop=True)


# =============================================================================
# Prediction
# =============================================================================


def _predict_catboost(
    block: dict,
    view: pd.DataFrame,
    label: str,
) -> np.ndarray:
    """L1 예측. E5 시드평균 — block["models"] 를 산술평균한다.

    구 artifact 호환: "models" 가 없으면 단일 "model" 을 1개짜리 리스트로 본다.
    모델이 1개면 평균은 항등이므로 시드평균 도입 이전과 비트 동일하다.
    """
    features = list(block["features"])
    cat_cols = list(block["cat_cols"])

    missing = sorted(set(features) - set(view.columns))
    if missing:
        raise KeyError(f"{label} features missing: {missing}")

    X = prepare_features(view[features], cat_cols)

    models = block.get("models")
    if models is None:
        models = [block["model"]]
    if not isinstance(models, (list, tuple)) or len(models) == 0:
        raise RuntimeError(f"{label} 모델 목록이 비었다")

    acc = None
    for k, m in enumerate(models):
        if hasattr(m, "feature_names_") and list(m.feature_names_) != features:
            raise RuntimeError(f"{label} feature order mismatch (model {k})")
        p = np.asarray(m.predict(X), dtype=np.float64)
        if len(p) != len(view) or not np.isfinite(p).all():
            raise RuntimeError(f"{label} prediction invalid (model {k})")
        acc = p if acc is None else acc + p
    return acc / float(len(models))


def predict(test: pd.DataFrame, art: dict) -> dict[str, np.ndarray]:
    """Predict every row independently using frozen artifact state."""
    test = test.reset_index(drop=True)
    n = len(test)
    target_year = int(art["expected_target_year"])

    # -----------------------------------------------------------------
    # L1 Control / Treatment views
    # -----------------------------------------------------------------
    control_view = build_control_view(test, art)
    treatment_view = build_treatment_view(control_view, test, art)

    pred_control = np.clip(
        _predict_catboost(art["control"], control_view, "CONTROL"),
        0.0,
        1.0,
    )
    pred_treatment = np.clip(
        _predict_catboost(art["treatment"], treatment_view, "TREATMENT"),
        0.0,
        1.0,
    )

    # E6 TRACK member — 동결 은행 lookup + CatBoost 시드평균.
    track_enabled = bool(art["track_enabled"]) and float(art["track_w"]) != 0.0
    track_w = float(art["track_w"])
    if track_enabled:
        track_view = build_track_view(treatment_view, test, art)
        pred_track = np.clip(
            _predict_catboost(art["track"], track_view, "TRACK"),
            0.0,
            1.0,
        )
    else:
        pred_track = None

    # -----------------------------------------------------------------
    # Dynamic-ASOF F expert
    # -----------------------------------------------------------------
    dyn = art["dynamic"]
    prev_anchor = compute_prev_anchor(
        rows=test,
        pitcher_season_stats=dyn["pitcher_season_stats"],
        season_league_means=dyn["season_league_means"],
        target_year=target_year,
        anchor_strength=float(dyn["anchor_strength"]),
        fallback_rate=float(dyn["anchor_fallback_rate"]),
    )
    dynamic_anchor = compute_dynamic_anchor(
        exact_view=control_view,
        prev_anchor=prev_anchor,
        dynamic_lambda=float(dyn["dynamic_lambda"]),
        dynamic_kappa=float(dyn["dynamic_kappa"]),
    )
    dynamic_correction = _predict_catboost(
        dyn,
        control_view,
        "DYNAMIC_ASOF",
    )
    pred_dynamic = np.clip(
        dynamic_anchor + dynamic_correction,
        0.0,
        1.0,
    )

    # -----------------------------------------------------------------
    # L1 routing
    # -----------------------------------------------------------------
    gt = (
        test[GAME_TYPE_COL]
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )
    r_mask = gt != "F"
    f_mask = gt == "F"

    # 행별 blend 가중치. 마스크를 곱셈 계수로만 쓴다 (train.py blend_weights).
    w_f = float(art["l1_routing"]["f_blend_weight"])
    w_r = float(art["w_r"])          # 단일 진실원 = artifact 최상위 (freeze())
    if not (0.0 <= w_r <= 1.0) or not (0.0 <= w_f <= 1.0):
        raise ValueError(f"blend weight out of [0,1]: w_r={w_r}, w_f={w_f}")

    # l1_routing 안의 사본은 기술용이다. 두 값이 갈라지면 artifact 가 자기
    # 구조를 잘못 기술하고 있다는 뜻이므로 추론을 진행하지 않는다.
    w_r_doc = art["l1_routing"].get("w_r")
    if w_r_doc is not None and float(w_r_doc) != w_r:
        raise RuntimeError(
            f"artifact w_r 불일치: top-level={w_r} vs l1_routing={float(w_r_doc)}"
        )

    w_row = np.where(f_mask, w_f, w_r).astype(np.float64)

    # E6: R 주 예측기 = (1-track_w)*TREATMENT + track_w*TRACK.
    # track 중립이면 경로 자체를 우회한다 (부동소수 연산 없음 → v005 비트 동일).
    if track_enabled:
        r_member = (1.0 - track_w) * pred_treatment + track_w * pred_track
    else:
        r_member = pred_treatment

    # 주 예측기는 game_type 으로 고른다 (train.py p_main24 와 동일).
    p_main = np.where(f_mask, pred_control, r_member)

    # train.py blend() 와 동일한 식·동일한 연산 순서.
    pred_l1 = np.clip(
        (1.0 - w_row) * p_main + w_row * pred_dynamic,
        0.0,
        1.0,
    )

    # 진단용. F 행 정의는 기존과 같고, w_r=0 이면 pred_l1[F] 와 일치한다.
    pred_f_blend = np.clip(
        (1.0 - w_f) * pred_control + w_f * pred_dynamic,
        0.0,
        1.0,
    )

    # -----------------------------------------------------------------
    # L2 ExtraTrees residual adapter — R only
    # -----------------------------------------------------------------
    l2 = art["l2"]
    core = transform_adapter_core(
        prep=l2["adapter_prep"],
        rows=test,
        base_pred=pred_l1,
    )

    if list(core.columns) != list(l2["adapter_core_feature_cols"]):
        raise RuntimeError("L2 adapter core feature order mismatch")

    exact = exact5_matrix(
        rows=test,
        pitcher_season_stats=l2["pitcher_season_stats"],
        target_year=target_year,
    )
    if list(exact.columns) != list(l2["exact_cols"]):
        raise RuntimeError("L2 Exact-ASOF feature order mismatch")

    X_l2 = pd.concat(
        [core.reset_index(drop=True), exact.reset_index(drop=True)],
        axis=1,
    )

    if list(X_l2.columns) != list(l2["adapter_feature_cols"]):
        raise RuntimeError("L2 full feature order mismatch")
    if not np.isfinite(X_l2.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("L2 inference matrix contains NaN/Inf")

    corrections = []
    models = l2["adapter_models"]
    for seed in l2["adapter_seeds"]:
        key = int(seed)
        if key not in models:
            # Defensive support for stringified keys, though joblib normally preserves ints.
            if str(key) in models:
                key = str(key)
            else:
                raise KeyError(f"L2 model seed {seed} missing from artifact")
        # 방어적 재설정. load_artifact 에서 이미 고정하지만, artifact 를 다른 경로로
        # 로드했을 때도 결정론이 깨지지 않도록 예측 직전에 한 번 더 건다.
        models[key].n_jobs = 1
        corr = np.asarray(models[key].predict(X_l2), dtype=np.float64)
        if len(corr) != n or not np.isfinite(corr).all():
            raise RuntimeError(f"L2 correction invalid for seed={seed}")
        corrections.append(corr)

    if corrections:
        l2_correction = np.mean(np.column_stack(corrections), axis=1)
    else:
        l2_correction = np.zeros(n, dtype=np.float64)

    # L2 적용 범위도 곱셈 가중치다 (train.py l2_weights 와 동일 정의).
    # l2_apply_f = False 이면 v001/v002 의 `final[f_mask] = pred_l1[f_mask]` 와
    # 비트 동일하다 — pred_l1 은 이미 [0,1] 이므로 clip 이 항등이고,
    # w_l2[F] = 0.0 이므로 0.0 * 유한값 = 0.0 이다.
    w_l2 = np.full(n, float(l2["adapter_shrink"]), dtype=np.float64)
    if not bool(art["l2_apply_f"]):
        w_l2[f_mask] = 0.0

    # --- E3c SHIFT: 절편 없는 Ridge 가산 (R 도메인) ---------------------------
    # 적용 범위도 곱셈 가중치다. shift_scale = 0 이면 w_shift 가 전부 0.0 이고
    # 0.0 * 유한값 = 0.0 이므로 danker_v004 와 **비트 단위로 동일**하다.
    shift_scale = float(art.get("shift_scale", 0.0))
    if shift_scale != 0.0:
        shift_raw = compute_shift_correction(test, art)
        if len(shift_raw) != n:
            raise RuntimeError(f"SHIFT 길이 불일치: {len(shift_raw)} vs {n}")
    else:
        shift_raw = np.zeros(n, dtype=np.float64)

    w_shift = np.full(n, shift_scale, dtype=np.float64)
    if not bool(art.get("shift_apply_f", False)):
        w_shift[f_mask] = 0.0

    # 단 한 번만 clip 한다 — 46_* 의 walk-forward 측정과 같은 형태다.
    final = np.clip(pred_l1 + w_l2 * l2_correction + w_shift * shift_raw, 0.0, 1.0)

    if not np.isfinite(final).all():
        raise RuntimeError("final prediction contains NaN/Inf")
    if ((final < 0.0) | (final > 1.0)).any():
        raise RuntimeError("final prediction outside [0, 1]")

    out = {
        "control": pred_control,
        "treatment": pred_treatment,
        "prev_anchor": prev_anchor,
        "dynamic_anchor": dynamic_anchor,
        "dynamic": pred_dynamic,
        "f_blend": pred_f_blend,
        "l1": pred_l1,
        "l2_correction": l2_correction,
        "shift_correction": shift_raw,
        "final": final,
        # 진단용 (selfcheck 대상 아님 — 상수 벡터라 행 독립 자명)
        "_w_row": w_row,
        "_w_l2": w_l2,
        "_w_shift": w_shift,
    }
    if track_enabled:
        out["track"] = pred_track
        out["r_member"] = r_member
    return out


# =============================================================================
# Row-independence selfcheck
# =============================================================================


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if len(a) else 0.0


def selfcheck(
    test: pd.DataFrame,
    art: dict,
    n_single: int = 30,
    seed: int = 0,
) -> dict:
    """Verify row prediction invariance.

    A: selected rows alone vs same rows inside full test
    B: shuffled full test vs original
    C: split/concat vs original

    Pass condition is exact max|difference| == 0.0 for final prediction and all
    major prediction-stage outputs.
    """
    keys = [
        "control",
        "treatment",
        "prev_anchor",
        "dynamic_anchor",
        "dynamic",
        "f_blend",
        "l1",
        "l2_correction",
        "shift_correction",
        "final",
    ]

    full = predict(test, art)
    # E6: track 이 켜져 있으면 그 중간 산출물도 행 독립 검사 대상이다.
    for extra in ("track", "r_member"):
        if extra in full:
            keys.insert(keys.index("l1"), extra)
    rng = np.random.default_rng(seed)

    selected = rng.choice(
        len(test),
        size=min(int(n_single), len(test)),
        replace=False,
    )
    selected = np.sort(selected)

    singles = {
        int(i): predict(test.iloc[[int(i)]].copy(), art)
        for i in selected
    }

    perm = rng.permutation(len(test))
    shuffled = predict(test.iloc[perm].copy(), art)

    cut = len(test) // 2
    left = predict(test.iloc[:cut].copy(), art)
    right = predict(test.iloc[cut:].copy(), art)

    result = {
        "experiment_name": EXPERIMENT_NAME,
        "n_test": int(len(test)),
        "n_single_rows": int(len(selected)),
    }

    all_pass = True
    for key in keys:
        a = max(
            abs(float(singles[int(i)][key][0]) - float(full[key][int(i)]))
            for i in selected
        ) if len(selected) else 0.0
        b = _max_abs(shuffled[key], full[key][perm])
        c = _max_abs(
            np.concatenate([left[key], right[key]]),
            full[key],
        )

        result[f"A_single_vs_full_{key}"] = float(a)
        result[f"B_shuffle_{key}"] = float(b)
        result[f"C_split_concat_{key}"] = float(c)

        key_pass = (a == 0.0) and (b == 0.0) and (c == 0.0)
        all_pass = all_pass and key_pass

        print(
            f"  {key:<14} "
            f"A={a:.3e}  B={b:.3e}  C={c:.3e}"
        )

    result["pass"] = bool(all_pass)
    return result


# =============================================================================
# Submission
# =============================================================================


def build_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    if ID not in sample.columns or TARGET not in sample.columns:
        raise KeyError(
            f"sample_submission must contain [{ID}, {TARGET}]"
        )
    if sample[ID].duplicated().any():
        raise RuntimeError("sample_submission row_id contains duplicates")

    pred_df = pd.DataFrame(
        {
            ID: test[ID].to_numpy(),
            TARGET: np.asarray(prediction, dtype=np.float64),
        }
    )

    out = (
        sample[[ID]]
        .merge(
            pred_df,
            on=ID,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if out[TARGET].isna().any():
        n_missing = int(out[TARGET].isna().sum())
        raise RuntimeError(
            f"submission prediction missing for {n_missing} row_id"
        )

    if len(out) != len(sample):
        raise RuntimeError("submission row count mismatch")

    return out[[ID, TARGET]]


# =============================================================================
# Main
# =============================================================================


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--test", default=str(TEST_PATH))
    parser.add_argument("--sample", default=str(SAMPLE_SUB_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--selfcheck-rows", type=int, default=30)
    args = parser.parse_args(argv)

    t0 = time.time()

    model_path = Path(args.model)
    test_path = Path(args.test)
    sample_path = Path(args.sample)
    output_path = Path(args.output)

    print("=" * 100)
    print(EXPERIMENT_NAME)
    print("=" * 100)
    print(f"Model      : {model_path}")
    print(f"Test       : {test_path}")
    print(f"Submission : {output_path}")
    print("Rule       : row-independent inference")
    print("=" * 100)

    art = load_artifact(model_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    test = validate_test(test, art)

    print("Inference...")
    outputs = predict(test, art)

    if args.selfcheck:
        print("Row-independence selfcheck...")
        result = selfcheck(
            test=test,
            art=art,
            n_single=args.selfcheck_rows,
            seed=0,
        )

        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        selfcheck_path = VALIDATION_DIR / "script_selfcheck.json"
        selfcheck_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Selfcheck saved: {selfcheck_path}")
        print(f"Selfcheck pass : {result['pass']}")

        if not result["pass"]:
            print("행 독립성 검사 실패 — submission 생성 금지")
            sys.exit(2)

    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    submission = build_submission(
        sample=sample,
        test=test,
        prediction=outputs["final"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved      : {output_path}")
    print(f"Rows       : {len(submission):,}")
    print(f"Elapsed    : {time.time() - t0:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()