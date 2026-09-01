#train.py 

from __future__ import annotations

import gc
import json
import os
import shutil
import time
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


# =============================================================================
# Config
# =============================================================================

EXPERIMENT_NAME = "dankerTrack_v001"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
META_PATH = MODEL_DIR / "final_metadata.json"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME
OOF_2024_PATH = VALIDATION_DIR / "l1_2024_oof.csv"
FOLD4_META_PATH = VALIDATION_DIR / "fold4_metadata.json"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

BASE_SEED = 42
SOURCE_YEAR = 2024
EXPECTED_TARGET_YEAR = 2025

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
]

EXACT_FEATURES = [
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

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": BASE_SEED,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}
EARLY_STOPPING_ROUNDS = 80

# -----------------------------------------------------------------------------
# 실행 환경 — GPU (Kaggle T4 x2, git clone 실행)
# -----------------------------------------------------------------------------
# "auto": GPU 가 보이면 GPU(T4 2개), 아니면 기존 CPU 경로 그대로.
# 주의:
#   1) GPU 학습은 CPU 와 비트 동일하지 않다. 같은 시드라도 fold Brier 가
#      CPU 기준선(v005 D0)과 미세하게 어긋난다 — 러닝 내부의 paired 측정
#      (TRACK_W 격자 등)은 같은 런 안에서 짝지어지므로 그대로 유효하다.
#   2) border_count=254 로 CPU 기본 quantization 에 정렬한다 (GPU 기본 128).
#   3) random_strength 는 CPU 전용 파라미터라 GPU 에서는 제거한다
#      (설정값 1.0 = 기본값이므로 의미 변화 없음).
#   추론(script.py)은 평가 서버 CPU 그대로이며 GPU 학습 모델 예측에 영향 없다.
USE_GPU = "auto"                     # "auto" | True | False
GPU_DEVICES = "0:1"                  # T4 2개


def _gpu_device_count() -> int:
    try:
        from catboost.utils import get_gpu_device_count
        return int(get_gpu_device_count())
    except Exception:
        return 0


if USE_GPU == "auto":
    GPU_ENABLED = _gpu_device_count() > 0
else:
    GPU_ENABLED = bool(USE_GPU)

if GPU_ENABLED:
    MODEL_PARAMS.update({
        "task_type": "GPU",
        "devices": GPU_DEVICES,
        "border_count": 254,
    })
    MODEL_PARAMS.pop("random_strength", None)

EXPECTED_CONTROL_FEATURES = 52
EXPECTED_TREATMENT_FEATURES = 86

# -----------------------------------------------------------------------------
# Validated F Dynamic-ASOF blend (fanchor_blend_v001 -> danker_v001)
# -----------------------------------------------------------------------------
# Previous anchor:
#   career-to-date pitcher history + previous-season league prior (EB strength=100)
# Dynamic update:
#   A_t = (1-alpha_t) * A_prev + alpha_t * R_t
#   alpha_t = lambda * n_t / (n_t + kappa)
# Final F route:
#   0.75 * Control + 0.25 * Dynamic-ASOF residual model
#
# These values are intentionally frozen from the confirmatory experiment.
ANCHOR_STRENGTH = 100.0
ANCHOR_FALLBACK_RATE = 0.5
DYNAMIC_LAMBDA = 1.0
DYNAMIC_KAPPA = 100.0
F_BLEND_WEIGHT = 0.25

PC_CONTEXTS = [
    ("v15_count", 28.0),
    ("batter_hand", 35.0),
    ("v15_inning_phase", 35.0),
    ("base_state", 35.0),
    ("v15_pressure", 28.0),
    ("v15_runners", 35.0),
    ("v15_li_bucket", 35.0),
]
PC_PARENT_STRENGTH = 80.0
PC_FIELDS = ("rate", "reliability", "delta")
PC_FEATURES = [
    f"pc__{ctx}__{field}"
    for ctx, _ in PC_CONTEXTS
    for field in PC_FIELDS
]

# L2 reference-style adapter settings used in the validated D0 line.
V15_CF_CFG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}
FINAL_ADAPTER_CFG = {
    "n_estimators": 120,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}
ADAPTER_SEEDS = [42, 2026, 314]
ADAPTER_SHRINK = 0.50

# =============================================================================
# danker_v003 — 구조 플래그 (38_danker_v002_구조개편_설계_v001.md §7-0)
# =============================================================================
# 이 파일이 단일 진실원이다. E0 중립 설정(W_R=0, L2_APPLY_F=False,
# L2_TRAIN_ON_ALL=False, SHIFT_SCALE=0, AFFINE_ALPHA=1)에서 danker_v002 와
# 비트 단위 동일해야 한다. 각 실험은 아래 중 "정확히 하나"만 바꾼다.

SEEDS = (42, 2026, 314)
FOLDS = (2021, 2022, 2023, 2024)
PRIMARY_FOLD = 2024
F_LABEL = "F"
BRANCHES = ("ALL", "R", "F")

LS_METRICS_PATH = VALIDATION_DIR / "ls_metrics.csv"
GATE_PATH = VALIDATION_DIR / "gate.json"

# --- E5: L1 시드평균 -----------------------------------------------------------
# L2 는 ADAPTER_SEEDS 로 이미 3 시드를 평균하는데 L1 은 단일 시드였다.
# 43_* §4 실측(2024 R행 Δshape, 단일시드 대비):
#     W_R=0.25 단독 +12.06   |   L1 3시드평균 +16.81   |   6시드평균 +20.96
#     그러나 2023 체제단절에서는 시드평균 +3.2 vs W_R +190.6 — 둘은 보완재다.
# fold 도 같은 시드로 돌려 **시드평균된 L1 OOF** 로 L2 를 적합한다.
#     그렇지 않으면 L2 가 학습한 base_pred(단일시드)와 배포 base_pred(시드평균)가
#     어긋난다 — 2024 실측 std 기준 13.3% (W_R 도입 때의 10.2% 보다 크다).
# 트리 수는 각 시드가 자기 fold 의 best iteration 을 쓴다.
# L1_SEEDS = (BASE_SEED,) 이면 시드평균 도입 이전과 비트 단위 동일하다.
#
# 🚩 danker_v004 = 제출 A (E5 단독).
#     v003(W_R=0.25, L1_SEEDS=(42,)) 대비 **이 상수 하나만** 바뀐다.
#     45_* §1 실측: 배포 기준(seed 42) 대비 R행 Δshape
#       2021 +9.71   2022 +27.00   2023 +10.94   2024 +14.90   → 4/4 양수
#     Jensen 부등식으로 "개별 시드 평균보다 나쁘지 않음"이 보장된다 (4/4 확인).
L1_SEEDS = (42, 2026, 314)
# L1_SEEDS = (42,)                # E5 를 끄고 v003 재현할 때

# --- E2b: 라우팅을 곱셈 가중치로. W_R=0 이면 v002 와 수학적으로 동일 -------------
W_R = 0.25
# W_F 는 아래 F_BLEND_WEIGHT (fanchor_blend_v001 검증 완료, 고정)

# --- E1a / E1b: L2 잔차 어댑터의 적용 범위와 학습 범위 --------------------------
L2_APPLY_F = False
L2_TRAIN_ON_ALL = False

# --- E2a: Dynamic 잔차 모델을 전 행으로 학습 -----------------------------------
DYNAMIC_TRAIN_ON_ALL = False

# --- E3c: 가산 보정항 (절편 없는 Ridge) ----------------------------------------
# 45_* §3 의 구성 ⑤ 를 그대로 구현한다.
#     도메인   R 전체 (game_type != "F")
#     적합     과거 전 연도 pooled OOF 잔차   (배포는 2021~2024 전부)
#     절편     없음
#     η        0.5
#
# 46_* 사전실측 — v005 의 실제 base(3시드평균 L1 + W_R 블렌드) 위에서 walk-forward,
# **total** BSS pt (shape 전용 아님):
#     2022  +15.11    2023  +12.90    2024  +18.92      3/3 양수
#     α 1e1~1e5 전 구간 3/3 양수,  η 0.30~0.60 전 구간 3/3 양수
#
# ⚠️ 이 측정의 base 는 L1 이고 L2 는 빠져 있다. 실제로는 L1+L2 위에 얹히므로
#    L2 와 겹치는 만큼 이득이 줄 수 있다. train 로그에 상관계수를 남긴다.
#
# 🚩 48_* §0 재결정: 겹침 실측 ρ=+0.3011 에서 η=0.30 이 minimax
#    (기대 +7.23 vs +5.18, 음수 꼬리 ρ≥0.62 vs ≥0.44). 0.50 은 폐기값.
SHIFT_SCALE = 0.30
SHIFT_RIDGE_ALPHA = 1e3
SHIFT_FIT_INTERCEPT = False          # 영구 False. 변경 금지.
SHIFT_CLIP = 8.0                     # 표준화된 **특징**을 자른다 (출력이 아니다)
SHIFT_FIT_YEARS = (2021, 2022, 2023, 2024)
SHIFT_APPLY_F = False                # F 는 Dynamic anchor 가 맡는다

# --- E3d: 대비 가산표 (06 방식) ------------------------------------------------
ADD_TABLE_SPECS = ()

# --- E3b: 최종 아핀 (절편 0 판본) ----------------------------------------------
AFFINE_ALPHA = 1.0
AFFINE_CENTER = 0.5
AFFINE_OFFSET = 0.0                  # 영구 0. 탐색 대상 아님. (38_* §5)
AFFINE_ALPHA_SEARCH = (0.95, 1.00, 1.05, 1.10)

# --- E4: L1 타겟 정의 ----------------------------------------------------------
L1_TARGET_MODE = "direct"            # "direct" | "anchor_residual"

# --- E6: TRACK — 공식 투구추적 로그의 선수 프로파일 state 멤버 (51_* 이식 설계) ----
# 공식 `trackman_history.csv`(2019~2024, data_description §3 이 파생 사용을 명시)를
# **선수별 과거 프로파일 요약**으로 얼려 네 번째 L1 멤버의 입력에 더한다.
#
# 형태 근거 (51_* §0):
#   - 물리값을 현재 행 회귀 입력으로 쓰는 형태·후처리 보정축 형태는 외부/내부에서
#     반복 실패했다. 과거 프로파일 state + **blend 멤버** 형태만 성립했고,
#     09_* 자기실측에서도 열 추가는 pooled 악화 / blend 는 4 fold 전부 양수였다.
#   - 그래서 TREATMENT 에 열을 넣지 않는다. 별도 멤버 TRACK 을 학습해
#     R 브랜치에서만 소수 가중으로 섞는다. F 경로는 무접촉.
#
#   R 행:  p_R = (1-W_R) * [ (1-TRACK_W)*TREATMENT + TRACK_W*TRACK ] + W_R*DYNAMIC
#   F 행:  기존과 동일 (CONTROL/DYNAMIC)  — TRACK 은 어디에도 안 들어간다
#
# 시간 규율: 행의 season = s 이면 프로파일은 **season < s** 로그만 집계한 은행을
# 쓴다 (ExactASOF 의 "완결된 과거만" 규율과 동일). 2025 추론 은행은 ≤2024 전체.
# 매핑: preprocess_trackman.py 산출 `pitcher_mapping.csv` 의 Tier A 만 (08_* 게이트).
#   매핑 identity 는 전체 이력 evidence 다 — 08_* §D 실측상 fold-frozen 과
#   95.5~99.0% 일치하므로 fold 검증의 낙관 편의는 작다고 보고 주석으로 남긴다.
# TRACK_ENABLED=False 또는 TRACK_W=0 이면 경로 자체를 우회해 v005 와 비트 동일.
TRACK_ENABLED = True
TRACK_W = 0.15                       # 비음수·사전 잠금. 단일 fold 최적화 금지 (51_* §5)
TRACK_W_GRID = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30)   # 진단 전용 (재학습 0회 재합성)
TRACK_TIERS_OK = ("A",)              # 보수 운용. B(저신뢰 팀) 는 넣지 않는다
TRACK_DIR = DATA_DIR / "trackman"
TRACK_HISTORY_CANDIDATES = (
    TRACK_DIR / "trackman_history.csv",
    DATA_DIR / "trackman_history.csv",     # 배포 원본 위치 (아직 안 옮긴 경우)
)
TRACK_MAPPING_PATH = TRACK_DIR / "pitcher_mapping.csv"
TRACK_ID_COL = "pitcher_trackman_id"
TRACK_GROUP_COL = "pitch_type_group"
TRACK_SEASON_COL_ = "season"         # trackman_history 쪽 season 컬럼
TRACK_METRICS = (
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
)
TRACK_GROUPS = ("fastball", "breaking", "offspeed", "other")
TRACK_COLS = (
    ["track_hist_n_log"]
    + [f"track_{m}_{s}" for m in TRACK_METRICS for s in ("avg", "dev")]
    + [f"track_mix_{g}" for g in TRACK_GROUPS]
    + [f"track_{g}_{m}_avg" for g in TRACK_GROUPS for m in TRACK_METRICS]
    + ["track_hist_missing"]
)                                    # 1 + 16 + 4 + 32 + 1 = 54열

# --- 게이트 상수 (38_* §8-5) — 단위는 within-branch ----------------------------
R_SIZE_THRESHOLD = 5.0
F_SIZE_THRESHOLD = 10.0
R_LEVEL_ABS_MAX = 5.0
F_LEVEL_MIN = -10.0
F_SHAPE_MIN = -5.0
R_SHAPE_MIN = -10.0
UNCERTAIN_Z = 2.0

# --- 참고값 (진단용, 판정에 직접 쓰지 않는다) ----------------------------------
B0_ALL_REF = 0.2498390                # 1 BSS pt = 2.4984e-6 Brier
V001_F_GAIN_WITHIN_REF = 22.166       # blend_gain_2024_F_pt (E2a 하한 기준)


def assert_invariants():
    """설정 자체가 설계 규약을 어기지 않는지 검사한다."""
    if AFFINE_OFFSET != 0.0:
        raise ValueError("AFFINE_OFFSET must stay 0.0 — 절편은 시즌을 건너지 못한다 (38_* §5)")
    if SHIFT_FIT_INTERCEPT:
        raise ValueError("SHIFT_FIT_INTERCEPT must stay False — 가산항의 중립값은 정확히 0")
    if not (0.0 <= W_R <= 1.0) or not (0.0 <= F_BLEND_WEIGHT <= 1.0):
        raise ValueError("W_R / F_BLEND_WEIGHT must lie in [0, 1]")
    if L1_TARGET_MODE not in ("direct", "anchor_residual"):
        raise ValueError(f"unknown L1_TARGET_MODE: {L1_TARGET_MODE!r}")
    if len(SEEDS) < 2:
        raise ValueError("seed 표준편차를 낼 수 없다 — SEEDS 는 2개 이상")
    if len(L1_SEEDS) < 1 or len(set(L1_SEEDS)) != len(L1_SEEDS):
        raise ValueError(f"L1_SEEDS 가 비었거나 중복이다: {L1_SEEDS}")
    if SHIFT_SCALE != 0.0:
        if not SHIFT_FIT_YEARS:
            raise ValueError("SHIFT_SCALE != 0 인데 SHIFT_FIT_YEARS 가 비었다")
        if sorted(set(SHIFT_FIT_YEARS)) != sorted(SHIFT_FIT_YEARS):
            raise ValueError(f"SHIFT_FIT_YEARS 에 중복: {SHIFT_FIT_YEARS}")
        if max(SHIFT_FIT_YEARS) > SOURCE_YEAR:
            raise ValueError(
                f"SHIFT_FIT_YEARS 최대 {max(SHIFT_FIT_YEARS)} > SOURCE_YEAR {SOURCE_YEAR} "
                "— 라벨이 없는 연도로는 적합할 수 없다"
            )
        if SHIFT_RIDGE_ALPHA <= 0.0:
            raise ValueError("SHIFT_RIDGE_ALPHA 는 양수여야 한다")
    if not (0.0 <= TRACK_W <= 0.5):
        raise ValueError("TRACK_W must lie in [0, 0.5] — 소수 가중 blend 전용 (51_*)")
    if TRACK_ENABLED:
        if not TRACK_TIERS_OK:
            raise ValueError("TRACK_ENABLED 인데 TRACK_TIERS_OK 가 비었다")
        n_expected = 1 + 2 * len(TRACK_METRICS) + len(TRACK_GROUPS) \
            + len(TRACK_GROUPS) * len(TRACK_METRICS) + 1
        if len(TRACK_COLS) != n_expected or len(set(TRACK_COLS)) != len(TRACK_COLS):
            raise ValueError(f"TRACK_COLS 구성 오류: {len(TRACK_COLS)} != {n_expected}")


def is_neutral():
    """E0 중립 설정인가? True 면 danker_v002 와 비트 동일해야 한다."""
    return (
        tuple(L1_SEEDS) == (BASE_SEED,)
        and W_R == 0.0
        and not L2_APPLY_F
        and not L2_TRAIN_ON_ALL
        and not DYNAMIC_TRAIN_ON_ALL
        and SHIFT_SCALE == 0.0
        and len(ADD_TABLE_SPECS) == 0
        and AFFINE_ALPHA == 1.0
        and AFFINE_OFFSET == 0.0
        and L1_TARGET_MODE == "direct"
        and (not TRACK_ENABLED or TRACK_W == 0.0)
    )


def freeze():
    """artifact(pkl)에 담을 상수 묶음.

    script.py 는 추론 시 **이 값들만** 참조한다. test 데이터에서 어떤 상수도
    재추정하지 않는다 (평가 데이터 독립 추론 규칙).
    """
    assert_invariants()
    return {
        "experiment_name": EXPERIMENT_NAME,
        "expected_target_year": EXPECTED_TARGET_YEAR,
        "w_r": float(W_R),
        "l1_seeds": [int(x) for x in L1_SEEDS],
        "w_f": float(F_BLEND_WEIGHT),
        "l2_apply_f": bool(L2_APPLY_F),
        "l2_train_on_all": bool(L2_TRAIN_ON_ALL),
        "l2_shrink": float(ADAPTER_SHRINK),
        "dynamic_train_on_all": bool(DYNAMIC_TRAIN_ON_ALL),
        "dynamic_lambda": float(DYNAMIC_LAMBDA),
        "dynamic_kappa": float(DYNAMIC_KAPPA),
        "shift_scale": float(SHIFT_SCALE),
        "shift_clip": float(SHIFT_CLIP),
        "shift_alpha": float(SHIFT_RIDGE_ALPHA),
        "shift_apply_f": bool(SHIFT_APPLY_F),
        "shift_fit_years": [int(x) for x in SHIFT_FIT_YEARS],
        "affine_alpha": float(AFFINE_ALPHA),
        "affine_center": float(AFFINE_CENTER),
        "affine_offset": float(AFFINE_OFFSET),
        "l1_target_mode": str(L1_TARGET_MODE),
        "track_enabled": bool(TRACK_ENABLED),
        "track_w": float(TRACK_W),
        "track_tiers": list(TRACK_TIERS_OK),
        "track_cols": list(TRACK_COLS),
        "is_neutral": is_neutral(),
    }



# =============================================================================
# danker_v003 — Brier Level/Shape 분해 및 ls_metrics (38_* §8-1~§8-2)
# =============================================================================

BSS_SCALE = 1e5


# =========================================================
# 기본 분해
# =========================================================
def brier_parts(p, y) -> tuple[float, float, float]:
    """(brier, level, shape) 를 반환. shape 는 brier - level 로 정의한다."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    if p.size == 0:
        return float("nan"), float("nan"), float("nan")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("non-finite value in p or y")

    d = p - y
    brier = float(np.mean(d * d))
    level = float(np.mean(d) ** 2)
    shape = brier - level
    return brier, level, shape


def b0_of(y) -> float:
    """기준 Brier B0 = r(1-r). 이진 y 에서 Var(y) 와 같다."""
    y = np.asarray(y, dtype=np.float64)
    r = float(np.mean(y))
    return r * (1.0 - r)


def bss_of(brier: float, b0: float) -> float:
    """공식 산식. max(0, .) 클리핑은 여기서 하지 않는다 (진단용 원값 보존)."""
    if not np.isfinite(b0) or b0 <= 0.0:
        return float("nan")
    return BSS_SCALE * (1.0 - brier / b0)


# =========================================================
# 한 (fold, seed, branch) 행
# =========================================================
@dataclass(frozen=True)
class BranchRow:
    experiment: str
    fold: int
    seed: int
    branch: str
    n: int
    b0: float
    brier: float
    level: float
    shape: float
    bss: float

    @staticmethod
    def build(experiment: str, fold: int, seed: int, branch: str, p, y) -> "BranchRow":
        brier, level, shape = brier_parts(p, y)
        b0 = b0_of(y)
        return BranchRow(
            experiment=str(experiment),
            fold=int(fold),
            seed=int(seed),
            branch=str(branch),
            n=int(np.asarray(y).size),
            b0=b0,
            brier=brier,
            level=level,
            shape=shape,
            bss=bss_of(brier, b0),
        )


def split_branches(
    y,
    p,
    game_type,
    f_label: str = "F",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ALL / R / F 로 나눈 (p, y) 쌍을 돌려준다. 행 순서는 보존한다."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    gt = pd.Series(game_type).fillna("__MISSING__").astype(str).to_numpy()
    if not (len(y) == len(p) == len(gt)):
        raise ValueError("length mismatch among y / p / game_type")

    f_mask = gt == f_label
    r_mask = ~f_mask
    return {
        "ALL": (p, y),
        "R": (p[r_mask], y[r_mask]),
        "F": (p[f_mask], y[f_mask]),
    }


def fold_seed_rows(
    experiment: str,
    fold: int,
    seed: int,
    y,
    p,
    game_type,
    f_label: str = "F",
) -> list[BranchRow]:
    parts = split_branches(y, p, game_type, f_label=f_label)
    return [
        BranchRow.build(experiment, fold, seed, br, pp, yy)
        for br, (pp, yy) in parts.items()
        if len(yy) > 0
    ]


def rows_to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


# =========================================================
# paired delta
# =========================================================
def paired_delta(
    base: pd.DataFrame,
    cand: pd.DataFrame,
    b0_all_by_fold: dict[int, float] | None = None,
    n_all_by_fold: dict[int, float] | None = None,
) -> pd.DataFrame:
    """동일 (fold, seed, branch) 끼리 짝지어 delta 를 계산한다.

    개선(= Brier 감소)이 양수가 되도록 부호를 잡는다.

        d_bss_within = (brier_base - brier_cand) / b0_branch * 1e5
        d_bss_total  = (brier_base - brier_cand) * (n_b/n_all) / b0_all * 1e5

    level / shape 도 같은 분모로 나누므로 d_total = d_level + d_shape 가 성립한다.
    b0 는 y 에만 의존하고 base/cand 가 같은 행을 보므로 두 표에서 동일하다.
    """
    keys = ["fold", "seed", "branch"]
    for name, df in (("base", base), ("cand", cand)):
        missing = set(keys) - set(df.columns)
        if missing:
            raise KeyError(f"{name} frame missing {sorted(missing)}")

    m = base.merge(cand, on=keys, suffixes=("_base", "_cand"), how="inner")
    if len(m) == 0:
        raise ValueError("paired_delta: 짝지어진 행이 없다 (fold/seed/branch 불일치)")
    if not np.allclose(m["n_base"], m["n_cand"]):
        raise ValueError("paired_delta: base/cand 의 행 수가 다르다 — paired 비교 불가")
    if not np.allclose(m["b0_base"], m["b0_cand"], atol=1e-12):
        raise ValueError("paired_delta: base/cand 의 B0 가 다르다 — 같은 행이 아니다")

    if b0_all_by_fold is None:
        b0_all_by_fold = (
            m.loc[m.branch == "ALL"].set_index("fold")["b0_base"].to_dict()
        )
    if n_all_by_fold is None:
        n_all_by_fold = (
            m.loc[m.branch == "ALL"].set_index("fold")["n_base"].to_dict()
        )

    b0_branch = m["b0_base"].to_numpy(dtype=np.float64)
    b0_all = m["fold"].map(b0_all_by_fold).to_numpy(dtype=np.float64)
    n_all = m["fold"].map(n_all_by_fold).to_numpy(dtype=np.float64)
    share = m["n_base"].to_numpy(dtype=np.float64) / n_all

    out = m[keys].copy()
    out["n"] = m["n_base"].to_numpy()
    out["b0"] = b0_branch
    out["share"] = share

    for comp in ("brier", "level", "shape"):
        d_raw = m[f"{comp}_base"].to_numpy(np.float64) - m[f"{comp}_cand"].to_numpy(np.float64)
        out[f"d_{comp}"] = d_raw
        name = "total" if comp == "brier" else comp
        out[f"d_{name}_within"] = d_raw / b0_branch * BSS_SCALE
        out[f"d_{name}_total"] = d_raw * share / b0_all * BSS_SCALE

    out["bss_base"] = m["bss_base"].to_numpy()
    out["bss_cand"] = m["bss_cand"].to_numpy()

    # 항등식 검사: total = level + shape (수치 오차 범위)
    resid = out["d_total_within"] - (out["d_level_within"] + out["d_shape_within"])
    if not np.allclose(resid, 0.0, atol=1e-6):
        raise AssertionError(
            f"Level/Shape 가산 항등식 위반: max |resid| = {np.abs(resid).max():.3e}"
        )
    return out


# =========================================================
# 산출
# =========================================================
LS_COLUMNS = [
    "experiment", "fold", "seed", "branch", "n", "share", "b0",
    "brier", "level", "shape", "bss",
    "d_brier", "d_level", "d_shape",
    "d_total_within", "d_level_within", "d_shape_within",
    "d_total_total", "d_level_total", "d_shape_total",
]


def build_ls_metrics(
    cand_rows,
    base_rows=None,
    experiment: str | None = None,
) -> pd.DataFrame:
    """ls_metrics.csv 로 저장할 표를 만든다.

    base_rows 가 없으면 절대값만 담는다(D0 기준선 생성용).
    있으면 paired delta 열을 붙인다.
    """
    cand = rows_to_frame(cand_rows) if not isinstance(cand_rows, pd.DataFrame) else cand_rows.copy()
    if experiment is not None:
        cand["experiment"] = experiment

    n_all = cand.loc[cand.branch == "ALL"].set_index(["fold", "seed"])["n"]
    cand["share"] = [
        row["n"] / n_all.get((row["fold"], row["seed"]), np.nan)
        for _, row in cand.iterrows()
    ]

    if base_rows is None:
        for c in LS_COLUMNS:
            if c not in cand.columns:
                cand[c] = np.nan
        return cand[LS_COLUMNS].sort_values(["fold", "seed", "branch"]).reset_index(drop=True)

    base = rows_to_frame(base_rows) if not isinstance(base_rows, pd.DataFrame) else base_rows.copy()
    d = paired_delta(base, cand)
    merged = cand.merge(
        d.drop(columns=["n", "b0", "share"]),
        on=["fold", "seed", "branch"],
        how="left",
    )
    for c in LS_COLUMNS:
        if c not in merged.columns:
            merged[c] = np.nan
    return merged[LS_COLUMNS].sort_values(["fold", "seed", "branch"]).reset_index(drop=True)


def save_ls_metrics(df: pd.DataFrame, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


# =============================================================================
# danker_v003 — R/F 게이트 판정 (38_* §8-5, 2026-08-30 정정본)
# =============================================================================

PASS = "PASS"
WEAK_POSITIVE = "WEAK_POSITIVE"
UNCERTAIN = "UNCERTAIN"
FAIL = "FAIL"


# =========================================================
# 시드 통계
# =========================================================
@dataclass(frozen=True)
class SeedStat:
    metric: str
    values: tuple
    k: int
    mean: float
    std: float          # 표본표준편차 (ddof=1)
    t_u: float          # 2*s/sqrt(k)
    all_positive: bool

    @staticmethod
    def build(metric: str, values) -> "SeedStat":
        v = np.asarray(list(values), dtype=np.float64)
        if v.size == 0:
            raise ValueError(f"{metric}: 시드 값이 없다")
        if not np.isfinite(v).all():
            raise ValueError(f"{metric}: 비유한 값 포함 {v}")
        k = int(v.size)
        std = float(np.std(v, ddof=1)) if k > 1 else 0.0
        return SeedStat(
            metric=metric,
            values=tuple(float(x) for x in v),
            k=k,
            mean=float(np.mean(v)),
            std=std,
            t_u=float(UNCERTAIN_Z * std / math.sqrt(k)),
            all_positive=bool(np.all(v > 0.0)),
        )


def _pull(df: pd.DataFrame, branch: str, column: str, fold: int) -> SeedStat:
    """(fold, branch) 의 시드별 값을 뽑는다. seed 순서는 config.SEEDS 를 따른다."""
    sub = df[(df.fold == fold) & (df.branch == branch)]
    if len(sub) == 0:
        raise KeyError(f"fold={fold} branch={branch} 행이 없다")
    sub = sub.set_index("seed")
    seeds = [s for s in SEEDS if s in sub.index]
    missing = [s for s in SEEDS if s not in sub.index]
    if missing:
        raise KeyError(f"fold={fold} branch={branch}: seed {missing} 누락")
    return SeedStat.build(f"{branch}.{column}", [float(sub.loc[s, column]) for s in seeds])


# =========================================================
# 판정 결과
# =========================================================
@dataclass
class Verdict:
    gate_type: str                  # "R" | "F"
    experiment: str
    fold: int
    decision: str                   # PASS / WEAK_POSITIVE / UNCERTAIN / FAIL
    primary_metric: str
    primary: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _diag_block(df: pd.DataFrame, fold: int, names: list[tuple[str, str]]) -> dict:
    out = {}
    for branch, col in names:
        try:
            st = _pull(df, branch, col, fold)
            out[f"{branch}.{col}"] = {"mean": st.mean, "std": st.std, "values": list(st.values)}
        except KeyError:
            out[f"{branch}.{col}"] = None
    return out


# =========================================================
# R 게이트
# =========================================================
def evaluate_r_gate(
    df: pd.DataFrame,
    experiment: str = EXPERIMENT_NAME,
    fold: int = PRIMARY_FOLD,
) -> Verdict:
    primary = _pull(df, "R", "d_shape_within", fold)
    t_u = primary.t_u
    t_p = max(R_SIZE_THRESHOLD, t_u)

    v = Verdict(
        gate_type="R",
        experiment=experiment,
        fold=fold,
        decision=FAIL,
        primary_metric="R.d_shape_within",
        primary={
            "mean": primary.mean,
            "std": primary.std,
            "values": list(primary.values),
            "all_positive": primary.all_positive,
            "k": primary.k,
        },
        thresholds={"T_u": t_u, "T_p": t_p, "size": R_SIZE_THRESHOLD},
    )

    r_level = _pull(df, "R", "d_level_within", fold)
    f_level = _pull(df, "F", "d_level_within", fold)
    v.gates = {
        "gate1_abs_R_level<=5": {
            "value": abs(r_level.mean),
            "limit": R_LEVEL_ABS_MAX,
            "ok": bool(abs(r_level.mean) <= R_LEVEL_ABS_MAX),
            "raw_mean": r_level.mean,
        },
        "gate2_F_level>=-10": {
            "value": f_level.mean,
            "limit": F_LEVEL_MIN,
            "ok": bool(f_level.mean >= F_LEVEL_MIN),
        },
    }
    v.diag = _diag_block(
        df, fold,
        [("R", "d_total_within"), ("F", "d_shape_within"), ("ALL", "d_total_within")],
    )

    m = primary.mean
    if m <= 0.0:
        v.decision = FAIL
        v.reasons.append(f"mean dR_shape = {m:.4f} <= 0")
        return v
    if m < t_u:
        v.decision = UNCERTAIN
        v.reasons.append(f"0 < mean {m:.4f} < T_u {t_u:.4f} — 시드 산포 대비 효과가 작다")
        return v
    if m < t_p:
        v.decision = WEAK_POSITIVE
        v.reasons.append(
            f"T_u {t_u:.4f} <= mean {m:.4f} < 크기문턱 {R_SIZE_THRESHOLD} — 안정적이나 효과 크기 부족"
        )
        return v
    if not primary.all_positive:
        v.decision = FAIL
        v.reasons.append(f"3/3 seed 양수 조건 미달: {list(primary.values)}")
        return v

    failed = [k for k, g in v.gates.items() if not g["ok"]]
    if failed:
        v.decision = FAIL
        v.reasons.extend(f"{k} 위반 (value={v.gates[k]['value']:.4f})" for k in failed)
        return v

    v.decision = PASS
    v.reasons.append(f"PRIMARY 통과 (mean {m:.4f} >= T_p {t_p:.4f}), 보조 게이트 전부 통과")
    return v


# =========================================================
# F 게이트
# =========================================================
def evaluate_f_gate(
    df: pd.DataFrame,
    experiment: str = EXPERIMENT_NAME,
    fold: int = PRIMARY_FOLD,
) -> Verdict:
    primary = _pull(df, "F", "d_total_within", fold)
    t_u = primary.t_u
    t_p = max(F_SIZE_THRESHOLD, t_u)

    v = Verdict(
        gate_type="F",
        experiment=experiment,
        fold=fold,
        decision=FAIL,
        primary_metric="F.d_total_within",
        primary={
            "mean": primary.mean,
            "std": primary.std,
            "values": list(primary.values),
            "all_positive": primary.all_positive,
            "k": primary.k,
        },
        thresholds={"T_u": t_u, "T_p": t_p, "size": F_SIZE_THRESHOLD},
    )

    f_shape = _pull(df, "F", "d_shape_within", fold)
    r_shape = _pull(df, "R", "d_shape_within", fold)
    v.gates = {
        "gate1_F_shape>=-5": {
            "value": f_shape.mean,
            "limit": F_SHAPE_MIN,
            "ok": bool(f_shape.mean >= F_SHAPE_MIN),
        },
        "gate2_R_shape>=-10": {
            "value": r_shape.mean,
            "limit": R_SHAPE_MIN,
            "ok": bool(r_shape.mean >= R_SHAPE_MIN),
        },
    }
    v.diag = _diag_block(
        df, fold,
        [("F", "d_level_within"), ("R", "d_level_within"), ("ALL", "d_total_within")],
    )

    m = primary.mean
    if m <= 0.0:
        v.decision = FAIL
        v.reasons.append(f"mean dF_total = {m:.4f} <= 0")
        return v
    if m < t_u:
        v.decision = UNCERTAIN
        v.reasons.append(f"0 < mean {m:.4f} < T_u {t_u:.4f} — 시드 산포 대비 효과가 작다")
        return v
    if m < F_SIZE_THRESHOLD:
        v.decision = WEAK_POSITIVE
        v.reasons.append(
            f"T_u {t_u:.4f} <= mean {m:.4f} < 크기문턱 {F_SIZE_THRESHOLD} — 안정적이나 효과 크기 부족"
        )
        return v
    if not primary.all_positive:
        v.decision = FAIL
        v.reasons.append(f"3/3 seed 양수 조건 미달: {list(primary.values)}")
        return v

    failed = [k for k, g in v.gates.items() if not g["ok"]]
    if failed:
        v.decision = FAIL
        v.reasons.extend(f"{k} 위반 (value={v.gates[k]['value']:.4f})" for k in failed)
        return v

    v.decision = PASS
    v.reasons.append(f"PRIMARY 통과 (mean {m:.4f} >= {t_p:.4f}), 보조 게이트 전부 통과")
    return v


# =========================================================
# LB 업로드 3조건 (38_* §8-3)
# =========================================================
def upload_conditions(df: pd.DataFrame) -> dict:
    """4 fold x k seed 전부 dBSS >= 0 인지, Total gain 이 양수인지 확인한다."""
    all_rows = df[df.branch == "ALL"]
    per_fold = {}
    for fold in FOLDS:
        sub = all_rows[all_rows.fold == fold]
        if len(sub) == 0:
            per_fold[fold] = {"present": False}
            continue
        vals = sub["d_total_within"].to_numpy(dtype=np.float64)
        wins = int(np.sum(vals >= 0.0))
        per_fold[fold] = {
            "present": True,
            "k": int(vals.size),
            "wins": wins,
            "record": f"{wins}/{vals.size}",
            "ok": bool(wins == vals.size),
            "mean": float(np.mean(vals)),
        }
    cond1 = all(v.get("ok", False) for v in per_fold.values())

    primary = all_rows[all_rows.fold == PRIMARY_FOLD]
    total_gain = float(primary["d_total_within"].mean()) if len(primary) else float("nan")
    cond2 = bool(np.isfinite(total_gain) and total_gain > 0.0)

    return {
        "cond1_all_folds_all_seeds_nonneg": cond1,
        "per_fold": per_fold,
        "cond2_total_gain_positive": cond2,
        "total_gain_2024_within": total_gain,
    }


# =========================================================
# 저장
# =========================================================
def run_gates(
    df: pd.DataFrame,
    experiment: str = EXPERIMENT_NAME,
    gate_types: tuple = ("R", "F"),
    fold: int = PRIMARY_FOLD,
) -> dict:
    result = {
        "experiment": experiment,
        "protocol": "LS_BRANCH_GATE_V1 (38_* §8-5, 2026-08-30)",
        "unit": "within-branch BSS points",
        "primary_fold": fold,
        "seeds": list(SEEDS),
        "upload_conditions": upload_conditions(df),
        "verdicts": {},
    }
    if "R" in gate_types:
        result["verdicts"]["R"] = evaluate_r_gate(df, experiment, fold).to_dict()
    if "F" in gate_types:
        result["verdicts"]["F"] = evaluate_f_gate(df, experiment, fold).to_dict()

    decisions = [v["decision"] for v in result["verdicts"].values()]
    if not decisions:
        result["final"] = FAIL
    elif all(d == PASS for d in decisions):
        result["final"] = PASS
    elif any(d == FAIL for d in decisions):
        result["final"] = FAIL
    elif any(d == UNCERTAIN for d in decisions):
        result["final"] = UNCERTAIN
    else:
        result["final"] = WEAK_POSITIVE

    result["submit_recommended"] = bool(
        result["final"] == PASS and result["upload_conditions"]["cond1_all_folds_all_seeds_nonneg"]
    )
    return result


def save_gate(result: dict, path=GATE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def format_report(result: dict) -> str:
    lines = [
        f"experiment : {result['experiment']}",
        f"unit       : {result['unit']}",
        f"final      : {result['final']}  (submit_recommended={result['submit_recommended']})",
        "",
        "-- LB 업로드 조건 --",
    ]
    uc = result["upload_conditions"]
    for fold, info in uc["per_fold"].items():
        if info.get("present"):
            mark = "OK " if info["ok"] else "NG "
            lines.append(f"  {mark}{fold}: {info['record']}  mean={info['mean']:+.4f}")
        else:
            lines.append(f"  -- {fold}: 없음")
    lines.append(f"  cond2 total gain 2024 = {uc['total_gain_2024_within']:+.4f}")
    for gt, v in result["verdicts"].items():
        lines += [
            "",
            f"-- {gt} 게이트 : {v['decision']} --",
            f"  PRIMARY {v['primary_metric']}  mean={v['primary']['mean']:+.4f}"
            f"  std={v['primary']['std']:.4f}  values={['%+.4f' % x for x in v['primary']['values']]}",
            f"  T_u={v['thresholds']['T_u']:.4f}  T_p={v['thresholds']['T_p']:.4f}",
        ]
        for k, g in v["gates"].items():
            lines.append(f"  {'OK ' if g['ok'] else 'NG '}{k}: {g['value']:+.4f} (limit {g['limit']})")
        for r in v["reasons"]:
            lines.append(f"  · {r}")
    return "\n".join(lines)


# =============================================================================
# danker_v003 — 후처리 슬롯 (가산 + 아핀, 절편 0) (38_* §5, §6)
# =============================================================================

# =========================================================
# 대비 가산표 (06 방식)
# =========================================================
@dataclass
class TableSpec:
    """개체 x 이진맥락 잔차 차등표.

    담는 값은 수준이 아니라 **대비**다. 부모 집단 대비 편차이므로 리그 연도 이동이
    뺄셈에서 소거된다. 반대칭(+-0.5v)이라 표 항목의 평균이 0이다.
    """
    name: str
    entity_col: str                 # 예: "pitcher_id"
    ctx_col: str                    # 0/1 이진 맥락 컬럼명
    weight: float
    table: dict                     # {(entity, ctx): value}
    row_mean: float = 0.0           # 학습 행 가중 평균 (중심화용)
    unseen: float = 0.0             # 미지 조합의 값. 반드시 0 (중립)


def build_contrast_table(
    entity,
    ctx,
    resid,
    k: float,
    name: str = "table",
    entity_col: str = "pitcher_id",
    ctx_col: str = "ctx",
    weight: float = 1.0,
    center_on_rows: bool = True,
) -> TableSpec:
    """OOF 잔차로 개체 x 이진맥락 대비표를 만든다.

        d  = mean(r | ctx=1) - mean(r | ctx=0)
        ne = n0*n1 / (n0+n1)                    # 대비의 유효 표본수
        v  = d * ne/(ne+k)                      # EB 축소
        표 = +0.5v (ctx=1) / -0.5v (ctx=0)      # 반대칭

    center_on_rows
        06 의 반대칭 인코딩은 **표 항목의 평균**을 0으로 만들지만, 개체별
        n0 != n1 이면 **행 가중 평균**은 0이 아니다. 즉 Level 이 미세하게 움직인다.
        이 옵션을 켜면 학습 행 기준 가중 평균을 빼서 Level 불변을 정확히 만든다.
        (38_* §8-8 의 "E3d 는 Level 불변이 설계 목표" 를 실제로 보장하는 장치)
    """
    entity = np.asarray(entity)
    ctx = np.asarray(ctx).astype(np.int64)
    resid = np.asarray(resid, dtype=np.float64)
    if not (len(entity) == len(ctx) == len(resid)):
        raise ValueError("entity / ctx / resid 길이 불일치")
    if not np.isin(ctx, (0, 1)).all():
        raise ValueError("ctx 는 0/1 이진이어야 한다")

    g = pd.DataFrame({"e": entity, "c": ctx, "r": resid}).groupby(["e", "c"])["r"]
    agg = g.agg(["mean", "size"]).unstack()

    m0 = agg[("mean", 0)] if ("mean", 0) in agg.columns else np.nan
    m1 = agg[("mean", 1)] if ("mean", 1) in agg.columns else np.nan
    n0 = agg[("size", 0)] if ("size", 0) in agg.columns else 0.0
    n1 = agg[("size", 1)] if ("size", 1) in agg.columns else 0.0
    n0 = pd.Series(n0, index=agg.index).fillna(0.0)
    n1 = pd.Series(n1, index=agg.index).fillna(0.0)

    d = (pd.Series(m1, index=agg.index) - pd.Series(m0, index=agg.index))
    denom = (n0 + n1).replace(0.0, np.nan)
    ne = (n0 * n1) / denom
    v = (d * ne / (ne + float(k))).dropna()

    table = {}
    for e, val in v.items():
        table[(e, 1)] = float(0.5 * val)
        table[(e, 0)] = float(-0.5 * val)

    spec = TableSpec(
        name=name, entity_col=entity_col, ctx_col=ctx_col,
        weight=float(weight), table=table, row_mean=0.0, unseen=0.0,
    )
    if center_on_rows:
        vals = _lookup_table(spec, entity, ctx)
        spec.row_mean = float(np.mean(vals))
    return spec


def _lookup_table(spec: TableSpec, entity, ctx) -> np.ndarray:
    """미지 조합은 정확히 0. 그 행 자신의 키만 본다."""
    tab = spec.table
    unseen = float(spec.unseen)
    out = np.fromiter(
        (tab.get((e, int(c)), unseen) for e, c in zip(np.asarray(entity), np.asarray(ctx))),
        dtype=np.float64,
        count=len(entity),
    )
    return out - float(spec.row_mean)


def apply_tables(pred: np.ndarray, rows: pd.DataFrame, specs) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float64).copy()
    for spec in specs or ():
        if spec.entity_col not in rows.columns or spec.ctx_col not in rows.columns:
            raise KeyError(f"table {spec.name!r}: 컬럼 {spec.entity_col}/{spec.ctx_col} 없음")
        v = _lookup_table(spec, rows[spec.entity_col].to_numpy(), rows[spec.ctx_col].to_numpy())
        out = out + float(spec.weight) * v
    return out


# =========================================================
# 표준화 (학습 시 동결된 mu/sd 만 사용)
# =========================================================
def standardize(mat, mu, sd, clip: float = 8.0) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
    z = (mat - mu) / sd
    return np.clip(z, -clip, clip)


# =========================================================
# E3c — SHIFT 설계행렬 (30열) 과 절편 없는 Ridge
# =========================================================
#
# 규칙 준수: 30열 전부 **현재 행의 공식 컬럼만** 쓴다. 다른 행을 참조하는 항이
# 하나도 없다. median / mu / sd / beta 는 전부 공식 학습 데이터에서만 나오며
# artifact 에 동결된다 — script.py 는 test 분포에서 어떤 상수도 재추정하지 않는다.
#
# 열 구성은 46_* 의 walk-forward 검증에 쓴 것과 **동일**하다. 열 하나라도 바꾸면
# 그 측정치는 무효다. 특히 base_pred 계열(base, base-0.5, (base-0.5)^2)을
# 넣으면 안 된다 — 46_* §2 에서 정상 연도 2/3 을 음수로 뒤집는 것을 확인했다.

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
    """E3c 민감도 설계행렬. 결측은 채우지 않고 NaN 으로 돌려준다.

    채우기는 artifact 에 동결된 median 으로 별도 단계에서 한다 — 그래야
    train 과 script 가 정확히 같은 값을 쓴다.
    """
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

    if "pitcher_hand" in frame.columns and "batter_hand" in frame.columns:
        x["same_hand"] = (
            frame["pitcher_hand"].astype(str) == frame["batter_hand"].astype(str)
        ).astype(float)
    else:
        raise KeyError("SHIFT: pitcher_hand / batter_hand 가 없다")

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


def shift_correction(matrix, median, mu, sd, beta, clip: float) -> np.ndarray:
    """설계행렬 -> 보정량. train.py 와 script.py 가 공유하는 정의."""
    mat = np.asarray(matrix, dtype=np.float64)
    med = np.asarray(median, dtype=np.float64)
    filled = np.where(np.isnan(mat), med, mat)
    if not np.isfinite(filled).all():
        raise RuntimeError("SHIFT 행렬에 median 으로도 못 채운 NaN/Inf 가 있다")
    z = standardize(filled, mu, sd, clip)
    return z @ np.asarray(beta, dtype=np.float64)


def fit_shift_ridge(matrix, residual, alpha: float, clip: float) -> dict:
    """절편 없는 Ridge. 표준화 -> 특징 clip -> 정규방정식 정해."""
    if SHIFT_FIT_INTERCEPT:
        raise ValueError("SHIFT_FIT_INTERCEPT must stay False")
    mat = np.asarray(matrix, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if len(mat) != len(r):
        raise ValueError(f"SHIFT 적합 길이 불일치: {len(mat)} vs {len(r)}")

    median = np.nanmedian(mat, axis=0)
    if not np.isfinite(median).all():
        raise RuntimeError("SHIFT median 에 NaN — 전부 결측인 열이 있다")
    filled = np.where(np.isnan(mat), median, mat)

    mu = filled.mean(axis=0)
    sd = filled.std(axis=0)
    sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
    z = standardize(filled, mu, sd, clip)

    gram = z.T @ z + float(alpha) * np.eye(z.shape[1])
    beta = np.linalg.solve(gram, z.T @ r)
    if not np.isfinite(beta).all():
        raise RuntimeError("SHIFT beta 에 NaN/Inf")
    return {"median": median, "mu": mu, "sd": sd, "beta": beta,
            "clip": float(clip), "alpha": float(alpha), "n_fit": int(len(r))}


# =========================================================
# 후처리 파이프라인
# =========================================================
def apply_postproc(
    pred,
    art: dict,
    rows: pd.DataFrame | None = None,
    sens_matrix=None,
) -> np.ndarray:
    """가산 -> 아핀 -> clip. 모든 상수는 art 에서만 온다.

    art 필수 키
        shift_scale, affine_alpha, affine_center, affine_offset
    art 선택 키
        shift_model (predict 가능), sens_mu, sens_sd, shift_clip, add_tables
    """
    for key in ("shift_scale", "affine_alpha", "affine_center", "affine_offset"):
        if key not in art:
            raise KeyError(f"artifact 에 {key} 가 없다 — 후처리 상수는 pkl 에서만 온다")

    offset = float(art["affine_offset"])
    if offset != 0.0:
        raise ValueError(
            f"affine_offset must be 0.0, got {offset} — 절편은 시즌을 건너지 못한다 (38_* §5)"
        )

    out = np.asarray(pred, dtype=np.float64).copy()

    # ---- 가산 1: 절편 없는 Ridge 민감도 보정 ----
    scale = float(art["shift_scale"])
    if scale != 0.0:
        beta = art.get("shift_beta")
        if beta is None or sens_matrix is None:
            raise ValueError("shift_scale != 0 인데 shift_beta 또는 sens_matrix 가 없다")
        corr = shift_correction(
            sens_matrix, art["shift_median"], art["sens_mu"], art["sens_sd"],
            beta, float(art.get("shift_clip", 8.0)),
        )
        if corr.shape != out.shape:
            raise ValueError(f"shift 보정 길이 불일치: {corr.shape} vs {out.shape}")
        out = out + scale * corr

    # ---- 가산 2: 대비 보정표 ----
    specs = art.get("add_tables") or ()
    if specs:
        if rows is None:
            raise ValueError("add_tables 가 있는데 rows 가 없다")
        out = apply_tables(out, rows, specs)

    # ---- 아핀 (절편 0) ----
    alpha = float(art["affine_alpha"])
    center = float(art["affine_center"])
    if alpha != 1.0:
        out = center + alpha * (out - center)

    out = np.clip(out, 0.0, 1.0)
    if not np.isfinite(out).all():
        raise RuntimeError("후처리 결과에 NaN/Inf")
    return out


# =========================================================
# L1 라우팅 -> 가중치 (I2)
# =========================================================
def blend_weights(game_type, w_r: float, w_f: float, f_label: str = "F") -> np.ndarray:
    """행별 blend 가중치. 마스크를 곱셈 계수로만 쓴다.

    w_r = 0, w_f = 0.25 이면 danker_v001 의 where 라우팅과 수학적으로 동일하다.
    """
    gt = pd.Series(game_type).fillna("__MISSING__").astype(str).to_numpy()
    return np.where(gt == f_label, float(w_f), float(w_r)).astype(np.float64)


def blend(p_main, p_dyn, weights) -> np.ndarray:
    p_main = np.asarray(p_main, dtype=np.float64)
    p_dyn = np.asarray(p_dyn, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if not (p_main.shape == p_dyn.shape == w.shape):
        raise ValueError("blend 입력 길이 불일치")
    return np.clip((1.0 - w) * p_main + w * p_dyn, 0.0, 1.0)


def apply_l2(pred, correction, weights) -> np.ndarray:
    """L2 잔차 보정. weights 로 적용 범위를 건다 (R만 = F에 0).

    danker_v001 의 `final[f_mask] = pred_l1[f_mask]` 는 weights[f]=0 과 동일하다.
    """
    pred = np.asarray(pred, dtype=np.float64)
    corr = np.asarray(correction, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if not (pred.shape == corr.shape == w.shape):
        raise ValueError("apply_l2 입력 길이 불일치")
    return np.clip(pred + w * corr, 0.0, 1.0)


def l2_weights(game_type, shrink: float, apply_f: bool, f_label: str = "F") -> np.ndarray:
    gt = pd.Series(game_type).fillna("__MISSING__").astype(str).to_numpy()
    f_mask = gt == f_label
    w = np.full(len(gt), float(shrink), dtype=np.float64)
    if not apply_f:
        w[f_mask] = 0.0
    return w
N_JOBS = max(1, min(6, os.cpu_count() or 1))

REFERENCE_RAWNUM = [
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "li",
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
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "pitcher_team_id",
    "batter_team_id",
]

REFERENCE_CAT_SMALL = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

ENG_COLS = [
    "eng_recent35",
    "eng_recent_delta",
    "eng_sb",
    "eng_same",
    "eng_close",
    "eng_late",
]


def mask_pitcher_conditional_to_r_only(pc_frame, source_rows):
    """
    R(game_type != 'F') 행에서만 PITCHER_CONDITIONAL을 활성화한다.
    F 행은 21개 PC feature를 모두 0으로 둔다.

    주의:
    - 이것은 single global CatBoost의 feature masking이다.
    - shared tree structure 때문에 F prediction이 Control과 완전히 같아지는 것은 보장되지 않는다.
    - 따라서 별도로 routed prediction도 평가한다:
        R -> masked treatment prediction
        F -> control prediction
    """
    out = pc_frame.copy()
    gt = source_rows[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    f_mask = gt == "F"

    out.loc[f_mask, PC_FEATURES] = 0.0

    values = out[PC_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("R-only masked PITCHER_CONDITIONAL contains NaN/Inf")

    if f_mask.any():
        f_values = out.loc[f_mask, PC_FEATURES].to_numpy(dtype=np.float64)
        if not np.all(f_values == 0.0):
            raise RuntimeError("F rows are not fully neutralized in R-only PC mask")

    return out

def prepare_features(df, cat_cols):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return df

def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

def calculate_bss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    brier = brier_score(y_true, y_pred)
    p = float(np.mean(y_true))
    baseline = p * (1.0 - p)

    if baseline <= 0.0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier / baseline
        )
    )

def build_pitcher_season_stats(df_train):
    """
    공식 train label만으로 pitcher × season 완결 통계를 만든다.

    Exact-ASOF 계산 시에는 target row의 season보다
    과거 season만 사용한다.
    """

    required = {
        "pitcher_id",
        SEASON_COL,
        TARGET,
    }

    missing = sorted(
        required - set(df_train.columns)
    )

    if missing:
        raise KeyError(
            f"pitcher season stats 필수 컬럼 없음: {missing}"
        )

    stats = (
        df_train
        .groupby(
            ["pitcher_id", SEASON_COL],
            dropna=False,
            sort=False,
        )[TARGET]
        .agg(
            n="size",
            success="sum",
        )
        .reset_index()
    )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats["n"] = (
        stats["n"]
        .astype(np.float64)
    )

    stats["success"] = (
        stats["success"]
        .astype(np.float64)
    )

    return stats

def add_pitcher_exact_asof_features(
    df,
    pitcher_season_stats,
):
    """
    해당 row의 official cumulative ASOF에서
    과거 완결 시즌 누적치를 빼서
    현재 시즌 시점 직전 상태를 복원한다.

    다른 validation/test row는 사용하지 않는다.
    """

    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF 필수 컬럼 없음: {missing}"
        )

    out = df.copy()

    for col in EXACT_FEATURES:
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    diagnostics = []

    for target_year in sorted(
        seasons.unique()
    ):
        row_mask = (
            seasons == int(target_year)
        )

        rows = out.loc[row_mask]

        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist) > 0:
            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[["n", "success"]]
                .sum()
            )

            hist_n = (
                rows["pitcher_id"]
                .map(hist_agg["n"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

            hist_success = (
                rows["pitcher_id"]
                .map(hist_agg["success"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            hist_n = np.zeros(
                len(rows),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(rows),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                rows["asof_pitcher_n"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64)
        )

        asof_rate = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(dtype=np.float64)
        )

        asof_success = (
            asof_n * asof_rate
        )

        raw_current_n = (
            asof_n - hist_n
        )

        current_n = np.maximum(
            raw_current_n,
            0.0,
        )

        raw_current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                raw_current_success
            ),
            np.clip(
                raw_current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_current = (
            (current_n > 0.0)
            & np.isfinite(
                current_success
            )
        )

        current_rate[
            valid_current
        ] = (
            current_success[
                valid_current
            ]
            / current_n[
                valid_current
            ]
        )

        past_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_past = (
            hist_n > 0.0
        )

        past_rate[
            valid_past
        ] = (
            hist_success[
                valid_past
            ]
            / hist_n[
                valid_past
            ]
        )

        idx = rows.index

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = current_n

        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = np.log1p(
            current_n
        )

        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = current_rate

        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = past_rate

        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = (
            current_rate
            - past_rate
        )

        diagnostics.append(
            {
                "season": int(target_year),
                "n_rows": int(len(rows)),
                "negative_current_n": int(
                    (
                        raw_current_n
                        < -1e-8
                    ).sum()
                ),
                "current_rate_available": int(
                    np.isfinite(
                        current_rate
                    ).sum()
                ),
                "past_rate_available": int(
                    np.isfinite(
                        past_rate
                    ).sum()
                ),
            }
        )

    return (
        out,
        pd.DataFrame(diagnostics),
    )

def compute_hbeb_priors(
    df_train,
    group_cols,
    rate_col,
):
    """
    train subset만으로 HB-EB prior를 추정한다.
    """

    g_mu = float(
        df_train[rate_col].mean()
    )

    g_var = float(
        df_train[rate_col].var()
    )

    max_bernoulli_var = (
        g_mu * (1.0 - g_mu)
    )

    if (
        not np.isfinite(g_var)
        or g_var <= 0.0
        or g_var >= max_bernoulli_var
    ):
        g_var = (
            max_bernoulli_var
            * 0.5
        )

    g_m = (
        max_bernoulli_var
        / g_var
    ) - 1.0

    g_alpha = (
        g_mu * g_m
    )

    g_beta = (
        (1.0 - g_mu)
        * g_m
    )

    stats = (
        df_train
        .groupby(
            group_cols,
            dropna=False,
        )[rate_col]
        .agg(
            ["mean", "var", "count"]
        )
        .reset_index()
    )

    group_max_var = (
        stats["mean"]
        * (
            1.0
            - stats["mean"]
        )
    )

    valid_mask = (
        stats["var"].notna()
        & np.isfinite(
            stats["var"]
        )
        & (
            stats["var"] > 0.0
        )
        & (
            stats["var"]
            < group_max_var
        )
        & (
            stats["count"] >= 30
        )
    )

    stats["alpha"] = g_alpha
    stats["beta"] = g_beta

    if valid_mask.any():
        mu = stats.loc[
            valid_mask,
            "mean",
        ]

        var = stats.loc[
            valid_mask,
            "var",
        ]

        m = (
            mu
            * (1.0 - mu)
            / var
        ) - 1.0

        stats.loc[
            valid_mask,
            "alpha",
        ] = (
            mu * m
        )

        stats.loc[
            valid_mask,
            "beta",
        ] = (
            (1.0 - mu)
            * m
        )

    prior_df = stats[
        group_cols
        + ["alpha", "beta"]
    ].copy()

    global_prior = (
        float(g_alpha),
        float(g_beta),
    )

    return (
        prior_df,
        global_prior,
    )

def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
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

    alpha = (
        lookup["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(dtype=np.float64)
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .to_numpy(dtype=np.float64)
    )

    success_count = (
        n * rate
    )

    denominator = (
        n + alpha + beta
    )

    return np.divide(
        success_count + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=(
            denominator > 0.0
        ),
    )

def apply_hbeb_pair(
    df_train,
    df_apply,
    base_features,
):
    """
    df_train만으로 prior를 만들고
    df_apply에 적용한다.
    """

    transformed = (
        df_apply.copy()
    )

    features = list(
        base_features
    )

    priors = {}

    for config in HBEB_CONFIGS:
        (
            prior_df,
            global_prior,
        ) = compute_hbeb_priors(
            df_train=df_train,
            group_cols=config[
                "group_cols"
            ],
            rate_col=config[
                "rate_col"
            ],
        )

        transformed[
            config["new_col"]
        ] = apply_hbeb(
            df=transformed,
            group_cols=config[
                "group_cols"
            ],
            n_col=config[
                "n_col"
            ],
            rate_col=config[
                "rate_col"
            ],
            prior_df=prior_df,
            global_prior=global_prior,
        )

        if (
            config["rate_col"]
            in features
        ):
            features.remove(
                config["rate_col"]
            )

        if (
            config["new_col"]
            not in features
        ):
            features.append(
                config["new_col"]
            )

        priors[
            config["new_col"]
        ] = {
            "prior_df": prior_df,
            "global_prior": (
                global_prior
            ),
            "group_cols": list(
                config[
                    "group_cols"
                ]
            ),
            "n_col": config[
                "n_col"
            ],
            "rate_col": config[
                "rate_col"
            ],
        }

    return (
        transformed,
        features,
        priors,
    )

def _ctx_frame_v15(frame):
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

def _pc_eb_table(history, keys, parent_keys=None, strength=40.0, parent_strength=80.0):
    # Recovered V15 code와 동일한 empirical-Bayes shrinkage 식.
    h = history
    if len(h) == 0:
        return pd.DataFrame(columns=keys + ["rate", "reliability", "delta", "log_n"])

    league = float(h[TARGET].mean())
    g = h.groupby(keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()

    if parent_keys is None:
        g["parent"] = league
    else:
        pg = h.groupby(parent_keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()
        pg["parent_rate"] = (
            pg["sum"] + parent_strength * league
        ) / (
            pg["size"] + parent_strength
        )
        g = g.merge(pg[parent_keys + ["parent_rate"]], on=parent_keys, how="left")
        g["parent"] = g["parent_rate"].fillna(league)

    g["rate"] = (
        g["sum"] + strength * g["parent"]
    ) / (
        g["size"] + strength
    )
    g["reliability"] = g["size"] / (g["size"] + strength)
    g["delta"] = g["rate"] - g["parent"]
    g["log_n"] = np.log1p(g["size"])
    return g[keys + ["rate", "reliability", "delta", "log_n"]]

def _pc_lookup(table, rows, keys, field, default=0.0):
    # 원본 V15의 unseen default=0.0을 그대로 유지.
    if len(table) == 0:
        return np.full(len(rows), default, dtype=np.float32)

    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)

    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)

def _pc_coverage(table, rows, keys):
    if len(rows) == 0 or len(table) == 0:
        return 0.0
    target_idx = pd.MultiIndex.from_frame(rows[keys].astype(str))
    table_idx = pd.MultiIndex.from_frame(table[keys].astype(str))
    return float(np.mean(target_idx.isin(table_idx)))

def build_pitcher_conditional_for_rows(rows, available_history, scope):
    """
    각 target season y의 PC는 오직 y-1 labeled season으로 생성한다.

    training rows:
      2020 row <- 2019 labels
      2021 row <- 2020 labels ...

    validation rows:
      2024 row <- fold-train의 2023 labels

    같은 target season label은 절대 사용하지 않는다.
    """
    required = {
        SEASON_COL, TARGET, "pitcher_id", "batter_hand", "base_state",
        "balls_before", "strikes_before", "inning", "num_runners_on", "li",
    }
    missing_rows = sorted((required - {TARGET}) - set(rows.columns))
    missing_hist = sorted(required - set(available_history.columns))
    if missing_rows:
        raise KeyError(f"PITCHER_CONDITIONAL row source missing: {missing_rows}")
    if missing_hist:
        raise KeyError(f"PITCHER_CONDITIONAL history source missing: {missing_hist}")

    r_all = _ctx_frame_v15(rows.reset_index(drop=False).rename(columns={"index": "__orig_index"}))
    h_all = _ctx_frame_v15(available_history.reset_index(drop=True))

    out = pd.DataFrame(0.0, index=np.arange(len(r_all)), columns=PC_FEATURES, dtype=np.float32)
    coverage_rows = []

    seasons = pd.to_numeric(r_all[SEASON_COL], errors="raise").astype(int)
    hist_seasons = pd.to_numeric(h_all[SEASON_COL], errors="raise").astype(int)

    for target_year in sorted(seasons.unique()):
        target_mask = seasons.eq(int(target_year)).to_numpy()
        target_pos = np.flatnonzero(target_mask)
        rr = r_all.loc[target_mask].reset_index(drop=True)
        source_year = int(target_year) - 1
        hh = h_all.loc[hist_seasons.eq(source_year)].reset_index(drop=True)

        # source season 부재(예: 2019)는 원본 lookup의 neutral 0 vector.
        if len(hh) == 0:
            for ctx, strength in PC_CONTEXTS:
                coverage_rows.append({
                    "scope": scope,
                    "target_season": int(target_year),
                    "source_season": int(source_year),
                    "context": ctx,
                    "strength": float(strength),
                    "n_rows": int(len(rr)),
                    "source_rows": 0,
                    "coverage": 0.0,
                })
            continue

        for ctx, strength in PC_CONTEXTS:
            table = _pc_eb_table(
                hh,
                ["pitcher_id", ctx],
                ["pitcher_id"],
                strength=float(strength),
                parent_strength=PC_PARENT_STRENGTH,
            )
            prefix = f"pc__{ctx}"
            keys = ["pitcher_id", ctx]
            for field in PC_FIELDS:
                out.loc[target_pos, f"{prefix}__{field}"] = _pc_lookup(
                    table, rr, keys, field, 0.0
                )

            coverage_rows.append({
                "scope": scope,
                "target_season": int(target_year),
                "source_season": int(source_year),
                "context": ctx,
                "strength": float(strength),
                "n_rows": int(len(rr)),
                "source_rows": int(len(hh)),
                "coverage": _pc_coverage(table, rr, keys),
            })

    # 원래 rows index로 복원.
    out.index = rows.index
    if list(out.columns) != PC_FEATURES:
        raise RuntimeError("PITCHER_CONDITIONAL feature order mismatch")
    if out.columns.duplicated().any():
        raise RuntimeError("PITCHER_CONDITIONAL duplicate columns")
    if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("PITCHER_CONDITIONAL contains NaN/Inf")

    return out, pd.DataFrame(coverage_rows)

# =============================================================================
# L2 adapter core
# =============================================================================

def numeric_series(frame: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(
            np.full(len(frame), default, dtype=np.float64),
            index=frame.index,
        )
    return pd.to_numeric(frame[col], errors="coerce")


def add_eng(frame: pd.DataFrame) -> pd.DataFrame:
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
            x["pitcher_hand"].astype(str)
            .eq(x["batter_hand"].astype(str))
            .astype(np.float32)
        )
    else:
        x["eng_same"] = 0.0
    x["eng_close"] = (
        numeric_series(x, "score_diff_pitcher_team", np.nan)
        .abs().le(1).astype(np.float32)
    )
    x["eng_late"] = (
        numeric_series(x, "inning", np.nan)
        .ge(7).astype(np.float32)
    )
    return x


def fit_adapter_prep(map_rows: pd.DataFrame) -> dict:
    m = add_eng(map_rows.reset_index(drop=True))
    numeric_cols = [c for c in REFERENCE_RAWNUM if c in m.columns] + ENG_COLS
    med = (
        m[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .fillna(0.0)
        .to_dict()
    )
    cat_cols = [c for c in REFERENCE_CAT_SMALL if c in m.columns]
    maps = {}
    for col in cat_cols:
        s = m[col].astype("string").fillna("<NA>").astype(str)
        maps[col] = {value: idx for idx, value in enumerate(pd.Index(s.unique()))}
    return {"numeric_cols": numeric_cols, "med": med, "maps": maps}


def transform_adapter_core(prep: dict, rows: pd.DataFrame, base_pred) -> pd.DataFrame:
    r = add_eng(rows.reset_index(drop=True))
    A = (
        r[prep["numeric_cols"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    for col in prep["numeric_cols"]:
        A[col] = A[col].fillna(float(prep["med"].get(col, 0.0))).astype(np.float32)
    for col, mapping in prep["maps"].items():
        A["code_" + col] = (
            r[col].astype("string").fillna("<NA>").astype(str)
            .map(mapping).fillna(-1).astype(np.float32)
        )
    A["base_pred"] = np.asarray(base_pred, dtype=np.float32)
    if not np.isfinite(A.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("adapter core contains NaN/Inf")
    return A.reset_index(drop=True)


def exact5_matrix(rows: pd.DataFrame, pitcher_season_stats: pd.DataFrame) -> pd.DataFrame:
    transformed, _ = add_pitcher_exact_asof_features(rows, pitcher_season_stats)
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


def fit_final_l2_models(X24, raw24, residual24):
    gt = raw24[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    r_mask = gt != "F"
    f_mask = gt == "F"
    n_r = int(r_mask.sum())
    minimum = int(FINAL_ADAPTER_CFG["min_samples_leaf"]) * 2
    if n_r <= minimum:
        raise RuntimeError(f"Final L2 R rows insufficient: {n_r} <= {minimum}")

    # E1b: 학습 표본을 R 로 제한할지 전 행으로 열지.
    # L2_TRAIN_ON_ALL=False 면 r_mask 와 동일하므로 v002 와 비트 동일하다.
    fit_mask = np.ones(len(raw24), dtype=bool) if L2_TRAIN_ON_ALL else r_mask

    models = {}
    diagnostics = []
    for seed in ADAPTER_SEEDS:
        model = ExtraTreesRegressor(
            n_jobs=N_JOBS,
            random_state=16200 + int(seed),
            **FINAL_ADAPTER_CFG,
        )
        t0 = time.time()
        model.fit(X24.loc[fit_mask], residual24[fit_mask])
        elapsed = time.time() - t0
        corr = np.asarray(model.predict(X24.loc[fit_mask]), dtype=np.float64)
        models[int(seed)] = model
        diagnostics.append({
            "adapter_seed": int(seed),
            "n_train_R": n_r,
            "n_train_used": int(fit_mask.sum()),
            "l2_train_on_all": bool(L2_TRAIN_ON_ALL),
            "n_protected_F": int(f_mask.sum()),
            "fit_elapsed_seconds": float(elapsed),
            "train_correction_mean_R": float(corr.mean()),
            "train_correction_std_R": float(corr.std()),
            "train_correction_abs_mean_R": float(np.mean(np.abs(corr))),
        })
        print(
            f"[L2 seed={seed}] R rows={n_r:,} | fit={elapsed:.1f}s | "
            f"corr mean={corr.mean():+.6f} | corr abs={np.mean(np.abs(corr)):.6f}"
        )
    return models, pd.DataFrame(diagnostics), r_mask, f_mask


def build_pc_tables_by_source_season(train: pd.DataFrame):
    train_ctx = _ctx_frame_v15(train.reset_index(drop=True))
    seasons = sorted(
        pd.to_numeric(train_ctx[SEASON_COL], errors="raise").astype(int).unique().tolist()
    )
    result = {}
    for source_year in seasons:
        hh = train_ctx.loc[
            pd.to_numeric(train_ctx[SEASON_COL], errors="raise").astype(int).eq(int(source_year))
        ].copy().reset_index(drop=True)
        season_tables = {}
        for context, strength in PC_CONTEXTS:
            table = _pc_eb_table(
                hh,
                ["pitcher_id", context],
                ["pitcher_id"],
                strength=float(strength),
                parent_strength=float(PC_PARENT_STRENGTH),
            )
            season_tables[context] = table
        result[int(source_year)] = season_tables
    return result


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj



# =============================================================================
# Dynamic F Anchor
# =============================================================================

def build_season_league_means(df_train: pd.DataFrame) -> dict[int, float]:
    """Build season-level league target means from official labeled train only."""
    means = (
        df_train.groupby(SEASON_COL, sort=True)[TARGET]
        .mean()
        .astype(np.float64)
    )
    return {int(k): float(v) for k, v in means.items()}


def league_prev_for(target_year: int, season_league_means: dict[int, float]) -> float:
    """Previous-season league prior with an older-season fallback."""
    prev = int(target_year) - 1
    if prev in season_league_means:
        return float(season_league_means[prev])

    older = [
        value
        for year, value in season_league_means.items()
        if int(year) < int(target_year)
    ]
    if older:
        return float(np.mean(older))
    return float(ANCHOR_FALLBACK_RATE)


def compute_prev_anchor(
    rows: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
    season_league_means: dict[int, float],
) -> np.ndarray:
    """
    Reproduce the validated previous-season EB anchor.

    For a row in target season t:
      hist = the pitcher's official labeled history from seasons < t
      league_prev = official league mean from t-1

      A_prev = (hist_success + S * league_prev) / (hist_n + S)

    No other validation/test row is used.
    """
    seasons = pd.to_numeric(rows[SEASON_COL], errors="raise").astype(int)
    stats = pitcher_season_stats.copy()
    stats[SEASON_COL] = pd.to_numeric(stats[SEASON_COL], errors="raise").astype(int)

    anchor = np.full(len(rows), np.nan, dtype=np.float64)
    positions = np.arange(len(rows))

    for target_year in sorted(seasons.unique().tolist()):
        mask = seasons.eq(int(target_year)).to_numpy()
        pos = positions[mask]
        sub = rows.loc[mask]

        hist = stats.loc[stats[SEASON_COL] < int(target_year)]
        if len(hist) > 0:
            agg = (
                hist.groupby("pitcher_id", sort=False, dropna=False)[["n", "success"]]
                .sum()
            )
            hist_n = (
                sub["pitcher_id"]
                .map(agg["n"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            hist_success = (
                sub["pitcher_id"]
                .map(agg["success"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            hist_n = np.zeros(len(sub), dtype=np.float64)
            hist_success = np.zeros(len(sub), dtype=np.float64)

        league_prev = league_prev_for(int(target_year), season_league_means)
        anchor[pos] = (
            hist_success + ANCHOR_STRENGTH * league_prev
        ) / (
            hist_n + ANCHOR_STRENGTH
        )

    if not np.isfinite(anchor).all():
        raise RuntimeError("Previous anchor contains NaN/Inf")
    return np.clip(anchor, 0.0, 1.0)


def compute_dynamic_anchor(
    exact_view: pd.DataFrame,
    prev_anchor: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Row-local Dynamic ASOF anchor.

      A_t = (1-alpha_t) * A_prev + alpha_t * R_t
      alpha_t = lambda * n_t / (n_t + kappa)

    R_t and n_t are derived from the current row's official ASOF fields through
    Exact-ASOF5. No other test row is referenced.
    """
    prev_anchor = np.asarray(prev_anchor, dtype=np.float64)
    if len(prev_anchor) != len(exact_view):
        raise RuntimeError("Dynamic anchor length mismatch")

    n_t = (
        pd.to_numeric(exact_view["exact_pitcher_season_n"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    r_t = pd.to_numeric(
        exact_view["exact_pitcher_season_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    valid = (n_t > 0.0) & np.isfinite(r_t)
    reliability = np.zeros(len(exact_view), dtype=np.float64)
    reliability[valid] = n_t[valid] / (n_t[valid] + float(DYNAMIC_KAPPA))
    alpha = np.clip(float(DYNAMIC_LAMBDA) * reliability, 0.0, 1.0)

    current_state = prev_anchor.copy()
    current_state[valid] = np.clip(r_t[valid], 0.0, 1.0)
    dynamic_anchor = (1.0 - alpha) * prev_anchor + alpha * current_state

    if not np.isfinite(dynamic_anchor).all():
        raise RuntimeError("Dynamic anchor contains NaN/Inf")

    diagnostics = {
        "lambda": float(DYNAMIC_LAMBDA),
        "kappa": float(DYNAMIC_KAPPA),
        "asof_available_rate": float(np.mean(valid)),
        "mean_current_n": float(np.mean(n_t)),
        "median_current_n": float(np.median(n_t)),
        "mean_alpha": float(np.mean(alpha)),
        "p90_alpha": float(np.quantile(alpha, 0.90)),
        "mean_abs_anchor_shift": float(np.mean(np.abs(dynamic_anchor - prev_anchor))),
        "max_abs_anchor_shift": float(np.max(np.abs(dynamic_anchor - prev_anchor))),
    }
    return np.clip(dynamic_anchor, 0.0, 1.0), diagnostics


def fit_dynamic_fold_model(
    train_view,
    val_view,
    features,
    cat_cols,
    y_train,
    y_val,
    anchor_train,
    anchor_val,
    label,
    seed=BASE_SEED,
):
    """Fit CatBoost to residual y - DynamicAnchor and return probability prediction."""
    X_tr = prepare_features(train_view[features], cat_cols)
    X_va = prepare_features(val_view[features], cat_cols)
    # 3 seed 프로토콜
    _params = dict(MODEL_PARAMS)
    _params["random_seed"] = int(seed)
    model = CatBoostRegressor(**_params)
    print(
        f"\n[{label}] fold4 training | rows={len(X_tr):,} -> {len(X_va):,} "
        f"| features={len(features)}"
    )
    t0 = time.time()
    model.fit(
        X_tr,
        np.asarray(y_train, dtype=np.float64) - np.asarray(anchor_train, dtype=np.float64),
        eval_set=(
            X_va,
            np.asarray(y_val, dtype=np.float64) - np.asarray(anchor_val, dtype=np.float64),
        ),
        cat_features=cat_cols,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )
    elapsed = time.time() - t0
    correction = np.asarray(model.predict(X_va), dtype=np.float64)
    pred = np.clip(np.asarray(anchor_val, dtype=np.float64) + correction, 0.0, 1.0)
    best_zero = int(model.get_best_iteration())
    if best_zero < 0:
        raise RuntimeError(f"{label} best iteration invalid: {best_zero}")
    return model, pred, correction, best_zero, elapsed


def fit_dynamic_full_model(
    view,
    features,
    cat_cols,
    y,
    anchor,
    iterations,
    label,
):
    """Final full-data Dynamic-ASOF residual CatBoost. E5: L1_SEEDS 시드평균."""
    seeds = tuple(L1_SEEDS)
    iters = (list(iterations) if isinstance(iterations, (list, tuple))
             else [int(iterations)] * len(seeds))
    if len(iters) != len(seeds):
        raise ValueError(f"{label}: iterations {len(iters)} != seeds {len(seeds)}")
    X = prepare_features(view[features], cat_cols)
    target = np.asarray(y, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    models, params_list, total = [], [], 0.0
    for sd, it in zip(seeds, iters):
        params = {**MODEL_PARAMS, "iterations": int(it), "random_seed": int(sd)}
        model = CatBoostRegressor(**params)
        print(
            f"\n[{label} seed={sd}] final full training | rows={len(X):,} "
            f"| features={len(features)} | trees={it}"
        )
        t0 = time.time()
        model.fit(X, target, cat_features=cat_cols, verbose=100)
        total += time.time() - t0
        if hasattr(model, "feature_names_") and list(model.feature_names_) != list(features):
            raise RuntimeError(f"{label} final feature order mismatch (seed={sd})")
        models.append(model)
        params_list.append(params)
    return models, params_list, total


# =============================================================================
# L1 view construction / training
# =============================================================================

# =============================================================================
# danker_v003 <- danker_v002(= danker_sj_v001): SJ v042 SUCCESS_FULL (원문 이식)
# =============================================================================

SEASON = SEASON_COL          # SJ 코드 호환 별칭
GAME_TYPE = GAME_TYPE_COL

SUCCESS_SPEC = {
    "pitcher": {
        "id_col": "pitcher_id",
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "strengths": [40.0, 80.0, 160.0],
        "reliability_strength": 80.0,
    },
    "batter": {
        "id_col": "batter_id",
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "strengths": [40.0, 80.0, 160.0],
        "reliability_strength": 100.0,
    },
}

SUCCESS_FEATURES = [
    # pitcher
    "gseason__pitcher_rate_k40",
    "gseason__pitcher_rate_k80",
    "gseason__pitcher_rate_k160",
    "gseason__pitcher_reliability",
    "gseason__pitcher_minus_career",

    # batter
    "gseason__batter_rate_k40",
    "gseason__batter_rate_k80",
    "gseason__batter_rate_k160",
    "gseason__batter_raw",
    "gseason__batter_log_n",
    "gseason__batter_reliability",
    "gseason__batter_minus_career",

    # matchup current-state diff
    "gseason__pitcher_minus_batter_k80",
]


def rounded_cumulative_count(
    n,
    rate,
):
    """
    v038 current-state semantics:
    cumulative event count ~= round(n * rate)
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


def build_success_bank(
    history: pd.DataFrame,
    entity: str,
):
    spec = (
        SUCCESS_SPEC[
            entity
        ]
    )

    id_col = (
        spec[
            "id_col"
        ]
    )

    if len(
        history
    ) == 0:
        return {
            "n": {},
            "success": {},
            "prior": 0.5,
        }

    agg = (
        history
        .groupby(
            id_col,
            sort=False,
            dropna=False,
        )[TARGET]
        .agg(
            [
                "size",
                "sum",
            ]
        )
    )

    max_year = int(
        pd.to_numeric(
            history[
                SEASON
            ],
            errors="raise",
        )
        .max()
    )

    prior = float(
        history.loc[
            pd.to_numeric(
                history[
                    SEASON
                ],
                errors="coerce",
            )
            .eq(
                max_year
            ),
            TARGET,
        ]
        .mean()
    )

    if not np.isfinite(
        prior
    ):
        prior = 0.5

    return {
        "n": (
            agg[
                "size"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "success": (
            agg[
                "sum"
            ]
            .astype(
                np.float64
            )
            .to_dict()
        ),
        "prior": float(
            prior
        ),
    }


def build_state_bank(
    available_history: pd.DataFrame,
    target_year: int,
):
    hist = (
        available_history.loc[
            pd.to_numeric(
                available_history[
                    SEASON
                ],
                errors="raise",
            )
            .astype(int)
            .lt(
                int(
                    target_year
                )
            )
        ]
        .copy()
    )

    return {
        "target_year": int(
            target_year
        ),
        "pitcher_success": (
            build_success_bank(
                hist,
                "pitcher",
            )
        ),
        "batter_success": (
            build_success_bank(
                hist,
                "batter",
            )
        ),
    }


def transform_success_state(
    rows: pd.DataFrame,
    bank: dict,
    entity: str,
    out: pd.DataFrame,
):
    spec = (
        SUCCESS_SPEC[
            entity
        ]
    )

    sb = (
        bank[
            f"{entity}_success"
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

    ids = (
        rows[
            id_col
        ]
    )

    hist_n = (
        ids
        .map(
            sb[
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
            sb[
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
        sb[
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

    rk = float(
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
            + rk
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
        "current_n": (
            current_n
        ),
        "raw": (
            raw
        ),
        "k80": (
            out[
                f"{prefix}_rate_k80"
            ]
            .to_numpy(
                dtype=np.float64
            )
        ),
    }


def build_success_state_for_rows(
    rows: pd.DataFrame,
    available_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    SUCCESS_FULL 13 features.

    row target season y:
      historical constants = labeled history season < y
      current state = row-local official cumulative asof - historical constants
    """
    out = pd.DataFrame(
        index=rows.index
    )

    for col in (
        SUCCESS_FEATURES
    ):
        out[
            col
        ] = np.nan

    seasons = (
        pd.to_numeric(
            rows[
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
            seasons.eq(
                int(
                    target_year
                )
            )
        )

        rr = (
            rows.loc[
                mask
            ]
            .copy()
        )

        bank = (
            build_state_bank(
                available_history=(
                    available_history
                ),
                target_year=int(
                    target_year
                ),
            )
        )

        z = pd.DataFrame(
            index=rr.index
        )

        pitcher_state = (
            transform_success_state(
                rows=rr,
                bank=bank,
                entity="pitcher",
                out=z,
            )
        )

        batter_state = (
            transform_success_state(
                rows=rr,
                bank=bank,
                entity="batter",
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

        z = (
            z.reindex(
                columns=SUCCESS_FEATURES
            )
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
            rr.index,
            SUCCESS_FEATURES,
        ] = z[
            SUCCESS_FEATURES
        ]

    out = (
        out
        .fillna(0.0)
        .astype(
            np.float32
        )
    )

    if not np.isfinite(
        out.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "SUCCESS_FULL contains NaN/Inf."
        )

    return out


def build_success_history_artifact(
    full_train: pd.DataFrame,
) -> dict:
    """
    inference 시 raw train.csv 없이도 target season의
    historical n/success/prior를 복원할 수 있게
    season-level aggregated labeled history를 저장한다.
    """
    result = {}

    for entity in [
        "pitcher",
        "batter",
    ]:
        id_col = (
            SUCCESS_SPEC[
                entity
            ][
                "id_col"
            ]
        )

        stats = (
            full_train
            .groupby(
                [
                    id_col,
                    SEASON,
                ],
                sort=False,
                dropna=False,
            )[TARGET]
            .agg(
                n="size",
                success="sum",
            )
            .reset_index()
        )

        stats[
            SEASON
        ] = (
            pd.to_numeric(
                stats[
                    SEASON
                ],
                errors="raise",
            )
            .astype(int)
        )

        result[
            entity
        ] = stats

    season_prior = (
        full_train
        .groupby(
            SEASON,
            sort=False,
        )[TARGET]
        .mean()
    )

    result[
        "season_prior"
    ] = {
        int(k): float(v)
        for k, v in (
            season_prior
            .to_dict()
            .items()
        )
    }

    return result


# =============================================================================
# E6 TRACK — 공식 투구추적 로그의 선수별 프로파일 은행 (51_* 이식 설계)
# =============================================================================
#
# 규칙 준수 요약:
#   - 원천은 공식 `trackman_history.csv`(2019~2024) 와, 공식 train/trackman 두
#     테이블의 통계 결합만으로 만든 `pitcher_mapping.csv` 다. test.csv 는 이 절
#     어디에서도 읽지 않는다.
#   - 행의 season = s 인 학습/검증 행은 season < s 로그만 집계한 은행을 쓴다.
#   - 추론(2025)은 ≤2024 전체 은행을 artifact 에 동결해 **조회만** 한다.
#     한 행의 조회 키는 그 행 자신의 pitcher_id 뿐 — 다른 test 행 참조가 없다.


def resolve_track_history_path() -> Path:
    for cand in TRACK_HISTORY_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "trackman_history.csv 를 찾을 수 없다. 시도한 위치: "
        + ", ".join(str(c) for c in TRACK_HISTORY_CANDIDATES)
    )


def load_track_history() -> pd.DataFrame:
    path = resolve_track_history_path()
    usecols = [TRACK_SEASON_COL_, TRACK_ID_COL, TRACK_GROUP_COL, *TRACK_METRICS]
    hist = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    hist[TRACK_SEASON_COL_] = pd.to_numeric(
        hist[TRACK_SEASON_COL_], errors="raise"
    ).astype(int)
    lo, hi = int(hist[TRACK_SEASON_COL_].min()), int(hist[TRACK_SEASON_COL_].max())
    if lo < 2019 or hi > SOURCE_YEAR:
        raise RuntimeError(f"track history season 범위 이상: {lo}~{hi}")
    hist[TRACK_GROUP_COL] = (
        hist[TRACK_GROUP_COL].astype(str).where(
            hist[TRACK_GROUP_COL].astype(str).isin(TRACK_GROUPS), "other"
        )
    )
    for m in TRACK_METRICS:
        hist[m] = pd.to_numeric(hist[m], errors="coerce")
    print(f"TRACK history : {path}  rows={len(hist):,}  seasons {lo}~{hi}")
    return hist


def load_track_mapping() -> pd.DataFrame:
    if not TRACK_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"{TRACK_MAPPING_PATH} 가 없다. preprocess_trackman.py 를 먼저 실행해\n"
            "공식 train/trackman 만으로 매핑을 생성하라 (08_* 게이트 산출물)."
        )
    mp = pd.read_csv(TRACK_MAPPING_PATH, encoding="utf-8-sig")
    need = {"pitcher_id", TRACK_ID_COL, "confidence_tier"}
    missing = sorted(need - set(mp.columns))
    if missing:
        raise KeyError(f"pitcher_mapping.csv 필수 컬럼 누락: {missing}")
    if "mapping_valid" in mp.columns:
        # CSV 왕복 시 bool 이 "True"/"False" 문자열로 남을 수 있다.
        # astype(bool) 은 비어있지 않은 문자열을 전부 True 로 만들므로 쓰지 않는다.
        mv = mp["mapping_valid"].map({
            True: True, False: False, "True": True, "False": False,
            "true": True, "false": False, 1: True, 0: False,
        })
        if mv.isna().any():
            bad = mp.loc[mv.isna(), "mapping_valid"].unique()[:5]
            raise ValueError(f"mapping_valid 에 해석 불가 값: {bad}")
        mp = mp[mv.to_numpy(dtype=bool)]
    mp = mp[mp["confidence_tier"].astype(str).isin([str(t) for t in TRACK_TIERS_OK])]
    mp = mp.dropna(subset=["pitcher_id", TRACK_ID_COL]).copy()
    mp["pitcher_id"] = pd.to_numeric(mp["pitcher_id"], errors="raise").astype("int64")
    if mp["pitcher_id"].duplicated().any() or mp[TRACK_ID_COL].duplicated().any():
        raise RuntimeError(
            "pitcher_mapping.csv 에 중복 매핑이 있다 — duplicate resolution 이전 파일이다"
        )
    print(
        f"TRACK mapping : {TRACK_MAPPING_PATH}  tier {tuple(TRACK_TIERS_OK)}  "
        f"pitchers={len(mp):,}"
    )
    return mp[["pitcher_id", TRACK_ID_COL]]


def build_track_banks(history: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    """season → (pitcher_id 인덱스, TRACK_COLS) 은행. bank[s] 는 season < s 만 집계."""
    hist = history.merge(mapping, on=TRACK_ID_COL, how="inner")
    if len(hist) == 0:
        raise RuntimeError("TRACK: 매핑 후 남은 로그가 0행이다")

    metric_cols = list(TRACK_METRICS)
    banks: dict[int, pd.DataFrame] = {}
    for s in range(2019, EXPECTED_TARGET_YEAR + 1):
        sub = hist.loc[hist[TRACK_SEASON_COL_] < s]
        if len(sub) == 0:
            banks[s] = pd.DataFrame(columns=TRACK_COLS)
            continue
        g = sub.groupby("pitcher_id", sort=True)
        out = pd.DataFrame(index=g.size().index)
        out["track_hist_n_log"] = np.log1p(g.size().astype(np.float64))
        for m in metric_cols:
            out[f"track_{m}_avg"] = g[m].mean()
            out[f"track_{m}_dev"] = g[m].std().fillna(0.0)
        counts = (
            sub.groupby(["pitcher_id", TRACK_GROUP_COL], sort=True, observed=True)
            .size().unstack(fill_value=0).reindex(columns=list(TRACK_GROUPS), fill_value=0)
        )
        denom = counts.sum(axis=1).replace(0, np.nan)
        for gname in TRACK_GROUPS:
            out[f"track_mix_{gname}"] = (
                (counts[gname] / denom).reindex(out.index).fillna(0.0)
            )
        gm = sub.groupby(["pitcher_id", TRACK_GROUP_COL], sort=True, observed=True)[
            metric_cols
        ].mean()
        for gname in TRACK_GROUPS:
            try:
                block = gm.xs(gname, level=TRACK_GROUP_COL)
            except KeyError:
                block = pd.DataFrame(index=out.index, columns=metric_cols, dtype=float)
            for m in metric_cols:
                col = block[m] if m in block.columns else pd.Series(dtype=float)
                out[f"track_{gname}_{m}_avg"] = col.reindex(out.index)
        out["track_hist_missing"] = 0.0
        banks[s] = out[TRACK_COLS].astype(np.float64)
    return banks


def track_matrix(rows: pd.DataFrame, banks: dict) -> pd.DataFrame:
    """행별 프로파일 조회. 각 행은 자기 (pitcher_id, season) 만 사용한다."""
    if rows[SEASON_COL].isna().any() or rows["pitcher_id"].isna().any():
        raise RuntimeError("TRACK: season/pitcher_id 에 결측이 있다")
    seasons = pd.to_numeric(rows[SEASON_COL], errors="raise").astype(int)
    pids = pd.to_numeric(rows["pitcher_id"], errors="raise").astype("int64")

    parts = []
    for s, idx in rows.groupby(seasons).groups.items():
        bank = banks.get(int(s))
        sel_pids = pids.loc[idx]
        if bank is None or len(bank) == 0:
            mat = pd.DataFrame(np.nan, index=idx, columns=TRACK_COLS)
            mat["track_hist_missing"] = 1.0
        else:
            mat = bank.reindex(sel_pids.to_numpy())
            mat.index = idx
            found = mat["track_hist_n_log"].notna()
            mat["track_hist_missing"] = np.where(found, 0.0, 1.0)
        parts.append(mat)

    out = pd.concat(parts).loc[rows.index, TRACK_COLS]
    if out.isna().all(axis=None) and len(out):
        print("TRACK ⚠️ 프로파일이 전 행 결측이다 — 매핑/은행을 확인하라")
    return out


def build_track_member_views(views: dict, rows_train, rows_apply, banks):
    """TREATMENT 뷰 + TRACK 54열 = TRACK 멤버 뷰. TREATMENT 원본은 불변.

    rows_apply=None 이면 train 쪽만 만든다 — 최종 전체 적합에서 apply 쪽 사본
    (~수 GB)을 만들 이유가 없다. 첫 실행이 정확히 이 지점에서 OOM 으로 죽었다.
    """
    overlap = set(TRACK_COLS) & set(views["treatment_features"])
    if overlap:
        raise RuntimeError(f"TRACK 열이 TREATMENT 와 겹친다: {sorted(overlap)}")

    mat_tr = track_matrix(rows_train, banks).reset_index(drop=True)
    cover_tr = float((mat_tr["track_hist_missing"] == 0.0).mean())
    train_view = pd.concat(
        [views["train_treatment"].reset_index(drop=True), mat_tr], axis=1
    )
    del mat_tr

    if rows_apply is None:
        apply_view = None
        cover_ap = float("nan")
        print(f"TRACK coverage: train {cover_tr:.4f} (apply 생략)")
    else:
        mat_ap = track_matrix(rows_apply, banks).reset_index(drop=True)
        cover_ap = float((mat_ap["track_hist_missing"] == 0.0).mean())
        apply_view = pd.concat(
            [views["apply_treatment"].reset_index(drop=True), mat_ap], axis=1
        )
        del mat_ap
        print(f"TRACK coverage: train {cover_tr:.4f} / apply {cover_ap:.4f}")

    features = list(views["treatment_features"]) + list(TRACK_COLS)
    return train_view, apply_view, features, (cover_tr, cover_ap)


def make_l1_views(df_train_raw, df_apply_raw, base_features, scope):
    pitcher_stats = build_pitcher_season_stats(df_train_raw)
    tr_exact, tr_exact_diag = add_pitcher_exact_asof_features(df_train_raw, pitcher_stats)
    ap_exact, ap_exact_diag = add_pitcher_exact_asof_features(df_apply_raw, pitcher_stats)

    exact_features = list(base_features)
    for col in EXACT_FEATURES:
        if col not in exact_features:
            exact_features.append(col)

    tr_control, control_features, hbeb_priors = apply_hbeb_pair(
        df_train=tr_exact,
        df_apply=tr_exact,
        base_features=exact_features,
    )
    ap_control, apply_features, _ = apply_hbeb_pair(
        df_train=tr_exact,
        df_apply=ap_exact,
        base_features=exact_features,
    )
    if list(apply_features) != list(control_features):
        raise RuntimeError("HB-EB feature order mismatch")

    pc_tr, pc_cov_tr = build_pitcher_conditional_for_rows(
        rows=df_train_raw,
        available_history=df_train_raw,
        scope=f"{scope}_train",
    )
    pc_ap, pc_cov_ap = build_pitcher_conditional_for_rows(
        rows=df_apply_raw,
        available_history=df_train_raw,
        scope=f"{scope}_apply",
    )
    pc_tr = mask_pitcher_conditional_to_r_only(pc_tr, df_train_raw)
    pc_ap = mask_pitcher_conditional_to_r_only(pc_ap, df_apply_raw)

    tr_treat = tr_control.copy()
    ap_treat = ap_control.copy()
    for col in PC_FEATURES:
        if col in tr_treat.columns or col in ap_treat.columns:
            raise RuntimeError(f"Duplicate PC feature: {col}")
        tr_treat[col] = pc_tr[col]
        ap_treat[col] = pc_ap[col]

    # --- SUCCESS_FULL 13 (v002 = SJ v042 이식본에서 승계) ---
    state_tr = build_success_state_for_rows(
        rows=df_train_raw, available_history=df_train_raw)
    state_ap = build_success_state_for_rows(
        rows=df_apply_raw, available_history=df_train_raw)
    for col in SUCCESS_FEATURES:
        if col in tr_treat.columns or col in ap_treat.columns:
            raise RuntimeError(f"Duplicate SUCCESS feature: {col}")
        tr_treat[col] = state_tr[col]
        ap_treat[col] = state_ap[col]

    treatment_features = (
        list(control_features) + list(PC_FEATURES) + list(SUCCESS_FEATURES)
    )
    control_cat_cols = [c for c in CAT_COLS if c in control_features]
    treatment_cat_cols = [c for c in CAT_COLS if c in treatment_features]

    if len(control_features) != EXPECTED_CONTROL_FEATURES:
        raise RuntimeError(
            f"control feature count {len(control_features)} != {EXPECTED_CONTROL_FEATURES}"
        )
    if len(treatment_features) != EXPECTED_TREATMENT_FEATURES:
        raise RuntimeError(
            f"treatment feature count {len(treatment_features)} != {EXPECTED_TREATMENT_FEATURES}"
        )

    return {
        "train_control": tr_control,
        "apply_control": ap_control,
        "train_treatment": tr_treat,
        "apply_treatment": ap_treat,
        "control_features": control_features,
        "treatment_features": treatment_features,
        "control_cat_cols": control_cat_cols,
        "treatment_cat_cols": treatment_cat_cols,
        "pitcher_season_stats": pitcher_stats,
        "hbeb_priors": hbeb_priors,
        "exact_diag_train": tr_exact_diag,
        "exact_diag_apply": ap_exact_diag,
        "pc_coverage_train": pc_cov_tr,
        "pc_coverage_apply": pc_cov_ap,
    }


def fit_fold_model(
    train_view, val_view, features, cat_cols, y_train, y_val, label, seed=BASE_SEED
):
    X_tr = prepare_features(train_view[features], cat_cols)
    X_va = prepare_features(val_view[features], cat_cols)
    # 3 seed 프로토콜 (38_* §8-3 조건 1)
    _params = dict(MODEL_PARAMS)
    _params["random_seed"] = int(seed)
    model = CatBoostRegressor(**_params)
    print(f"\n[{label}] fold4 training | rows={len(X_tr):,} -> {len(X_va):,} | features={len(features)}")
    t0 = time.time()
    model.fit(
        X_tr,
        y_train,
        eval_set=(X_va, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )
    elapsed = time.time() - t0
    pred = np.clip(np.asarray(model.predict(X_va), dtype=np.float64), 0.0, 1.0)
    best_zero = int(model.get_best_iteration())
    if best_zero < 0:
        raise RuntimeError(f"{label} best iteration invalid: {best_zero}")
    return model, pred, best_zero, elapsed


def fit_fold_seeded(train_view, val_view, features, cat_cols, y_train, y_val,
                    label, seeds):
    """fold 를 시드마다 돌려 (평균 예측, 시드별 트리 수, 총 소요) 를 돌려준다 (E5).

    배포 L1 이 시드평균이므로 L2 가 학습할 base_pred 도 시드평균이어야 한다.
    단일 시드 fold 로 L2 를 적합하면 배포 시 base_pred 가 13.3%(std 기준) 어긋난다.
    """
    preds, trees, total = [], [], 0.0
    for sd in tuple(seeds):
        _, pred, best_zero, sec = fit_fold_model(
            train_view, val_view, features, cat_cols, y_train, y_val,
            f"{label}[seed={sd}]", seed=int(sd),
        )
        preds.append(np.asarray(pred, dtype=np.float64))
        trees.append(int(best_zero) + 1)
        total += sec
    return np.mean(np.column_stack(preds), axis=1), trees, total


def fit_dynamic_fold_seeded(train_view, val_view, features, cat_cols,
                            y_train, y_val, anchor_train, anchor_val,
                            label, seeds):
    """DYNAMIC fold 시드 루프. 예측과 보정을 각각 평균한다 (E5)."""
    preds, corrs, trees, total = [], [], [], 0.0
    for sd in tuple(seeds):
        _, pred, corr, best_zero, sec = fit_dynamic_fold_model(
            train_view=train_view, val_view=val_view,
            features=features, cat_cols=cat_cols,
            y_train=y_train, y_val=y_val,
            anchor_train=anchor_train, anchor_val=anchor_val,
            label=f"{label}[seed={sd}]", seed=int(sd),
        )
        preds.append(np.asarray(pred, dtype=np.float64))
        corrs.append(np.asarray(corr, dtype=np.float64))
        trees.append(int(best_zero) + 1)
        total += sec
    return (np.mean(np.column_stack(preds), axis=1),
            np.mean(np.column_stack(corrs), axis=1),
            trees, total)


def fit_full_model(view, features, cat_cols, y, iterations, label, seeds=None):
    """최종 full-fit. L1_SEEDS 개수만큼 학습해 모델 리스트를 돌려준다 (E5).

    seeds=(BASE_SEED,) 이면 시드평균 도입 이전과 비트 단위 동일하다.
    """
    seeds = tuple(L1_SEEDS if seeds is None else seeds)
    iters = (list(iterations) if isinstance(iterations, (list, tuple))
             else [int(iterations)] * len(seeds))
    if len(iters) != len(seeds):
        raise ValueError(f"{label}: iterations {len(iters)} != seeds {len(seeds)}")
    X = prepare_features(view[features], cat_cols)
    models, params_list, total = [], [], 0.0
    for sd, it in zip(seeds, iters):
        params = {**MODEL_PARAMS, "iterations": int(it), "random_seed": int(sd)}
        model = CatBoostRegressor(**params)
        print(f"\n[{label} seed={sd}] final full training | rows={len(X):,} "
              f"| features={len(features)} | trees={it}")
        t0 = time.time()
        model.fit(X, y, cat_features=cat_cols, verbose=100)
        total += time.time() - t0
        if hasattr(model, "feature_names_") and list(model.feature_names_) != list(features):
            raise RuntimeError(f"{label} final feature order mismatch (seed={sd})")
        models.append(model)
        params_list.append(params)
    return models, params_list, total


# =============================================================================
# Main
# =============================================================================

def run_shift_fold(train, base_features, target_year: int, track_banks=None) -> dict:
    """SHIFT 적합용 과거 fold 하나를 돌린다 (E3c).

    2024 fold 와 **완전히 같은 절차**다. 다만 SHIFT 에 필요한 것만 돌려준다.
    < target_year 로 학습해 target_year 를 예측하므로, 각 연도의 L1 예측은
    그 연도를 보지 않은 정직한 OOF 다.
    """
    fold_train = train.loc[train[SEASON_COL] < target_year].copy().reset_index(drop=True)
    fold_val = train.loc[train[SEASON_COL] == target_year].copy().reset_index(drop=True)
    if len(fold_train) == 0 or len(fold_val) == 0:
        raise RuntimeError(f"SHIFT fold {target_year}: 학습 또는 검증 행이 없다")

    print("\n" + "-" * 112)
    print(f"SHIFT FOLD  {int(fold_train[SEASON_COL].min())}~{target_year - 1} -> {target_year}"
          f"   (train {len(fold_train):,} / val {len(fold_val):,})")
    print("-" * 112)

    views = make_l1_views(
        df_train_raw=fold_train,
        df_apply_raw=fold_val,
        base_features=base_features,
        scope=f"shift{target_year}",
    )
    y_tr = fold_train[TARGET].to_numpy(dtype=np.float64)
    y_va = fold_val[TARGET].to_numpy(dtype=np.float64)

    pred_control, _, sec_c = fit_fold_seeded(
        views["train_control"], views["apply_control"],
        views["control_features"], views["control_cat_cols"],
        y_tr, y_va, f"CONTROL[{target_year}]", L1_SEEDS,
    )
    pred_treat, _, sec_t = fit_fold_seeded(
        views["train_treatment"], views["apply_treatment"],
        views["treatment_features"], views["treatment_cat_cols"],
        y_tr, y_va, f"TREATMENT[{target_year}]", L1_SEEDS,
    )

    # E6: 배포 L1 과 같은 합성으로 잔차를 만들려면 SHIFT fold 도 TRACK 을 포함해야
    # 한다 (47_* 규약 4 — 사전분석은 배포 축에서 잰다).
    sec_k = 0.0
    if TRACK_ENABLED and TRACK_W != 0.0:
        if track_banks is None:
            raise RuntimeError("run_shift_fold: TRACK_ENABLED 인데 track_banks 가 없다")
        tr_view_k, ap_view_k, feats_k, _ = build_track_member_views(
            views, fold_train, fold_val, track_banks,
        )
        pred_track, _, sec_k = fit_fold_seeded(
            tr_view_k, ap_view_k, feats_k, views["treatment_cat_cols"],
            y_tr, y_va, f"TRACK[{target_year}]", L1_SEEDS,
        )
        r_member = (1.0 - TRACK_W) * pred_treat + TRACK_W * pred_track
        del tr_view_k, ap_view_k
    else:
        r_member = pred_treat

    league_means = build_season_league_means(fold_train)
    prev_tr = compute_prev_anchor(
        rows=fold_train,
        pitcher_season_stats=views["pitcher_season_stats"],
        season_league_means=league_means,
    )
    prev_va = compute_prev_anchor(
        rows=fold_val,
        pitcher_season_stats=views["pitcher_season_stats"],
        season_league_means=league_means,
    )
    anchor_tr, _ = compute_dynamic_anchor(
        exact_view=views["train_control"], prev_anchor=prev_tr,
    )
    anchor_va, _ = compute_dynamic_anchor(
        exact_view=views["apply_control"], prev_anchor=prev_va,
    )
    pred_dynamic, _, _, sec_d = fit_dynamic_fold_seeded(
        train_view=views["train_control"],
        val_view=views["apply_control"],
        features=views["control_features"],
        cat_cols=views["control_cat_cols"],
        y_train=y_tr,
        y_val=y_va,
        anchor_train=anchor_tr,
        anchor_val=anchor_va,
        label=f"DYNAMIC_ASOF[{target_year}]",
        seeds=L1_SEEDS,
    )

    gt = fold_val[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    f_mask = gt == "F"
    p_main = np.where(f_mask, pred_control, r_member)
    weights = blend_weights(gt, w_r=W_R, w_f=F_BLEND_WEIGHT)
    l1_oof = blend(p_main, pred_dynamic, weights)

    print(f"  L1 {target_year} Brier : {brier_score(y_va, l1_oof):.10f}"
          f"   R행 {int((~f_mask).sum()):,}   {sec_c + sec_t + sec_k + sec_d:.1f}s")
    return {
        "year": int(target_year),
        "rows": fold_val,
        "y": y_va,
        "l1_oof": l1_oof,
        "r_mask": ~f_mask,
        "seconds": sec_c + sec_t + sec_k + sec_d,
    }


def main():
    total_start = time.time()

    print("=" * 112)
    print(EXPERIMENT_NAME)
    print("=" * 112)
    print("Standalone    : YES (no existing model/validation artifact)")
    if GPU_ENABLED:
        print(f"Compute       : GPU (devices={GPU_DEVICES}, n={_gpu_device_count()}, "
              f"border_count=254) — CPU 기준선과 비트 비동일 주의")
    else:
        print("Compute       : CPU")
    print("L1 base       : CatBoost Exact-ASOF5 + HB-EB + R-only PC FULL")
    print(
        f"R branch      : (1-{W_R:g})*TREATMENT + {W_R:g}*DYNAMIC_ASOF"
        + ("   [W_R=0 → v002 라우팅과 비트 동일]" if W_R == 0.0 else "   [E2b]")
    )
    print("F branch      : 0.75 * CONTROL + 0.25 * DYNAMIC_ASOF")
    print(f"Dynamic λ     : {DYNAMIC_LAMBDA}")
    print(f"Dynamic κ     : {DYNAMIC_KAPPA}")
    print(f"F blend w     : {F_BLEND_WEIGHT}")
    print("b_F           : OFF")
    print("L2            : ExtraTrees residual adapter")
    print("L2 features   : adapter core + Exact-ASOF5")
    print("PC in L2      : NO")
    print("Exact in L2   : YES")
    print("L2 routing    : R=L1+L2 / F=L1 protected")
    print(f"Adapter seeds : {ADAPTER_SEEDS}")
    print(f"FINAL ET cfg  : {FINAL_ADAPTER_CFG}")
    print(f"Shrink        : {ADAPTER_SHRINK}")
    print("L3            : OFF")
    if TRACK_ENABLED:
        print(f"TRACK member  : ON  w={TRACK_W:g}  tier={tuple(TRACK_TIERS_OK)}  "
              f"cols={len(TRACK_COLS)}  (R blend 전용, F 무접촉)")
    else:
        print("TRACK member  : OFF")
    print("Test usage    : HEADER ONLY")
    print("=" * 112)

    # -----------------------------------------------------------------
    # 0) Input / artifact guards
    # -----------------------------------------------------------------
    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"기존 model 보호를 위해 중단: {MODEL_DIR}\n"
            "기존 폴더를 이동/삭제하거나 EXPERIMENT_NAME을 변경하세요."
        )
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 validation 보호를 위해 중단: {VALIDATION_DIR}\n"
            "기존 폴더를 이동/삭제하거나 EXPERIMENT_NAME을 변경하세요."
        )

    # test.csv는 schema(header)만 사용한다.
    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns.tolist()
    base_features = [c for c in test_cols if c != ID]
    if "asof_pitcher_pitchmix_n" not in base_features:
        raise RuntimeError("canonical feature asof_pitcher_pitchmix_n missing from test schema")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    train[SEASON_COL] = pd.to_numeric(train[SEASON_COL], errors="raise").astype(int)
    if int(train[SEASON_COL].max()) != SOURCE_YEAR:
        raise RuntimeError(
            f"latest train season={int(train[SEASON_COL].max())}; expected {SOURCE_YEAR}"
        )

    required = {
        ID,
        TARGET,
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise KeyError(f"train required columns missing: {missing}")

    # -----------------------------------------------------------------
    # 0b) TRACK 프로파일 은행 (season < s 규율, 51_* §2)
    # -----------------------------------------------------------------
    track_banks = None
    if TRACK_ENABLED:
        track_history = load_track_history()
        track_mapping = load_track_mapping()
        track_banks = build_track_banks(track_history, track_mapping)
        for s in (2021, SOURCE_YEAR, EXPECTED_TARGET_YEAR):
            print(f"TRACK bank[<{s}] pitchers: {len(track_banks[s]):,}")
        del track_history            # 은행만 남긴다 — 원로그는 더 안 쓴다

    # -----------------------------------------------------------------
    # 1) Latest OOT fold: 2019~2023 -> 2024
    #    - CONTROL tree count
    #    - TREATMENT tree count
    #    - DYNAMIC tree count
    #    - actual danker L1 OOT prediction
    #    - R-only L2 residual source
    # -----------------------------------------------------------------
    fold_train = train.loc[train[SEASON_COL] < SOURCE_YEAR].copy().reset_index(drop=True)
    fold_val = train.loc[train[SEASON_COL] == SOURCE_YEAR].copy().reset_index(drop=True)
    if len(fold_train) == 0 or len(fold_val) == 0:
        raise RuntimeError("fold4 source/validation rows missing")

    fold_views = make_l1_views(
        df_train_raw=fold_train,
        df_apply_raw=fold_val,
        base_features=base_features,
        scope="fold4",
    )

    y_tr = fold_train[TARGET].to_numpy(dtype=np.float64)
    y24 = fold_val[TARGET].to_numpy(dtype=np.float64)

    # CONTROL
    pred_control24, control_trees, control_fold_time = fit_fold_seeded(
        fold_views["train_control"],
        fold_views["apply_control"],
        fold_views["control_features"],
        fold_views["control_cat_cols"],
        y_tr,
        y24,
        "CONTROL",
        L1_SEEDS,
    )

    # R TREATMENT
    pred_treat24, treatment_trees, treatment_fold_time = fit_fold_seeded(
        fold_views["train_treatment"],
        fold_views["apply_treatment"],
        fold_views["treatment_features"],
        fold_views["treatment_cat_cols"],
        y_tr,
        y24,
        "TREATMENT",
        L1_SEEDS,
    )

    # E6 TRACK member (TREATMENT 피처 + 프로파일 54열, R blend 전용)
    pred_track24 = None
    track_trees = None
    track_fold_time = 0.0
    track_cover_fold = None
    if TRACK_ENABLED:
        (
            train_track_view,
            apply_track_view,
            track_features_fold,
            track_cover_fold,
        ) = build_track_member_views(fold_views, fold_train, fold_val, track_banks)
        pred_track24, track_trees, track_fold_time = fit_fold_seeded(
            train_track_view,
            apply_track_view,
            track_features_fold,
            fold_views["treatment_cat_cols"],
            y_tr,
            y24,
            "TRACK",
            L1_SEEDS,
        )
        del train_track_view, apply_track_view

    # F DYNAMIC-ASOF expert
    fold_season_league_means = build_season_league_means(fold_train)
    prev_anchor_train = compute_prev_anchor(
        rows=fold_train,
        pitcher_season_stats=fold_views["pitcher_season_stats"],
        season_league_means=fold_season_league_means,
    )
    prev_anchor24 = compute_prev_anchor(
        rows=fold_val,
        pitcher_season_stats=fold_views["pitcher_season_stats"],
        season_league_means=fold_season_league_means,
    )
    dynamic_anchor_train, dynamic_diag_train = compute_dynamic_anchor(
        exact_view=fold_views["train_control"],
        prev_anchor=prev_anchor_train,
    )
    dynamic_anchor24, dynamic_diag24 = compute_dynamic_anchor(
        exact_view=fold_views["apply_control"],
        prev_anchor=prev_anchor24,
    )
    (
        pred_dynamic24,
        dynamic_correction24,
        dynamic_trees,
        dynamic_fold_time,
    ) = fit_dynamic_fold_seeded(
        train_view=fold_views["train_control"],
        val_view=fold_views["apply_control"],
        features=fold_views["control_features"],
        cat_cols=fold_views["control_cat_cols"],
        y_train=y_tr,
        y_val=y24,
        anchor_train=dynamic_anchor_train,
        anchor_val=dynamic_anchor24,
        label="DYNAMIC_ASOF",
        seeds=L1_SEEDS,
    )

    gt24 = fold_val[GAME_TYPE_COL].fillna("__MISSING__").astype(str).to_numpy()
    r24 = gt24 != "F"
    f24 = gt24 == "F"
    if not r24.any() or not f24.any():
        raise RuntimeError("2024 R/F routing rows missing")

    f_blend24 = np.clip(
        (1.0 - F_BLEND_WEIGHT) * pred_control24
        + F_BLEND_WEIGHT * pred_dynamic24,
        0.0,
        1.0,
    )
    # --- L1 합성: 라우팅을 곱셈 가중치로 (I2 + E6) ---------------------------
    # E6: R 주 예측기가 TREATMENT 단독에서 소수 가중 blend 로 바뀐다.
    #   R 행 : (1-W_R) * [ (1-TRACK_W)*TREATMENT + TRACK_W*TRACK ] + W_R*DYNAMIC
    #   F 행 : (1-0.25)*control + 0.25*dynamic      = 기존 f_blend (TRACK 무접촉)
    # TRACK_W = 0 또는 TRACK_ENABLED=False 면 경로 자체를 우회해 (부동소수 연산
    # 없이 pred_treat24 를 그대로 쓴다) v005 와 비트 단위 동일하다.
    # W_R = 0 이고 TRACK 중립이면 v002 의 np.where 라우팅과도 비트 단위 동일하다.
    if TRACK_ENABLED and TRACK_W != 0.0:
        if pred_track24 is None:
            raise RuntimeError("TRACK_ENABLED 인데 TRACK fold 예측이 없다")
        r_member24 = (1.0 - TRACK_W) * pred_treat24 + TRACK_W * pred_track24
    else:
        r_member24 = pred_treat24
    p_main24 = np.where(f24, pred_control24, r_member24)
    w24 = blend_weights(gt24, w_r=W_R, w_f=F_BLEND_WEIGHT)
    l1_oof24 = blend(p_main24, pred_dynamic24, w24)

    if W_R == 0.0:                                  # I5 회귀 (TRACK 반영판)
        _ref = np.clip(np.where(r24, r_member24, f_blend24), 0.0, 1.0)
        if not np.array_equal(l1_oof24, _ref):
            raise RuntimeError(
                "L1 합성 회귀 실패: W_R=0 인데 참조 라우팅과 값이 다르다 "
                f"(max|diff|={float(np.abs(l1_oof24 - _ref).max()):.3e})"
            )

    # control_trees / treatment_trees / dynamic_trees 는 fit_fold_seeded 가
    # 시드별 리스트로 돌려준다 (E5). 각 시드가 자기 fold best iteration 을 쓴다.

    # Paired 2024 F diagnostics.
    y24_f = y24[f24]
    control_f_brier = brier_score(y24_f, pred_control24[f24])
    dynamic_f_brier = brier_score(y24_f, pred_dynamic24[f24])
    blend_f_brier = brier_score(y24_f, f_blend24[f24])
    f_baseline = float(np.mean(y24_f) * (1.0 - np.mean(y24_f)))
    if f_baseline > 0.0:
        dynamic_f_gain_pt = 100000.0 * (control_f_brier - dynamic_f_brier) / f_baseline
        blend_f_gain_pt = 100000.0 * (control_f_brier - blend_f_brier) / f_baseline
    else:
        dynamic_f_gain_pt = 0.0
        blend_f_gain_pt = 0.0

    print("\n" + "=" * 112)
    print("LATEST FOLD RESULT (2019~2023 -> 2024)")
    print("=" * 112)
    print(f"CONTROL best tree count  : {control_trees}")
    print(f"TREATMENT best tree count: {treatment_trees}")
    print(f"DYNAMIC best tree count  : {dynamic_trees}")
    print(f"CONTROL 2024 Brier       : {brier_score(y24, pred_control24):.10f}")
    print(f"TREATMENT 2024 Brier     : {brier_score(y24, pred_treat24):.10f}")
    print(f"DYNAMIC 2024 Brier       : {brier_score(y24, pred_dynamic24):.10f}")
    print(f"DANKER L1 2024 Brier     : {brier_score(y24, l1_oof24):.10f}")
    print(f"DANKER L1 2024 BSS       : {calculate_bss(y24, l1_oof24):+.4f}")
    print(f"2024 R/F rows            : {int(r24.sum()):,} / {int(f24.sum()):,}")
    print(f"CONTROL F Brier          : {control_f_brier:.10f}")
    print(f"DYNAMIC F Brier          : {dynamic_f_brier:.10f}")
    print(f"BLEND F Brier            : {blend_f_brier:.10f}")
    print(f"DYNAMIC F gain           : {dynamic_f_gain_pt:+.3f} pt")
    print(f"BLEND F gain             : {blend_f_gain_pt:+.3f} pt")
    print(f"Dynamic mean alpha       : {dynamic_diag24['mean_alpha']:.6f}")
    print(f"Dynamic mean |shift|     : {dynamic_diag24['mean_abs_anchor_shift']:.6f}")
    if TRACK_ENABLED and pred_track24 is not None:
        print(f"TRACK best tree count    : {track_trees}")
        print(f"TRACK 2024 Brier         : {brier_score(y24, pred_track24):.10f}")

    # --- E6 진단: TRACK_W 격자 재합성 (재학습 0회, 2024 R행 paired vs w=0) -----
    # 배포축(47_* 규약 4): base = 배포되는 L1 합성 그대로, 여기서는 R행 within 지표.
    # F 행은 TRACK 이 구조적으로 못 건드린다 — 아래에서 실측으로도 확인한다.
    track_w_diag = []
    if TRACK_ENABLED and pred_track24 is not None:
        y24_r = y24[r24]
        b0_r = float(np.mean(y24_r) * (1.0 - np.mean(y24_r)))
        base_main = np.where(f24, pred_control24, pred_treat24)
        base_l1 = blend(base_main, pred_dynamic24, w24)
        base_b, base_lv, base_sh = brier_parts(base_l1[r24], y24_r)
        print("\n  TRACK_W 격자 (2024 R행, within-R pt, paired vs w=0):")
        print(f"  {'w':>6}{'Δtotal':>12}{'Δlevel':>12}{'Δshape':>12}")
        for wt in TRACK_W_GRID:
            mem = (1.0 - wt) * pred_treat24 + wt * pred_track24 if wt != 0.0 \
                else pred_treat24
            l1w = blend(np.where(f24, pred_control24, mem), pred_dynamic24, w24)
            bb, lv, sh = brier_parts(l1w[r24], y24_r)
            row = {
                "w": float(wt),
                "d_total_pt": 1e5 * (base_b - bb) / b0_r,
                "d_level_pt": 1e5 * (base_lv - lv) / b0_r,
                "d_shape_pt": 1e5 * (base_sh - sh) / b0_r,
            }
            track_w_diag.append(row)
            print(f"  {wt:>6.2f}{row['d_total_pt']:>+12.3f}"
                  f"{row['d_level_pt']:>+12.3f}{row['d_shape_pt']:>+12.3f}")
        # F 무접촉 실측 — 배포 합성과 w=0 합성이 F 행에서 완전히 같아야 한다.
        if not np.array_equal(l1_oof24[f24], base_l1[f24]):
            raise RuntimeError("E6 위반: TRACK 이 F 행 예측을 바꿨다")

    # Save the actual OOT L1 source before final training.
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    oof_cols = {
        ID: fold_val[ID].to_numpy(),
        SEASON_COL: fold_val[SEASON_COL].to_numpy(),
        GAME_TYPE_COL: fold_val[GAME_TYPE_COL].to_numpy(),
        "y_true": y24,
        "CONTROL": pred_control24,
        "TREATMENT": pred_treat24,
        "DYNAMIC_ANCHOR": dynamic_anchor24,
        "DYNAMIC": pred_dynamic24,
        "F_BLEND": f_blend24,
        "L1_ROUTED": l1_oof24,
    }
    if TRACK_ENABLED and pred_track24 is not None:
        oof_cols["TRACK"] = pred_track24
        oof_cols["R_MEMBER"] = r_member24
    oof_df = pd.DataFrame(oof_cols)
    oof_df.to_csv(OOF_2024_PATH, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # 1b) E3c SHIFT — 과거 전 연도 OOF 잔차로 절편 없는 Ridge 를 적합
    # -----------------------------------------------------------------
    shift_fit = None
    shift_corr_2024 = None
    if SHIFT_SCALE != 0.0:
        print("\n" + "=" * 112)
        print(f"E3c SHIFT — 적합 연도 {tuple(SHIFT_FIT_YEARS)} · 도메인 R 전체 · 절편 없음")
        print(f"            η={SHIFT_SCALE}  α={SHIFT_RIDGE_ALPHA:.0e}  feature clip=±{SHIFT_CLIP}")
        print("=" * 112)

        folds = {}
        for year in sorted(SHIFT_FIT_YEARS):
            if int(year) == SOURCE_YEAR:
                # 2024 는 위에서 이미 돌렸다. 다시 돌리지 않는다.
                folds[int(year)] = {
                    "year": SOURCE_YEAR, "rows": fold_val, "y": y24,
                    "l1_oof": l1_oof24, "r_mask": r24, "seconds": 0.0,
                }
            else:
                folds[int(year)] = run_shift_fold(
                    train, base_features, int(year), track_banks=track_banks,
                )

        mats, resids, years, sizes = [], [], [], []
        for year in sorted(folds):
            fd = folds[year]
            m = fd["r_mask"]
            mat = build_shift_matrix(fd["rows"]).to_numpy(dtype=np.float64)[m]
            res = (fd["y"] - fd["l1_oof"])[m]
            mats.append(mat)
            resids.append(res)
            years.append(np.full(int(m.sum()), year, dtype=np.int64))
            sizes.append(int(m.sum()))

        X_shift = np.vstack(mats)
        r_shift = np.concatenate(resids)
        year_vec = np.concatenate(years)

        shift_fit = fit_shift_ridge(
            X_shift, r_shift, alpha=SHIFT_RIDGE_ALPHA, clip=SHIFT_CLIP,
        )
        shift_fit["cols"] = list(SHIFT_COLS)
        shift_fit["fit_years"] = [int(y) for y in sorted(folds)]
        shift_fit["fit_rows_by_year"] = {int(y): int(n) for y, n in zip(sorted(folds), sizes)}

        print(f"\n적합 표본     : {shift_fit['n_fit']:,} R행  "
              f"({', '.join(f'{y}:{n:,}' for y, n in shift_fit['fit_rows_by_year'].items())})")
        print(f"잔차 평균/표준편차: {r_shift.mean():+.8f} / {r_shift.std():.8f}")

        # ---- 진단: 각 적합 연도에서의 in-sample 이득 (정직한 측정이 아니다) ----
        # 정직한 walk-forward 는 46_* 에서 재학습 없이 이미 쟀다.
        # 여기서는 계수가 죽지 않았는지 확인하는 용도로만 본다.
        corr_all = shift_correction(
            X_shift, shift_fit["median"], shift_fit["mu"], shift_fit["sd"],
            shift_fit["beta"], shift_fit["clip"],
        )
        print(f"보정량 |ηc|   : 평균 {np.abs(SHIFT_SCALE * corr_all).mean():.6f}  "
              f"최대 {np.abs(SHIFT_SCALE * corr_all).max():.6f}")

        rows_diag = []
        base_all = np.concatenate([folds[y]["l1_oof"][folds[y]["r_mask"]] for y in sorted(folds)])
        y_all = np.concatenate([folds[y]["y"][folds[y]["r_mask"]] for y in sorted(folds)])
        print(f"\n{'연도':>6}{'R행':>10}{'in-sample BSS pt':>20}")
        for year in sorted(folds):
            sel = year_vec == year
            yv = y_all[sel]
            p0 = base_all[sel]
            p1 = np.clip(p0 + SHIFT_SCALE * corr_all[sel], 0.0, 1.0)
            b0 = float(np.mean(yv) * (1.0 - np.mean(yv)))
            gain = 1e5 * (float(np.mean((yv - p0) ** 2)) - float(np.mean((yv - p1) ** 2))) / b0
            print(f"{year:>6}{int(sel.sum()):>10,}{gain:>+20.2f}")
            rows_diag.append({
                "year": int(year), "n_R": int(sel.sum()),
                "brier_before": float(np.mean((yv - p0) ** 2)),
                "brier_after": float(np.mean((yv - p1) ** 2)),
                "in_sample_bss_pt": float(gain),
                "mean_abs_corr": float(np.abs(SHIFT_SCALE * corr_all[sel]).mean()),
            })

        pd.DataFrame(rows_diag).to_csv(
            VALIDATION_DIR / "shift_fit_diag.csv", index=False, encoding="utf-8-sig",
        )
        pd.DataFrame({
            "feature": list(SHIFT_COLS),
            "beta": shift_fit["beta"],
            "median": shift_fit["median"],
            "mu": shift_fit["mu"],
            "sd": shift_fit["sd"],
        }).to_csv(
            VALIDATION_DIR / "shift_coefficients.csv", index=False, encoding="utf-8-sig",
        )

        sel24 = year_vec == SOURCE_YEAR
        shift_corr_2024 = corr_all[sel24]

        # 메모리 반환 (OOM 방지): SHIFT 적합 원천은 이후 단계에서 쓰지 않는다.
        # folds 는 각 연도의 fold_val DataFrame(~25만 행)을 물고 있다.
        del folds, mats, resids, X_shift, r_shift, base_all, y_all, corr_all
        gc.collect()

    # -----------------------------------------------------------------
    # 2) Final full L1 from all official train (2019~2024)
    # -----------------------------------------------------------------
    # 메모리 반환 (OOM 방지): fold4 학습/적용 뷰는 이후 단계에서 쓰지 않는다.
    # (L2 는 raw24=fold_val 원본 행에서 자체 변환을 다시 만든다.)
    fold_train_seasons = sorted(fold_train[SEASON_COL].unique().tolist())
    n_fold_train = int(len(fold_train))
    for _k in ("train_control", "train_treatment", "apply_control", "apply_treatment"):
        fold_views.pop(_k, None)
    del fold_train
    gc.collect()

    final_views = make_l1_views(
        df_train_raw=train,
        df_apply_raw=train,
        base_features=base_features,
        scope="final",
    )
    y_full = train[TARGET].to_numpy(dtype=np.float64)

    control_models, control_final_params, control_final_time = fit_full_model(
        final_views["train_control"],
        final_views["control_features"],
        final_views["control_cat_cols"],
        y_full,
        control_trees,
        "CONTROL",
    )

    treatment_models, treatment_final_params, treatment_final_time = fit_full_model(
        final_views["train_treatment"],
        final_views["treatment_features"],
        final_views["treatment_cat_cols"],
        y_full,
        treatment_trees,
        "TREATMENT",
    )

    # E6 TRACK member final fit (train 전 행, 각 행은 자기 season 이전 은행만 본다)
    track_models = None
    track_final_params = None
    track_final_time = 0.0
    track_features_full = None
    track_cover_full = None
    if TRACK_ENABLED:
        (
            train_track_full,
            _unused_apply,
            track_features_full,
            track_cover_full,
        ) = build_track_member_views(final_views, train, None, track_banks)
        track_models, track_final_params, track_final_time = fit_full_model(
            train_track_full,
            track_features_full,
            final_views["treatment_cat_cols"],
            y_full,
            track_trees,
            "TRACK",
        )
        del train_track_full

    # Dynamic-ASOF expert final fit.  The full-history bank is saved in the artifact
    # so script.py can reconstruct A_prev and A_t for each 2025 row independently.
    full_season_league_means = build_season_league_means(train)
    full_pitcher_stats = final_views["pitcher_season_stats"]
    prev_anchor_full = compute_prev_anchor(
        rows=train,
        pitcher_season_stats=full_pitcher_stats,
        season_league_means=full_season_league_means,
    )
    dynamic_anchor_full, dynamic_diag_full = compute_dynamic_anchor(
        exact_view=final_views["train_control"],
        prev_anchor=prev_anchor_full,
    )
    dynamic_models, dynamic_final_params, dynamic_final_time = fit_dynamic_full_model(
        view=final_views["train_control"],
        features=final_views["control_features"],
        cat_cols=final_views["control_cat_cols"],
        y=y_full,
        anchor=dynamic_anchor_full,
        iterations=dynamic_trees,
        label="DYNAMIC_ASOF",
    )

    # E3c 진단: L2 보정과 SHIFT 보정이 같은 것을 배우고 있는가?
    # 46_* 의 walk-forward 는 base 가 L1 이라 L2 를 보지 않았다. 여기서 겹침을
    # 실제로 재서 다음 실험의 근거로 남긴다. 실패해도 학습을 막지 않는다.
    def _report_shift_l2_overlap(l2_models_dict, X_matrix, r_mask_24):
        if shift_corr_2024 is None:
            return
        try:
            preds = [
                np.asarray(m.predict(X_matrix), dtype=np.float64)
                for m in l2_models_dict.values()
            ]
            l2c = np.mean(np.column_stack(preds), axis=1)[r_mask_24]
            a = ADAPTER_SHRINK * l2c
            b = SHIFT_SCALE * shift_corr_2024
            if len(a) != len(b):
                print(f"\n[진단] SHIFT/L2 겹침 측정 생략 — 길이 {len(a)} vs {len(b)}")
                return
            rho = float(np.corrcoef(a, b)[0, 1])
            print("\n" + "-" * 112)
            print("E3c 진단 — L2 보정과 SHIFT 보정의 겹침 (2024 R행, in-sample L2)")
            print("-" * 112)
            print(f"  corr(w_l2·L2corr, η·shift) : {rho:+.4f}")
            print(f"  |w_l2·L2corr| 평균          : {np.abs(a).mean():.6f}")
            print(f"  |η·shift|     평균          : {np.abs(b).mean():.6f}")
            print("  ⚠️ L2 는 2024 에 in-sample 이므로 실제 겹침의 상한으로 읽는다.")
        except Exception as exc:                       # 진단 실패가 학습을 막지 않는다
            print(f"\n[진단] SHIFT/L2 겹침 측정 실패: {exc}")

    pc_tables_by_source_season = build_pc_tables_by_source_season(train)
    success_history_artifact = build_success_history_artifact(train)

    # -----------------------------------------------------------------
    # 3) Final R-only L2 on genuine 2024 OOT residuals from danker L1
    # -----------------------------------------------------------------
    raw24 = fold_val.copy().reset_index(drop=True)
    residual24 = y24 - l1_oof24

    adapter_prep = fit_adapter_prep(raw24)
    core24 = transform_adapter_core(adapter_prep, raw24, l1_oof24)

    # Safe: Exact-ASOF transform uses each row's own official ASOF plus only
    # completed prior-season labeled history.  It does not aggregate test rows.
    exact24 = exact5_matrix(raw24, full_pitcher_stats)
    X24 = pd.concat([core24, exact24], axis=1)

    if X24.columns.duplicated().any():
        dup = X24.columns[X24.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate L2 columns: {dup}")
    if not np.isfinite(X24.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("L2 X24 contains NaN/Inf")

    print("\n" + "=" * 112)
    print("FINAL L2 TRAINING")
    print("=" * 112)
    print(f"2024 rows      : {len(raw24):,}")
    print(f"R residual rows: {int(r24.sum()):,}")
    print(f"L2 features    : {X24.shape[1]} ({core24.shape[1]} core + {exact24.shape[1]} Exact)")
    print(f"Residual mean  : {residual24.mean():+.8f}")
    print(f"Residual std   : {residual24.std():.8f}")

    l2_models, l2_fit_diag, l2_r_mask, l2_f_mask = fit_final_l2_models(
        X24=X24,
        raw24=raw24,
        residual24=residual24,
    )

    _report_shift_l2_overlap(l2_models, X24, l2_r_mask)

    # -----------------------------------------------------------------
    # 4) Save one self-contained artifact
    # -----------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "Danker weighted-blend CatBoost L1 + R-only ExtraTrees L2",
        "random_seed": BASE_SEED,
        "expected_target_year": EXPECTED_TARGET_YEAR,

        "control": {
            "models": control_models,
            "features": list(final_views["control_features"]),
            "cat_cols": list(final_views["control_cat_cols"]),
            "final_iterations": control_trees,
            "model_params": control_final_params,
            "seeds": [int(x) for x in L1_SEEDS],
        },

        "treatment": {
            "models": treatment_models,
            "features": list(final_views["treatment_features"]),
            "cat_cols": list(final_views["treatment_cat_cols"]),
            "final_iterations": treatment_trees,
            "model_params": treatment_final_params,
            "seeds": [int(x) for x in L1_SEEDS],
            "pc_features": list(PC_FEATURES),
        },

        # E6 TRACK — 추론은 이 블록의 은행 lookup 만 쓴다 (재계산 0).
        "track": None if not TRACK_ENABLED else {
            "models": track_models,
            "features": list(track_features_full),
            "cat_cols": list(final_views["treatment_cat_cols"]),
            "final_iterations": track_trees,
            "model_params": track_final_params,
            "seeds": [int(x) for x in L1_SEEDS],
            "track_cols": list(TRACK_COLS),
            # 2025 추론용 은행: 공식 로그 season<=2024 전체 집계. pitcher_id 인덱스.
            "bank_target": track_banks[EXPECTED_TARGET_YEAR],
            "tiers": list(TRACK_TIERS_OK),
            "history_source": "official trackman_history.csv (2019~2024)",
            "mapping_source": (
                "data/trackman/pitcher_mapping.csv — 공식 train/trackman 통계 결합, "
                "test 미사용 (08_* 안정성 게이트, Tier A)"
            ),
        },

        "dynamic": {
            "models": dynamic_models,
            "features": list(final_views["control_features"]),
            "cat_cols": list(final_views["control_cat_cols"]),
            "final_iterations": dynamic_trees,
            "model_params": dynamic_final_params,
            "seeds": [int(x) for x in L1_SEEDS],
            "target_definition": "control_success - dynamic_anchor",
            "anchor_strength": float(ANCHOR_STRENGTH),
            "anchor_fallback_rate": float(ANCHOR_FALLBACK_RATE),
            "dynamic_lambda": float(DYNAMIC_LAMBDA),
            "dynamic_kappa": float(DYNAMIC_KAPPA),
            "f_blend_weight": float(F_BLEND_WEIGHT),
            "season_league_means": full_season_league_means,
            "pitcher_season_stats": full_pitcher_stats,
            "formula": "A_t=(1-alpha_t)*A_prev+alpha_t*R_t; alpha_t=lambda*n_t/(n_t+kappa)",
            "R_t": "exact_pitcher_season_success_rate",
            "n_t": "exact_pitcher_season_n",
        },

        "shared_l1_state": {
            "base_features": list(base_features),
            "exact_features": list(EXACT_FEATURES),
            "pitcher_season_stats": full_pitcher_stats,
            "hbeb_priors": final_views["hbeb_priors"],
            "hbeb_configs": HBEB_CONFIGS,
            "pc_contexts": [
                {"context": c, "strength": float(s)} for c, s in PC_CONTEXTS
            ],
            "pc_parent_strength": float(PC_PARENT_STRENGTH),
            "pc_fields": list(PC_FIELDS),
            "pc_tables_by_source_season": pc_tables_by_source_season,
            "pc_unseen_default": 0.0,
            "success_spec": SUCCESS_SPEC,
            "success_features": list(SUCCESS_FEATURES),
            "success_history": success_history_artifact,
        },

        "l1_routing": {
            "column": GAME_TYPE_COL,
            "R_rule": "game_type != 'F'",
            "F_rule": "game_type == 'F'",
            # 라우팅은 곱셈 가중치다 (blend_weights / blend).
            #   p_main = where(F, CONTROL, TREATMENT)
            #   w      = where(F, w_f, w_r)
            #   L1     = clip((1 - w) * p_main + w * DYNAMIC_ASOF, 0, 1)
            "R_main": (
                "(1 - track_w)*TREATMENT + track_w*TRACK"
                if TRACK_ENABLED and TRACK_W != 0.0 else "TREATMENT"
            ),
            "F_main": "CONTROL",
            "R_prediction": (
                "(1 - w_r) * [(1 - track_w)*TREATMENT + track_w*TRACK] + w_r * DYNAMIC_ASOF"
                if TRACK_ENABLED and TRACK_W != 0.0
                else "(1 - w_r) * TREATMENT + w_r * DYNAMIC_ASOF"
            ),
            "F_prediction": "(1 - w_f) * CONTROL + w_f * DYNAMIC_ASOF",
            # 기술용 사본. 추론 측 단일 진실원은 artifact 최상위 "w_r"/"track_w" 이며,
            # script.py 가 두 값의 일치를 검사한다.
            "w_r": float(W_R),
            "track_w": float(TRACK_W),
            "f_blend_weight": float(F_BLEND_WEIGHT),
            "b_F": False,
        },

        "l2": {
            "source_year": SOURCE_YEAR,
            "residual_definition": "official_y_2024 - danker_L1_2019_2023_to_2024_routed_prediction",
            "adapter_models": l2_models,
            "adapter_seeds": list(ADAPTER_SEEDS),
            "validation_crossfit_cfg": dict(V15_CF_CFG),
            "final_adapter_cfg": dict(FINAL_ADAPTER_CFG),
            "adapter_shrink": float(ADAPTER_SHRINK),
            "adapter_prep": adapter_prep,
            "adapter_feature_cols": list(X24.columns),
            "adapter_core_feature_cols": list(core24.columns),
            "exact_cols": list(EXACT_FEATURES),
            "pitcher_season_stats": full_pitcher_stats,
            "reference_rawnum": list(REFERENCE_RAWNUM),
            "reference_cat_small": list(REFERENCE_CAT_SMALL),
            "eng_cols": list(ENG_COLS),
            "pc_in_l2": False,
            "exact_in_l2": True,
            "fit_scope": "all official 2024 R rows",
        },

        "l2_routing": {
            "column": GAME_TYPE_COL,
            "L2_rule": "game_type != 'F'",
            "protected_rule": "game_type == 'F'",
            "R_prediction": "clip(L1 + 0.50 * mean(3 ExtraTrees corrections), 0, 1)",
            "F_prediction": "L1 exactly (Control/Dynamic blend protected)",
        },

        "prediction_policy": "np.clip(prediction, 0.0, 1.0)",
        "rule_compliance": {
            "test_rows_used_during_training": False,
            "test_usage_during_training": "header_only",
            "test_row_aggregation": False,
            "other_test_rows_used": False,
            "dynamic_state_source": "current row official ASOF only",
            "dynamic_historical_source": "official labeled train only",
            "l2_residual_source": "official 2024 labels + genuine danker 2024 L1 OOT prediction",
            "exact_asof": "row-local official ASOF minus completed prior-season train labels",
            "pc_source": "exactly previous labeled season only",
            "track_profile_source": (
                "official trackman_history.csv, row season 미만 시즌만 집계한 "
                "동결 은행; 2025 추론 은행은 <=2024 전체"
            ),
            "track_mapping_source": (
                "official train+trackman 통계 결합으로만 생성 (08_* 게이트, Tier A); "
                "test.csv 는 어느 단계에서도 미사용"
            ),
            "track_inference": "row-local pitcher_id lookup only; no test aggregation",
        },
    }

    # 후처리 상수를 artifact 에 동결한다.
    # script.py 는 test 분포에서 어떤 상수도 재추정하지 않는다 (평가 데이터 독립 추론).
    artifact.update(freeze())
    if shift_fit is None:
        artifact.setdefault("shift_beta", None)
        artifact.setdefault("shift_median", None)
        artifact.setdefault("shift_cols", None)
        artifact.setdefault("sens_mu", None)
        artifact.setdefault("sens_sd", None)
    else:
        artifact["shift_beta"] = np.asarray(shift_fit["beta"], dtype=np.float64)
        artifact["shift_median"] = np.asarray(shift_fit["median"], dtype=np.float64)
        artifact["shift_cols"] = list(shift_fit["cols"])
        artifact["sens_mu"] = np.asarray(shift_fit["mu"], dtype=np.float64)
        artifact["sens_sd"] = np.asarray(shift_fit["sd"], dtype=np.float64)
        artifact["shift_meta"] = {
            "n_fit": int(shift_fit["n_fit"]),
            "alpha": float(shift_fit["alpha"]),
            "clip": float(shift_fit["clip"]),
            "fit_years": list(shift_fit["fit_years"]),
            "fit_rows_by_year": dict(shift_fit["fit_rows_by_year"]),
            "domain": "game_type != 'F' (R 전체)",
            "fit_intercept": False,
            "residual_source": "각 연도 정직한 OOF L1 잔차 (< 연도로 학습)",
        }
        if len(artifact["shift_cols"]) != len(artifact["shift_beta"]):
            raise RuntimeError("SHIFT 열 수와 beta 길이가 다르다")
    artifact.setdefault("add_tables", tuple())
    if float(artifact["affine_offset"]) != 0.0:
        raise RuntimeError("affine_offset must be 0.0 (38_* §5)")

    joblib.dump(artifact, MODEL_PATH, compress=3)

    # Keep the exact training source next to the artifact for reproducibility.
    try:
        shutil.copy2(Path(__file__), MODEL_DIR / "train.py")
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 5) Metadata
    # -----------------------------------------------------------------
    fold_meta = {
        "train_seasons": fold_train_seasons,
        "validation_season": SOURCE_YEAR,
        "n_train": n_fold_train,
        "n_validation": int(len(fold_val)),
        "n_R": int(r24.sum()),
        "n_F": int(f24.sum()),

        # E5: 시드별 리스트다. L1_SEEDS 와 같은 순서.
        "l1_seeds": [int(x) for x in L1_SEEDS],
        "control_final_tree_count": list(control_trees),
        "treatment_final_tree_count": list(treatment_trees),
        "dynamic_final_tree_count": list(dynamic_trees),

        "control_brier_2024": brier_score(y24, pred_control24),
        "treatment_brier_2024": brier_score(y24, pred_treat24),
        "dynamic_brier_2024": brier_score(y24, pred_dynamic24),
        "danker_l1_brier_2024": brier_score(y24, l1_oof24),
        "danker_l1_bss_2024": calculate_bss(y24, l1_oof24),

        "control_brier_2024_F": control_f_brier,
        "dynamic_brier_2024_F": dynamic_f_brier,
        "blend_brier_2024_F": blend_f_brier,
        "dynamic_gain_2024_F_pt": dynamic_f_gain_pt,
        "blend_gain_2024_F_pt": blend_f_gain_pt,

        "dynamic_lambda": float(DYNAMIC_LAMBDA),
        "dynamic_kappa": float(DYNAMIC_KAPPA),
        "f_blend_weight": float(F_BLEND_WEIGHT),
        "dynamic_diag_train": dynamic_diag_train,
        "dynamic_diag_2024": dynamic_diag24,

        "control_fold_seconds": control_fold_time,
        "treatment_fold_seconds": treatment_fold_time,
        "dynamic_fold_seconds": dynamic_fold_time,
        "dynamic_correction_mean_2024": float(np.mean(dynamic_correction24)),
        "dynamic_correction_std_2024": float(np.std(dynamic_correction24)),
        "dynamic_correction_abs_mean_2024": float(np.mean(np.abs(dynamic_correction24))),

        # E6 TRACK
        "track_enabled": bool(TRACK_ENABLED),
        "track_w": float(TRACK_W),
        "track_final_tree_count": list(track_trees) if track_trees else None,
        "track_brier_2024": (
            brier_score(y24, pred_track24) if pred_track24 is not None else None
        ),
        "track_fold_seconds": track_fold_time,
        "track_coverage_fold": (
            {"train": track_cover_fold[0], "apply": track_cover_fold[1]}
            if track_cover_fold else None
        ),
        "track_w_grid_2024_R_paired": track_w_diag,
    }
    FOLD4_META_PATH.write_text(
        json.dumps(to_jsonable(fold_meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "standalone_training": True,
        "external_model_dependency": False,
        "external_validation_dependency": False,
        "compute": {
            "gpu_enabled": bool(GPU_ENABLED),
            "devices": GPU_DEVICES if GPU_ENABLED else "cpu",
            "note": "GPU 학습은 CPU 기준선과 비트 비동일 (border_count=254 정렬)",
        },
        "architecture": {
            "L1_R": (
                f"(1-{W_R:g})*[(1-{TRACK_W:g})*Treatment + {TRACK_W:g}*Track] "
                f"+ {W_R:g}*Dynamic-ASOF  "
                "| Treatment = CatBoost Exact-ASOF5 + HB-EB + R-only PC FULL"
                " | Track = Treatment 피처 + 공식 투구추적 프로파일 54열"
                if TRACK_ENABLED and TRACK_W != 0.0 else
                f"(1-{W_R:g})*Treatment + {W_R:g}*Dynamic-ASOF  "
                "| Treatment = CatBoost Exact-ASOF5 + HB-EB + R-only PC FULL"
            ),
            "L1_F": (
                f"(1-{F_BLEND_WEIGHT:g})*Control + {F_BLEND_WEIGHT:g}*Dynamic-ASOF"
            ),
            "Dynamic": "A_t=(1-alpha)A_prev+alpha*R_t; lambda=1; kappa=100; no b_F",
            "L2": "ExtraTrees residual adapter; core + Exact-ASOF5; no PC; R-only correction",
            "L3": False,
        },
        "validated_source": {
            "f_blend_experiment": "fanchor_blend_v001",
            "f_blend_primary_gate": "2024 F gain > 0 for 3/3 CatBoost seeds",
            "f_blend_temporal_mean_gate": "2021~2024 positive in 4/4 folds",
            "settings_frozen": {
                "dynamic_lambda": float(DYNAMIC_LAMBDA),
                "dynamic_kappa": float(DYNAMIC_KAPPA),
                "f_blend_weight": float(F_BLEND_WEIGHT),
            },
        },
        "latest_fold": fold_meta,
        "final_l1": {
            "control_tree_count": control_trees,
            "treatment_tree_count": treatment_trees,
            "dynamic_tree_count": dynamic_trees,
            "control_fit_seconds": control_final_time,
            "treatment_fit_seconds": treatment_final_time,
            "dynamic_fit_seconds": dynamic_final_time,
            "control_features": len(final_views["control_features"]),
            "treatment_features": len(final_views["treatment_features"]),
            "dynamic_features": len(final_views["control_features"]),
            "dynamic_full_diag": dynamic_diag_full,
            "track_tree_count": list(track_trees) if track_trees else None,
            "track_fit_seconds": track_final_time,
            "track_features": (
                len(track_features_full) if track_features_full else None
            ),
            "track_coverage_full": (
                {"train": track_cover_full[0]} if track_cover_full else None
            ),
        },
        "l2": {
            "source_year": SOURCE_YEAR,
            "n_source_rows": int(len(raw24)),
            "n_R": int(l2_r_mask.sum()),
            "n_F_protected": int(l2_f_mask.sum()),
            "n_features": int(X24.shape[1]),
            "n_core_features": int(core24.shape[1]),
            "n_exact_features": int(exact24.shape[1]),
            "residual_mean_all": float(residual24.mean()),
            "residual_std_all": float(residual24.std()),
            "residual_mean_R": float(residual24[r24].mean()),
            "residual_std_R": float(residual24[r24].std()),
            "adapter_seeds": list(ADAPTER_SEEDS),
            "final_cfg": dict(FINAL_ADAPTER_CFG),
            "shrink": float(ADAPTER_SHRINK),
            "fit_diagnostics": l2_fit_diag.to_dict(orient="records"),
        },
        "rule_compliance": artifact["rule_compliance"],
        "elapsed_seconds": float(time.time() - total_start),
    }
    META_PATH.write_text(
        json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 112)
    print("DONE")
    print("=" * 112)
    print(f"Model         : {MODEL_PATH}")
    print(f"Metadata      : {META_PATH}")
    print(f"2024 L1 OOF   : {OOF_2024_PATH}")
    if shift_fit is not None:
        print(f"SHIFT 계수    : {VALIDATION_DIR / 'shift_coefficients.csv'}")
        print(f"SHIFT 진단    : {VALIDATION_DIR / 'shift_fit_diag.csv'}")
    print(f"Fold4 meta    : {FOLD4_META_PATH}")
    print(f"CONTROL trees : {control_trees}")
    print(f"TREATMENT trees: {treatment_trees}")
    print(f"DYNAMIC trees : {dynamic_trees}")
    if TRACK_ENABLED:
        print(f"TRACK trees   : {track_trees}   (w={TRACK_W:g}, R 전용)")
    print(f"F blend       : {1.0 - F_BLEND_WEIGHT:.2f} CONTROL + {F_BLEND_WEIGHT:.2f} DYNAMIC")
    print(f"Final ET cfg  : {FINAL_ADAPTER_CFG}")
    print(f"Total elapsed : {time.time() - total_start:.1f}s")
    print("=" * 112)


if __name__ == "__main__":
    main()