"""V18 Reproduction — Phase 2 (Feature) + Phase 3 (Anchor / Residual Target).

원본        : kaggle_kbo_v18_RECOVERED_FROM_V20_exact.ipynb
             (= kaggle_kbo_v20_TEMPORAL_ROLE_USAGE_ONE_SHOT_submit.ipynb 의
               cell 1~15 + cell 19, byte 단위 동일)
목표        : V18 의 Feature pipeline 과 Layer 0 Anchor 를 numerical 수준에서 재현한다.
범위        : Phase 2 + Phase 3 까지만. Layer 1 crossfit / Layer 2 / artifact /
             script.py / submission 은 이 파일에서 구현하지 않는다.

최우선 규칙 : 평가 데이터 독립 추론.
             모든 통계 / lookup 은 공식 train.csv 만으로 생성한다.
             test row 는 자기 자신의 입력값 + frozen artifact 로만 예측된다.
             predict(test.iloc[[i]])[0] == predict(test)[i]

재현 원칙   : reproduction accuracy > code cleanliness.
             vectorization / fillna / dtype / categorical encoding / sorting /
             clip 위치 / smoothing 을 원본에서 바꾸지 않는다.
             이상해 보이는 구현도 prediction path 에 있으면 그대로 유지한다.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# lightgbm / scikit-learn 은 Phase 3 에서만 필요하다.
# Phase 2 만 실행하는 환경에서도 import 실패로 죽지 않도록 지연 import 한다.
lgb = None
Ridge = None


def _import_phase3_libs():
    global lgb, Ridge
    if lgb is not None and Ridge is not None:
        return
    import lightgbm as _lgb
    from sklearn.linear_model import Ridge as _Ridge

    lgb = _lgb
    Ridge = _Ridge


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "v18_repro_v001"

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

REFERENCE_DIR = VALIDATION_DIR / "reference"
FEATURE_PARITY_DIR = VALIDATION_DIR / "feature_parity"


# =========================================================
# 실행 제어
# =========================================================

# 2 = Feature 까지, 3 = Anchor / Residual 까지 (lightgbm + sklearn 필요)
PHASE = int(os.environ.get("V18_REPRO_PHASE", "3"))

# 미투입 3블록(CROSS_DOMAIN_PROFILE / BATTER_CONDITIONAL / ROLE_DEPLOYMENT)을
# 계산은 하되 모델 입력에는 포함하지 않는다. Phase 2~8 동안 True 를 유지하고,
# 모든 gate 통과 후 False 로 전환한 뒤 R7 parity gate 를 수행한다.
REPRO_BUILD_UNUSED_BLOCKS = os.environ.get("V18_REPRO_UNUSED", "1") == "1"

# 원본은 screen 에 train.csv 전체 컬럼을 싣는다.
# 아래 8개 컬럼은 prediction path 어디에서도 참조되지 않음이 감사로 확인되었다.
#   run_top_before / run_bot_before / run_total_before / score_diff_home /
#   runner_on_1b / runner_on_2b / runner_on_3b / asof_pitcher_pitchmix_n
# 메모리 절감을 위해 참조 컬럼만 적재하며, 이 선택이 값에 영향을 주지 않는다는 것은
# reference(전 컬럼 적재) 대비 R2 gate 로 실증한다.
LOAD_ONLY_REFERENCED_COLUMNS = os.environ.get("V18_REPRO_SUBSET_COLS", "1") == "1"

# 처리 대상 연도. 환경변수로 1개 연도만 지정해 subprocess 단위 실행이 가능하다.
_YEARS_ENV = os.environ.get("V18_REPRO_YEARS", "")
TARGET_YEARS = (
    [int(y) for y in _YEARS_ENV.split(",") if y.strip()]
    if _YEARS_ENV
    else [2020, 2021, 2022, 2023, 2024]
)

# Phase 2 산출물 저장 여부 (대용량). npz(float32) 로 저장한다.
DUMP_FEATURE_MATRICES = os.environ.get("V18_REPRO_DUMP", "1") == "1"
FEATURE_DUMP_DIR = Path(os.environ.get("V18_REPRO_DUMP_DIR", str(VALIDATION_DIR / "matrices")))

_SCREEN_CACHE_ENV = os.environ.get("V18_REPRO_SCREEN_CACHE", "").strip()
SEED_CACHE_PATH = Path(_SCREEN_CACHE_ENV) if _SCREEN_CACHE_ENV else None  # 선택적 pickle 캐시


# =========================================================
# V18 원본 상수 (notebook cell 1 그대로)
# =========================================================

PROTOCOL_VERSION = "V18_TEMPORAL_STABLE_HAND_PRESSURE_COUNT_RIDGE"
SOURCE_PROTOCOL = "V20_TEMPORAL_ROLE_USAGE_V1"

TARGET = "control_success"
ID = "row_id"
SEASON_COL = "season"

ANCHOR_DISCOVERY_MAX_SEASON = 2022
BASE_SOURCE_YEAR = 2023
VALID_YEAR = 2024

SMOKE_MODE = os.environ.get("V20_SMOKE", "0") == "1"
N_JOBS = max(1, min(6, os.cpu_count() or 1))

BASE_CFG = {
    "exact_lgb": dict(
        n_estimators=120 if not SMOKE_MODE else 20,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=350 if not SMOKE_MODE else 40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=6.0,
        max_bin=127,
        random_state=918,
    ),
    "ridge_alpha": 10000.0,
    "trend_lgb": dict(
        n_estimators=180 if not SMOKE_MODE else 25,
        learning_rate=0.04,
        num_leaves=7,
        min_child_samples=300 if not SMOKE_MODE else 40,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=5.0,
        max_bin=127,
        random_state=77,
    ),
}

V15_CF_CFG = dict(
    n_estimators=100 if not SMOKE_MODE else 12,
    max_depth=10,
    min_samples_leaf=200 if not SMOKE_MODE else 25,
    max_features=0.7,
)
FINAL_ADAPTER_CFG = dict(
    n_estimators=120 if not SMOKE_MODE else 15,
    max_depth=10,
    min_samples_leaf=200 if not SMOKE_MODE else 25,
    max_features=0.7,
)
ADAPTER_SHRINK = 0.50
CF_SEEDS = (42, 2026, 314)

V18_HISTORY_YEARS = 3
V18_DECAY = 0.65
V18_PARENT_STRENGTH = 100.0
V18_HAND_STRENGTH = 35.0
V18_PRESSURE_HAND_STRENGTH = 45.0
V18_RIDGE_ALPHA = 1000.0
V18_GAMMA = 0.75
V18_GAMMA_STRESS = (0.55, 0.65, 0.75, 0.85, 0.95)
V18_ROUTE_DOMAIN = "R_CORE"

MIN_GAIN_2023 = 8.0
MIN_GAIN_2024 = 10.0
MIN_CORE_GAIN_2024 = 15.0
MIN_BOOT_P05 = 3.0
MIN_BOOT_PROB = 0.97
MIN_LATE_GAIN = 0.0

# V20 전용 상수는 이식하지 않는다. prediction path 참조 0회로 감사되었다.
OMITTED_V20_CONSTANTS = [
    "V20_HISTORY_YEARS", "V20_DECAY", "V20_VOLUME_K", "V20_ROLE_K",
    "V20_ALPHA_GRID", "V20_GAMMA_GRID", "V20_ROUTE_DOMAIN",
    "V20_DISCOVERY_TARGETS", "V20_STRESS_TARGET", "V20_MIN_DISCOVERY_GAIN",
    "V20_MIN_STRESS_GAIN", "V20_MIN_CORE_STRESS_GAIN", "V20_MIN_BOOT_P05",
    "V20_MIN_BOOT_PROB", "V20_MIN_LATE_GAIN", "V20_WORST_MONTH_GAIN",
]

# 호출 0회 + 미정의 dependency 를 가진 dead function. 구현하지 않는다.
OMITTED_DEAD_CODE = [
    "global_season_features_for_training",
    "exact_numeric_pair",
    "one_adapter_crossfit",       # V14_ADAPTER_CFG 미정의
    "build_disagreement_block",
    "score_prediction",
    "assemble_blocks",            # X_V14 미정의
]

# 계산은 되지만 모델 입력에서 제외된 executable block
UNUSED_BLOCK_NAMES = {
    "CROSS_DOMAIN_PROFILE",
    "BATTER_CONDITIONAL",
    "ROLE_DEPLOYMENT",
}
DEPLOY_SELECTED_BLOCKS = ["PITCHER_CONDITIONAL", "EXACT_ASOF_STATE"]
assert UNUSED_BLOCK_NAMES.isdisjoint(DEPLOY_SELECTED_BLOCKS)

# 기대 차원 (Audit 확정치)
EXPECTED_DIMS = {
    "self_state": 34,
    "gseason": 28,
    "exact_asof_state": 62,
    "pitcher_conditional": 21,
    "adapter_base": 45,
    "adapter_matrix": 128,
    "v18_stable": 25,
    "exact_lgb_features": 98,
    "ridge_features": 89,
    "trend_features": 42,
}


# =========================================================
# 컬럼 정의 (notebook cell 2 / cell 4)
# =========================================================

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

RAWNUM_SPEC = [
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
CAT_SMALL_SPEC = [
    "game_dayofweek", "top_bottom", "game_type", "domain3", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]
ENG_COLS = [
    "eng_recent35", "eng_recent_delta", "eng_sb", "eng_same", "eng_close", "eng_late",
]

# build_self_state 가 추가로 읽는 컬럼
_SELF_STATE_EXTRA = ["top_bottom", "pitcher_hand", "batter_hand", "away_win_expectancy"]

# screen 에 반드시 있어야 하는 컬럼 (LOAD_ONLY_REFERENCED_COLUMNS 시 적재 목록)
REFERENCED_COLUMNS = list(
    dict.fromkeys(
        [ID, SEASON_COL, TARGET, "pitcher_id", "batter_id", "game_type",
         "pitcher_team_id", "batter_team_id"]
        + RAWNUM_SPEC
        + [c for c in CAT_SMALL_SPEC if c != "domain3"]
        + _SELF_STATE_EXTRA
    )
)

PC_CONTEXTS = [
    ("v15_count", 28.0),
    ("batter_hand", 35.0),
    ("v15_inning_phase", 35.0),
    ("base_state", 35.0),
    ("v15_pressure", 28.0),
    ("v15_runners", 35.0),
    ("v15_li_bucket", 35.0),
]

# 런타임에 확정 (screen 컬럼 존재 여부 필터)
RAWNUM: list[str] = []
CAT_SMALL: list[str] = []
ADAPTER_RAWNUM: list[str] = []
BANK_COLS: list[str] = []


# =========================================================
# 공통 유틸
# =========================================================


def _n(frame, c, default=np.nan):
    """notebook cell 2 원본."""
    if c in frame:
        return pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    return np.full(len(frame), default, np.float64)


def raw_bss(y, p):
    """notebook cell 4 원본. max(0, ·) 를 적용하지 않는 V18 정의."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    ref = max(float(y.mean() * (1 - y.mean())), 1e-12)
    return float(100000.0 * (1.0 - np.mean((p - y) ** 2) / ref))


def official_score(y, p, base_rate=None):
    """대회 공식 정의. r 은 전체 평가셋 제구율로 고정하고 max(0, ·) 를 적용한다."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    r = float(y.mean()) if base_rate is None else float(base_rate)
    ref = max(r * (1 - r), 1e-12)
    return float(max(0.0, 100000.0 * (1.0 - np.mean((p - y) ** 2) / ref)))


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (Path,)):
        return str(x)
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def library_versions() -> dict:
    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for name, mod in [("scikit_learn", "sklearn"), ("lightgbm", "lightgbm"),
                      ("joblib", "joblib"), ("scipy", "scipy")]:
        try:
            m = __import__(mod)
            out[name] = getattr(m, "__version__", "?")
        except Exception:
            out[name] = None
    out["n_jobs"] = N_JOBS
    out["cpu_count"] = os.cpu_count()
    return out


def save_matrix(path: Path, df: pd.DataFrame) -> None:
    """float32 행렬 + 컬럼명을 npz 로 저장한다 (pyarrow 비의존)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=df.to_numpy(np.float32),
        columns=np.array(list(df.columns), dtype=object),
        dtypes=np.array([str(t) for t in df.dtypes], dtype=object),
    )


# =========================================================
# 데이터 적재 / 스키마 / 라우터
# =========================================================


def load_train() -> pd.DataFrame:
    """notebook cell 3 의 screen 적재를 재현한다.

    원본은 discover_csv 로 파일을 찾지만, 프로젝트 표준 경로를 사용한다.
    파일 탐색 방식만 다르고 읽는 파일과 dtype 처리는 동일하다.
    """
    if SEED_CACHE_PATH and SEED_CACHE_PATH.exists():
        with open(SEED_CACHE_PATH, "rb") as f:
            screen = pickle.load(f)
    else:
        usecols = REFERENCED_COLUMNS if LOAD_ONLY_REFERENCED_COLUMNS else None
        screen = pd.read_csv(
            TRAIN_PATH, encoding="utf-8-sig", low_memory=False, usecols=usecols
        )
        if usecols is not None:
            # usecols 는 파일 순서를 유지하므로 컬럼 순서를 명시적으로 고정한다.
            screen = screen[[c for c in REFERENCED_COLUMNS if c in screen.columns]]

    screen[SEASON_COL] = pd.to_numeric(screen[SEASON_COL], errors="raise").astype(int)
    screen["game_type"] = screen["game_type"].astype("string").fillna("<NA>").astype(str)
    return screen


def validate_schema(screen: pd.DataFrame) -> None:
    """notebook cell 3 의 assert + 프로젝트 test.csv 스키마 대조."""
    assert screen[ID].is_unique, "row_id is not unique"
    assert TARGET in screen, "target missing"
    assert set(screen.game_type.unique()) == {"R", "F"}, "unexpected game_type"

    missing = [c for c in REFERENCED_COLUMNS if c not in screen.columns]
    assert not missing, f"referenced columns missing from train.csv: {missing}"

    if TEST_PATH.exists():
        test_head = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0)
        test_cols = set(test_head.columns)
        need = set(REFERENCED_COLUMNS) - {TARGET}
        missing_in_test = sorted(need - test_cols)
        assert not missing_in_test, (
            "test.csv 에 없는 참조 컬럼: " + str(missing_in_test)
        )
        assert TARGET not in test_cols, "test.csv 에 target 이 있다"

        # §15 context 문자열 dtype 안전성: train/test dtype 이 같아야
        # v15_count 같은 문자열 키가 어긋나지 않는다.
        test_sample = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=5000)
        ctx_int_cols = ["balls_before", "strikes_before", "inning", "num_runners_on"]
        for c in ctx_int_cols:
            if c in test_sample.columns:
                a, b = str(screen[c].dtype), str(test_sample[c].dtype)
                assert a == b, f"context dtype mismatch {c}: train={a} test={b}"


def fit_router(screen: pd.DataFrame) -> int:
    """notebook cell 3. ANCHOR_TEAM_ID 는 F ∧ season<=2022 의 구조에서만 학습한다."""
    f_hist = screen.loc[
        screen.game_type.eq("F") & screen[SEASON_COL].le(ANCHOR_DISCOVERY_MAX_SEASON)
    ]
    teams = sorted(
        set(f_hist.pitcher_team_id.unique()) | set(f_hist.batter_team_id.unique())
    )
    involvement = {
        int(t): float(((f_hist.pitcher_team_id == t) | (f_hist.batter_team_id == t)).mean())
        for t in teams
    }
    anchors = [t for t, v in involvement.items() if v > 0.999999]
    assert len(anchors) == 1, ("expected one persistent F anchor", involvement)
    return int(anchors[0])


def attach_domain3(frame: pd.DataFrame, anchor_team_id: int) -> pd.DataFrame:
    """row-local 3-way router. train / test 공통."""
    frame["domain3"] = np.where(
        frame.game_type.eq("F"),
        "F",
        np.where(
            (frame.pitcher_team_id == anchor_team_id)
            | (frame.batter_team_id == anchor_team_id),
            "R_ANCHOR",
            "R_CORE",
        ),
    )
    return frame


def finalize_column_lists(screen: pd.DataFrame) -> None:
    """notebook cell 4 의 `if c in screen` 필터를 재현한다."""
    global RAWNUM, CAT_SMALL, ADAPTER_RAWNUM, BANK_COLS
    RAWNUM = [c for c in RAWNUM_SPEC if c in screen]
    CAT_SMALL = [c for c in CAT_SMALL_SPEC if c in screen]
    ADAPTER_RAWNUM = list(
        dict.fromkeys(
            RAWNUM + [c for c in ["pitcher_team_id", "batter_team_id"] if c in screen]
        )
    )
    BANK_COLS = list(
        dict.fromkeys(
            [SEASON_COL, "pitcher_id", "batter_id", TARGET,
             "asof_pitcher_n", "asof_batter_n"]
            + P_COMPONENT_RATES + B_COMPONENT_RATES
        )
    )
    assert len(RAWNUM) == 27, len(RAWNUM)
    assert len(CAT_SMALL) == 9, len(CAT_SMALL)
    assert len(ADAPTER_RAWNUM) == 29, len(ADAPTER_RAWNUM)


# =========================================================
# Phase 2 — Self State (notebook cell 2, 34열)
# =========================================================


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


# =========================================================
# Phase 2 — Global season decomposition (notebook cell 2, 28열)
# =========================================================


def _rounded_cumulative_count(n, rate):
    n = np.asarray(n, np.float64)
    rate = np.asarray(rate, np.float64)
    out = np.zeros(len(n), np.float64)
    ok = (n > 0) & np.isfinite(rate)
    out[ok] = np.rint(n[ok] * rate[ok])
    return out


def fit_global_season_bank(history: pd.DataFrame, cutoff: int) -> dict:
    """원본명 build_global_season_bank.

    history 는 공식 train 만. cutoff 미만 시즌만 사용한다.
    산출물은 frozen lookup 이며 inference 시 test 를 전혀 보지 않는다.
    """
    h = history.loc[history[SEASON_COL].astype(int) < int(cutoff)].copy()
    prior = float(h.loc[h[SEASON_COL].eq(h[SEASON_COL].max()), TARGET].mean()) if len(h) else 0.5
    pg = (
        h.groupby("pitcher_id", sort=False)[TARGET].agg(["size", "sum"])
        if len(h) else pd.DataFrame(columns=["size", "sum"])
    )
    bg = (
        h.groupby("batter_id", sort=False)[TARGET].agg(["size", "sum"])
        if len(h) else pd.DataFrame(columns=["size", "sum"])
    )
    bank = {
        "cutoff": int(cutoff),
        "prior": prior,
        "pitcher_n": pg["size"].to_dict(),
        "pitcher_s": pg["sum"].to_dict(),
        "batter_n": bg["size"].to_dict(),
        "batter_s": bg["sum"].to_dict(),
        "source_max_season": int(h[SEASON_COL].max()) if len(h) else None,
    }

    prev = h.loc[h[SEASON_COL].eq(h[SEASON_COL].max())] if len(h) else h
    bank["component_priors"] = {
        c: (
            float(pd.to_numeric(prev[c], errors="coerce").mean())
            if c in prev and pd.to_numeric(prev[c], errors="coerce").notna().any()
            else 0.5
        )
        for c in P_COMPONENT_RATES + B_COMPONENT_RATES
    }
    if len(h):
        pidx = h.groupby("pitcher_id", sort=False)["asof_pitcher_n"].idxmax()
        plast = h.loc[pidx, ["pitcher_id", "asof_pitcher_n"] + P_COMPONENT_RATES].set_index(
            "pitcher_id"
        )
        pbase_n = pd.to_numeric(plast["asof_pitcher_n"], errors="coerce").fillna(0).astype(np.float64)
        bank["pitcher_component_base_n"] = pbase_n.to_dict()
        bank["pitcher_component_base"] = {}
        for c in P_COMPONENT_RATES:
            rr = pd.to_numeric(plast[c], errors="coerce")
            cnt = np.rint(pbase_n * rr)
            bank["pitcher_component_base"][c] = cnt.to_dict()
        bidx = h.groupby("batter_id", sort=False)["asof_batter_n"].idxmax()
        blast = h.loc[bidx, ["batter_id", "asof_batter_n"] + B_COMPONENT_RATES].set_index(
            "batter_id"
        )
        bbase_n = pd.to_numeric(blast["asof_batter_n"], errors="coerce").fillna(0).astype(np.float64)
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


def year_state(screen: pd.DataFrame, rows: pd.DataFrame, year: int):
    """notebook cell 4 year_state. bank 는 season < year 의 공식 train 만 사용한다."""
    selfx = build_self_state(rows.drop(columns=[TARGET], errors="ignore"))
    hist = screen.loc[screen[SEASON_COL].lt(int(year)), BANK_COLS]
    bank = fit_global_season_bank(hist, int(year))
    gx = transform_global_season(bank, rows.drop(columns=[TARGET], errors="ignore"))
    assert len(selfx) == len(gx) == len(rows)
    del hist
    return selfx.reset_index(drop=True), gx.reset_index(drop=True), bank


def build_exact_asof_state(screen: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.DataFrame:
    """EXACT_ASOF_STATE (62열). notebook cell 7 / cell 16 exact_adapter_block."""
    sx, gx, _ = year_state(screen, rows, year)
    out = pd.concat([sx.add_prefix("exact__"), gx.add_prefix("exact__")], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    assert out.shape[1] == EXPECTED_DIMS["exact_asof_state"], out.shape
    return out


# =========================================================
# Phase 2 — engineered / categorical (notebook cell 4)
# =========================================================


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


def encode_cats(source, target, cats=None, prefix="code_"):
    """범주 level 은 source(공식 train) 에서만 만든다. 미등장은 -1."""
    cats = CAT_SMALL if cats is None else cats
    A = pd.DataFrame(index=np.arange(len(source)))
    B = pd.DataFrame(index=np.arange(len(target)))
    maps = {}
    for c in cats:
        s = source[c].astype("string").fillna("<NA>").astype(str)
        levels = pd.Index(s.unique())
        mp = {v: i for i, v in enumerate(levels)}
        maps[c] = mp
        A[prefix + c] = s.map(mp).astype(np.float32)
        B[prefix + c] = (
            target[c].astype("string").fillna("<NA>").astype(str).map(mp).fillna(-1).astype(np.float32)
        )
    return A, B, maps


def simple_pair(source, target):
    s = add_eng(source)
    t = add_eng(target)
    cols = RAWNUM + ENG_COLS
    A = s[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    B = t[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    ca, cb, _ = encode_cats(s, t)
    return (
        pd.concat([A, ca], axis=1).astype(np.float32),
        pd.concat([B, cb], axis=1).astype(np.float32),
    )


# =========================================================
# Phase 2 — EB lookup helpers (notebook cell 7)
# =========================================================


def _ctx_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["v15_count"] = x["balls_before"].astype(str) + "_" + x["strikes_before"].astype(str)
    x["v15_inning_phase"] = (
        pd.cut(
            pd.to_numeric(x["inning"], errors="coerce"),
            [-np.inf, 3, 6, 9, np.inf],
            labels=["early", "middle", "late", "extra"],
        ).astype(str)
    )
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
        pd.to_numeric(x["num_runners_on"], errors="coerce").fillna(0) > 0, "on", "empty"
    )
    x["v15_li_bucket"] = (
        pd.cut(
            pd.to_numeric(x["li"], errors="coerce"),
            [-np.inf, 0.7, 1.5, np.inf],
            labels=["low", "mid", "high"],
        ).astype(str)
    )
    return x


def _lookup(table, rows, keys, field, default=0.0):
    if len(keys) == 1:
        s = table.set_index(keys[0])[field]
        return rows[keys[0]].map(s).fillna(default).to_numpy(np.float32)
    idx = pd.MultiIndex.from_frame(rows[keys])
    s = table.set_index(keys)[field]
    return s.reindex(idx).fillna(default).to_numpy(np.float32)


def _eb_table(history, keys, parent_keys=None, strength=40.0, parent_strength=80.0):
    h = history
    league = float(h[TARGET].mean())
    g = h.groupby(keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()
    if parent_keys is None:
        g["parent"] = league
    else:
        pg = h.groupby(parent_keys, sort=False)[TARGET].agg(["sum", "size"]).reset_index()
        pg["parent_rate"] = (pg["sum"] + parent_strength * league) / (pg["size"] + parent_strength)
        g = g.merge(pg[parent_keys + ["parent_rate"]], on=parent_keys, how="left")
        g["parent"] = g["parent_rate"].fillna(league)
    g["rate"] = (g["sum"] + strength * g["parent"]) / (g["size"] + strength)
    g["reliability"] = g["size"] / (g["size"] + strength)
    g["delta"] = g["rate"] - g["parent"]
    g["log_n"] = np.log1p(g["size"])
    return g[keys + ["rate", "reliability", "delta", "log_n"]]


def _table_features(table, rows, keys, prefix, fields=("rate", "reliability", "delta", "log_n")):
    z = pd.DataFrame(index=np.arange(len(rows)))
    for f in fields:
        z[prefix + "__" + f] = _lookup(table, rows, keys, f, 0.0)
    return z


# =========================================================
# Phase 2 — PITCHER_CONDITIONAL (21열)
# =========================================================


def make_pc_specs(screen: pd.DataFrame, source_year: int) -> list:
    """직전 1시즌(source_year) 의 공식 train 라벨로만 EB 표를 만든다."""
    h = _ctx_frame(screen.loc[screen[SEASON_COL].eq(int(source_year))].reset_index(drop=True))
    specs = []
    for ctx, strength in PC_CONTEXTS:
        table = _eb_table(h, ["pitcher_id", ctx], ["pitcher_id"], strength)
        specs.append({"ctx": ctx, "prefix": "pc__" + ctx, "table": table})
    del h
    gc.collect()
    return specs


def transform_pc_specs(specs: list, rows: pd.DataFrame) -> pd.DataFrame:
    r = _ctx_frame(rows.reset_index(drop=True))
    parts = []
    for spec in specs:
        z = _table_features(
            spec["table"], r, ["pitcher_id", spec["ctx"]], spec["prefix"],
            ("rate", "reliability", "delta"),
        )
        parts.append(z)
    out = pd.concat(parts, axis=1).astype(np.float32)
    del r
    assert out.shape[1] == EXPECTED_DIMS["pitcher_conditional"], out.shape
    return out


def build_pitcher_conditional(screen: pd.DataFrame, source_year: int, target_rows: pd.DataFrame):
    specs = make_pc_specs(screen, source_year)
    return transform_pc_specs(specs, target_rows), specs


# =========================================================
# Phase 2 — 미투입 블록 (계산만 수행, 모델 입력 제외)
# =========================================================


def build_unused_blocks_for_audit(screen: pd.DataFrame, source_year: int,
                                  target_rows: pd.DataFrame) -> dict:
    """notebook cell 7 build_latest_history_blocks 중 선택되지 않은 3블록.

    V18 원본은 이 3개를 계산하지만 DEPLOY_SELECTED_BLOCKS 에 포함하지 않는다.
    reproduction 완전성을 위해 계산은 하되, 모델 입력에는 절대 넣지 않는다.
    """
    if not REPRO_BUILD_UNUSED_BLOCKS:
        return {}

    h = _ctx_frame(screen.loc[screen[SEASON_COL].eq(int(source_year))].reset_index(drop=True))
    r = _ctx_frame(target_rows.reset_index(drop=True))
    out = {}

    # CROSS_DOMAIN_PROFILE
    xd = []
    for ent, tag in [("pitcher_id", "p"), ("batter_id", "b")]:
        tg = _eb_table(h, [ent], None, 80.0)
        td = _eb_table(h, [ent, "domain3"], ["domain3"], 60.0)
        a = _table_features(tg, r, [ent], f"xd__{tag}_global")
        b = _table_features(td, r, [ent, "domain3"], f"xd__{tag}_domain")
        b[f"xd__{tag}_domain_minus_global"] = (
            b[f"xd__{tag}_domain__rate"].to_numpy() - a[f"xd__{tag}_global__rate"].to_numpy()
        )
        b[f"xd__{tag}_transfer_confidence"] = (
            b[f"xd__{tag}_domain_minus_global"].to_numpy()
            * b[f"xd__{tag}_domain__reliability"].to_numpy()
        )
        xd += [a, b]
    out["CROSS_DOMAIN_PROFILE"] = pd.concat(xd, axis=1).astype(np.float32)

    # BATTER_CONDITIONAL
    bc = []
    for ctx, strength in [("pitcher_hand", 35.0), ("v15_count", 35.0), ("domain3", 45.0)]:
        t = _eb_table(h, ["batter_id", ctx], ["batter_id"], strength)
        bc.append(_table_features(t, r, ["batter_id", ctx], f"bc__{ctx}",
                                  ("rate", "reliability", "delta")))
    out["BATTER_CONDITIONAL"] = pd.concat(bc, axis=1).astype(np.float32)

    # ROLE_DEPLOYMENT
    rg = (
        h.groupby("pitcher_id", sort=False)
        .agg(
            role_n=("inning", "size"),
            role_inning_mean=("inning", "mean"),
            role_inning_std=("inning", "std"),
            role_li_mean=("li", "mean"),
            role_li_std=("li", "std"),
            role_late_share=("inning", lambda s: float((s >= 7).mean())),
            role_highli_share=("li", lambda s: float((s >= 1.5).mean())),
        )
        .reset_index()
    )
    rd = (
        h.groupby(["pitcher_id", "domain3"], sort=False)
        .agg(
            drole_n=("inning", "size"),
            drole_inning_mean=("inning", "mean"),
            drole_inning_std=("inning", "std"),
            drole_late_share=("inning", lambda s: float((s >= 7).mean())),
        )
        .reset_index()
    )
    t = r[["pitcher_id", "domain3", "inning", "li"]].copy()
    t["_i"] = np.arange(len(t))
    t = t.merge(rg, on="pitcher_id", how="left").merge(
        rd, on=["pitcher_id", "domain3"], how="left"
    ).sort_values("_i")
    role = pd.DataFrame(index=np.arange(len(r)))
    im = t.role_inning_mean.fillna(float(h.inning.mean()))
    isd = t.role_inning_std.fillna(2.0).clip(lower=0.5)
    lm = t.role_li_mean.fillna(float(h.li.mean()))
    ls = t.role_li_std.fillna(1.0).clip(lower=0.2)
    role["role__log_n"] = np.log1p(t.role_n.fillna(0))
    role["role__inning_z"] = (t.inning - im) / isd
    role["role__inning_abs_z"] = np.abs(role["role__inning_z"])
    role["role__li_z"] = (t.li - lm) / ls
    role["role__li_abs_z"] = np.abs(role["role__li_z"])
    role["role__late_surprise"] = (t.inning >= 7).astype(float) - t.role_late_share.fillna(0.3)
    role["role__highli_surprise"] = (t.li >= 1.5).astype(float) - t.role_highli_share.fillna(0.2)
    dim = t.drole_inning_mean.fillna(im)
    disd = t.drole_inning_std.fillna(isd).clip(lower=0.5)
    role["role__domain_log_n"] = np.log1p(t.drole_n.fillna(0))
    role["role__domain_inning_abs_z"] = np.abs((t.inning - dim) / disd)
    role["role__domain_late_surprise"] = (t.inning >= 7).astype(float) - t.drole_late_share.fillna(
        t.role_late_share
    ).fillna(0.3)
    out["ROLE_DEPLOYMENT"] = role.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    for k, v in out.items():
        assert len(v) == len(target_rows) and not v.columns.duplicated().any()
        assert np.isfinite(v.to_numpy(float)).all(), k
    assert set(out) == UNUSED_BLOCK_NAMES, set(out)
    del h, r
    gc.collect()
    return out


# =========================================================
# Phase 2 — adapter matrix (128열)
# =========================================================


def fit_adapter_prep(map_rows: pd.DataFrame) -> dict:
    m = add_eng(map_rows.reset_index(drop=True))
    cols = ADAPTER_RAWNUM + ENG_COLS
    med = m[cols].apply(pd.to_numeric, errors="coerce").median().fillna(0.0).to_dict()
    maps = {}
    for c in CAT_SMALL:
        s = m[c].astype("string").fillna("<NA>").astype(str)
        maps[c] = {v: i for i, v in enumerate(pd.Index(s.unique()))}
    del m
    return {"cols": cols, "med": med, "maps": maps}


def transform_adapter(prep: dict, rows: pd.DataFrame, base_pred) -> pd.DataFrame:
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
    assert A.shape[1] == EXPECTED_DIMS["adapter_base"], A.shape
    del r
    return A


def build_adapter_matrix(adapter_base: pd.DataFrame, pitcher_conditional: pd.DataFrame,
                         exact_asof: pd.DataFrame,
                         unused_blocks: dict | None = None) -> pd.DataFrame:
    """V18 원본 결합 순서: adapter_base | PITCHER_CONDITIONAL | EXACT_ASOF_STATE."""
    X = pd.concat(
        [
            adapter_base.reset_index(drop=True),
            pitcher_conditional.reset_index(drop=True),
            exact_asof.reset_index(drop=True),
        ],
        axis=1,
    )
    assert not X.columns.duplicated().any()
    assert np.isfinite(X.to_numpy(float)).all()
    assert X.shape[1] == EXPECTED_DIMS["adapter_matrix"], X.shape

    # 미투입 3블록이 실제 모델 입력에 절대 포함되지 않음을 검증한다.
    if unused_blocks:
        cols = set(X.columns)
        for name, blk in unused_blocks.items():
            leaked = cols & set(blk.columns)
            assert not leaked, f"unused block {name} leaked into adapter matrix: {sorted(leaked)}"
    return X


# =========================================================
# Phase 2 — V18 STABLE (25열)
# =========================================================


def _v18_pressure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame.copy()
    balls = pd.to_numeric(r["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(r["strikes_before"], errors="coerce").fillna(0).astype(int)
    r["v18_pressure"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(balls == 3, "3ball", np.where(strikes == 2, "2strike", "normal")),
    )
    r["v18_pressure_bhand"] = r["v18_pressure"].astype(str) + "|" + r["batter_hand"].astype(str)
    return r


def _v18_weighted_conditional_table(history, ctx, strength, cutoff_year):
    h = _v18_pressure_frame(history.reset_index(drop=True))
    h["_w"] = np.power(
        V18_DECAY, np.maximum(int(cutoff_year) - 1 - h[SEASON_COL].astype(int), 0)
    ).astype(np.float64)
    h["_wy"] = h["_w"] * pd.to_numeric(h[TARGET], errors="coerce").to_numpy(np.float64)
    mu = float(np.average(h[TARGET].to_numpy(np.float64), weights=h["_w"]))
    pg = h.groupby("pitcher_id", sort=False).agg(sw=("_w", "sum"), sy=("_wy", "sum")).reset_index()
    pg["parent_rate"] = (pg["sy"] + V18_PARENT_STRENGTH * mu) / (pg["sw"] + V18_PARENT_STRENGTH)
    g = (
        h.groupby(["pitcher_id", ctx], sort=False)
        .agg(sw=("_w", "sum"), sy=("_wy", "sum"))
        .reset_index()
        .merge(pg[["pitcher_id", "parent_rate"]], on="pitcher_id", how="left")
    )
    g["rate"] = (g["sy"] + float(strength) * g["parent_rate"]) / (g["sw"] + float(strength))
    g["delta"] = g["rate"] - g["parent_rate"]
    g["reliability"] = g["sw"] / (g["sw"] + float(strength))
    del h
    return g[["pitcher_id", ctx, "delta", "reliability"]]


def build_v18_stable_specs(screen: pd.DataFrame, cutoff_year: int) -> list:
    """cutoff_year 직전 최대 3시즌의 공식 train 만 사용한다."""
    lo = max(int(screen[SEASON_COL].min()), int(cutoff_year) - V18_HISTORY_YEARS)
    h = screen.loc[
        screen[SEASON_COL].lt(int(cutoff_year)) & screen[SEASON_COL].ge(lo)
    ].reset_index(drop=True)
    if h.empty:
        raise RuntimeError(f"no stable history for cutoff {cutoff_year}")
    specs = []
    for ctx, strength, prefix in [
        ("batter_hand", V18_HAND_STRENGTH, "stable__batter_hand"),
        ("v18_pressure_bhand", V18_PRESSURE_HAND_STRENGTH, "stable__pressure_bhand"),
    ]:
        specs.append(
            {
                "ctx": ctx,
                "prefix": prefix,
                "table": _v18_weighted_conditional_table(h, ctx, strength, cutoff_year),
            }
        )
    del h
    gc.collect()
    return specs


def transform_v18_stable_specs(specs: list, rows: pd.DataFrame) -> pd.DataFrame:
    r = _v18_pressure_frame(rows.reset_index(drop=True))
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
    assert z.shape[1] == EXPECTED_DIMS["v18_stable"], z.shape
    del r
    return z


# =========================================================
# Phase 3 — temporal / domain drift (notebook cell 5)
# =========================================================


class DomainMeans:
    """V18 원본은 전 시즌 groupby 로 DOMAIN_MEANS 를 만든 뒤 source_year 이하만 읽는다.

    §14 정책에 따라 계산 방식은 바꾸지 않고, 읽기 시점에 season <= source_year 인지
    assert 한다. 접근 이력을 남겨 미래 시즌 참조 0건임을 산출물로 증명한다.
    """

    def __init__(self, screen: pd.DataFrame):
        self.table = screen.groupby([SEASON_COL, "domain3"])[TARGET].mean().unstack()
        self.access_log: list[tuple[int, str, int]] = []

    def value(self, season: int, domain: str, source_year: int) -> float:
        assert int(season) <= int(source_year), (
            f"DOMAIN_MEANS future access: season={season} > source_year={source_year}"
        )
        self.access_log.append((int(season), str(domain), int(source_year)))
        return float(self.table.loc[int(season), domain])

    def trailing_core_slope(self, source_year: int) -> float:
        s = self.table.loc[self.table.index <= int(source_year), "R_CORE"].dropna().tail(4)
        assert len(s) == 0 or int(s.index.max()) <= int(source_year)
        for yr in s.index:
            self.access_log.append((int(yr), "R_CORE", int(source_year)))
        if len(s) < 2:
            return 0.0
        return float(np.polyfit(s.index.to_numpy(float), s.to_numpy(float), 1)[0])

    def max_accessed_season(self) -> int | None:
        return max((a[0] for a in self.access_log), default=None)

    def future_access_count(self) -> int:
        return sum(1 for season, _, src in self.access_log if season > src)


# =========================================================
# Phase 3 — Layer 0 Anchor (notebook cell 5 fit_recent_base)
# =========================================================


def build_anchor(screen: pd.DataFrame, domain_means: DomainMeans, source_year: int,
                 target_rows: pd.DataFrame, target_year: int, verbose: bool = True):
    """3-모델 등가중 anchor. 가중치 최적화를 추가하지 않는다."""
    _import_phase3_libs()

    source = screen.loc[screen[SEASON_COL].eq(int(source_year))].reset_index(drop=True)
    target = target_rows.reset_index(drop=True)
    y = source[TARGET].to_numpy(np.float64)

    ss, gs, _ = year_state(screen, source, source_year)
    st, gt, _ = year_state(screen, target, target_year)
    A0 = pd.concat(
        [source[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True), ss, gs],
        axis=1,
    )
    B0 = pd.concat(
        [target[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True), st, gt],
        axis=1,
    )
    ca, cb, exact_maps = encode_cats(source, target)
    A = pd.concat([A0, ca], axis=1)
    B = pd.concat([B0, cb], axis=1)
    assert A.shape[1] == EXPECTED_DIMS["exact_lgb_features"], A.shape
    exact_feature_cols = list(A.columns)

    # 1) exact-ASOF binary LightGBM
    cfg = BASE_CFG["exact_lgb"]
    m1 = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_JOBS, **cfg)
    m1.fit(A, y)
    p1 = m1.predict_proba(B)[:, 1]
    del m1, A, B
    gc.collect()

    # 2) exact-ASOF Ridge — 범주 코드 의도적 제외
    A = A0.copy()
    B = B0.copy()
    ridge_feature_cols = list(A0.columns)
    assert A.shape[1] == EXPECTED_DIMS["ridge_features"], A.shape
    del A0, B0, ss, gs, st, gt
    gc.collect()
    A = A.replace([np.inf, -np.inf], np.nan)
    B = B.replace([np.inf, -np.inf], np.nan)
    med = A.median().fillna(0.0)
    A = A.fillna(med).to_numpy(np.float32)
    B = B.fillna(med).to_numpy(np.float32)
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    A = np.clip((A - mu) / sd, -8, 8)
    B = np.clip((B - mu) / sd, -8, 8)
    m2 = Ridge(alpha=BASE_CFG["ridge_alpha"], solver="lsqr", max_iter=100, tol=1e-3)
    m2.fit(A, y)
    p2 = np.clip(m2.predict(B), 0.001, 0.999)
    del m2, A, B
    gc.collect()

    # 3) domain-level trend-centered LightGBM
    A, B = simple_pair(source, target)
    assert A.shape[1] == EXPECTED_DIMS["trend_features"], A.shape
    trend_feature_cols = list(A.columns)
    source_dom = source.domain3.to_numpy()
    target_dom = target.domain3.to_numpy()
    slope = domain_means.trailing_core_slope(source_year)
    base = np.array(
        [domain_means.value(int(source_year), d, source_year) for d in source_dom],
        dtype=np.float64,
    )
    prior = {
        d: domain_means.value(int(source_year), d, source_year) + slope
        for d in ["R_CORE", "R_ANCHOR", "F"]
    }
    bt = np.array([prior[d] for d in target_dom], dtype=np.float64)
    cfg = BASE_CFG["trend_lgb"]
    m3 = lgb.LGBMRegressor(objective="regression_l2", verbosity=-1, n_jobs=N_JOBS, **cfg)
    m3.fit(A, y - base)
    p3 = np.clip(bt + m3.predict(B), 0.001, 0.999)
    del m3, A, B
    gc.collect()

    p = (np.asarray(p1, float) + np.asarray(p2, float) + np.asarray(p3, float)) / 3.0
    meta = {
        "source_year": int(source_year),
        "target_year": int(target_year),
        "trend_slope": slope,
        "trend_prior": prior,
        "n_source": int(len(source)),
        "n_target": int(len(target)),
        "exact_feature_cols": exact_feature_cols,
        "ridge_feature_cols": ridge_feature_cols,
        "trend_feature_cols": trend_feature_cols,
        "exact_cat_levels": {c: len(v) for c, v in exact_maps.items()},
    }
    if verbose:
        print("  base meta", {k: meta[k] for k in
                              ("source_year", "target_year", "trend_slope", "trend_prior")})
    del source, target
    gc.collect()
    return (
        np.clip(p, 0, 1),
        {"exact_lgb": p1, "exact_ridge": p2, "trend_center_lgb": p3},
        meta,
    )


def build_training_target(y, p_base):
    """Layer 1 residual target. logit 변환 없음, 확률공간 순수 차이."""
    return np.asarray(y, np.float64) - np.asarray(p_base, np.float64)


# =========================================================
# 교차적합 분할 (notebook cell 6) — Phase 4 에서 사용, R1 gate 용으로 미리 검증
# =========================================================


def pitcher_half(frame: pd.DataFrame, seed: int) -> np.ndarray:
    pids = np.array(sorted(frame.pitcher_id.astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pids)
    A = set(pids[: len(pids) // 2])
    return np.where(frame.pitcher_id.astype(str).isin(A).to_numpy(), 0, 1).astype(np.int8)


def surrogate_base_pred(n: int) -> np.ndarray:
    """Phase 2 단독 실행용 결정적 base_pred 대체값.

    lightgbm 없이도 adapter matrix 의 열 이름 / 열 순서 / dtype / 행 정렬을
    reference 와 비교할 수 있게 한다. 행마다 값이 다르므로 정렬 버그를 잡아낸다.
    Phase 3 에서는 실제 p_base 로 다시 만든다.
    """
    return (((np.arange(n) % 997) / 997.0) * 0.98 + 0.01).astype(np.float64)


# =========================================================
# 실행
# =========================================================


def run_phase2_year(screen: pd.DataFrame, year: int, anchor_team_id: int) -> dict:
    """한 연도의 Feature 블록을 모두 생성하고 검증한다."""
    t0 = time.time()
    source_year = int(year) - 1
    rows = screen.loc[screen[SEASON_COL].eq(int(year))].reset_index(drop=True)
    n = len(rows)

    self_state = build_self_state(rows.drop(columns=[TARGET], errors="ignore"))
    _, gseason, bank = year_state(screen, rows, year)
    exact_asof = build_exact_asof_state(screen, rows, year)
    pitcher_conditional, pc_specs = build_pitcher_conditional(screen, source_year, rows)

    stable_specs = build_v18_stable_specs(screen, year)
    stable = transform_v18_stable_specs(stable_specs, rows)

    unused = build_unused_blocks_for_audit(screen, source_year, rows)

    # adapter prep 은 직전 시즌(source_year) 에서 적합한다 (validation 규약).
    prep = fit_adapter_prep(screen.loc[screen[SEASON_COL].eq(source_year)].reset_index(drop=True))

    # Phase 2 단독 실행 시 결정적 surrogate base_pred 로 adapter matrix 를 만들어
    # 열 이름 / 열 순서 / dtype / 행 정렬을 reference 와 대조한다.
    adapter_base_surrogate = transform_adapter(prep, rows, surrogate_base_pred(n))
    adapter_surrogate = build_adapter_matrix(
        adapter_base_surrogate, pitcher_conditional, exact_asof, unused
    )

    out = {
        "year": int(year),
        "source_year": source_year,
        "n_rows": n,
        "row_id": rows[ID].to_numpy(),
        "target": rows[TARGET].to_numpy(np.float64),
        "domain3": rows["domain3"].to_numpy(),
        "self_state": self_state,
        "gseason": gseason,
        "exact_asof_state": exact_asof,
        "pitcher_conditional": pitcher_conditional,
        "v18_stable": stable,
        "adapter_matrix_surrogate": adapter_surrogate,
        "unused_blocks": unused,
        "adapter_prep": prep,
        "pc_specs": pc_specs,
        "stable_specs": stable_specs,
        "bank_prior": float(bank["prior"]),
        "bank_source_max_season": bank["source_max_season"],
        "pitcher_half": {int(s): pitcher_half(rows, s) for s in CF_SEEDS},
        "rows": rows,
        "sec": time.time() - t0,
    }
    return out


def summarize_block(name: str, df: pd.DataFrame) -> dict:
    arr = df.to_numpy(np.float64)
    return {
        "block": name,
        "shape": list(df.shape),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": sorted({str(t) for t in df.dtypes}),
        "n_nan": int(np.isnan(arr).sum()),
        "min": float(np.nanmin(arr)) if arr.size else None,
        "max": float(np.nanmax(arr)) if arr.size else None,
        "mean": float(np.nanmean(arr)) if arr.size else None,
        "checksum": float(np.nansum(arr.astype(np.float32), dtype=np.float64)),
    }


def main():
    global_start = time.time()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_PARITY_DIR.mkdir(parents=True, exist_ok=True)
    if DUMP_FEATURE_MATRICES:
        FEATURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Experiment                : {EXPERIMENT_NAME}")
    print(f"Protocol                  : {PROTOCOL_VERSION}")
    print(f"Phase                     : {PHASE}  (2=Feature, 3=+Anchor/Residual)")
    print(f"REPRO_BUILD_UNUSED_BLOCKS : {REPRO_BUILD_UNUSED_BLOCKS}")
    print(f"LOAD_ONLY_REFERENCED_COLS : {LOAD_ONLY_REFERENCED_COLUMNS}")
    print(f"Target years              : {TARGET_YEARS}")
    print(f"N_JOBS                    : {N_JOBS}")
    print("=" * 78)

    screen = load_train()
    validate_schema(screen)
    anchor_team_id = fit_router(screen)
    screen = attach_domain3(screen, anchor_team_id)
    finalize_column_lists(screen)
    domain_means = DomainMeans(screen)

    print(f"ANCHOR_TEAM_ID            : {anchor_team_id}")
    print(f"TRAIN_ROWS                : {len(screen)}")
    print(f"SEASONS                   : {sorted(screen[SEASON_COL].unique().tolist())}")
    print(f"RAWNUM / CAT_SMALL        : {len(RAWNUM)} / {len(CAT_SMALL)}")
    print(f"domain3 share             : "
          f"{screen.domain3.value_counts(normalize=True).round(4).to_dict()}")
    print()

    phase2_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "anchor_team_id": anchor_team_id,
        "train_rows": int(len(screen)),
        "seasons": sorted(int(s) for s in screen[SEASON_COL].unique()),
        "rawnum": RAWNUM,
        "cat_small": CAT_SMALL,
        "adapter_rawnum": ADAPTER_RAWNUM,
        "eng_cols": ENG_COLS,
        "bank_cols": BANK_COLS,
        "pc_contexts": [[c, s] for c, s in PC_CONTEXTS],
        "expected_dims": EXPECTED_DIMS,
        "repro_build_unused_blocks": REPRO_BUILD_UNUSED_BLOCKS,
        "unused_block_names": sorted(UNUSED_BLOCK_NAMES),
        "deploy_selected_blocks": DEPLOY_SELECTED_BLOCKS,
        "library_versions": library_versions(),
        "years": {},
    }
    phase3_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "library_versions": library_versions(),
        "years": {},
    }

    for year in TARGET_YEARS:
        print("-" * 78)
        print(f"[Phase 2] year {year}")
        st = run_phase2_year(screen, year, anchor_team_id)
        rows = st["rows"]

        blocks = {
            "self_state": st["self_state"],
            "gseason": st["gseason"],
            "exact_asof_state": st["exact_asof_state"],
            "pitcher_conditional": st["pitcher_conditional"],
            "v18_stable": st["v18_stable"],
            "adapter_matrix_surrogate": st["adapter_matrix_surrogate"],
        }
        ysum = {
            "n_rows": st["n_rows"],
            "source_year": st["source_year"],
            "bank_prior": st["bank_prior"],
            "bank_source_max_season": st["bank_source_max_season"],
            "sec_features": round(st["sec"], 2),
            "blocks": {k: summarize_block(k, v) for k, v in blocks.items()},
            "unused_blocks": {
                k: summarize_block(k, v) for k, v in st["unused_blocks"].items()
            },
            "pitcher_half_fold1_share": {
                str(s): float(v.mean()) for s, v in st["pitcher_half"].items()
            },
        }

        # Layer 0 의 temporal scalar 는 lightgbm 없이 계산 가능하므로 Phase 2 에서도
        # 산출해 R3 Trend Scalars gate 를 미리 측정한다.
        _slope = domain_means.trailing_core_slope(st["source_year"])
        ysum["trend_slope"] = _slope
        ysum["trend_prior"] = {
            d: domain_means.value(int(st["source_year"]), d, st["source_year"]) + _slope
            for d in ["R_CORE", "R_ANCHOR", "F"]
        }
        ysum["domain_means_source_year"] = {
            d: domain_means.value(int(st["source_year"]), d, st["source_year"])
            for d in ["R_CORE", "R_ANCHOR", "F"]
        }
        for k, v in blocks.items():
            print(f"    {k:24s} {str(list(v.shape)):>16s}  nan={ysum['blocks'][k]['n_nan']}")
        for k in st["unused_blocks"]:
            print(f"    [unused] {k:17s} {str(list(st['unused_blocks'][k].shape)):>16s}")

        if DUMP_FEATURE_MATRICES:
            for k, v in blocks.items():
                save_matrix(FEATURE_DUMP_DIR / f"{k}_{year}.npz", v)
            for k, v in st["unused_blocks"].items():
                save_matrix(FEATURE_DUMP_DIR / f"{k.lower()}_{year}.npz", v)
            np.savez_compressed(
                FEATURE_DUMP_DIR / f"data_{year}.npz",
                row_id=st["row_id"].astype(str),
                target=st["target"],
                season=np.full(st["n_rows"], int(year), dtype=np.int64),
                domain3=st["domain3"].astype(str),
                **{f"pitcher_half_{s}": v for s, v in st["pitcher_half"].items()},
            )

        # ---------------- Phase 3 ----------------
        if PHASE >= 3:
            print(f"[Phase 3] year {year}  anchor  ({st['source_year']} -> {year})")
            p_base, parts, meta = build_anchor(
                screen, domain_means, st["source_year"], rows, year
            )
            y = st["target"]
            residual_l1 = build_training_target(y, p_base)

            adapter_base = transform_adapter(st["adapter_prep"], rows, p_base)
            X_adapter = build_adapter_matrix(
                adapter_base,
                st["pitcher_conditional"],
                st["exact_asof_state"],
                st["unused_blocks"],
            )
            assert X_adapter.shape[1] == 128
            EXPECTED_ADAPTER_FEATURE_COLS = (
                list(adapter_base.columns)
                + list(st["pitcher_conditional"].columns)
                + list(st["exact_asof_state"].columns)
            )
            assert list(X_adapter.columns) == EXPECTED_ADAPTER_FEATURE_COLS

            base_df = pd.DataFrame(
                {
                    ID: st["row_id"],
                    "season": int(year),
                    "domain3": st["domain3"],
                    "target": y,
                    "p_exact_lgb": np.asarray(parts["exact_lgb"], np.float64),
                    "p_exact_ridge": np.asarray(parts["exact_ridge"], np.float64),
                    "p_trend_lgb": np.asarray(parts["trend_center_lgb"], np.float64),
                    "p_base": np.asarray(p_base, np.float64),
                    "residual_l1": residual_l1,
                }
            )
            base_df.to_csv(VALIDATION_DIR / f"base_{year}.csv", index=False)
            save_matrix(FEATURE_DUMP_DIR / f"adapter_matrix_{year}.npz", X_adapter)

            bss = raw_bss(y, p_base)
            print(f"    base_bss={bss:.2f}  pred_mean={float(p_base.mean()):.6f}  "
                  f"target_mean={float(y.mean()):.6f}")

            phase3_summary["years"][str(year)] = {
                "source_year": st["source_year"],
                "n_rows": st["n_rows"],
                "trend_slope": meta["trend_slope"],
                "trend_prior": meta["trend_prior"],
                "exact_feature_cols_n": len(meta["exact_feature_cols"]),
                "ridge_feature_cols_n": len(meta["ridge_feature_cols"]),
                "trend_feature_cols_n": len(meta["trend_feature_cols"]),
                "exact_cat_levels": meta["exact_cat_levels"],
                "adapter_matrix_shape": list(X_adapter.shape),
                "adapter_columns": list(X_adapter.columns),
                "p_exact_lgb": {"mean": float(np.mean(parts["exact_lgb"])),
                                "min": float(np.min(parts["exact_lgb"])),
                                "max": float(np.max(parts["exact_lgb"]))},
                "p_exact_ridge": {"mean": float(np.mean(parts["exact_ridge"])),
                                  "min": float(np.min(parts["exact_ridge"])),
                                  "max": float(np.max(parts["exact_ridge"]))},
                "p_trend_lgb": {"mean": float(np.mean(parts["trend_center_lgb"])),
                                "min": float(np.min(parts["trend_center_lgb"])),
                                "max": float(np.max(parts["trend_center_lgb"]))},
                "p_base": {"mean": float(p_base.mean()), "min": float(p_base.min()),
                           "max": float(p_base.max())},
                "residual_l1": {"mean": float(residual_l1.mean()),
                                "std": float(residual_l1.std())},
                "base_bss": bss,
                "official_score": official_score(y, p_base),
            }
            ysum["adapter_matrix"] = summarize_block("adapter_matrix", X_adapter)
            del p_base, parts, adapter_base, X_adapter, base_df
            gc.collect()

        phase2_summary["years"][str(year)] = ysum
        del st, rows, blocks
        gc.collect()

    # DOMAIN_MEANS 미래 접근 감사 (§14)
    phase3_summary["domain_means_audit"] = {
        "n_access": len(domain_means.access_log),
        "future_access_count": domain_means.future_access_count(),
        "max_accessed_season": domain_means.max_accessed_season(),
        "policy": "V18 원본 전 시즌 groupby 유지. 읽기는 season <= source_year 만 assert.",
    }
    assert domain_means.future_access_count() == 0

    phase2_summary["elapsed_sec"] = round(time.time() - global_start, 1)
    phase3_summary["elapsed_sec"] = round(time.time() - global_start, 1)

    write_json(VALIDATION_DIR / "phase2_summary.json", phase2_summary)
    if PHASE >= 3:
        write_json(VALIDATION_DIR / "phase3_summary.json", phase3_summary)

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "source": "kaggle_kbo_v18_RECOVERED_FROM_V20_exact.ipynb",
        "source_verified_against": "kaggle_kbo_v20_TEMPORAL_ROLE_USAGE_ONE_SHOT_submit.ipynb",
        "protocol": PROTOCOL_VERSION,
        "source_protocol": SOURCE_PROTOCOL,
        "target": TARGET,
        "phase_implemented": PHASE,
        "architecture": {
            "layers": 3,
            "layer0": "recent 1-season anchor: LGBMClassifier + Ridge(a=1e4) + trend-centered LGBMRegressor, equal 1/3",
            "layer1": "three-domain ExtraTrees residual adapter, shrink 0.50  [NOT IMPLEMENTED IN PHASE 2/3]",
            "layer2": "temporal-stable conditional Ridge(a=1e3) on prev-year OOF residual, gamma 0.75, R_CORE only  [NOT IMPLEMENTED IN PHASE 2/3]",
            "router": "domain3 = F / R_ANCHOR / R_CORE, anchor from F & season<=2022",
        },
        "features": {
            "raw": {"rawnum": len(RAWNUM), "cat_small": len(CAT_SMALL), "eng": len(ENG_COLS)},
            "temporal": {"domain_means": True, "trailing_core_slope": True,
                         "v18_stable_decay": V18_DECAY},
            "historical": {"self_state": EXPECTED_DIMS["self_state"],
                           "gseason": EXPECTED_DIMS["gseason"],
                           "exact_asof_state": EXPECTED_DIMS["exact_asof_state"]},
            "conditional": {"pitcher_conditional": EXPECTED_DIMS["pitcher_conditional"],
                            "v18_stable": EXPECTED_DIMS["v18_stable"]},
            "anchor": {"exact_lgb": EXPECTED_DIMS["exact_lgb_features"],
                       "exact_ridge": EXPECTED_DIMS["ridge_features"],
                       "trend_lgb": EXPECTED_DIMS["trend_features"],
                       "adapter": EXPECTED_DIMS["adapter_matrix"]},
        },
        "model_params": {
            "base_cfg": BASE_CFG,
            "v15_cf_cfg": V15_CF_CFG,
            "final_adapter_cfg": FINAL_ADAPTER_CFG,
            "adapter_shrink": ADAPTER_SHRINK,
            "cf_seeds": list(CF_SEEDS),
            "v18": {
                "history_years": V18_HISTORY_YEARS, "decay": V18_DECAY,
                "parent_strength": V18_PARENT_STRENGTH, "hand_strength": V18_HAND_STRENGTH,
                "pressure_hand_strength": V18_PRESSURE_HAND_STRENGTH,
                "ridge_alpha": V18_RIDGE_ALPHA, "gamma": V18_GAMMA,
                "gamma_stress": list(V18_GAMMA_STRESS), "route_domain": V18_ROUTE_DOMAIN,
            },
            "gate": {"min_gain_2023": MIN_GAIN_2023, "min_gain_2024": MIN_GAIN_2024,
                     "min_core_gain_2024": MIN_CORE_GAIN_2024, "min_boot_p05": MIN_BOOT_P05,
                     "min_boot_prob": MIN_BOOT_PROB, "min_late_gain": MIN_LATE_GAIN},
        },
        "validation": {
            "scheme": "adjacent-season transition (2020->21, 21->22, 22->23, 23->24)",
            "regime": "within_year_pitcher_crossfit for layer1; strict next-year for layer2",
            "metric": "raw_bss = 100000*(1 - MSE/(ybar*(1-ybar)))",
            "official_metric_also_reported": True,
            "warning": "V18 original validation BSS is not directly comparable to the "
                       "project's expanding-window validation.",
        },
        "reproduction": {
            "structural_reproducibility": "CONFIRMED",
            "numerical_exact_reproducibility": "PHASE 2/3 GATE PENDING",
            "v18_original_lb": 1081.6824647538,
            "v18_original_lb_source": "V20 notebook cell 0 markdown + metadata v19_design/v20",
            "v15_parent_lb": 1044.25629,
            "omitted_dead_code": OMITTED_DEAD_CODE,
            "omitted_v20_constants": OMITTED_V20_CONSTANTS,
            "unused_but_computed_blocks": sorted(UNUSED_BLOCK_NAMES),
            "library_versions": library_versions(),
        },
    }
    write_json(MODEL_DIR / "metadata.json", metadata)
    write_json(VALIDATION_DIR / "metadata.json", metadata)

    print()
    print("=" * 78)
    print(f"DONE  phase={PHASE}  elapsed={time.time() - global_start:.1f}s")
    print(f"  -> {VALIDATION_DIR}/phase2_summary.json")
    if PHASE >= 3:
        print(f"  -> {VALIDATION_DIR}/phase3_summary.json")
        print(f"  -> {VALIDATION_DIR}/base_{{year}}.csv")
    print(f"  -> {MODEL_DIR}/metadata.json")
    print("=" * 78)


if __name__ == "__main__":
    main()
