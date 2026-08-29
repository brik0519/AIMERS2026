"""Anchor / Residual / Sensitivity-Shift pipeline — 실험 shift_v002 (기본) / anchor_v001.

구조        : Layer 0 anchor(직전 1시즌, 3모델 평균)
             → Layer 1 residual adjuster(ExtraTrees, pitcher-half crossfit) — 검증용으로만 유지, 배포 shrink 0
             → Layer 2 sensitivity shift(Ridge α=1e3, 25열, fit_intercept=False, CORE 만, scale 0.75)
변경점      : (1) Layer 2 의 절편(= 직전 시즌 잔차 평균)을 제거한다 (shift_v001).
             (2) 배포 예측 경로의 행 독립성 보정: Ridge 순차 누적, LightGBM 배치 경로 고정,
                 forest 단일 스레드 예측 (shift_v002). 예측값 변화는 1e-8 이하.
             (3) SHIFT_DEPLOY_SHRINK=0 이면 배포에서 Layer 1 을 끈다 (anchor_v001). 배포 regime
                 전이 검증(deploy_validation.py) 에서 Layer 1 이 4/4 전이에서 해로웠기 때문이다. 절편은 다음 시즌으로
             전이되지 않으며(2021 전이 손실의 91%), 조건부 성분만이 이득을 만든다.
             reference(절편 포함) 를 같은 실행에서 함께 계산해 전이별 차이를 출력하고,
             reference 를 넘을 때만 ADOPT=True 로 기록한다. 그 외 Layer 0/1/feature 는 불변.

최우선 규칙 : 평가 데이터 독립 추론.
             모든 통계 / lookup 은 공식 train.csv 만으로 생성한다.
             test row 는 자기 자신의 입력값 + frozen artifact 로만 예측된다.
             predict(test.iloc[[i]])[0] == predict(test)[i]

재현 원칙   : vectorization / fillna / dtype / categorical encoding / sorting /
             clip 위치 / smoothing 을 바꾸지 않는다.
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


ExtraTreesRegressor = None


def _import_phase3_libs():
    global lgb, Ridge
    if lgb is not None and Ridge is not None:
        return
    import lightgbm as _lgb
    from sklearn.linear_model import Ridge as _Ridge

    lgb = _lgb
    Ridge = _Ridge


def _import_phase4_libs():
    global ExtraTreesRegressor
    if ExtraTreesRegressor is not None:
        return
    from sklearn.ensemble import ExtraTreesRegressor as _ET

    ExtraTreesRegressor = _ET


# =========================================================
# 경로
# =========================================================

# 실험 이름과 배포 Layer 1 shrink 는 환경변수로 바꿀 수 있다.
#   shift_v002 : SHIFT_EXPERIMENT=shift_v002 SHIFT_DEPLOY_SHRINK=0.5  (원본 구조 + 행 독립성 보정)
#   anchor_v001: SHIFT_EXPERIMENT=anchor_v001 SHIFT_DEPLOY_SHRINK=0.0  (Layer 1 배포 제거)
EXPERIMENT_NAME = os.environ.get("SHIFT_EXPERIMENT", "shift_v002").strip()

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

# 2 = Feature, 3 = +Anchor/Residual, 4 = +Layer 1 adjusted OOF crossfit,
# 5 = +Layer 2 sensitivity shift (PASS 2), 6 = +deploy artifact (2024->2025)
# (3 이상은 lightgbm + scikit-learn 필요)
#
# 기본값 6 — `python train.py` 만으로 학습 전 과정이 끝나고
# MODEL_PATH(단일 pkl) 와 OUTPUT_PATH(submission.csv) 까지 생성된다.
# 중간 단계만 보려면 SHIFT_PHASE=2|3|4|5 로 낮춘다.
PHASE = int(os.environ.get("SHIFT_PHASE", "6"))

# Phase 4 는 ExtraTrees 90회 적합(5년 × 3seed × 2fold × 3segment)이라 매우 비싸다.
# 이미 adjusted_oof_{year}.npz 가 있으면 그 연도를 건너뛴다.
RESUME = os.environ.get("SHIFT_RESUME", "0") == "1"

# 미투입 3블록(CROSS_DOMAIN_PROFILE / BATTER_CONDITIONAL / ROLE_DEPLOYMENT)을
# 계산은 하되 모델 입력에는 포함하지 않는다. Phase 2~8 동안 True 를 유지하고,
# 모든 gate 통과 후 False 로 전환한 뒤 R7 parity gate 를 수행한다.
REPRO_BUILD_UNUSED_BLOCKS = os.environ.get("SHIFT_UNUSED", "1") == "1"

# 원본은 screen 에 train.csv 전체 컬럼을 싣는다.
# 아래 8개 컬럼은 prediction path 어디에서도 참조되지 않음이 감사로 확인되었다.
#   run_top_before / run_bot_before / run_total_before / score_diff_home /
#   runner_on_1b / runner_on_2b / runner_on_3b / asof_pitcher_pitchmix_n
# 메모리 절감을 위해 참조 컬럼만 적재하며, 이 선택이 값에 영향을 주지 않는다는 것은
# reference(전 컬럼 적재) 대비 R2 gate 로 실증한다.
LOAD_ONLY_REFERENCED_COLUMNS = os.environ.get("SHIFT_SUBSET_COLS", "1") == "1"

# 처리 대상 연도. 환경변수로 1개 연도만 지정해 subprocess 단위 실행이 가능하다.
_YEARS_ENV = os.environ.get("SHIFT_YEARS", "")
TARGET_YEARS = (
    [int(y) for y in _YEARS_ENV.split(",") if y.strip()]
    if _YEARS_ENV
    else [2020, 2021, 2022, 2023, 2024]
)

# Phase 2 산출물 저장 여부 (대용량). npz(float32) 로 저장한다.
DUMP_FEATURE_MATRICES = os.environ.get("SHIFT_DUMP", "1") == "1"
FEATURE_DUMP_DIR = Path(os.environ.get("SHIFT_DUMP_DIR", str(VALIDATION_DIR / "matrices")))

_SCREEN_CACHE_ENV = os.environ.get("SHIFT_SCREEN_CACHE", "").strip()
SEED_CACHE_PATH = Path(_SCREEN_CACHE_ENV) if _SCREEN_CACHE_ENV else None  # 선택적 pickle 캐시

# Phase 3/4 산출물(anchor_*.npz, adjusted_oof_*.npz 등)을 다른 실험 디렉터리에서 읽어 재사용한다.
# Layer 0/1 은 불변이므로 이전 실험의 산출물을 그대로 쓸 수 있다 (구 파일명/키는 별칭으로 허용).
# 쓰기는 항상 FEATURE_DUMP_DIR 로 간다.
_CACHE_ENV = os.environ.get("SHIFT_CACHE_DIR", "").strip()
CACHE_DIR = Path(_CACHE_ENV) if _CACHE_ENV else None

# 1 이면 Phase 2~4 루프를 건너뛰고 캐시만으로 Phase 5(+6) 를 실행한다. Layer 2 반복 실험용.
LAYER2_ONLY = os.environ.get("SHIFT_LAYER2_ONLY", "0") == "1"


# =========================================================
# 모델 상수 (원본 notebook cell 1 값 유지)
# =========================================================

PROTOCOL_VERSION = "ANCHOR_RESIDUAL_SENSITIVITY_SHIFT_V1"

TARGET = "control_success"
ID = "row_id"
SEASON_COL = "season"

ANCHOR_DISCOVERY_MAX_SEASON = 2022
BASE_SOURCE_YEAR = 2023
VALID_YEAR = 2024

SMOKE_MODE = os.environ.get("SHIFT_SMOKE", "0") == "1"
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

RESIDUAL_CF_CFG = dict(
    n_estimators=100 if not SMOKE_MODE else 12,
    max_depth=10,
    min_samples_leaf=200 if not SMOKE_MODE else 25,
    max_features=0.7,
)
RESIDUAL_FINAL_CFG = dict(
    n_estimators=120 if not SMOKE_MODE else 15,
    max_depth=10,
    min_samples_leaf=200 if not SMOKE_MODE else 25,
    max_features=0.7,
)
RESIDUAL_SHRINK = 0.50          # Phase 4 (같은 해 crossfit 검증) 의 상수. 원본과 동일하게 유지한다.
CF_SEEDS = (42, 2026, 314)

# 배포(Phase 6) 에서 Layer 1 에 곱하는 shrink. Phase 4 의 상수와 분리한다.
# 근거 — 배포 regime 전이 검증 (deploy_validation.py / validation/deployval_v001):
#   Layer 1 의 target `y_S − anchor(S-1→S)` 는 "S 가 S-1 과 어떻게 달랐는가" 이며 다음 해에는
#   전이되지 않는다 (2021..2024 전이에서 L1 gain −479/−115/−747/−388, 조건부 성분과 다음 해
#   잔차의 상관 −0.035/+0.009/−0.087/−0.0005). shrink 격자 {0,.05,.1,.15,.2,.25,.5} 에서
#   0 이 3/4 전이 최적, 합 2224→3992. 같은 해 crossfit(Phase 4) 는 당해 라벨 흡수분을 이득으로
#   잘못 계산하므로 이 결정은 Phase 4 수치가 아니라 배포 regime 전이 수치로만 내린다.
#   0.0 이면 residual 모델 학습을 생략하고 anchor + Layer 2 shift 만 배포한다.
DEPLOY_RESIDUAL_SHRINK = float(os.environ.get("SHIFT_DEPLOY_SHRINK", "0.5"))

SENS_HISTORY_YEARS = 3
SENS_DECAY = 0.65
SENS_PARENT_STRENGTH = 100.0
SENS_HAND_STRENGTH = 35.0
SENS_PRESSURE_HAND_STRENGTH = 45.0
SHIFT_RIDGE_ALPHA = 1000.0
SHIFT_SCALE = 0.75
SHIFT_SCALE_STRESS = (0.55, 0.65, 0.75, 0.85, 0.95)
SHIFT_SEGMENT = "CORE"

MIN_GAIN_2023 = 8.0
MIN_GAIN_2024 = 10.0
MIN_CORE_GAIN_2024 = 15.0
MIN_BOOT_P05 = 3.0
MIN_BOOT_PROB = 0.97
MIN_LATE_GAIN = 0.0

# reference(절편 포함 Layer 2) 대비 상대 gate
MIN_RELATIVE_BOOT_P05 = -5.0        # 2024 bootstrap(final − reference) p05 하한: 유의한 악화가 아니어야 한다
MIN_TRANSITIONS_NOT_WORSE = 2       # 4개 전이 중 reference 이상인 전이 수 하한

# 호출 0회 + 미정의 dependency 를 가진 dead function. 구현하지 않는다.
OMITTED_DEAD_CODE = [
    "global_season_features_for_training",
    "exact_numeric_pair",
    "one_residual_crossfit",
    "build_disagreement_block",
    "score_prediction",
    "assemble_blocks",
]

# 계산은 되지만 모델 입력에서 제외된 executable block
UNUSED_BLOCK_NAMES = {
    "CROSS_DOMAIN_PROFILE",
    "BATTER_CONDITIONAL",
    "ROLE_DEPLOYMENT",
}
DEPLOY_SELECTED_BLOCKS = ["PITCHER_CONTEXT", "ASOF_STATE"]
assert UNUSED_BLOCK_NAMES.isdisjoint(DEPLOY_SELECTED_BLOCKS)

# 기대 차원 (Audit 확정치)
EXPECTED_DIMS = {
    "self_state": 34,
    "gseason": 28,
    "asof_state": 62,
    "pitcher_context": 21,
    "residual_base": 45,
    "residual_matrix": 128,
    "sensitivity": 25,
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
    "game_dayofweek", "top_bottom", "game_type", "route_segment", "base_state",
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
        + [c for c in CAT_SMALL_SPEC if c != "route_segment"]
        + _SELF_STATE_EXTRA
    )
)

CONTEXT_KEYS = [
    ("ctx_count", 28.0),
    ("batter_hand", 35.0),
    ("ctx_inning_phase", 35.0),
    ("base_state", 35.0),
    ("ctx_pressure", 28.0),
    ("ctx_runners", 35.0),
    ("ctx_li_bucket", 35.0),
]

# 런타임에 확정 (screen 컬럼 존재 여부 필터)
RAWNUM: list[str] = []
CAT_SMALL: list[str] = []
RESIDUAL_RAWNUM: list[str] = []
BANK_COLS: list[str] = []


# =========================================================
# 공통 유틸
# =========================================================


def _n(frame, c, default=np.nan):
    """notebook cell 2 원본."""
    if c in frame:
        return pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    return np.full(len(frame), default, np.float64)


def brier_skill(y, p):
    """notebook cell 4 원본. max(0, ·) 를 적용하지 않는 원본 정의."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    ref = max(float(y.mean() * (1 - y.mean())), 1e-12)
    return float(100000.0 * (1.0 - np.mean((p - y) ** 2) / ref))


def pitcher_bootstrap_gain(frame, y, p_new, p_base, draws=1000, seed=979):
    """notebook cell 4 원본. 투수 단위 cluster bootstrap."""
    pid = frame.pitcher_id.astype(str).to_numpy()
    y = np.asarray(y, float)
    p_new = np.asarray(p_new, float)
    p_base = np.asarray(p_base, float)
    codes, uniq = pd.factorize(pid, sort=False)
    G = len(uniq)
    n = np.bincount(codes, minlength=G).astype(float)
    sy = np.bincount(codes, weights=y, minlength=G)
    en = np.bincount(codes, weights=(p_new - y) ** 2, minlength=G)
    eb = np.bincount(codes, weights=(p_base - y) ** 2, minlength=G)
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(int(draws)):
        draw = rng.integers(0, G, size=G)
        w = np.bincount(draw, minlength=G).astype(float)
        N = float(w @ n)
        if N < 100:
            continue
        ym = float(w @ sy / N)
        ref = max(ym * (1 - ym), 1e-12)
        g.append(float(100000.0 * ((w @ eb - w @ en) / N) / ref))
    a = np.asarray(g, float)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p05": float(np.quantile(a, 0.05)), "p10": float(np.quantile(a, 0.10)),
        "p95": float(np.quantile(a, 0.95)), "prob_positive": float((a > 0).mean()),
    }


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


def merge_years_json(path: Path, obj: dict) -> None:
    """연도별 subprocess 실행(`SHIFT_YEARS`)을 지원하기 위해 years 를 병합 저장한다.

    통짜로 덮어쓰면 마지막 연도만 남아 gate 가 나머지 연도를 SKIP 한다.
    top-level 키는 최신 실행 값으로 갱신하고 `years` 만 누적한다.
    """
    merged = dict(obj)
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        years = dict(old.get("years", {}))
        years.update(obj.get("years", {}))
        merged["years"] = years
        # library_versions 를 통째로 덮어쓰면 "이 산출물을 만든 버전" 이력이 사라진다.
        # 실행마다 다른 버전을 썼는지 사후에 알 수 있도록 이력을 누적한다.
        hist = list(old.get("library_versions_history", []))
        cur = obj.get("library_versions")
        if cur and (not hist or hist[-1].get("versions") != cur):
            hist.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "years": sorted(obj.get("years", {})), "versions": cur})
        if hist:
            merged["library_versions_history"] = hist
    elif obj.get("library_versions"):
        merged["library_versions_history"] = [
            {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "years": sorted(obj.get("years", {})), "versions": obj["library_versions"]}
        ]
    write_json(path, merged)


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


def load_cached_npz(primary_dir: Path, name: str, aliases: list[str] | None = None):
    """FEATURE_DUMP_DIR → CACHE_DIR 순으로 npz 를 찾는다. 구 파일명은 aliases 로 허용한다."""
    names = [name] + list(aliases or [])
    for d in [primary_dir, CACHE_DIR]:
        if d is None:
            continue
        for n in names:
            p = Path(d) / n
            if p.exists():
                return np.load(p, allow_pickle=True), p
    return None, None


def npz_key(z, key: str, aliases: list[str] | None = None):
    for k in [key] + list(aliases or []):
        if k in z.files:
            return z[k]
    raise KeyError((key, aliases, z.files))


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
        # ctx_count 같은 문자열 키가 어긋나지 않는다.
        test_sample = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=5000)
        ctx_int_cols = ["balls_before", "strikes_before", "inning", "num_runners_on"]
        for c in ctx_int_cols:
            if c in test_sample.columns:
                a, b = str(screen[c].dtype), str(test_sample[c].dtype)
                assert a == b, f"context dtype mismatch {c}: train={a} test={b}"

        # base_state 는 `_lookup` 의 MultiIndex 키다. 값이 '___','1__',…,'123' 인데
        # **'123' 만 들어있는 파일**을 읽으면 pandas 가 int64 로 추론하고,
        # object "123" 로 만들어진 EB 표와 키가 어긋나 pc__base_state__* 가
        # 조용히 0.0 이 된다 (T2 검사 D 에서 실측: 0.489 -> 0.0).
        # 계산은 바꾸지 않고, 그 조건이 되면 시끄럽게 실패시킨다.
        for name, frame in [("train", screen), ("test", test_sample)]:
            if "base_state" in frame.columns:
                dt = str(frame["base_state"].dtype)
                assert dt == "object", (
                    f"{name}.base_state dtype={dt} (object 여야 한다). "
                    "숫자로 추론되면 pitcher_context 의 base_state 조회가 "
                    "조용히 실패한다. 입력 파일의 base_state 값 구성을 확인할 것."
                )


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


def finalize_column_lists(screen: pd.DataFrame) -> None:
    """notebook cell 4 의 `if c in screen` 필터를 재현한다."""
    global RAWNUM, CAT_SMALL, RESIDUAL_RAWNUM, BANK_COLS
    RAWNUM = [c for c in RAWNUM_SPEC if c in screen]
    CAT_SMALL = [c for c in CAT_SMALL_SPEC if c in screen]
    RESIDUAL_RAWNUM = list(
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
    assert len(RESIDUAL_RAWNUM) == 29, len(RESIDUAL_RAWNUM)


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


def build_asof_state(screen: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.DataFrame:
    """ASOF_STATE (62열). notebook cell 7 / cell 16 exact_residual_block."""
    sx, gx, _ = year_state(screen, rows, year)
    out = pd.concat([sx.add_prefix("exact__"), gx.add_prefix("exact__")], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    assert out.shape[1] == EXPECTED_DIMS["asof_state"], out.shape
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
# Phase 2 — PITCHER_CONTEXT (21열)
# =========================================================


def make_context_specs(screen: pd.DataFrame, source_year: int) -> list:
    """직전 1시즌(source_year) 의 공식 train 라벨로만 EB 표를 만든다."""
    h = _context_frame(screen.loc[screen[SEASON_COL].eq(int(source_year))].reset_index(drop=True))
    specs = []
    for ctx, strength in CONTEXT_KEYS:
        table = _eb_table(h, ["pitcher_id", ctx], ["pitcher_id"], strength)
        specs.append({"ctx": ctx, "prefix": "pc__" + ctx, "table": table})
    del h
    gc.collect()
    return specs


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


def build_pitcher_context(screen: pd.DataFrame, source_year: int, target_rows: pd.DataFrame):
    specs = make_context_specs(screen, source_year)
    return transform_context_specs(specs, target_rows), specs


# =========================================================
# Phase 2 — 미투입 블록 (계산만 수행, 모델 입력 제외)
# =========================================================


def build_unused_blocks_for_audit(screen: pd.DataFrame, source_year: int,
                                  target_rows: pd.DataFrame) -> dict:
    """notebook cell 7 build_latest_history_blocks 중 선택되지 않은 3블록.

    원본은 이 3개를 계산하지만 DEPLOY_SELECTED_BLOCKS 에 포함하지 않는다.
    reproduction 완전성을 위해 계산은 하되, 모델 입력에는 절대 넣지 않는다.
    """
    if not REPRO_BUILD_UNUSED_BLOCKS:
        return {}

    h = _context_frame(screen.loc[screen[SEASON_COL].eq(int(source_year))].reset_index(drop=True))
    r = _context_frame(target_rows.reset_index(drop=True))
    out = {}

    # CROSS_DOMAIN_PROFILE
    xd = []
    for ent, tag in [("pitcher_id", "p"), ("batter_id", "b")]:
        tg = _eb_table(h, [ent], None, 80.0)
        td = _eb_table(h, [ent, "route_segment"], ["route_segment"], 60.0)
        a = _table_features(tg, r, [ent], f"xd__{tag}_global")
        b = _table_features(td, r, [ent, "route_segment"], f"xd__{tag}_domain")
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
    for ctx, strength in [("pitcher_hand", 35.0), ("ctx_count", 35.0), ("route_segment", 45.0)]:
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
        h.groupby(["pitcher_id", "route_segment"], sort=False)
        .agg(
            drole_n=("inning", "size"),
            drole_inning_mean=("inning", "mean"),
            drole_inning_std=("inning", "std"),
            drole_late_share=("inning", lambda s: float((s >= 7).mean())),
        )
        .reset_index()
    )
    t = r[["pitcher_id", "route_segment", "inning", "li"]].copy()
    t["_i"] = np.arange(len(t))
    t = t.merge(rg, on="pitcher_id", how="left").merge(
        rd, on=["pitcher_id", "route_segment"], how="left"
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
# Phase 2 — residual matrix (128열)
# =========================================================


def fit_residual_prep(map_rows: pd.DataFrame) -> dict:
    m = add_eng(map_rows.reset_index(drop=True))
    cols = RESIDUAL_RAWNUM + ENG_COLS
    med = m[cols].apply(pd.to_numeric, errors="coerce").median().fillna(0.0).to_dict()
    maps = {}
    for c in CAT_SMALL:
        s = m[c].astype("string").fillna("<NA>").astype(str)
        maps[c] = {v: i for i, v in enumerate(pd.Index(s.unique()))}
    del m
    return {"cols": cols, "med": med, "maps": maps}


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


def build_residual_matrix(residual_base: pd.DataFrame, pitcher_context: pd.DataFrame,
                         asof: pd.DataFrame,
                         unused_blocks: dict | None = None) -> pd.DataFrame:
    """원본 결합 순서: residual_base | PITCHER_CONTEXT | ASOF_STATE."""
    X = pd.concat(
        [
            residual_base.reset_index(drop=True),
            pitcher_context.reset_index(drop=True),
            asof.reset_index(drop=True),
        ],
        axis=1,
    )
    assert not X.columns.duplicated().any()
    assert np.isfinite(X.to_numpy(float)).all()
    assert X.shape[1] == EXPECTED_DIMS["residual_matrix"], X.shape

    # 미투입 3블록이 실제 모델 입력에 절대 포함되지 않음을 검증한다.
    if unused_blocks:
        cols = set(X.columns)
        for name, blk in unused_blocks.items():
            leaked = cols & set(blk.columns)
            assert not leaked, f"unused block {name} leaked into residual matrix: {sorted(leaked)}"
    return X


# =========================================================
# Phase 2 — sensitivity (25열)
# =========================================================


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


def _weighted_sensitivity_table(history, ctx, strength, cutoff_year):
    h = _pressure_frame(history.reset_index(drop=True))
    h["_w"] = np.power(
        SENS_DECAY, np.maximum(int(cutoff_year) - 1 - h[SEASON_COL].astype(int), 0)
    ).astype(np.float64)
    h["_wy"] = h["_w"] * pd.to_numeric(h[TARGET], errors="coerce").to_numpy(np.float64)
    mu = float(np.average(h[TARGET].to_numpy(np.float64), weights=h["_w"]))
    pg = h.groupby("pitcher_id", sort=False).agg(sw=("_w", "sum"), sy=("_wy", "sum")).reset_index()
    pg["parent_rate"] = (pg["sy"] + SENS_PARENT_STRENGTH * mu) / (pg["sw"] + SENS_PARENT_STRENGTH)
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


def build_sensitivity_specs(screen: pd.DataFrame, cutoff_year: int) -> list:
    """cutoff_year 직전 최대 3시즌의 공식 train 만 사용한다."""
    lo = max(int(screen[SEASON_COL].min()), int(cutoff_year) - SENS_HISTORY_YEARS)
    h = screen.loc[
        screen[SEASON_COL].lt(int(cutoff_year)) & screen[SEASON_COL].ge(lo)
    ].reset_index(drop=True)
    if h.empty:
        raise RuntimeError(f"no sensitivity history for cutoff {cutoff_year}")
    specs = []
    for ctx, strength, prefix in [
        ("batter_hand", SENS_HAND_STRENGTH, "sens__batter_hand"),
        ("sens_pressure_bhand", SENS_PRESSURE_HAND_STRENGTH, "sens__pressure_bhand"),
    ]:
        specs.append(
            {
                "ctx": ctx,
                "prefix": prefix,
                "table": _weighted_sensitivity_table(h, ctx, strength, cutoff_year),
            }
        )
    del h
    gc.collect()
    return specs


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


# =========================================================
# Phase 3 — temporal / segment drift (notebook cell 5)
# =========================================================


class SegmentMeans:
    """원본은 전 시즌 groupby 로 SEGMENT_MEANS 를 만든 뒤 source_year 이하만 읽는다.

    §14 정책에 따라 계산 방식은 바꾸지 않고, 읽기 시점에 season <= source_year 인지
    assert 한다. 접근 이력을 남겨 미래 시즌 참조 0건임을 산출물로 증명한다.
    """

    def __init__(self, screen: pd.DataFrame):
        self.table = screen.groupby([SEASON_COL, "route_segment"])[TARGET].mean().unstack()
        self.access_log: list[tuple[int, str, int]] = []

    def value(self, season: int, segment: str, source_year: int) -> float:
        assert int(season) <= int(source_year), (
            f"SEGMENT_MEANS future access: season={season} > source_year={source_year}"
        )
        self.access_log.append((int(season), str(segment), int(source_year)))
        return float(self.table.loc[int(season), segment])

    def trailing_core_slope(self, source_year: int) -> float:
        s = self.table.loc[self.table.index <= int(source_year), "CORE"].dropna().tail(4)
        assert len(s) == 0 or int(s.index.max()) <= int(source_year)
        for yr in s.index:
            self.access_log.append((int(yr), "CORE", int(source_year)))
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


def build_anchor(screen: pd.DataFrame, segment_means: SegmentMeans, source_year: int,
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

    # 3) segment-level trend-centered LightGBM
    A, B = simple_pair(source, target)
    assert A.shape[1] == EXPECTED_DIMS["trend_features"], A.shape
    trend_feature_cols = list(A.columns)
    source_dom = source.route_segment.to_numpy()
    target_dom = target.route_segment.to_numpy()
    slope = segment_means.trailing_core_slope(source_year)
    base = np.array(
        [segment_means.value(int(source_year), d, source_year) for d in source_dom],
        dtype=np.float64,
    )
    prior = {
        d: segment_means.value(int(source_year), d, source_year) + slope
        for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]
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

    lightgbm 없이도 residual matrix 의 열 이름 / 열 순서 / dtype / 행 정렬을
    reference 와 비교할 수 있게 한다. 행마다 값이 다르므로 정렬 버그를 잡아낸다.
    Phase 3 에서는 실제 p_base 로 다시 만든다.
    """
    return (((np.arange(n) % 997) / 997.0) * 0.98 + 0.01).astype(np.float64)


# =========================================================
# Phase 4 — Layer 1 : residual adjuster (notebook cell 12 / 13)
# =========================================================


def residual_crossfit(seed, X, y, base, frame, cfg):
    """notebook cell 12 원본.

    · 같은 seed 안에서 6개 모델(2 fold × 3 segment)이 동일한 random_state 를 공유한다.
    · corr 은 (fold × segment) 조합으로 모든 행이 정확히 1회 채워진다.
    · 반환값은 seed 단위로 clip 된 예측이다. 평균은 호출부에서 한다.
    """
    _import_phase4_libs()
    fold = pitcher_half(frame, seed)
    corr = np.zeros(len(frame), dtype=np.float64)
    filled = np.zeros(len(frame), dtype=np.int8)  # 재현 무관 감사용
    fold_rows = []
    for target_fold in [0, 1]:
        src = fold != target_fold
        tgt = fold == target_fold
        for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]:
            tr = src & frame.route_segment.eq(d).to_numpy()
            te = tgt & frame.route_segment.eq(d).to_numpy()
            min_tr = 1000 if not SMOKE_MODE else 80
            min_te = 500 if not SMOKE_MODE else 40
            assert tr.sum() > min_tr and te.sum() > min_te, (d, tr.sum(), te.sum())
            m = ExtraTreesRegressor(
                n_jobs=N_JOBS, random_state=16200 + int(seed), **cfg
            )
            m.fit(X.loc[tr], (y - base)[tr])
            corr[te] = m.predict(X.loc[te])
            filled[te] += 1
            del m
            gc.collect()
        pp = np.clip(base[tgt] + RESIDUAL_SHRINK * corr[tgt], 0, 1)
        fold_rows.append(
            {
                "seed": int(seed),
                "target_fold": int(target_fold),
                "n": int(tgt.sum()),
                "base_bss": brier_skill(y[tgt], base[tgt]),
                "adapted_bss": brier_skill(y[tgt], pp),
                "gain": brier_skill(y[tgt], pp) - brier_skill(y[tgt], base[tgt]),
            }
        )
    # 모든 행이 정확히 한 번씩 예측되었는가 (fold × segment 이 완전 분할인가)
    assert filled.min() == 1 and filled.max() == 1, (
        "corr coverage broken",
        int(filled.min()), int(filled.max()),
    )
    return np.clip(base + RESIDUAL_SHRINK * corr, 0, 1), fold_rows, corr


def run_adjusted_oof(year, X, y, base, frame, verbose=True):
    """notebook cell 13 adjusted_oof_year 원본.

    seed 별로 clip 한 뒤 평균한다. 평균 후 clip 이 아니다.
    """
    preds = []
    corrs = {}
    seed_rows = []
    all_fold_rows = []
    for seed in CF_SEEDS:
        t0 = time.time()
        p, fold_rows, corr = residual_crossfit(seed, X, y, base, frame, RESIDUAL_CF_CFG)
        preds.append(p)
        corrs[int(seed)] = corr
        all_fold_rows.extend(fold_rows)
        row = {
            "year": int(year),
            "seed": int(seed),
            "bss": brier_skill(y, p),
            "gain_vs_base": brier_skill(y, p) - brier_skill(y, base),
            "sec": round(time.time() - t0, 1),
        }
        seed_rows.append(row)
        if verbose:
            print(f"    {row}")
        gc.collect()
    consensus = np.mean(np.column_stack(preds), axis=1)
    if verbose:
        print(f"    adjusted OOF CONSENSUS {year} {brier_skill(y, consensus):.2f}")
    return consensus, preds, corrs, seed_rows, all_fold_rows


# =========================================================
# Phase 5 — Layer 2 : sensitivity shift (intercept-free Ridge)
#                     (notebook cell 14 = PASS 2, cell 15 = 승격 gate)
# =========================================================

PHASE5_SOURCE_YEARS = [2020, 2021, 2022, 2023, 2024]
PHASE5_TRANSITIONS = [2021, 2022, 2023, 2024]

# Layer 2 는 sensitivity / bootstrap / month 집계에만 쓰이는 소수 컬럼만 필요하다.
PHASE5_SLIM_COLS = ["pitcher_id", "batter_hand", "balls_before", "strikes_before",
                    "route_segment", "game_month"]


def forest_predict_rowwise(model, X) -> np.ndarray:
    """행 독립·결정론이 보장되는 sklearn forest 예측.

    n_jobs>1 이면 트리 출력을 여러 스레드가 공유 배열에 더하므로 누적 순서가 실행마다 달라져
    2^-53 수준의 잡음이 생긴다(재현 감사에서 실측). 단일 스레드로 예측하면 트리 순서가 고정되어
    결과가 결정론적이고 배치 크기와도 무관하다. 배포(Phase 6)·script.py 가 같은 함수를 쓴다.
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


def _fit_shift_pair(Az, Bz, resid, rowwise: bool = False):
    """같은 표준화 행렬 위에서 reference(intercept 포함)와 shift(intercept 제거)를 함께 적합한다.

    reference : 원본 Layer 2 그대로. 절편 = source 시즌 잔차 평균이 target 시즌에 상수로 전이된다.
    shift     : fit_intercept=False. 절편(상수 성분)만 제거하고 조건부 성분은 동일하게 학습한다.
                절편은 train(source 시즌)만으로 정해지는 상수이므로 제거 역시 train-only 결정이다.
    """
    ref = Ridge(alpha=SHIFT_RIDGE_ALPHA, solver="lsqr", max_iter=200, tol=1e-5)
    ref.fit(Az, resid)
    sh = Ridge(alpha=SHIFT_RIDGE_ALPHA, solver="lsqr", max_iter=200, tol=1e-5, fit_intercept=False)
    sh.fit(Az, resid)
    if rowwise:   # 배포 경로: 행 독립 보장
        return ref, sh, ridge_predict_rowwise(ref, Bz), ridge_predict_rowwise(sh, Bz)
    return ref, sh, np.asarray(ref.predict(Bz), np.float64), np.asarray(sh.predict(Bz), np.float64)


def _apply_shift(base, corr, core, scale):
    p = base.copy()
    p[core] = np.clip(p[core] + scale * corr[core], 0, 1)
    return p


def run_phase5(screen: pd.DataFrame, feature_dump_dir: Path, validation_dir: Path) -> dict:
    """PASS 2. ADJUSTED_OOF 는 Phase 4 산출물(npz)에서 읽는다.

    reference(원본 Layer 2)와 shift(절편 제거)를 **같은 실행에서 함께** 계산해
    전이별 성능 차이를 출력하고, reference 를 넘을 때만 ADOPT=True 로 기록한다.
    """
    _import_phase3_libs()

    # ---------- 입력 적재 ----------
    ROWS, Y, ADJUSTED_OOF = {}, {}, {}
    for y in PHASE5_SOURCE_YEARS:
        rows = screen.loc[screen[SEASON_COL].eq(int(y))].reset_index(drop=True)
        Y[y] = rows[TARGET].to_numpy(np.float64)
        z, path = load_cached_npz(feature_dump_dir, f"adjusted_oof_{y}.npz", [f"v15_oof_{y}.npz"])
        if z is None:
            raise FileNotFoundError(
                f"Phase 5 는 Phase 4 산출물이 전부 필요하다. 없음: adjusted_oof_{y}.npz "
                f"(검색: {feature_dump_dir}, {CACHE_DIR})\n  SHIFT_PHASE=4 로 {PHASE5_SOURCE_YEARS} 를 먼저 실행할 것."
            )
        ADJUSTED_OOF[y] = npz_key(z, "adjusted_oof", ["v15_oof"]).astype(np.float64)
        assert len(ADJUSTED_OOF[y]) == len(rows), (y, len(ADJUSTED_OOF[y]), len(rows))
        assert np.array_equal(z["target"].astype(np.float64), Y[y]), f"target mismatch {y}"
        print(f"    adjusted_oof {y} <- {path}")
        ROWS[y] = rows
        del z

    # ---------- sensitivity 행렬 ----------
    SENS, SENS_SPECS = {}, {}
    for y in PHASE5_SOURCE_YEARS:
        SENS_SPECS[y] = build_sensitivity_specs(screen, y)
        SENS[y] = transform_sensitivity_specs(SENS_SPECS[y], ROWS[y])
        print(f"    sensitivity matrix {y} {list(SENS[y].shape)}")
        ROWS[y] = ROWS[y][PHASE5_SLIM_COLS].copy()
        gc.collect()

    # ---------- PASS 2 : 전이별 reference / shift ----------
    SHIFT_MODELS, TRANSITION_ROWS = {}, []
    FINAL, REFERENCE, CORR_SHIFT, CORR_REF = {}, {}, {}, {}
    stress_rows = []
    for target_year in PHASE5_TRANSITIONS:
        source_year = target_year - 1
        A = SENS[source_year].to_numpy(np.float32)
        B = SENS[target_year].to_numpy(np.float32)
        mu = A.mean(0)
        sd = A.std(0)
        sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
        Az = np.clip((A - mu) / sd, -8, 8)
        Bz = np.clip((B - mu) / sd, -8, 8)
        resid = Y[source_year] - ADJUSTED_OOF[source_year]
        ref_model, shift_model, corr_ref, corr_shift = _fit_shift_pair(Az, Bz, resid)
        # 두 모델의 계수는 동일해야 한다 (절편만 다름) — 구조 확인
        coef_gap = float(np.abs(ref_model.coef_ - shift_model.coef_).max())

        base = ADJUSTED_OOF[target_year]
        frame = ROWS[target_year]
        yy = Y[target_year]
        core = frame.route_segment.eq(SHIFT_SEGMENT).to_numpy()
        p_ref = _apply_shift(base, corr_ref, core, SHIFT_SCALE)
        p_new = _apply_shift(base, corr_shift, core, SHIFT_SCALE)
        REFERENCE[target_year], FINAL[target_year] = p_ref, p_new
        CORR_REF[target_year], CORR_SHIFT[target_year] = corr_ref, corr_shift
        SHIFT_MODELS[target_year] = {
            "model": shift_model, "reference_model": ref_model,
            "mu": mu.astype(np.float32), "sd": sd, "source_year": source_year,
        }
        b_adj, b_ref, b_new = brier_skill(yy, base), brier_skill(yy, p_ref), brier_skill(yy, p_new)
        row = {
            "source_year": source_year, "target_year": target_year,
            "adjusted_bss": b_adj, "reference_bss": b_ref, "final_bss": b_new,
            "reference_gain": b_ref - b_adj,
            "final_gain": b_new - b_adj,
            "final_vs_reference": b_new - b_ref,
            "reference_core_gain": brier_skill(yy[core], p_ref[core]) - brier_skill(yy[core], base[core]),
            "final_core_gain": brier_skill(yy[core], p_new[core]) - brier_skill(yy[core], base[core]),
            "reference_intercept": float(ref_model.intercept_),
            "source_resid_mean": float(resid.mean()),
            "target_resid_mean": float((yy - base).mean()),
            "coef_gap": coef_gap,
        }
        TRANSITION_ROWS.append(row)
        print(f"    {source_year}->{target_year}  adjusted={b_adj:.2f}  reference={b_ref:.2f}  "
              f"final={b_new:.2f}  final_vs_reference={b_new - b_ref:+.2f}  "
              f"(ref_gain={b_ref - b_adj:+.2f}, final_gain={b_new - b_adj:+.2f}, "
              f"intercept={ref_model.intercept_:+.5f}, target_resid_mean={row['target_resid_mean']:+.5f})")
        stress = {}
        for g in SHIFT_SCALE_STRESS:
            stress[f"reference_{g}"] = brier_skill(yy, _apply_shift(base, corr_ref, core, g)) - b_adj
            stress[f"final_{g}"] = brier_skill(yy, _apply_shift(base, corr_shift, core, g)) - b_adj
        stress_rows.append({"target_year": target_year, **stress})
        del A, B, Az, Bz
        gc.collect()

    transition_df = pd.DataFrame(TRANSITION_ROWS)
    transition_df.to_csv(validation_dir / "transitions.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(validation_dir / "scale_stress.csv", index=False)

    # ---------- 성능 비교 표 ----------
    print()
    print("    " + "-" * 92)
    print(f"    {'transition':>10} {'adjusted':>10} {'reference':>10} {'final':>10} "
          f"{'ref_gain':>9} {'final_gain':>10} {'vs_ref':>8} {'core_vs_ref':>11}")
    for r in TRANSITION_ROWS:
        print(f"    {r['source_year']}->{r['target_year']:<4} {r['adjusted_bss']:>10.2f} "
              f"{r['reference_bss']:>10.2f} {r['final_bss']:>10.2f} {r['reference_gain']:>+9.2f} "
              f"{r['final_gain']:>+10.2f} {r['final_vs_reference']:>+8.2f} "
              f"{r['final_core_gain'] - r['reference_core_gain']:>+11.2f}")
    sum_ref = float(transition_df.reference_gain.sum())
    sum_new = float(transition_df.final_gain.sum())
    min_ref = float(transition_df.reference_gain.min())
    min_new = float(transition_df.final_gain.min())
    n_better = int((transition_df.final_vs_reference >= 0).sum())
    print("    " + "-" * 92)
    print(f"    sum gain   reference={sum_ref:+.2f}  final={sum_new:+.2f}  diff={sum_new - sum_ref:+.2f}")
    print(f"    worst      reference={min_ref:+.2f}  final={min_new:+.2f}  diff={min_new - min_ref:+.2f}")
    print(f"    transitions final >= reference : {n_better}/{len(TRANSITION_ROWS)}")

    # ---------- 2024 gate 진단 ----------
    p_adj = ADJUSTED_OOF[VALID_YEAR]
    p_ref = REFERENCE[VALID_YEAR]
    p_new = FINAL[VALID_YEAR]
    frame = ROWS[VALID_YEAR]
    yy = Y[VALID_YEAR]
    draws = 3000 if not SMOKE_MODE else 200
    boot_vs_adjusted = pitcher_bootstrap_gain(frame, yy, p_new, p_adj, draws=draws, seed=18018)
    boot_vs_reference = pitcher_bootstrap_gain(frame, yy, p_new, p_ref, draws=draws, seed=18018)
    boot_reference_vs_adjusted = pitcher_bootstrap_gain(frame, yy, p_ref, p_adj, draws=draws, seed=18018)
    print(f"    bootstrap {VALID_YEAR} final vs adjusted  : {boot_vs_adjusted}")
    print(f"    bootstrap {VALID_YEAR} final vs reference : {boot_vs_reference}")
    print(f"    bootstrap {VALID_YEAR} reference vs adjusted: {boot_reference_vs_adjusted}")

    month_rows = []
    for mo in sorted(frame.game_month.unique()):
        z = frame.game_month.eq(mo).to_numpy()
        month_rows.append({
            "month": int(mo), "n": int(z.sum()),
            "adjusted": brier_skill(yy[z], p_adj[z]), "reference": brier_skill(yy[z], p_ref[z]),
            "final": brier_skill(yy[z], p_new[z]),
            "final_gain": brier_skill(yy[z], p_new[z]) - brier_skill(yy[z], p_adj[z]),
            "final_vs_reference": brier_skill(yy[z], p_new[z]) - brier_skill(yy[z], p_ref[z]),
        })
    month_df = pd.DataFrame(month_rows)
    month_df.to_csv(validation_dir / "months.csv", index=False)
    early = frame.game_month.le(6).to_numpy()
    late = frame.game_month.ge(7).to_numpy()
    early_gain = brier_skill(yy[early], p_new[early]) - brier_skill(yy[early], p_adj[early])
    late_gain = brier_skill(yy[late], p_new[late]) - brier_skill(yy[late], p_adj[late])
    print(f"    early_gain {early_gain:+.2f}  late_gain {late_gain:+.2f}")

    segment_rows = []
    for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]:
        z = frame.route_segment.eq(d).to_numpy()
        segment_rows.append({
            "segment": d, "n": int(z.sum()),
            "adjusted": brier_skill(yy[z], p_adj[z]), "reference": brier_skill(yy[z], p_ref[z]),
            "final": brier_skill(yy[z], p_new[z]),
            "final_gain": brier_skill(yy[z], p_new[z]) - brier_skill(yy[z], p_adj[z]),
            "final_vs_reference": brier_skill(yy[z], p_new[z]) - brier_skill(yy[z], p_ref[z]),
        })
    segment_df = pd.DataFrame(segment_rows)
    segment_df.to_csv(validation_dir / "segments.csv", index=False)

    g23 = float(transition_df.loc[transition_df.target_year.eq(2023), "final_gain"].iloc[0])
    g24 = float(transition_df.loc[transition_df.target_year.eq(2024), "final_gain"].iloc[0])
    core24 = float(segment_df.loc[segment_df.segment.eq("CORE"), "final_gain"].iloc[0])
    stress_ok = True
    for ty in [2023, 2024]:
        for g in (0.65, 0.75, 0.85):
            pp = _apply_shift(ADJUSTED_OOF[ty], CORR_SHIFT[ty],
                              ROWS[ty].route_segment.eq(SHIFT_SEGMENT).to_numpy(), g)
            stress_ok &= (brier_skill(Y[ty], pp) - brier_skill(Y[ty], ADJUSTED_OOF[ty]) > 0)

    # ---------- 채택 판정 ----------
    # (A) 배포 layer 자체의 절대 기준 — 원본 gate 와 동일 조건
    layer_gate = {
        "g23>=8.0": g23 >= MIN_GAIN_2023,
        "g24>=10.0": g24 >= MIN_GAIN_2024,
        "core24>=15.0": core24 >= MIN_CORE_GAIN_2024,
        "boot_vs_adjusted_p05>=3.0": boot_vs_adjusted["p05"] >= MIN_BOOT_P05,
        "boot_vs_adjusted_prob>=0.97": boot_vs_adjusted["prob_positive"] >= MIN_BOOT_PROB,
        "late_gain>=0.0": late_gain >= MIN_LATE_GAIN,
        "stress_ok": bool(stress_ok),
    }
    # (B) reference 대비 상대 기준 — 이것을 넘어야 "다음 근거"로 채택한다
    relative_gate = {
        "sum_gain_final>sum_gain_reference": sum_new > sum_ref,
        "worst_transition_final>worst_transition_reference": min_new > min_ref,
        f"boot_{VALID_YEAR}_final_vs_reference_p05>={MIN_RELATIVE_BOOT_P05}":
            boot_vs_reference["p05"] >= MIN_RELATIVE_BOOT_P05,
        f"transitions_final>=reference_count>={MIN_TRANSITIONS_NOT_WORSE}":
            n_better >= MIN_TRANSITIONS_NOT_WORSE,
    }
    layer_ok = all(layer_gate.values())
    relative_ok = all(relative_gate.values())
    adopt = bool(layer_ok and relative_ok)
    summary = {
        "protocol": PROTOCOL_VERSION,
        "layer2_mode_if_adopted": "shift (fit_intercept=False)",
        "layer2_mode_if_rejected": "reference (fit_intercept=True)",
        "transition_2022_to_2023_gain": g23,
        "transition_2023_to_2024_gain": g24,
        "core_2024_gain": core24,
        "sum_gain_reference": sum_ref, "sum_gain_final": sum_new,
        "worst_gain_reference": min_ref, "worst_gain_final": min_new,
        "n_transitions_final_not_worse": n_better,
        "bootstrap_final_vs_adjusted": boot_vs_adjusted,
        "bootstrap_final_vs_reference": boot_vs_reference,
        "bootstrap_reference_vs_adjusted": boot_reference_vs_adjusted,
        "early_gain": float(early_gain), "late_gain": float(late_gain),
        "scale": SHIFT_SCALE, "shift_segment": SHIFT_SEGMENT,
        "stress_ok": bool(stress_ok),
        "layer_gate": layer_gate, "layer_gate_ok": layer_ok,
        "relative_gate": relative_gate, "relative_gate_ok": relative_ok,
        "adopt": adopt,
        "submit_recommended": adopt,
        "warning": "2023/2024 are development-exposed. Validation BSS is not an LB estimate. "
                   "Deploy Layer 2 is trained on 2024 residual whose mean is small; the value of "
                   "intercept removal is robustness to next-season base-rate shift, not a large LB gain.",
    }
    print()
    print("    === ADOPT 판정 ===")
    for k, v in layer_gate.items():
        print(f"    [layer]    {'PASS' if v else 'FAIL'}  {k}")
    for k, v in relative_gate.items():
        print(f"    [relative] {'PASS' if v else 'FAIL'}  {k}")
    print(f"    ADOPT = {adopt}   (deploy Layer 2 = {'shift' if adopt else 'reference'})")
    write_json(validation_dir / "gate.json", summary)

    # ---------- 산출물 ----------
    oof_parts = []
    for ty in PHASE5_TRANSITIONS:
        rows = ROWS[ty]
        fold = pd.DataFrame({
            "season": int(ty),
            "route_segment": rows.route_segment.to_numpy(),
            "target": Y[ty],
            "adjusted": ADJUSTED_OOF[ty],
            "reference_correction": CORR_REF[ty],
            "shift_correction": CORR_SHIFT[ty],
            "reference_prediction": REFERENCE[ty],
            "final_prediction": FINAL[ty],
            "adjusted_residual": Y[ty] - ADJUSTED_OOF[ty],
            "final_residual": Y[ty] - FINAL[ty],
        })
        fold.to_csv(validation_dir / f"fold_{ty}.csv", index=False, float_format="%.17g")
        oof_parts.append(fold)
        np.savez_compressed(
            feature_dump_dir / f"final_{ty}.npz",
            final=FINAL[ty], reference=REFERENCE[ty], adjusted_oof=ADJUSTED_OOF[ty],
            corr_shift=CORR_SHIFT[ty], corr_reference=CORR_REF[ty], target=Y[ty],
            core=ROWS[ty].route_segment.eq(SHIFT_SEGMENT).to_numpy(),
        )
    pd.concat(oof_parts, ignore_index=True).to_csv(
        validation_dir / "oof.csv", index=False, float_format="%.17g"
    )

    phase5 = {
        "experiment_name": EXPERIMENT_NAME,
        "layer": "layer2_sensitivity_shift",
        "regime": "strict next-year (fit on T-1 OOF residual, apply to T)",
        "transitions": PHASE5_TRANSITIONS,
        "ridge": {"alpha": SHIFT_RIDGE_ALPHA, "solver": "lsqr", "max_iter": 200, "tol": 1e-5,
                  "final_fit_intercept": False, "reference_fit_intercept": True},
        "scale": SHIFT_SCALE, "shift_segment": SHIFT_SEGMENT,
        "sensitivity_cols": list(SENS[VALID_YEAR].columns),
        "transition_rows": TRANSITION_ROWS,
        "scale_stress": stress_rows,
        "months": month_rows,
        "segments": segment_rows,
        "gate": summary,
        "library_versions": library_versions(),
    }
    write_json(validation_dir / "phase5_summary.json", phase5)
    return phase5


# =========================================================
# Phase 6 — Final artifact (deploy regime, 2024 -> 2025)
#           notebook cell 16 의 모델 구축부. zip / subprocess smoke 는 Phase 7.
# =========================================================

DEPLOY_SOURCE_YEAR = VALID_YEAR          # 2024
DEPLOY_PC_TRAIN_SOURCE = BASE_SOURCE_YEAR  # 2023 — 학습 feature 는 2023 EB


def run_phase6(screen: pd.DataFrame, segment_means: SegmentMeans,
               feature_dump_dir: Path, validation_dir: Path) -> dict:
    """배포 regime. validation 과 5가지가 다르다 — 그대로 재현한다.

      1) Layer 1 은 2024 **전체**로 학습한다 (crossfit 없음)
      2) `RESIDUAL_FINAL_CFG` (n_estimators=120). `RESIDUAL_CF_CFG`(100) 가 아니다
      3) `fit_residual_prep` 소스가 **raw24** 다 (학습 시즌과 동일)
      4) 학습 PC EB 는 2023, 추론 PC EB 는 2024
      5) Layer 2 는 **in-sample residual 출력**이 아니라 p_base25 기반 direct 예측을 보정한다
    """
    _import_phase3_libs()
    _import_phase4_libs()

    expected_year = int(screen[SEASON_COL].max()) + 1
    raw24 = screen.loc[screen[SEASON_COL].eq(DEPLOY_SOURCE_YEAR)].reset_index(drop=True)
    y24 = raw24[TARGET].to_numpy(np.float64)

    # ---- Phase 3 / 4 / 5 산출물 ----
    za, za_path = load_cached_npz(feature_dump_dir, f"anchor_{DEPLOY_SOURCE_YEAR}.npz")
    assert za is not None, f"anchor_{DEPLOY_SOURCE_YEAR}.npz 가 없다 ({feature_dump_dir}, {CACHE_DIR})"
    p_base24 = za["p_base"].astype(np.float64)
    zo, zo_path = load_cached_npz(feature_dump_dir, f"adjusted_oof_{DEPLOY_SOURCE_YEAR}.npz",
                                  [f"v15_oof_{DEPLOY_SOURCE_YEAR}.npz"])
    assert zo is not None, f"adjusted_oof_{DEPLOY_SOURCE_YEAR}.npz 가 없다"
    adjusted_oof24 = npz_key(zo, "adjusted_oof", ["v15_oof"]).astype(np.float64)
    assert np.array_equal(zo["target"].astype(np.float64), y24), "adjusted_oof target mismatch"
    assert float(np.abs(zo["p_base"].astype(np.float64) - p_base24).max()) == 0.0, (
        "anchor_2024 와 adjusted_oof_2024 의 p_base 가 다르다 — Artifact Chain 확인"
    )
    print(f"    anchor <- {za_path}\n    adjusted_oof <- {zo_path}")
    gate_path = validation_dir / "gate.json"
    assert gate_path.exists(), f"Phase 5 의 {gate_path} 가 필요하다"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    shift_adopted = bool(gate["adopt"])
    layer2_mode = "shift" if shift_adopted else "reference"
    print(f"    ADOPT (Phase 5) : {shift_adopted}  -> deploy Layer 2 = {layer2_mode}")

    # ---- test 적재 / 검증 ----
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", low_memory=False)
    test[SEASON_COL] = pd.to_numeric(test[SEASON_COL], errors="raise").astype(int)
    test["game_type"] = test["game_type"].astype("string").fillna("<NA>").astype(str)
    assert TARGET not in test, "test.csv 에 target 이 있다"
    if bool((test[SEASON_COL] != expected_year).any()):
        raise RuntimeError(f"evaluation season differs from expected {expected_year}")
    if bool((~test.game_type.isin(["R", "F"])).any()):
        raise RuntimeError("unexpected game_type")
    anchor_team_id = fit_router(screen)
    test = attach_route_segment(test, anchor_team_id)
    assert str(test["base_state"].dtype) == "object", (
        f"test.base_state dtype={test['base_state'].dtype} — object 여야 한다 (L-3 참조)"
    )
    print(f"    expected_year={expected_year}  test rows={len(test)}  "
          f"route_segment={test.route_segment.value_counts().to_dict()}")

    # ---- 1) 2024 -> 2025 shared base ----
    source = raw24
    ss = build_self_state(source.drop(columns=[TARGET], errors="ignore"))
    source_bank = fit_global_season_bank(
        screen.loc[screen[SEASON_COL].lt(DEPLOY_SOURCE_YEAR), BANK_COLS], DEPLOY_SOURCE_YEAR
    )
    gs = transform_global_season(source_bank, source.drop(columns=[TARGET], errors="ignore"))
    target_bank = fit_global_season_bank(
        screen.loc[screen[SEASON_COL].lt(expected_year), BANK_COLS], expected_year
    )
    st = build_self_state(test)
    gt = transform_global_season(target_bank, test)
    A0 = pd.concat(
        [source[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
         ss.reset_index(drop=True), gs.reset_index(drop=True)], axis=1)
    B0 = pd.concat(
        [test[RAWNUM].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
         st.reset_index(drop=True), gt.reset_index(drop=True)], axis=1)

    ca, cb, exact_maps = encode_cats(source, test)
    A_exact = pd.concat([A0, ca], axis=1)
    B_exact = pd.concat([B0, cb], axis=1)
    exact_feature_cols = list(A_exact.columns)
    assert len(exact_feature_cols) == EXPECTED_DIMS["exact_lgb_features"]
    m1 = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_JOBS,
                            **BASE_CFG["exact_lgb"])
    m1.fit(A_exact, y24)
    p1 = lgb_predict_batchpath(m1, B_exact, proba=True)   # 행 독립 (배포 경로)
    del A_exact, B_exact, ca, cb
    gc.collect()

    Ar = A0.replace([np.inf, -np.inf], np.nan)
    Br = B0.replace([np.inf, -np.inf], np.nan)
    ridge_feature_cols = list(A0.columns)
    ridge_med = Ar.median().fillna(0.0)
    Ar = Ar.fillna(ridge_med).to_numpy(np.float32)
    Br = Br.fillna(ridge_med).to_numpy(np.float32)
    ridge_mu = Ar.mean(0)
    ridge_sd = Ar.std(0)
    ridge_sd = np.where(ridge_sd > 1e-6, ridge_sd, 1.0)
    Ar = np.clip((Ar - ridge_mu) / ridge_sd, -8, 8)
    Br = np.clip((Br - ridge_mu) / ridge_sd, -8, 8)
    m2 = Ridge(alpha=BASE_CFG["ridge_alpha"], solver="lsqr", max_iter=100, tol=1e-3)
    m2.fit(Ar, y24)
    p2 = np.clip(ridge_predict_rowwise(m2, Br), 0.001, 0.999)   # 행 독립 (배포 경로)
    del Ar, Br, A0, B0, ss, gs
    gc.collect()

    s = add_eng(source)
    t = add_eng(test)
    trend_num_cols = RAWNUM + ENG_COLS
    An = s[trend_num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    Bn = t[trend_num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    cta, ctb, trend_maps = encode_cats(s, t)
    At = pd.concat([An, cta], axis=1).astype(np.float32)
    Bt = pd.concat([Bn, ctb], axis=1).astype(np.float32)
    trend_feature_cols = list(At.columns)
    slope = segment_means.trailing_core_slope(DEPLOY_SOURCE_YEAR)
    base_dom = np.array(
        [segment_means.value(DEPLOY_SOURCE_YEAR, d, DEPLOY_SOURCE_YEAR)
         for d in source.route_segment.to_numpy()], dtype=np.float64)
    trend_prior = {d: segment_means.value(DEPLOY_SOURCE_YEAR, d, DEPLOY_SOURCE_YEAR) + slope
                   for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]}
    bt = np.array([trend_prior[d] for d in test.route_segment.to_numpy()], dtype=np.float64)
    m3 = lgb.LGBMRegressor(objective="regression_l2", verbosity=-1, n_jobs=N_JOBS,
                           **BASE_CFG["trend_lgb"])
    m3.fit(At, y24 - base_dom)
    p3 = np.clip(bt + lgb_predict_batchpath(m3, Bt), 0.001, 0.999)   # 행 독립 (배포 경로)
    del At, Bt, An, Bn, cta, ctb, s, t
    gc.collect()

    p_base25 = np.clip(
        (np.asarray(p1, float) + np.asarray(p2, float) + np.asarray(p3, float)) / 3.0, 0, 1)
    print(f"    p_base25 mean={float(p_base25.mean()):.6f}  slope={slope:.10f}")

    # ---- 2) context 블록 : 학습은 2023 EB, 추론은 2024 EB ----
    pc23 = make_context_specs(screen, DEPLOY_PC_TRAIN_SOURCE)
    pc24 = make_context_specs(screen, DEPLOY_SOURCE_YEAR)
    pc24_train = transform_context_specs(pc23, raw24)
    pc25 = transform_context_specs(pc24, test)
    ex24 = build_asof_state(screen, raw24, DEPLOY_SOURCE_YEAR)
    ex25 = pd.concat([st.add_prefix("exact__"), gt.add_prefix("exact__")], axis=1) \
        .replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # notebook cell 16 line 94~97 의 자기검증: 배포 학습 feature 가
    # validation 에서 이미 gate 통과한 블록과 같은가
    parity = {}
    for name, built, old in [("pitcher_context", pc24_train, "pitcher_conditional"),
                             ("asof_state", ex24, "exact_asof_state")]:
        z, ref = load_cached_npz(feature_dump_dir, f"{name}_{DEPLOY_SOURCE_YEAR}.npz",
                                 [f"{old}_{DEPLOY_SOURCE_YEAR}.npz"])
        if z is not None:
            assert list(built.columns) == [str(c) for c in z["columns"]], name
            d = float(np.abs(built.to_numpy(np.float32).astype(np.float64)
                             - z["values"].astype(np.float64)).max())
            parity[name] = d
            assert d < 1e-7, (name, d)
    print(f"    deploy 학습 feature parity vs Phase2 산출물: {parity}")

    prep24 = fit_residual_prep(raw24)
    X24_base = transform_residual_features(prep24, raw24, p_base24)
    X25_base = transform_residual_features(prep24, test, p_base25)
    residual_base_feature_cols = list(X24_base.columns)
    X24_final = pd.concat([X24_base.reset_index(drop=True), pc24_train.reset_index(drop=True),
                           ex24.reset_index(drop=True)], axis=1)
    X25 = pd.concat([X25_base.reset_index(drop=True), pc25.reset_index(drop=True),
                     ex25.reset_index(drop=True)], axis=1)
    assert list(X24_final.columns) == list(X25.columns)
    assert not X24_final.columns.duplicated().any()
    assert np.isfinite(X24_final.to_numpy(float)).all() and np.isfinite(X25.to_numpy(float)).all()
    assert X24_final.shape[1] == EXPECTED_DIMS["residual_matrix"] == 128
    residual_feature_cols = list(X24_final.columns)
    print(f"    deploy residual {list(X24_final.shape)} -> test {list(X25.shape)}")

    # ---- 3) 최종 residual adjuster : 2024 전체 학습, crossfit 없음 ----
    #      DEPLOY_RESIDUAL_SHRINK == 0 이면 배포에 쓰이지 않으므로 학습을 생략한다 (artifact 에 빈 dict).
    res24 = y24 - p_base24
    residual_models = {}
    corrections = []
    residual_seeds = list(CF_SEEDS) if DEPLOY_RESIDUAL_SHRINK != 0.0 else []
    if not residual_seeds:
        print(f"    deploy residual adjuster SKIPPED (DEPLOY_RESIDUAL_SHRINK={DEPLOY_RESIDUAL_SHRINK})")
    for seed in residual_seeds:
        c = np.zeros(len(test), dtype=np.float64)
        for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]:
            tr = raw24.route_segment.eq(d).to_numpy()
            te = test.route_segment.eq(d).to_numpy()
            t0 = time.time()
            m = ExtraTreesRegressor(n_jobs=N_JOBS, random_state=16200 + int(seed),
                                    **RESIDUAL_FINAL_CFG)
            m.fit(X24_final.loc[tr], res24[tr])
            residual_models[(int(seed), d)] = m
            if te.any():
                c[te] = forest_predict_rowwise(m, X25.loc[te])   # 행 독립·결정론 (배포 경로)
            print(f"      residual seed={seed} segment={d:<8} n_tr={int(tr.sum()):>7} "
                  f"n_te={int(te.sum()):>4} sec={time.time() - t0:.1f}")
            gc.collect()
        corrections.append(c)
    correction = (np.mean(np.column_stack(corrections), axis=1) if corrections
                  else np.zeros(len(test), dtype=np.float64))
    adjusted_direct = np.clip(p_base25 + DEPLOY_RESIDUAL_SHRINK * correction, 0, 1)
    final_direct = adjusted_direct.copy()
    assert np.isfinite(final_direct).all() and ((final_direct >= 0) & (final_direct <= 1)).all()

    # ---- 4) Layer 2 : 2024 OOF 잔차로 학습해 2025 CORE 에만 적용 ----
    sens_specs24 = build_sensitivity_specs(screen, DEPLOY_SOURCE_YEAR)
    sens24 = transform_sensitivity_specs(sens_specs24, raw24)
    sens_specs25 = build_sensitivity_specs(screen, expected_year)
    sens25 = transform_sensitivity_specs(sens_specs25, test)
    assert list(sens24.columns) == list(sens25.columns)
    A = sens24.to_numpy(np.float32)
    B = sens25.to_numpy(np.float32)
    sens_mu = A.mean(0)
    sens_sd = A.std(0)
    sens_sd = np.where(sens_sd > 1e-6, sens_sd, 1.0).astype(np.float32)
    Az = np.clip((A - sens_mu) / sens_sd, -8, 8)
    Bz = np.clip((B - sens_mu) / sens_sd, -8, 8)
    reference_model, shift_model, reference_corr25, shift_corr25 = _fit_shift_pair(
        Az, Bz, y24 - adjusted_oof24, rowwise=True)
    deploy_model = shift_model if shift_adopted else reference_model
    deploy_corr25 = shift_corr25 if shift_adopted else reference_corr25
    core25 = test.route_segment.eq(SHIFT_SEGMENT).to_numpy()
    final_direct[core25] = np.clip(
        final_direct[core25] + SHIFT_SCALE * deploy_corr25[core25], 0, 1)
    removed_constant = float(reference_model.intercept_)
    print(f"    Layer 2 mode={layer2_mode}  sensitivity {list(sens24.shape)}  "
          f"core test rows={int(core25.sum())}  "
          f"reference intercept(제거되는 상수)={removed_constant:+.6f}  "
          f"mean|shift-reference| on core={float(np.abs(shift_corr25 - reference_corr25)[core25].mean()):.6f}")
    assert np.isfinite(final_direct).all() and ((final_direct >= 0) & (final_direct <= 1)).all()

    # ---- 5) 단일 pkl artifact ----
    artifact = {
        "experiment_name": EXPERIMENT_NAME,
        "protocol": PROTOCOL_VERSION,
        "expected_year": int(expected_year),
        "anchor_team_id": int(anchor_team_id),
        "selected_blocks": list(DEPLOY_SELECTED_BLOCKS),
        "rawnum": list(RAWNUM), "cat_small": list(CAT_SMALL),
        "eng_cols": list(ENG_COLS), "residual_rawnum": list(RESIDUAL_RAWNUM),
        # Layer 0
        "base_exact_lgb": m1, "exact_feature_cols": exact_feature_cols,
        "exact_cat_maps": exact_maps,
        "base_ridge": m2, "ridge_feature_cols": ridge_feature_cols,
        "ridge_med": ridge_med.to_dict(),
        "ridge_mu": ridge_mu.astype(np.float32), "ridge_sd": ridge_sd.astype(np.float32),
        "base_trend_lgb": m3, "trend_num_cols": list(trend_num_cols),
        "trend_cat_maps": trend_maps, "trend_feature_cols": trend_feature_cols,
        "trend_prior": trend_prior, "trend_slope": float(slope),
        "target_global_bank": target_bank,
        # Layer 1
        "residual_prep": prep24,
        "residual_base_feature_cols": residual_base_feature_cols,
        "residual_feature_cols": residual_feature_cols,
        "pitcher_context_specs": pc24,
        "pitcher_context_cols": list(pc25.columns),
        "asof_residual_cols": list(ex25.columns),
        "residual_seeds": residual_seeds, "residual_shrink": float(DEPLOY_RESIDUAL_SHRINK),
        "residual_models": residual_models,
        # Layer 2
        "layer2_mode": layer2_mode,
        "shift_enabled": True,
        "sensitivity_specs": sens_specs25,
        "sensitivity_cols": list(sens25.columns),
        "sens_mu": sens_mu.astype(np.float32), "sens_sd": sens_sd,
        "shift_model": deploy_model,
        "reference_model": reference_model,
        "reference_intercept": removed_constant,
        "shift_scale": float(SHIFT_SCALE), "shift_segment": SHIFT_SEGMENT,
        "postprocess": None,
        "library_versions": library_versions(),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"    artifact -> {MODEL_PATH}  ({size_mb:.1f} MB)")

    # 제출 zip 용 requirements.txt — artifact 를 unpickle 하려면 두 라이브러리가 필요하다
    # (notebook cell 16 은 joblib 도 넣지만 단일 pkl 은 표준 pickle 로 읽는다)
    lv = library_versions()
    req = "".join(
        f"{name}=={lv[key]}\n"
        for name, key in [("lightgbm", "lightgbm"), ("scikit-learn", "scikit_learn"),
                          ("numpy", "numpy"), ("pandas", "pandas")]
        if lv.get(key)
    )
    (MODEL_DIR / "requirements.txt").write_text(req, encoding="utf-8")
    print(f"    requirements -> {MODEL_DIR / 'requirements.txt'}\n{req.rstrip()}")

    # ---- 6) 산출물 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test[ID].to_numpy(), TARGET: final_direct}).to_csv(
        OUTPUT_PATH, index=False)
    np.savez_compressed(
        feature_dump_dir / "deploy_2025.npz",
        row_id=test[ID].to_numpy().astype(str),
        route_segment=test.route_segment.to_numpy().astype(str),
        p_base25=p_base25, p1=np.asarray(p1, np.float64), p2=np.asarray(p2, np.float64),
        p3=np.asarray(p3, np.float64), correction=correction,
        adjusted_direct=adjusted_direct, shift_corr25=shift_corr25,
        reference_corr25=reference_corr25, final_direct=final_direct,
    )
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "regime": "deploy (2024 full-fit -> 2025), NOT validation crossfit",
        "expected_year": int(expected_year),
        "n_test_rows": int(len(test)),
        "anchor_team_id": int(anchor_team_id),
        "layer2_mode": layer2_mode, "shift_adopted": shift_adopted,
        "reference_intercept_removed": removed_constant,
        "artifact_path": str(MODEL_PATH), "artifact_mb": round(size_mb, 2),
        "trend_slope": float(slope), "trend_prior": trend_prior,
        "deploy_feature_parity_vs_phase2": parity,
        "dims": {"exact_lgb": len(exact_feature_cols), "ridge": len(ridge_feature_cols),
                 "trend_lgb": len(trend_feature_cols),
                 "residual": len(residual_feature_cols),
                 "residual_base": len(residual_base_feature_cols),
                 "pitcher_context": len(pc25.columns),
                 "asof_residual": len(ex25.columns),
                 "sensitivity": len(sens25.columns)},
        "residual_cfg": RESIDUAL_FINAL_CFG, "residual_seeds": residual_seeds,
        "residual_shrink": DEPLOY_RESIDUAL_SHRINK,
        "validation_residual_shrink": RESIDUAL_SHRINK,
        "n_residual_models": len(residual_models),
        "residual_shrink_basis": "validation/deployval_v001 (deploy-regime transitions 2021..2024)",
        "pred": {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                     "max": float(np.max(v))}
                 for k, v in [("p_base25", p_base25), ("correction", correction),
                              ("adjusted_direct", adjusted_direct),
                              ("shift_corr25", shift_corr25),
                              ("reference_corr25", reference_corr25),
                              ("final_direct", final_direct)]},
        "library_versions": library_versions(),
    }
    write_json(validation_dir / "phase6_summary.json", summary)
    del X24_final, X25, raw24
    gc.collect()
    return summary


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
    asof = build_asof_state(screen, rows, year)
    pitcher_context, context_specs = build_pitcher_context(screen, source_year, rows)

    sens_specs = build_sensitivity_specs(screen, year)
    sens = transform_sensitivity_specs(sens_specs, rows)

    unused = build_unused_blocks_for_audit(screen, source_year, rows)

    # residual prep 은 직전 시즌(source_year) 에서 적합한다 (validation 규약).
    prep = fit_residual_prep(screen.loc[screen[SEASON_COL].eq(source_year)].reset_index(drop=True))

    # Phase 2 단독 실행 시 결정적 surrogate base_pred 로 residual matrix 를 만들어
    # 열 이름 / 열 순서 / dtype / 행 정렬을 reference 와 대조한다.
    residual_base_surrogate = transform_residual_features(prep, rows, surrogate_base_pred(n))
    residual_surrogate = build_residual_matrix(
        residual_base_surrogate, pitcher_context, asof, unused
    )

    out = {
        "year": int(year),
        "source_year": source_year,
        "n_rows": n,
        "row_id": rows[ID].to_numpy(),
        "target": rows[TARGET].to_numpy(np.float64),
        "route_segment": rows["route_segment"].to_numpy(),
        "self_state": self_state,
        "gseason": gseason,
        "asof_state": asof,
        "pitcher_context": pitcher_context,
        "sensitivity": sens,
        "residual_matrix_surrogate": residual_surrogate,
        "unused_blocks": unused,
        "residual_prep": prep,
        "context_specs": context_specs,
        "sens_specs": sens_specs,
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

    phase_names = {2: "Feature", 3: "+Layer 0 Anchor/Residual",
                   4: "+Layer 1 adjusted OOF crossfit", 5: "+Layer 2 sensitivity shift (PASS 2)",
                   6: "+Deploy artifact (2024->2025)"}
    print("=" * 78)
    print(f"Experiment                : {EXPERIMENT_NAME}")
    print(f"Protocol                  : {PROTOCOL_VERSION}")
    print(f"Phase                     : {PHASE}  "
          f"({phase_names.get(PHASE, '?')})")
    print(f"REPRO_BUILD_UNUSED_BLOCKS : {REPRO_BUILD_UNUSED_BLOCKS}")
    print(f"LOAD_ONLY_REFERENCED_COLS : {LOAD_ONLY_REFERENCED_COLUMNS}")
    print(f"Target years              : {TARGET_YEARS}")
    print(f"CACHE_DIR / LAYER2_ONLY   : {CACHE_DIR} / {LAYER2_ONLY}")
    print(f"RESUME                    : {RESUME}")
    print(f"N_JOBS                    : {N_JOBS}  (cpu {os.cpu_count()})")
    print(f"SMOKE_MODE                : {SMOKE_MODE}")
    print("-" * 78)
    print("실행 계획")
    for p in range(2, PHASE + 1):
        print(f"  Phase {p}  {phase_names[p]}")
    if PHASE >= 6:
        print(f"  -> {MODEL_PATH}   (단일 pkl artifact)")
        print(f"  -> {OUTPUT_PATH}  (submission)")
    if PHASE >= 4 and not SMOKE_MODE:
        n_fit = len(TARGET_YEARS) * len(CF_SEEDS) * 2 * 3
        print(f"  ExtraTrees 적합 {n_fit}회 (Layer 1 crossfit)"
              + (" + 9회 (deploy)" if PHASE >= 6 else ""))
        print("  예상 소요: 6코어 기준 약 15~25분. SHIFT_SMOKE=1 로 먼저 배선 확인 가능.")
    print("환경변수: SHIFT_PHASE / SHIFT_YEARS / SHIFT_RESUME /"
          " SHIFT_UNUSED / SHIFT_DUMP_DIR / SHIFT_SMOKE")
    print("=" * 78)
    if PHASE >= 5 and _YEARS_ENV and set(TARGET_YEARS) != set(PHASE5_SOURCE_YEARS):
        print("[note] Phase 5/6 은 2020~2024 전체 ADJUSTED_OOF 를 요구한다. "
              "디스크에 있으면 그대로 쓰고, 없으면 명확히 실패한다.")
        print("=" * 78)

    screen = load_train()
    validate_schema(screen)
    anchor_team_id = fit_router(screen)
    screen = attach_route_segment(screen, anchor_team_id)
    finalize_column_lists(screen)
    segment_means = SegmentMeans(screen)

    print(f"ANCHOR_TEAM_ID            : {anchor_team_id}")
    print(f"TRAIN_ROWS                : {len(screen)}")
    print(f"SEASONS                   : {sorted(screen[SEASON_COL].unique().tolist())}")
    print(f"RAWNUM / CAT_SMALL        : {len(RAWNUM)} / {len(CAT_SMALL)}")
    print(f"route_segment share             : "
          f"{screen.route_segment.value_counts(normalize=True).round(4).to_dict()}")
    print()

    phase2_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "anchor_team_id": anchor_team_id,
        "train_rows": int(len(screen)),
        "seasons": sorted(int(s) for s in screen[SEASON_COL].unique()),
        "rawnum": RAWNUM,
        "cat_small": CAT_SMALL,
        "residual_rawnum": RESIDUAL_RAWNUM,
        "eng_cols": ENG_COLS,
        "bank_cols": BANK_COLS,
        "pc_contexts": [[c, s] for c, s in CONTEXT_KEYS],
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
    phase4_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "layer": "layer1_residual_adjuster",
        "regime": "within_year_pitcher_half_crossfit (NOT strict next-year)",
        "library_versions": library_versions(),
        "smoke_mode": SMOKE_MODE,
        "years": {},
    }

    for year in ([] if LAYER2_ONLY else TARGET_YEARS):
        print("-" * 78)
        print(f"[Phase 2] year {year}")
        st = run_phase2_year(screen, year, anchor_team_id)
        rows = st["rows"]

        blocks = {
            "self_state": st["self_state"],
            "gseason": st["gseason"],
            "asof_state": st["asof_state"],
            "pitcher_context": st["pitcher_context"],
            "sensitivity": st["sensitivity"],
            "residual_matrix_surrogate": st["residual_matrix_surrogate"],
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
        _slope = segment_means.trailing_core_slope(st["source_year"])
        ysum["trend_slope"] = _slope
        ysum["trend_prior"] = {
            d: segment_means.value(int(st["source_year"]), d, st["source_year"]) + _slope
            for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]
        }
        ysum["segment_means_source_year"] = {
            d: segment_means.value(int(st["source_year"]), d, st["source_year"])
            for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]
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
                route_segment=st["route_segment"].astype(str),
                **{f"pitcher_half_{s}": v for s, v in st["pitcher_half"].items()},
            )

        # ---------------- Phase 3 ----------------
        if PHASE >= 3:
            print(f"[Phase 3] year {year}  anchor  ({st['source_year']} -> {year})")
            p_base, parts, meta = build_anchor(
                screen, segment_means, st["source_year"], rows, year
            )
            y = st["target"]
            residual_l1 = build_training_target(y, p_base)

            residual_base = transform_residual_features(st["residual_prep"], rows, p_base)
            X_residual = build_residual_matrix(
                residual_base,
                st["pitcher_context"],
                st["asof_state"],
                st["unused_blocks"],
            )
            assert X_residual.shape[1] == 128
            EXPECTED_RESIDUAL_FEATURE_COLS = (
                list(residual_base.columns)
                + list(st["pitcher_context"].columns)
                + list(st["asof_state"].columns)
            )
            assert list(X_residual.columns) == EXPECTED_RESIDUAL_FEATURE_COLS

            base_df = pd.DataFrame(
                {
                    ID: st["row_id"],
                    "season": int(year),
                    "route_segment": st["route_segment"],
                    "target": y,
                    "p_exact_lgb": np.asarray(parts["exact_lgb"], np.float64),
                    "p_exact_ridge": np.asarray(parts["exact_ridge"], np.float64),
                    "p_trend_lgb": np.asarray(parts["trend_center_lgb"], np.float64),
                    "p_base": np.asarray(p_base, np.float64),
                    "residual_l1": residual_l1,
                }
            )
            # base_{year}.csv 는 사람이 보는 산출물이다.
            # pandas 기본 to_csv 는 float64 round-trip 을 보장하지 않아
            # 읽어들일 때 최대 1 ulp(약 5.55e-17) 오차가 생긴다. gate 가 그 오차를
            # 재현 오차로 오인하지 않도록 %.17g 로 저장한다. 계산에는 영향이 없다.
            base_df.to_csv(
                VALIDATION_DIR / f"base_{year}.csv", index=False, float_format="%.17g"
            )
            # gate 용 무손실 산출물 (float64 원본 비트 그대로)
            np.savez_compressed(
                FEATURE_DUMP_DIR / f"anchor_{year}.npz",
                p_base=np.asarray(p_base, np.float64),
                p_exact_lgb=np.asarray(parts["exact_lgb"], np.float64),
                p_exact_ridge=np.asarray(parts["exact_ridge"], np.float64),
                p_trend_lgb=np.asarray(parts["trend_center_lgb"], np.float64),
                residual_l1=np.asarray(residual_l1, np.float64),
            )
            save_matrix(FEATURE_DUMP_DIR / f"residual_matrix_{year}.npz", X_residual)

            bss = brier_skill(y, p_base)
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
                "residual_matrix_shape": list(X_residual.shape),
                "residual_columns": list(X_residual.columns),
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
            ysum["residual_matrix"] = summarize_block("residual_matrix", X_residual)

            # ---------------- Phase 4 : Layer 1 adjusted OOF ----------------
            if PHASE >= 4:
                oof_npz = FEATURE_DUMP_DIR / f"adjusted_oof_{year}.npz"
                z, cached_path = (load_cached_npz(FEATURE_DUMP_DIR, f"adjusted_oof_{year}.npz",
                                                  [f"v15_oof_{year}.npz"]) if RESUME else (None, None))
                if RESUME and z is not None:
                    # 적합만 건너뛰고 보고는 한다. 통짜 덮어쓰기로 요약이 사라지는 것을 막는다.
                    oof_npz = cached_path
                    consensus = npz_key(z, "adjusted_oof", ["v15_oof"]).astype(np.float64)
                    assert float(np.abs(npz_key(z, "p_base").astype(np.float64) - p_base).max()) == 0.0, (
                        f"캐시 {cached_path} 의 p_base 가 현재 anchor 와 다르다 — Layer 0 불변 조건 위반"
                    )
                    oof_bss = brier_skill(y, consensus)
                    seed_rows, fold_rows = [], []
                    for name, box in [("residual_seed_scores", seed_rows),
                                      ("residual_fold_scores", fold_rows)]:
                        p_ = VALIDATION_DIR / f"{name}_{year}.csv"
                        if p_.exists():
                            box.extend(pd.read_csv(p_).to_dict("records"))
                    phase4_summary["years"][str(year)] = {
                        "source_year": st["source_year"],
                        "n_rows": st["n_rows"],
                        "residual_matrix_shape": list(X_residual.shape),
                        "cf_seeds": list(CF_SEEDS),
                        "residual_cf_cfg": RESIDUAL_CF_CFG,
                        "residual_shrink": RESIDUAL_SHRINK,
                        "base_bss": bss,
                        "adjusted_oof_bss": oof_bss,
                        "gain_vs_base": oof_bss - bss,
                        "official_score_base": official_score(y, p_base),
                        "official_score_adjusted_oof": official_score(y, consensus),
                        "seed_rows": seed_rows,
                        "fold_rows": fold_rows,
                        "adjusted_oof": {"mean": float(consensus.mean()),
                                    "min": float(consensus.min()),
                                    "max": float(consensus.max())},
                        "segment_bss": {
                            d: brier_skill(y[st["route_segment"] == d], consensus[st["route_segment"] == d])
                            for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]
                        },
                        "resumed_from_npz": True,
                    }
                    print(f"[Phase 4] year {year}  RESUME — {oof_npz.name} 재사용, "
                          f"adjusted_oof_bss={oof_bss:.2f} gain_vs_base={oof_bss - bss:+.2f}")
                    del z, consensus
                    gc.collect()
                else:
                    print(f"[Phase 4] year {year}  residual adjuster crossfit "
                          f"(seeds={list(CF_SEEDS)}, cfg n_estimators="
                          f"{RESIDUAL_CF_CFG['n_estimators']})")
                    t4 = time.time()
                    consensus, preds, corrs, seed_rows, fold_rows = run_adjusted_oof(
                        year, X_residual, y, p_base, rows
                    )
                    np.savez_compressed(
                        oof_npz,
                        adjusted_oof=np.asarray(consensus, np.float64),
                        p_base=np.asarray(p_base, np.float64),
                        target=np.asarray(y, np.float64),
                        **{f"p_seed{int(s)}": np.asarray(p, np.float64)
                           for s, p in zip(CF_SEEDS, preds)},
                        **{f"corr_seed{int(s)}": np.asarray(c, np.float64)
                           for s, c in corrs.items()},
                    )
                    pd.DataFrame(
                        {
                            ID: st["row_id"],
                            "season": int(year),
                            "route_segment": st["route_segment"],
                            "target": y,
                            "p_base": np.asarray(p_base, np.float64),
                            "adjusted_oof": np.asarray(consensus, np.float64),
                            "residual_l2_source": y - np.asarray(consensus, np.float64),
                        }
                    ).to_csv(
                        VALIDATION_DIR / f"adjusted_oof_{year}.csv",
                        index=False, float_format="%.17g",
                    )
                    pd.DataFrame(seed_rows).to_csv(
                        VALIDATION_DIR / f"residual_seed_scores_{year}.csv", index=False
                    )
                    pd.DataFrame(fold_rows).to_csv(
                        VALIDATION_DIR / f"residual_fold_scores_{year}.csv", index=False
                    )
                    oof_bss = brier_skill(y, consensus)
                    print(f"    adjusted_oof_bss={oof_bss:.2f}  "
                          f"gain_vs_base={oof_bss - bss:+.2f}  "
                          f"sec={time.time() - t4:.1f}")
                    phase4_summary["years"][str(year)] = {
                        "source_year": st["source_year"],
                        "n_rows": st["n_rows"],
                        "residual_matrix_shape": list(X_residual.shape),
                        "cf_seeds": list(CF_SEEDS),
                        "residual_cf_cfg": RESIDUAL_CF_CFG,
                        "residual_shrink": RESIDUAL_SHRINK,
                        "base_bss": bss,
                        "adjusted_oof_bss": oof_bss,
                        "gain_vs_base": oof_bss - bss,
                        "official_score_base": official_score(y, p_base),
                        "official_score_adjusted_oof": official_score(y, consensus),
                        "seed_rows": seed_rows,
                        "fold_rows": fold_rows,
                        "adjusted_oof": {"mean": float(consensus.mean()),
                                    "min": float(consensus.min()),
                                    "max": float(consensus.max())},
                        "segment_bss": {
                            d: brier_skill(y[st["route_segment"] == d], consensus[st["route_segment"] == d])
                            for d in ["CORE", "ANCHOR_TEAM", "FUTURES"]
                        },
                        "sec": round(time.time() - t4, 1),
                    }
                    del consensus, preds, corrs
                    gc.collect()

            del p_base, parts, residual_base, X_residual, base_df
            gc.collect()

        phase2_summary["years"][str(year)] = ysum
        del st, rows, blocks
        gc.collect()

    # ---------------- Phase 5 : PASS 2 (Layer 2) ----------------
    phase5_summary = None
    if PHASE >= 5:
        missing = [y for y in PHASE5_SOURCE_YEARS if set(TARGET_YEARS) and y not in TARGET_YEARS]
        if missing and _YEARS_ENV:
            print()
            print("!" * 78)
            print("Phase 5 는 2-pass 의 PASS 2 이며 2020~2024 전체 ADJUSTED_OOF 가 필요하다.")
            print(f"현재 SHIFT_YEARS={TARGET_YEARS} 로 제한되어 있으나, "
                  "Phase 4 산출물이 디스크(또는 SHIFT_CACHE_DIR)에 있으면 그대로 사용한다.")
            print("!" * 78)
        print()
        print("=" * 78)
        print("[Phase 5] Layer 2 — sensitivity shift (PASS 2)")
        print("=" * 78)
        t5 = time.time()
        phase5_summary = run_phase5(screen, FEATURE_DUMP_DIR, VALIDATION_DIR)
        print(f"[Phase 5] done  {time.time() - t5:.1f}s")

    # ---------------- Phase 6 : deploy artifact ----------------
    phase6_summary = None
    if PHASE >= 6:
        print()
        print("=" * 78)
        print("[Phase 6] Final artifact — deploy regime (2024 -> 2025)")
        print("=" * 78)
        t6 = time.time()
        phase6_summary = run_phase6(screen, segment_means, FEATURE_DUMP_DIR, VALIDATION_DIR)
        print(f"[Phase 6] done  {time.time() - t6:.1f}s")

    # SEGMENT_MEANS 미래 접근 감사 (§14)
    phase3_summary["segment_means_audit"] = {
        "n_access": len(segment_means.access_log),
        "future_access_count": segment_means.future_access_count(),
        "max_accessed_season": segment_means.max_accessed_season(),
        "policy": "전 시즌 groupby 유지. 읽기는 season <= source_year 만 assert.",
    }
    assert segment_means.future_access_count() == 0

    phase2_summary["elapsed_sec"] = round(time.time() - global_start, 1)
    phase3_summary["elapsed_sec"] = round(time.time() - global_start, 1)

    merge_years_json(VALIDATION_DIR / "phase2_summary.json", phase2_summary)
    if PHASE >= 3:
        merge_years_json(VALIDATION_DIR / "phase3_summary.json", phase3_summary)
    if PHASE >= 4 and phase4_summary["years"]:
        phase4_summary["elapsed_sec"] = round(time.time() - global_start, 1)
        merge_years_json(VALIDATION_DIR / "phase4_summary.json", phase4_summary)

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "protocol": PROTOCOL_VERSION,
        "target": TARGET,
        "phase_implemented": PHASE,
        "architecture": {
            "layers": 3,
            "layer0": "recent 1-season anchor: LGBMClassifier + Ridge(a=1e4) + trend-centered LGBMRegressor, equal 1/3",
            "layer1": (
                f"three-segment ExtraTrees residual adjuster, validation shrink {RESIDUAL_SHRINK}, "
                f"deploy shrink {DEPLOY_RESIDUAL_SHRINK} (0 = not deployed), pitcher-half crossfit x3 seeds"
                + ("" if PHASE >= 4 else "  [NOT IMPLEMENTED AT THIS PHASE]")
            ),
            "layer2": (
                "sensitivity shift: Ridge(a=1e3, fit_intercept=False) on prev-year OOF residual, "
                "scale 0.75, CORE only; reference(fit_intercept=True) computed alongside for comparison"
                + ("" if PHASE >= 5 else "  [NOT IMPLEMENTED AT THIS PHASE]")
            ),
            "router": "route_segment = FUTURES / ANCHOR_TEAM / CORE, anchor team from FUTURES & season<=2022",
        },
        "features": {
            "raw": {"rawnum": len(RAWNUM), "cat_small": len(CAT_SMALL), "eng": len(ENG_COLS)},
            "temporal": {"segment_means": True, "trailing_core_slope": True,
                         "sensitivity_decay": SENS_DECAY},
            "historical": {"self_state": EXPECTED_DIMS["self_state"],
                           "gseason": EXPECTED_DIMS["gseason"],
                           "asof_state": EXPECTED_DIMS["asof_state"]},
            "conditional": {"pitcher_context": EXPECTED_DIMS["pitcher_context"],
                            "sensitivity": EXPECTED_DIMS["sensitivity"]},
            "anchor": {"exact_lgb": EXPECTED_DIMS["exact_lgb_features"],
                       "exact_ridge": EXPECTED_DIMS["ridge_features"],
                       "trend_lgb": EXPECTED_DIMS["trend_features"],
                       "residual": EXPECTED_DIMS["residual_matrix"]},
        },
        "model_params": {
            "base_cfg": BASE_CFG,
            "residual_cf_cfg": RESIDUAL_CF_CFG,
            "residual_final_cfg": RESIDUAL_FINAL_CFG,
            "residual_shrink_validation": RESIDUAL_SHRINK,
            "residual_shrink_deploy": DEPLOY_RESIDUAL_SHRINK,
            "cf_seeds": list(CF_SEEDS),
            "sensitivity": {
                "history_years": SENS_HISTORY_YEARS, "decay": SENS_DECAY,
                "parent_strength": SENS_PARENT_STRENGTH, "hand_strength": SENS_HAND_STRENGTH,
                "pressure_hand_strength": SENS_PRESSURE_HAND_STRENGTH,
            },
            "shift": {
                "ridge_alpha": SHIFT_RIDGE_ALPHA, "fit_intercept": False, "scale": SHIFT_SCALE,
                "scale_stress": list(SHIFT_SCALE_STRESS), "segment": SHIFT_SEGMENT,
            },
            "gate": {"min_gain_2023": MIN_GAIN_2023, "min_gain_2024": MIN_GAIN_2024,
                     "min_core_gain_2024": MIN_CORE_GAIN_2024, "min_boot_p05": MIN_BOOT_P05,
                     "min_boot_prob": MIN_BOOT_PROB, "min_late_gain": MIN_LATE_GAIN,
                     "min_relative_boot_p05": MIN_RELATIVE_BOOT_P05,
                     "min_transitions_not_worse": MIN_TRANSITIONS_NOT_WORSE},
        },
        "validation": {
            "scheme": "adjacent-season transition (2020->21, 21->22, 22->23, 23->24)",
            "regime": "within_year_pitcher_crossfit for layer1; strict next-year for layer2",
            "metric": "brier_skill = 100000*(1 - MSE/(ybar*(1-ybar)))",
            "official_metric_also_reported": True,
            "warning": "validation BSS is not an LB estimate (deploy regime differs from validation).",
        },
        "layer2_adopted": (phase5_summary["gate"]["adopt"] if phase5_summary else None),
        "omitted_dead_code": OMITTED_DEAD_CODE,
        "unused_but_computed_blocks": sorted(UNUSED_BLOCK_NAMES),
        "library_versions": library_versions(),
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
    if PHASE >= 4 and phase4_summary["years"]:
        print(f"  -> {VALIDATION_DIR}/phase4_summary.json")
        print(f"  -> {VALIDATION_DIR}/adjusted_oof_{{year}}.csv")
        print(f"  -> {VALIDATION_DIR}/residual_seed_scores_{{year}}.csv")
        print(f"  -> {VALIDATION_DIR}/residual_fold_scores_{{year}}.csv")
        print(f"  -> {FEATURE_DUMP_DIR}/adjusted_oof_{{year}}.npz")
    if PHASE >= 5 and phase5_summary is not None:
        print(f"  -> {VALIDATION_DIR}/phase5_summary.json")
        print(f"  -> {VALIDATION_DIR}/gate.json  (ADOPT={phase5_summary['gate']['adopt']})")
        print(f"  -> {VALIDATION_DIR}/fold_{{2021..2024}}.csv · oof.csv")
        print(f"  -> {VALIDATION_DIR}/transitions.csv · "
              f"scale_stress.csv · months.csv · segments.csv")
        print(f"  -> {FEATURE_DUMP_DIR}/final_{{2021..2024}}.npz")
    if PHASE >= 6 and phase6_summary is not None:
        print(f"  -> {VALIDATION_DIR}/phase6_summary.json")
        print(f"  -> {MODEL_PATH}  ({phase6_summary['artifact_mb']} MB, "
              f"layer2_mode={phase6_summary['layer2_mode']})")
        print(f"  -> {OUTPUT_PATH}")
        print(f"  -> {FEATURE_DUMP_DIR}/deploy_2025.npz")
    print(f"  -> {MODEL_DIR}/metadata.json")
    print("=" * 78)


if __name__ == "__main__":
    main()