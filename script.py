# script.py: cb_reg_v011 추론 및 제출
#
# train.py(cb_reg_v011)가 저장한 단일 Artifact
#   model/cb_reg_v011/cb_reg_v011.pkl
# 를 로드하여 test.csv 각 행을 독립적으로 추론하고 output/submission.csv를 생성한다.
#
# 추론 파이프라인
# ----------------
# 1. Artifact의 HB-EB prior를 각 행에 lookup (train.py::apply_hbeb 와 동일 수식)
# 2. Artifact의 frozen 2025 platoon bank를 각 행에 lookup
#    rate / delta / reliability / n 네 컬럼을 모두 생성하고,
#    실제 모델 입력은 artifact["features"]가 선택한다.
# 3. CatBoostRegressor.predict()
# 4. np.clip(pred, 0.0, 1.0)
# 5. sample_submission.csv 의 row_id 순서에 맞춰 key 기반으로 정렬
#
# 평가 데이터 독립 추론 규칙
# -------------------------
# - test.csv의 다른 행이나 전체 분포를 Feature/Prediction 생성에 사용하지 않는다.
# - test 기반 groupby / mean / rank / rolling / lag / transform / target 통계 재추정 없음.
# - test prediction 평균을 이용한 calibration 없음. F correction 없음.
# - HB-EB prior와 platoon bank는 공식 train.csv(2019~2024)만으로 학습 시점에 확정된 값이다.
# - 한 행만 입력했을 때와 전체 test.csv에 포함했을 때 해당 행의 예측값이 동일하다.
#
# 설계 방어선
# -----------
# - join key를 양쪽 동일 규칙으로 정규화하여 dtype 추론 차이에 의한 merge 전멸을 차단한다.
# - lookup 붕괴(unseen_cell_rate ~ 1.0, nan_delta_rate ~ 1.0)를 하드 가드로 검출한다.
# - submission은 row_id merge(validate="one_to_one")로만 만든다. 위치 기반 대입 금지.
# - 모델 입력 feature 목록은 artifact가 단일 진실이며 스크립트에 하드코딩하지 않는다.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v011"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"

MISSING_TOKEN = "__MISSING__"

PREDICTION_CLIP_MIN = 0.0
PREDICTION_CLIP_MAX = 1.0

# platoon bank가 생성하는 전체 컬럼.
# 이 중 무엇을 모델에 넣을지는 artifact["features"]가 정한다.
CF_ALL_COLUMNS = [
    "cf_platoon_rate",
    "cf_platoon_delta",
    "cf_platoon_reliability",
    "cf_platoon_n",
]

# lookup 붕괴 검출 임계값. 형식 확인용 소규모 test.csv에서는 적용하지 않는다.
GUARD_MIN_ROWS = 1000
GUARD_UNSEEN_FATAL = 0.999
GUARD_UNSEEN_WARN = 0.30

# 행 독립성 감사 규모
AUDIT_FRAME_ROWS = 3000
AUDIT_SAMPLES = 25
AUDIT_TOLERANCE = 1e-12


# =========================================================
# 공통 유틸
# =========================================================

def ensure_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: 필수 컬럼 누락: {missing}")


def sanitize_categoricals(df, cat_cols):
    """train.py::prepare_features 와 동일한 범주형 전처리."""
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            raise KeyError(f"범주형 Feature 누락: {col}")
        out[col] = (
            out[col].astype("string").fillna(MISSING_TOKEN).astype(str)
        )
    return out


def normalize_join_key(values):
    """
    join key를 dtype 추론과 무관한 정규 문자열로 변환한다.

    pandas는 같은 컬럼을 train.csv에서 int64로, test.csv에서 object 또는
    float64로 추론할 수 있다. 그 상태로 merge하면 한 건도 매칭되지 않으면서
    예외가 발생하지 않는다. 양쪽에 동일 규칙을 적용해 이를 차단한다.

        3      -> "3"
        3.0    -> "3"
        "3"    -> "3"
        "003"  -> "3"
        "L"    -> "L"
        NaN    -> "__MISSING__"
    """
    s = pd.Series(values).reset_index(drop=True).astype("string")

    numeric = pd.to_numeric(s, errors="coerce")
    is_integral = numeric.notna() & (numeric == numeric.round())

    out = s.copy()
    if bool(is_integral.any()):
        out[is_integral] = (
            numeric[is_integral].astype("Int64").astype("string")
        )

    return out.fillna(MISSING_TOKEN).astype(str)


def attach_normalized_keys(df, key_cols, prefix="__nk_"):
    """원본 컬럼을 보존한 채 정규화된 join key 컬럼을 덧붙인다."""
    out = df.copy()
    names = []
    for col in key_cols:
        name = prefix + col
        out[name] = normalize_join_key(out[col]).to_numpy()
        names.append(name)
    return out, names


# =========================================================
# HB-EB — train.py::apply_hbeb 와 수식 동일
# =========================================================

def apply_hbeb(df, group_cols, n_col, rate_col, prior_df, global_prior):
    ensure_columns(df, list(group_cols) + [n_col, rate_col], "apply_hbeb")

    left, left_keys = attach_normalized_keys(
        df[list(group_cols)].reset_index(drop=True),
        list(group_cols),
        prefix="__hk_",
    )
    right, right_keys = attach_normalized_keys(
        prior_df.copy(),
        list(group_cols),
        prefix="__hk_",
    )
    right = right.drop(columns=list(group_cols))

    lookup = left.merge(right, on=left_keys, how="left", sort=False)

    alpha = (
        pd.to_numeric(lookup["alpha"], errors="coerce")
        .fillna(float(global_prior["alpha"]))
        .to_numpy(np.float64)
    )
    beta = (
        pd.to_numeric(lookup["beta"], errors="coerce")
        .fillna(float(global_prior["beta"]))
        .to_numpy(np.float64)
    )

    n = (
        pd.to_numeric(df[n_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    rate = (
        pd.to_numeric(df[rate_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )

    n = np.clip(n, 0.0, None)
    rate = np.clip(rate, 0.0, 1.0)

    success_count = n * rate
    denominator = n + alpha + beta

    if np.any(denominator <= 0.0):
        raise ValueError(f"HB-EB denominator가 0 이하입니다: {rate_col}")

    eb = (success_count + alpha) / denominator

    return pd.Series(eb, index=df.index, dtype=np.float64)


def apply_hbeb_from_artifact(df, base_features, hbeb_configs, hbeb_priors):
    """
    cb_reg_v003 호환: raw success_rate를 제거하고 hbeb_* 로 대체한다.
    prior 재추정은 하지 않는다.
    """
    out = df.copy()
    features = list(base_features)

    for cfg in hbeb_configs:
        new_col = cfg["new_col"]

        if new_col not in hbeb_priors:
            raise KeyError(f"Artifact에 HB-EB prior가 없습니다: {new_col}")

        saved = hbeb_priors[new_col]

        out[new_col] = apply_hbeb(
            out,
            cfg["group_cols"],
            cfg["n_col"],
            cfg["rate_col"],
            saved["prior_df"],
            saved["global_prior"],
        )

        if cfg["rate_col"] in features:
            features.remove(cfg["rate_col"])
        if new_col not in features:
            features.append(new_col)

    return out, features


# =========================================================
# Frozen platoon bank — train.py::apply_platoon_bank 와 수식 동일
# =========================================================

def apply_frozen_platoon_bank(df, bank, key_cols, parent_cols):
    """
    학습 시점에 공식 train 2019~2024 label로 확정된 frozen bank를
    각 행에 독립적으로 lookup한다.

        posterior_rate = (cell_success + K * parent_rate) / (cell_n + K)
        reliability    = cell_n / (cell_n + K)
        delta          = posterior_rate - parent_rate      (cell_n = 0 이면 NaN)

    unseen cell    : cell_n = 0        -> delta = NaN, reliability = 0
    unseen pitcher : parent_rate = global_rate

    네 컬럼을 모두 생성한다. 모델 입력 선택은 artifact["features"]가 한다.
    """
    key_cols = list(key_cols)
    parent_cols = list(parent_cols)

    ensure_columns(
        df,
        list(dict.fromkeys(key_cols + parent_cols)),
        "apply_frozen_platoon_bank",
    )

    for required in ("cell_table", "parent_table", "global_rate", "strength"):
        if required not in bank:
            raise KeyError(f"platoon bank 필수 key 누락: {required}")

    out = df.copy()
    n_rows = len(out)

    cell_table = bank["cell_table"]
    parent_table = bank["parent_table"]
    global_rate = float(bank["global_rate"])
    strength = float(bank["strength"])

    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError(f"platoon bank strength가 유효하지 않습니다: {strength}")

    # ---- parent lookup -----------------------------------------------
    if len(parent_table) > 0:
        p_left, p_keys = attach_normalized_keys(
            out[parent_cols].reset_index(drop=True), parent_cols, prefix="__pk_"
        )
        p_right, _ = attach_normalized_keys(
            parent_table.copy(), parent_cols, prefix="__pk_"
        )
        p_right = p_right.drop(columns=parent_cols)

        if p_right.duplicated(subset=p_keys).any():
            raise RuntimeError(
                "parent_table 정규화 키가 중복됩니다. 키 정규화가 서로 다른 "
                "선수를 병합했을 수 있습니다."
            )

        parent_lookup = p_left.merge(p_right, on=p_keys, how="left", sort=False)
        parent_rate = (
            pd.to_numeric(parent_lookup["parent_rate"], errors="coerce")
            .fillna(global_rate)
            .to_numpy(np.float64)
        )
    else:
        parent_rate = np.full(n_rows, global_rate, dtype=np.float64)

    # ---- cell lookup -------------------------------------------------
    if len(cell_table) > 0:
        c_left, c_keys = attach_normalized_keys(
            out[key_cols].reset_index(drop=True), key_cols, prefix="__ck_"
        )
        c_right, _ = attach_normalized_keys(
            cell_table.copy(), key_cols, prefix="__ck_"
        )
        c_right = c_right.drop(columns=key_cols)

        if c_right.duplicated(subset=c_keys).any():
            raise RuntimeError(
                "cell_table 정규화 키가 중복됩니다. 키 정규화가 서로 다른 "
                "셀을 병합했을 수 있습니다."
            )

        cell_lookup = c_left.merge(c_right, on=c_keys, how="left", sort=False)
        cell_n = (
            pd.to_numeric(cell_lookup["cell_n"], errors="coerce")
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        cell_success = (
            pd.to_numeric(cell_lookup["cell_success"], errors="coerce")
            .fillna(0.0)
            .to_numpy(np.float64)
        )
    else:
        cell_n = np.zeros(n_rows, dtype=np.float64)
        cell_success = np.zeros(n_rows, dtype=np.float64)

    cell_n = np.clip(cell_n, 0.0, None)
    cell_success = np.clip(cell_success, 0.0, None)

    posterior_rate = (cell_success + strength * parent_rate) / (cell_n + strength)
    reliability = cell_n / (cell_n + strength)

    # 미관측 셀은 0 이 아니라 결측으로 표현한다.
    # 0 으로 채우면 "편차가 없다"와 "정보가 없다"가 같은 값이 되어
    # 트리가 두 상태를 구분할 수 없다. train.py 와 반드시 동일해야 한다.
    delta = np.where(cell_n > 0.0, posterior_rate - parent_rate, np.nan)

    out["cf_platoon_rate"] = posterior_rate.astype(np.float64)
    out["cf_platoon_delta"] = delta.astype(np.float64)
    out["cf_platoon_reliability"] = reliability.astype(np.float64)
    out["cf_platoon_n"] = cell_n.astype(np.float64)

    # NaN 전파로 진단값이 무력화되지 않도록 유한값만 집계한다.
    finite = np.isfinite(delta)
    has_finite = bool(finite.any())

    diagnostics = {
        "n_rows": int(n_rows),
        "strength": strength,
        "global_rate": global_rate,
        "unseen_cell_rows": int(np.sum(cell_n <= 0.0)),
        "unseen_cell_rate": float(np.mean(cell_n <= 0.0)) if n_rows else 0.0,
        "unseen_parent_rows": int(np.sum(np.isclose(parent_rate, global_rate))),
        "nan_delta_rate": float(np.mean(~finite)) if n_rows else 0.0,
        "mean_cell_n": float(np.mean(cell_n)) if n_rows else 0.0,
        "median_cell_n": float(np.median(cell_n)) if n_rows else 0.0,
        "mean_reliability": float(np.mean(reliability)) if n_rows else 0.0,
        "mean_abs_delta": (
            float(np.mean(np.abs(delta[finite]))) if has_finite else 0.0
        ),
        "max_abs_delta": (
            float(np.max(np.abs(delta[finite]))) if has_finite else 0.0
        ),
    }

    return out, diagnostics


def guard_platoon_lookup(diagnostics):
    """
    dtype 불일치 등으로 bank lookup이 전멸했는지 검출한다.

    형식 확인용 소규모 test.csv에서는 unseen 비율이 정상적으로 높을 수 있으므로
    실제 평가 규모에서만 치명 판정을 내린다.
    """
    n_rows = int(diagnostics["n_rows"])
    unseen = float(diagnostics["unseen_cell_rate"])
    nan_rate = float(diagnostics["nan_delta_rate"])
    max_abs_delta = float(diagnostics["max_abs_delta"])

    if n_rows < GUARD_MIN_ROWS:
        return

    # delta 가 결측으로 표현되므로 lookup 전멸은 nan_delta_rate 로도 나타난다.
    if unseen >= GUARD_UNSEEN_FATAL and nan_rate >= GUARD_UNSEEN_FATAL:
        raise RuntimeError(
            "platoon bank lookup이 전멸했습니다. join key dtype 불일치가 "
            "가장 유력합니다. "
            f"unseen_cell_rate={unseen:.6f}, nan_delta_rate={nan_rate:.6f}, "
            f"max_abs_delta={max_abs_delta:.3e}, n_rows={n_rows:,}"
        )

    if unseen >= GUARD_UNSEEN_WARN:
        print(
            "[WARNING] unseen_cell_rate가 비정상적으로 높습니다: "
            f"{unseen:.4%} (n={n_rows:,}). bank 커버리지를 확인하십시오."
        )


# =========================================================
# 행 단위 변환 파이프라인
# =========================================================

def transform_rows(df, artifact):
    """
    입력 행들을 모델 입력 행렬 직전까지 변환한다.
    각 행의 변환은 다른 행에 의존하지 않는다.
    """
    transformed, _ = apply_hbeb_from_artifact(
        df,
        artifact["base_features"],
        artifact["hbeb_configs"],
        artifact["hbeb_priors"],
    )

    cf_meta = artifact["conditional_features"]

    transformed, cf_diag = apply_frozen_platoon_bank(
        transformed,
        cf_meta["final_2025_bank"],
        cf_meta["key_cols"],
        cf_meta["parent_cols"],
    )

    ensure_columns(transformed, artifact["features"], "변환된 test")

    transformed = sanitize_categoricals(transformed, artifact["cat_cols"])

    return transformed, cf_diag


def predict_rows(df, artifact):
    if len(df) == 0:
        return np.empty(0, dtype=np.float64), None

    transformed, cf_diag = transform_rows(df, artifact)

    pred_raw = artifact["model"].predict(transformed[artifact["features"]])

    pred = np.clip(
        np.asarray(pred_raw, dtype=np.float64),
        PREDICTION_CLIP_MIN,
        PREDICTION_CLIP_MAX,
    )

    if not np.all(np.isfinite(pred)):
        raise ValueError("최종 prediction에 NaN/Inf가 존재합니다.")

    return pred, cf_diag


# =========================================================
# 행 독립성 감사
# =========================================================

def verify_row_independence(frame, artifact, n_samples=AUDIT_SAMPLES, seed=42):
    """
    같은 행을 단독으로 넣었을 때와 배치에 포함해 넣었을 때
    예측값이 동일한지 확인한다.
    """
    if len(frame) == 0:
        return {"passed": True, "n_samples": 0, "max_abs_diff": 0.0, "failures": []}

    batch_pred, _ = predict_rows(frame, artifact)

    rng = np.random.default_rng(seed)
    k = min(int(n_samples), len(frame))
    positions = rng.choice(len(frame), size=k, replace=False)

    failures = []
    max_abs_diff = 0.0

    for pos in positions:
        one = frame.iloc[[int(pos)]].copy()
        one_pred, _ = predict_rows(one, artifact)

        diff = abs(float(batch_pred[int(pos)]) - float(one_pred[0]))
        max_abs_diff = max(max_abs_diff, diff)

        if diff > AUDIT_TOLERANCE:
            failures.append(
                {
                    "position": int(pos),
                    "batch": float(batch_pred[int(pos)]),
                    "single": float(one_pred[0]),
                    "abs_diff": diff,
                }
            )

    return {
        "passed": len(failures) == 0,
        "n_samples": int(k),
        "max_abs_diff": float(max_abs_diff),
        "failures": failures,
    }


# =========================================================
# Artifact 로드 및 검증
# =========================================================

def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Artifact를 찾을 수 없습니다: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    required = [
        "model",
        "features",
        "cat_cols",
        "base_features",
        "hbeb_priors",
        "hbeb_configs",
        "conditional_features",
        "experiment_name",
        "model_type",
    ]
    missing = [k for k in required if k not in artifact]
    if missing:
        raise KeyError(f"Artifact 필수 key 누락: {missing}")

    if artifact["experiment_name"] != EXPERIMENT_NAME:
        raise ValueError(
            "Artifact experiment_name 불일치: "
            f"{artifact['experiment_name']} != {EXPERIMENT_NAME}"
        )

    if artifact["model_type"] != "CatBoostRegressor":
        raise ValueError(f"지원하지 않는 model_type: {artifact['model_type']}")

    cf_meta = artifact["conditional_features"]
    for key in ("key_cols", "parent_cols", "final_2025_bank"):
        if key not in cf_meta:
            raise KeyError(f"conditional_features에 {key}가 없습니다.")

    # 모델이 요구하는 cf 컬럼이 이 스크립트가 생성 가능한 범위 안에 있는지 확인한다.
    requested = [c for c in artifact["features"] if c.startswith("cf_platoon_")]
    unsupported = [c for c in requested if c not in CF_ALL_COLUMNS]
    if unsupported:
        raise ValueError(
            "이 스크립트가 생성할 수 없는 conditional feature가 "
            f"artifact에 있습니다: {unsupported}. 생성 가능: {CF_ALL_COLUMNS}"
        )

    return artifact


def validate_input_schema(test_df, sample_sub, artifact):
    if ID not in test_df.columns:
        raise ValueError(f"test.csv에 {ID} 컬럼이 없습니다.")

    if TARGET in test_df.columns:
        raise ValueError("test.csv에 target 컬럼이 존재합니다.")

    ensure_columns(test_df, artifact["base_features"], "test.csv")

    if list(sample_sub.columns) != [ID, TARGET]:
        raise ValueError(
            f"sample_submission.csv 컬럼이 [{ID}, {TARGET}]와 다릅니다: "
            f"{list(sample_sub.columns)}"
        )

    if test_df[ID].duplicated().any():
        raise ValueError("test.csv의 row_id가 중복되어 있습니다.")
    if sample_sub[ID].duplicated().any():
        raise ValueError("sample_submission.csv의 row_id가 중복되어 있습니다.")

    test_ids = set(test_df[ID].astype(str))
    sub_ids = set(sample_sub[ID].astype(str))
    if test_ids != sub_ids:
        raise ValueError(
            "test.csv와 sample_submission.csv의 row_id 집합이 다릅니다. "
            f"only_test={len(test_ids - sub_ids)}, "
            f"only_submission={len(sub_ids - test_ids)}"
        )


# =========================================================
# Submission — 반드시 key 기반
# =========================================================

def build_submission(test_df, sample_sub, predictions):
    """
    row_id merge로만 생성한다. 위치 기반 대입은 순서가 다를 때
    무음으로 전 행을 오배정하므로 사용하지 않는다.
    """
    pred = np.asarray(predictions, dtype=np.float64)

    if len(pred) != len(test_df):
        raise ValueError(
            f"prediction 수와 test row 수가 다릅니다: {len(pred)} != {len(test_df)}"
        )

    pred_df = pd.DataFrame(
        {
            "__sub_key": test_df[ID].astype(str).to_numpy(),
            TARGET: pred,
        }
    )

    submission = sample_sub[[ID]].copy()
    submission["__sub_key"] = submission[ID].astype(str)

    submission = submission.merge(
        pred_df,
        on="__sub_key",
        how="left",
        sort=False,
        validate="one_to_one",
    ).drop(columns="__sub_key")

    if submission[TARGET].isna().any():
        n_missing = int(submission[TARGET].isna().sum())
        raise ValueError(f"submission prediction 누락: {n_missing}건")

    values = submission[TARGET].to_numpy(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("submission에 NaN/Inf prediction이 있습니다.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("submission prediction이 [0, 1] 범위를 벗어났습니다.")

    if list(submission.columns) != [ID, TARGET]:
        raise ValueError(f"submission 컬럼 구성 오류: {list(submission.columns)}")

    return submission


# =========================================================
# main
# =========================================================

def main():
    print("=" * 72)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Model      : {MODEL_PATH}")
    print(f"Test       : {TEST_PATH}")
    print(f"Output     : {OUTPUT_PATH}")
    print("=" * 72)

    for path in (MODEL_PATH, TEST_PATH, SAMPLE_SUB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"필수 파일이 없습니다: {path}")

    print()
    print("[1/6] Artifact 로드")
    artifact = load_artifact()

    cf_meta = artifact["conditional_features"]
    bank = cf_meta["final_2025_bank"]

    print(f"Model type       : {artifact['model_type']}")
    print(f"Final iterations : {artifact.get('final_iterations', '?')}")
    print(f"Feature 수       : {len(artifact['features'])}")
    print(f"HB-EB feature    : {sorted(artifact['hbeb_priors'].keys())}")
    print(
        "CF feature       : "
        f"{[c for c in artifact['features'] if c.startswith('cf_platoon_')]}"
    )
    print(f"Bank strength K  : {float(bank['strength']):.2f}")
    print(f"Bank cells       : {len(bank['cell_table']):,}")
    print(f"Bank parents     : {len(bank['parent_table']):,}")
    print(f"Bank seasons     : {bank.get('history_seasons', '?')}")

    print()
    print("[2/6] Test / Sample Submission 로드")
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")

    validate_input_schema(test_df, sample_sub, artifact)

    print(f"Test rows        : {len(test_df):,}")
    print(f"Submission rows  : {len(sample_sub):,}")

    print()
    print("[3/6] 행 독립성 감사")
    audit_frame = test_df.head(min(AUDIT_FRAME_ROWS, len(test_df)))
    audit = verify_row_independence(audit_frame, artifact)

    print(f"passed           : {audit['passed']}")
    print(f"n_samples        : {audit['n_samples']}")
    print(f"max_abs_diff     : {audit['max_abs_diff']:.3e}")

    if not audit["passed"]:
        raise RuntimeError(f"행 독립성 검사 실패: {audit['failures'][:3]}")

    print()
    print("[4/6] 전체 추론")
    predictions, cf_diag = predict_rows(test_df, artifact)

    print()
    print("[Conditional lookup 진단]")
    print(f"unseen cell rate : {cf_diag['unseen_cell_rate']:.4%}")
    print(f"nan delta rate   : {cf_diag['nan_delta_rate']:.4%}")
    print(f"mean cell n      : {cf_diag['mean_cell_n']:.2f}")
    print(f"mean reliability : {cf_diag['mean_reliability']:.6f}")
    print(f"mean |delta|     : {cf_diag['mean_abs_delta']:.6f}")
    print(f"max  |delta|     : {cf_diag['max_abs_delta']:.6f}")

    guard_platoon_lookup(cf_diag)

    print()
    print("[Prediction 진단]")
    print(f"n                : {len(predictions):,}")
    print(f"mean             : {predictions.mean():.6f}")
    print(f"std              : {predictions.std():.6f}")
    print(f"min              : {predictions.min():.6f}")
    print(f"max              : {predictions.max():.6f}")
    print("F correction     : disabled")
    print("test aggregate   : 진단 출력 전용. 예측을 수정하지 않음")

    print()
    print("[5/6] Submission 생성")
    submission = build_submission(test_df, sample_sub, predictions)

    print()
    print("[6/6] 저장")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("=" * 72)
    print("완료")
    print(f"Submission : {OUTPUT_PATH}")
    print(f"Rows       : {len(submission):,}")
    print("=" * 72)


if __name__ == "__main__":
    main()