# script.py: cb_reg_v010 inference / submission
#
# cb_reg_v010 train.py가 저장한 단일 Artifact를 로드하여 test.csv를 추론한다.
#
# 핵심:
# - 기존 cb_reg_v003 HB-EB prior는 artifact에서 읽어 lookup만 수행한다.
# - pitcher_id × batter_hand conditional feature는
#   train 2019~2024 label로 미리 만든 frozen 2025 bank를 artifact에서 읽는다.
# - CatBoost 입력 conditional feature는 cf_platoon_delta 1개만 사용한다.
# - test 내부 groupby / mean / rank / rolling / transform / target-stat 재추정 금지.
# - test prediction 전체 평균을 이용한 calibration / delta 재계산 없음.
# - F correction 없음.
# - 각 test 행은 독립적으로 변환/예측 가능해야 한다.
#
# 출력:
#   output/submission.csv

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v010"

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 상수
# =========================================================

CF_FEATURES_FOR_MODEL = [
    "cf_platoon_delta",
]

PREDICTION_CLIP_MIN = 0.0
PREDICTION_CLIP_MAX = 1.0


# =========================================================
# 공통 유틸
# =========================================================

def ensure_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns: {missing}")


def sanitize_categoricals(df, cat_cols):
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
    return out


# =========================================================
# 기존 HB-EB
# =========================================================

def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """
    train.py에서 동결된 HB-EB prior를 각 test row에 lookup한다.
    test label/statistics를 사용하지 않는다.
    """
    ensure_columns(
        df,
        list(group_cols) + [n_col, rate_col],
        "apply_hbeb",
    )

    lookup = (
        df[list(group_cols)]
        .reset_index(drop=True)
        .merge(
            prior_df,
            on=list(group_cols),
            how="left",
            sort=False,
        )
    )

    alpha = (
        pd.to_numeric(lookup["alpha"], errors="coerce")
        .fillna(global_prior["alpha"])
        .to_numpy(np.float64)
    )

    beta = (
        pd.to_numeric(lookup["beta"], errors="coerce")
        .fillna(global_prior["beta"])
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

    eb = (success_count + alpha) / denominator

    return pd.Series(
        eb,
        index=df.index,
        dtype=np.float64,
    )


def apply_hbeb_from_artifact(
    df,
    base_features,
    hbeb_configs,
    hbeb_priors,
):
    """
    cb_reg_v003 호환:
    raw success_rate를 제거하고 hbeb_* rate로 대체한다.
    """
    out = df.copy()
    features = list(base_features)

    for cfg in hbeb_configs:
        new_col = cfg["new_col"]

        if new_col not in hbeb_priors:
            raise KeyError(
                f"Artifact missing HB-EB prior: {new_col}"
            )

        saved = hbeb_priors[new_col]

        prior_df = saved["prior_df"]
        global_prior = saved["global_prior"]

        out[new_col] = apply_hbeb(
            out,
            cfg["group_cols"],
            cfg["n_col"],
            cfg["rate_col"],
            prior_df,
            global_prior,
        )

        if cfg["rate_col"] in features:
            features.remove(cfg["rate_col"])

        if new_col not in features:
            features.append(new_col)

    return out, features


# =========================================================
# Frozen Strictly-Past Platoon Bank
# =========================================================

def apply_frozen_platoon_bank(
    df,
    bank,
    key_cols,
    parent_cols,
):
    """
    train 2019~2024 label로 학습 시점에 생성된 frozen bank를
    test row에 lookup한다.

    이번 cb_reg_v010 모델에 실제로 공급하는 conditional feature는
    cf_platoon_delta 하나뿐이다.

    정의:
        posterior =
            (cell_success + K * parent_rate)
            / (cell_n + K)

        cf_platoon_delta =
            posterior - parent_rate

    unseen cell:
        cell_n=0, cell_success=0
        -> posterior=parent_rate
        -> delta=0

    unseen pitcher:
        parent_rate=global_rate
    """
    ensure_columns(
        df,
        list(dict.fromkeys(list(key_cols) + list(parent_cols))),
        "apply_frozen_platoon_bank",
    )

    required_bank_keys = [
        "cell_table",
        "parent_table",
        "global_rate",
        "strength",
    ]
    missing = [k for k in required_bank_keys if k not in bank]
    if missing:
        raise KeyError(
            f"Frozen platoon bank missing keys: {missing}"
        )

    out = df.copy()
    n_rows = len(out)

    cell_table = bank["cell_table"]
    parent_table = bank["parent_table"]
    global_rate = float(bank["global_rate"])
    strength = float(bank["strength"])

    # -----------------------------------------------------
    # Parent pitcher lookup
    # -----------------------------------------------------
    parent_lookup = (
        out[list(parent_cols)]
        .reset_index(drop=True)
        .merge(
            parent_table,
            on=list(parent_cols),
            how="left",
            sort=False,
        )
    )

    if "parent_rate" in parent_lookup.columns:
        parent_rate = (
            pd.to_numeric(
                parent_lookup["parent_rate"],
                errors="coerce",
            )
            .fillna(global_rate)
            .to_numpy(np.float64)
        )
    else:
        parent_rate = np.full(
            n_rows,
            global_rate,
            dtype=np.float64,
        )

    # -----------------------------------------------------
    # Conditional cell lookup
    # -----------------------------------------------------
    cell_lookup = (
        out[list(key_cols)]
        .reset_index(drop=True)
        .merge(
            cell_table,
            on=list(key_cols),
            how="left",
            sort=False,
        )
    )

    if "cell_n" in cell_lookup.columns:
        cell_n = (
            pd.to_numeric(
                cell_lookup["cell_n"],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(np.float64)
        )
    else:
        cell_n = np.zeros(n_rows, dtype=np.float64)

    if "cell_success" in cell_lookup.columns:
        cell_success = (
            pd.to_numeric(
                cell_lookup["cell_success"],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(np.float64)
        )
    else:
        cell_success = np.zeros(n_rows, dtype=np.float64)

    cell_n = np.clip(cell_n, 0.0, None)

    posterior_rate = (
        cell_success + strength * parent_rate
    ) / (cell_n + strength)

    delta = posterior_rate - parent_rate

    out["cf_platoon_delta"] = delta.astype(np.float64)

    # 진단값은 모델 입력이 아니다.
    diagnostics = {
        "n_rows": int(n_rows),
        "unseen_cell_rows": int(np.sum(cell_n <= 0.0)),
        "unseen_cell_rate": (
            float(np.mean(cell_n <= 0.0))
            if n_rows > 0
            else 0.0
        ),
        "mean_cell_n": (
            float(np.mean(cell_n))
            if n_rows > 0
            else 0.0
        ),
        "mean_abs_delta": (
            float(np.mean(np.abs(delta)))
            if n_rows > 0
            else 0.0
        ),
        "strength": strength,
        "global_rate": global_rate,
    }

    return out, diagnostics


# =========================================================
# Row independence audit
# =========================================================

def transform_one_pipeline(
    frame,
    artifact,
):
    """
    test 전처리만 수행한다.
    batch와 single-row 결과 동일성 검증에 사용.
    """
    base_features = artifact["base_features"]
    cat_cols = artifact["cat_cols"]

    hbeb_configs = artifact["hbeb_configs"]
    hbeb_priors = artifact["hbeb_priors"]

    transformed, hbeb_features = apply_hbeb_from_artifact(
        frame,
        base_features,
        hbeb_configs,
        hbeb_priors,
    )

    cf_meta = artifact["conditional_features"]
    bank = cf_meta["final_2025_bank"]

    transformed, cf_diag = apply_frozen_platoon_bank(
        transformed,
        bank,
        cf_meta["key_cols"],
        cf_meta["parent_cols"],
    )

    expected_features = artifact["features"]

    ensure_columns(
        transformed,
        expected_features,
        "transformed test",
    )

    transformed = sanitize_categoricals(
        transformed,
        cat_cols,
    )

    return transformed, cf_diag


def verify_row_independence(
    frame,
    artifact,
    n_samples=30,
    seed=42,
):
    if len(frame) == 0:
        return {
            "passed": True,
            "n_samples": 0,
            "max_abs_diff": 0.0,
            "failures": [],
        }

    model = artifact["model"]
    features = artifact["features"]

    batch_frame, _ = transform_one_pipeline(
        frame,
        artifact,
    )
    batch_pred = model.predict(
        batch_frame[features]
    )

    rng = np.random.default_rng(seed)
    k = min(int(n_samples), len(frame))
    positions = rng.choice(
        len(frame),
        size=k,
        replace=False,
    )

    failures = []
    max_abs_diff = 0.0

    for pos in positions:
        one = frame.iloc[[int(pos)]].copy()

        one_frame, _ = transform_one_pipeline(
            one,
            artifact,
        )
        one_pred = float(
            model.predict(
                one_frame[features]
            )[0]
        )

        diff = abs(
            float(batch_pred[int(pos)])
            - one_pred
        )
        max_abs_diff = max(
            max_abs_diff,
            diff,
        )

        if diff > 1e-12:
            failures.append(
                {
                    "position": int(pos),
                    "batch_pred": float(
                        batch_pred[int(pos)]
                    ),
                    "single_pred": one_pred,
                    "abs_diff": diff,
                }
            )

    return {
        "passed": len(failures) == 0,
        "n_samples": k,
        "max_abs_diff": max_abs_diff,
        "failures": failures,
    }


# =========================================================
# Submission 생성
# =========================================================

def build_submission(
    test_df,
    predictions,
):
    """
    sample_submission.csv가 있으면 그 포맷을 우선 사용한다.
    없으면 control_success 단일 컬럼으로 생성한다.
    """
    pred = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if SAMPLE_SUBMISSION_PATH.exists():
        sub = pd.read_csv(
            SAMPLE_SUBMISSION_PATH
        )

        if len(sub) != len(test_df):
            raise ValueError(
                "sample_submission row count mismatch: "
                f"{len(sub)} != {len(test_df)}"
            )

        # target 컬럼명이 이미 있으면 그 컬럼 사용
        if "control_success" in sub.columns:
            target_col = "control_success"
        else:
            # id성 컬럼을 제외하고 단일 예측 컬럼을 찾는다.
            candidate_cols = [
                c for c in sub.columns
                if c.lower() not in {
                    "id",
                    "row_id",
                    "index",
                }
            ]

            if len(candidate_cols) != 1:
                raise ValueError(
                    "Could not infer prediction column from "
                    f"sample_submission columns={list(sub.columns)}"
                )

            target_col = candidate_cols[0]

        sub[target_col] = pred
        return sub

    return pd.DataFrame(
        {
            "control_success": pred,
        }
    )


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Model      : {MODEL_PATH}")
    print(f"Test       : {TEST_PATH}")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test.csv not found: {TEST_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    required_artifact_keys = [
        "model",
        "features",
        "cat_cols",
        "base_features",
        "hbeb_priors",
        "hbeb_configs",
        "conditional_features",
    ]

    missing = [
        k
        for k in required_artifact_keys
        if k not in artifact
    ]
    if missing:
        raise KeyError(
            f"Artifact missing required keys: {missing}"
        )

    if artifact.get(
        "experiment_name"
    ) != EXPERIMENT_NAME:
        raise ValueError(
            "Artifact experiment mismatch: "
            f"{artifact.get('experiment_name')} "
            f"!= {EXPERIMENT_NAME}"
        )

    cf_meta = artifact["conditional_features"]

    # 최신 v010 정의 검증:
    # 실제 모델 입력에는 cf_platoon_delta 하나만 있어야 한다.
    artifact_cf_features = [
        c
        for c in artifact["features"]
        if c.startswith("cf_platoon_")
    ]

    if artifact_cf_features != CF_FEATURES_FOR_MODEL:
        raise ValueError(
            "Unexpected conditional feature manifest. "
            f"Expected={CF_FEATURES_FOR_MODEL}, "
            f"Artifact={artifact_cf_features}. "
            "Train v010 must use delta-only conditional feature."
        )

    test_df = pd.read_csv(
        TEST_PATH
    )

    ensure_columns(
        test_df,
        artifact["base_features"],
        "test.csv",
    )

    # -----------------------------------------------------
    # Row independence regression audit
    # -----------------------------------------------------
    audit_frame = test_df.head(
        min(5000, len(test_df))
    )

    audit = verify_row_independence(
        audit_frame,
        artifact,
        n_samples=30,
        seed=42,
    )

    print()
    print("[Row independence audit]")
    print(
        f"passed       : {audit['passed']}"
    )
    print(
        f"n_samples    : {audit['n_samples']}"
    )
    print(
        f"max_abs_diff : {audit['max_abs_diff']:.3e}"
    )

    if not audit["passed"]:
        raise RuntimeError(
            "Row independence check failed: "
            f"{audit['failures'][:3]}"
        )

    # -----------------------------------------------------
    # Full test transform
    # -----------------------------------------------------
    test_transformed, cf_diag = (
        transform_one_pipeline(
            test_df,
            artifact,
        )
    )

    print()
    print("[Conditional diagnostics]")
    print(
        f"unseen cell rate : "
        f"{cf_diag['unseen_cell_rate']:.4%}"
    )
    print(
        f"mean cell n      : "
        f"{cf_diag['mean_cell_n']:.2f}"
    )
    print(
        f"mean |delta|     : "
        f"{cf_diag['mean_abs_delta']:.6f}"
    )
    print(
        f"strength K       : "
        f"{cf_diag['strength']:.2f}"
    )

    model = artifact["model"]
    features = artifact["features"]

    pred_raw = model.predict(
        test_transformed[features]
    )

    pred_prob = np.clip(
        np.asarray(
            pred_raw,
            dtype=np.float64,
        ),
        PREDICTION_CLIP_MIN,
        PREDICTION_CLIP_MAX,
    )

    # -----------------------------------------------------
    # 중요: test 평균 기반 correction 없음
    # -----------------------------------------------------
    print()
    print("[Prediction diagnostics]")
    print(
        f"n              : {len(pred_prob):,}"
    )
    print(
        f"mean           : {pred_prob.mean():.6f}"
    )
    print(
        f"std            : {pred_prob.std():.6f}"
    )
    print(
        f"min            : {pred_prob.min():.6f}"
    )
    print(
        f"max            : {pred_prob.max():.6f}"
    )
    print(
        "F correction   : disabled"
    )
    print(
        "test aggregate : diagnostics only; "
        "never used to modify predictions"
    )

    submission = build_submission(
        test_df,
        pred_prob,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    submission.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 80)
    print("Submission saved")
    print(f"- {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
