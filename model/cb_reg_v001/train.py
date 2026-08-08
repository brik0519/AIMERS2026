# train.py: CatBoost Regressor 적용 버전
# 기존 CatBoostClassifier 실험과 최대한 동일한 조건에서
# Classification -> Regression objective 변경 효과를 검증한다.
#
# 추가 저장 항목:
# - raw prediction / clipped prediction
# - raw / clipped Brier Score 및 BSS
# - [0, 1] 범위 이탈 개수/비율
# - prediction 분포 통계

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =========================================================
# 실험 설정
# =========================================================

DATA_DIR = Path("./data")

ID = "row_id"
TARGET = "control_success"

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

# 실험마다 이 값만 변경
EXPERIMENT_NAME = "cb_reg_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# CatBoost 설정
# =========================================================

# 기존 CatBoostClassifier와 capacity 관련 파라미터는 동일하게 유지.
# 핵심 변경점은 loss_function: Logloss -> RMSE
MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(df, cat_cols):
    """
    기존 CatBoostClassifier 실험과 동일한 전처리.

    - 수치형 결측값은 CatBoost가 직접 처리
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 문자열 dtype으로 통일
    """

    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Brier / BSS 계산
# =========================================================

def calculate_brier_metrics(y_true, y_pred):
    """
    Validation 기준 Brier Score와 Brier Skill Score를 계산한다.

    BSS score:
        max(0, 100000 * (1 - BS / BS_baseline))

    baseline은 Validation target의 climatology를 사용한다.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    r = float(np.mean(y_true))

    brier = float(
        np.mean((y_pred - y_true) ** 2)
    )

    baseline_brier = float(
        r * (1.0 - r)
    )

    if baseline_brier <= 0:
        raise ValueError(
            "baseline_brier가 0 이하입니다. "
            "Validation target 분포를 확인하세요."
        )

    bss = float(
        max(
            0.0,
            100000.0 * (
                1.0
                - brier / baseline_brier
            ),
        )
    )

    return brier, baseline_brier, bss


# =========================================================
# Validation 결과 저장
# =========================================================

def save_validation(
    experiment_name,
    estimator,
    model_params,
    row_ids,
    y_true,
    y_pred_raw,
    y_pred_clip,
    raw_brier,
    clipped_brier,
    baseline_brier,
    raw_bss,
    clipped_bss,
    output_root,
    best_iteration=None,
    final_iterations=None,
):
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred_raw = np.asarray(y_pred_raw)
    y_pred_clip = np.asarray(y_pred_clip)

    assert (
        len(row_ids)
        == len(y_true)
        == len(y_pred_raw)
        == len(y_pred_clip)
    )

    output_dir = (
        Path(output_root)
        / experiment_name
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Validation experiment already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    below_zero = y_pred_raw < 0.0
    above_one = y_pred_raw > 1.0
    outside_range = below_zero | above_one

    # -----------------------------------------------------
    # pred.csv
    # -----------------------------------------------------

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction_raw": y_pred_raw,
        "prediction_clip": y_pred_clip,
    })

    pred_df.to_csv(
        output_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # validation.npz
    # -----------------------------------------------------

    np.savez_compressed(
        output_dir / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction_raw=y_pred_raw,
        prediction_clip=y_pred_clip,
    )

    # -----------------------------------------------------
    # metadata.json
    # -----------------------------------------------------

    metadata = {
        "experiment_name": experiment_name,
        "model_type": type(estimator).__name__,
        "model_params": model_params,
        "validation_season": 2024,
        "n_validation": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),

        # 평가 지표
        "raw_brier": float(raw_brier),
        "clipped_brier": float(clipped_brier),
        "baseline_brier": float(baseline_brier),
        "raw_bss": float(raw_bss),
        "clipped_bss": float(clipped_bss),

        # raw prediction 통계
        "prediction_raw_mean": float(np.mean(y_pred_raw)),
        "prediction_raw_std": float(np.std(y_pred_raw)),
        "prediction_raw_min": float(np.min(y_pred_raw)),
        "prediction_raw_max": float(np.max(y_pred_raw)),

        # clipped prediction 통계
        "prediction_clip_mean": float(np.mean(y_pred_clip)),
        "prediction_clip_std": float(np.std(y_pred_clip)),
        "prediction_clip_min": float(np.min(y_pred_clip)),
        "prediction_clip_max": float(np.max(y_pred_clip)),

        # [0, 1] 범위 이탈 진단
        "n_prediction_below_0": int(np.sum(below_zero)),
        "ratio_prediction_below_0": float(np.mean(below_zero)),
        "n_prediction_above_1": int(np.sum(above_one)),
        "ratio_prediction_above_1": float(np.mean(above_one)),
        "n_prediction_outside_0_1": int(np.sum(outside_range)),
        "ratio_prediction_outside_0_1": float(np.mean(outside_range)),
    }

    if best_iteration is not None:
        metadata["best_iteration"] = int(
            best_iteration
        )

    if final_iterations is not None:
        metadata["final_iterations"] = int(
            final_iterations
        )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("[Validation 저장 완료]")
    print(f"- {output_dir / 'pred.csv'}")
    print(f"- {output_dir / 'validation.npz'}")
    print(f"- {output_dir / 'metadata.json'}")


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation experiment already exists: {VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"Model experiment already exists: {MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # 데이터
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        DATA_DIR / "test.csv",
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    features = [
        c for c in test_cols
        if c != ID
    ]

    cat_cols = [
        c for c in CAT_COLS
        if c in features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[ID] + features + [TARGET],
    )

    print("=" * 60)
    print("데이터 로드 완료")
    print("train shape:", train.shape)
    print("Feature 수:", len(features))
    print("범주형 Feature 수:", len(cat_cols))
    print("범주형 Feature:", cat_cols)
    print(
        "시즌:",
        train["season"].min(),
        "~",
        train["season"].max(),
    )
    print(
        f"제구 성공률: {train[TARGET].mean():.6f}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # 2019~2023 Train / 2024 Validation
    # -----------------------------------------------------

    is_val = train["season"] == 2024

    X_train = prepare_features(
        train.loc[
            ~is_val,
            features,
        ],
        cat_cols,
    )

    # 회귀 모델이므로 target을 float으로 명시
    y_train = (
        train.loc[
            ~is_val,
            TARGET,
        ]
        .astype(np.float64)
    )

    X_val = prepare_features(
        train.loc[
            is_val,
            features,
        ],
        cat_cols,
    )

    y_val = (
        train.loc[
            is_val,
            TARGET,
        ]
        .astype(np.float64)
    )

    val_row_id = train.loc[
        is_val,
        ID,
    ]

    print(
        "학습 데이터:",
        X_train.shape,
    )

    print(
        "검증 데이터:",
        X_val.shape,
    )

    # -----------------------------------------------------
    # Validation 모델
    # -----------------------------------------------------

    estimator = CatBoostRegressor(
        **MODEL_PARAMS
    )

    print("=" * 60)
    print("Validation CatBoostRegressor 학습 시작")
    print("=" * 60)

    start_time = time.time()

    estimator.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(
            X_val,
            y_val,
        ),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Validation 모델 학습 완료: "
        f"{elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # Validation prediction
    # -----------------------------------------------------

    val_pred_raw = estimator.predict(
        X_val
    )

    val_pred_clip = np.clip(
        val_pred_raw,
        0.0,
        1.0,
    )

    # -----------------------------------------------------
    # Brier Skill Score
    # -----------------------------------------------------

    (
        raw_brier,
        baseline_brier,
        raw_score,
    ) = calculate_brier_metrics(
        y_val,
        val_pred_raw,
    )

    (
        clipped_brier,
        clipped_baseline_brier,
        clipped_score,
    ) = calculate_brier_metrics(
        y_val,
        val_pred_clip,
    )

    assert np.isclose(
        baseline_brier,
        clipped_baseline_brier,
    )

    below_zero = (
        val_pred_raw < 0.0
    )

    above_one = (
        val_pred_raw > 1.0
    )

    outside_range = (
        below_zero
        | above_one
    )

    print("=" * 60)
    print("Validation 결과")
    print("=" * 60)

    print("[Raw prediction]")
    print(
        f"Brier Score          : {raw_brier:.8f}"
    )
    print(
        f"Validation Score     : {raw_score:.2f}"
    )

    print()

    print("[Clipped prediction]")
    print(
        f"Brier Score          : {clipped_brier:.8f}"
    )
    print(
        f"Validation Score     : {clipped_score:.2f}"
    )

    print()

    print("[Baseline]")
    print(
        f"Baseline Brier       : {baseline_brier:.8f}"
    )
    print(
        f"Target mean          : {y_val.mean():.6f}"
    )

    print()

    print("[Raw prediction distribution]")
    print(
        f"Prediction mean      : {val_pred_raw.mean():.6f}"
    )
    print(
        f"Prediction std       : {val_pred_raw.std():.6f}"
    )
    print(
        f"Prediction min       : {val_pred_raw.min():.6f}"
    )
    print(
        f"Prediction max       : {val_pred_raw.max():.6f}"
    )

    print()

    print("[Range diagnostics]")
    print(
        f"Prediction < 0       : "
        f"{below_zero.sum()} "
        f"({below_zero.mean():.6%})"
    )
    print(
        f"Prediction > 1       : "
        f"{above_one.sum()} "
        f"({above_one.mean():.6%})"
    )
    print(
        f"Outside [0, 1]       : "
        f"{outside_range.sum()} "
        f"({outside_range.mean():.6%})"
    )

    # -----------------------------------------------------
    # 최적 iteration
    # -----------------------------------------------------

    best_iteration = (
        estimator.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        final_iterations = (
            MODEL_PARAMS["iterations"]
        )
    else:
        final_iterations = (
            best_iteration + 1
        )

    print()
    print(
        f"Best iteration       : {best_iteration}"
    )
    print(
        f"Final iterations     : {final_iterations}"
    )

    # -----------------------------------------------------
    # Validation 저장
    # -----------------------------------------------------

    save_validation(
        experiment_name=EXPERIMENT_NAME,
        estimator=estimator,
        model_params=MODEL_PARAMS,
        row_ids=val_row_id,
        y_true=y_val,
        y_pred_raw=val_pred_raw,
        y_pred_clip=val_pred_clip,
        raw_brier=raw_brier,
        clipped_brier=clipped_brier,
        baseline_brier=baseline_brier,
        raw_bss=raw_score,
        clipped_bss=clipped_score,
        output_root=VALIDATION_ROOT,
        best_iteration=best_iteration,
        final_iterations=final_iterations,
    )

    # -----------------------------------------------------
    # 전체 데이터 재학습
    # -----------------------------------------------------

    X_full = prepare_features(
        train[features],
        cat_cols,
    )

    y_full = (
        train[
            TARGET
        ]
        .astype(np.float64)
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": final_iterations,
    }

    final_model = CatBoostRegressor(
        **final_params
    )

    print("=" * 60)
    print("전체 데이터 CatBoostRegressor 재학습 시작")
    print("=" * 60)

    start_time = time.time()

    final_model.fit(
        X_full,
        y_full,
        cat_features=cat_cols,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"전체 데이터 재학습 완료: "
        f"{elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # 모델 저장
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
        compress=3,
    )

    shutil.copy2(
        "./train.py",
        MODEL_DIR / "train.py",
    )

    print("=" * 60)
    print("모델 저장 완료")
    print(f"- {MODEL_PATH}")
    print(f"- {MODEL_DIR / 'train.py'}")
    print("=" * 60)

    print()
    print("[중요]")
    print(
        "CatBoostRegressor의 predict() 출력은 "
        "[0, 1] 범위를 벗어날 수 있습니다."
    )
    print(
        "추론/제출 코드에서는 Validation 결과를 기준으로 "
        "필요 시 np.clip(pred, 0.0, 1.0)을 적용하세요."
    )


if __name__ == "__main__":
    main()
