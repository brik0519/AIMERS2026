# train.py: CatBoost 적용버전
# 또한 모델 분석을 위한 구체적인 수치까지 저장하게 하였다.

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


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
EXPERIMENT_NAME = "cb_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# CatBoost 설정
# =========================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
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
    기존 CatBoost 실험과 동일한 전처리.

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
# Validation 결과 저장
# =========================================================

def save_validation(
    experiment_name,
    estimator,
    model_params,
    row_ids,
    y_true,
    y_pred,
    brier,
    baseline_brier,
    bss,
    output_root,
    best_iteration=None,
    final_iterations=None,
):
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert len(row_ids) == len(y_true) == len(y_pred)

    output_dir = Path(output_root) / experiment_name

    if output_dir.exists():
        raise FileExistsError(
            f"Validation experiment already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # pred.csv
    # -----------------------------------------------------

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction": y_pred,
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
        prediction=y_pred,
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
        "brier": float(brier),
        "baseline_brier": float(baseline_brier),
        "bss": float(bss),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }

    if best_iteration is not None:
        metadata["best_iteration"] = int(best_iteration)

    if final_iterations is not None:
        metadata["final_iterations"] = int(final_iterations)

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

    y_train = train.loc[
        ~is_val,
        TARGET,
    ]

    X_val = prepare_features(
        train.loc[
            is_val,
            features,
        ],
        cat_cols,
    )

    y_val = train.loc[
        is_val,
        TARGET,
    ]

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

    estimator = CatBoostClassifier(
        **MODEL_PARAMS
    )

    print("=" * 60)
    print("Validation 모델 학습 시작")
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
    # Brier Skill Score
    # -----------------------------------------------------

    val_pred = estimator.predict_proba(
        X_val
    )[:, 1]

    r = y_val.mean()

    brier = (
        (val_pred - y_val.to_numpy()) ** 2
    ).mean()

    baseline_brier = (
        r * (1 - r)
    )

    score = max(
        0,
        100000 * (
            1 - brier / baseline_brier
        ),
    )

    print("=" * 60)
    print("Validation 결과")
    print("=" * 60)

    print(
        f"Brier Score          : {brier:.8f}"
    )

    print(
        f"Baseline Brier       : {baseline_brier:.8f}"
    )

    print(
        f"Validation Score     : {score:.2f}"
    )

    print(
        f"Target mean          : {y_val.mean():.6f}"
    )

    print(
        f"Prediction mean      : {val_pred.mean():.6f}"
    )

    print(
        f"Prediction std       : {val_pred.std():.6f}"
    )

    print(
        f"Prediction min       : {val_pred.min():.6f}"
    )

    print(
        f"Prediction max       : {val_pred.max():.6f}"
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
        y_pred=val_pred,
        brier=brier,
        baseline_brier=baseline_brier,
        bss=score,
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

    y_full = train[
        TARGET
    ]

    final_params = {
        **MODEL_PARAMS,
        "iterations": final_iterations,
    }

    final_model = CatBoostClassifier(
        **final_params
    )

    print("=" * 60)
    print("전체 데이터 재학습 시작")
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


if __name__ == "__main__":
    main()