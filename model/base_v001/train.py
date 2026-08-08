# train.py

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


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
]

# 실험마다 이 값만 변경
EXPERIMENT_NAME = "rf_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "n_jobs": -1,
    "random_state": 42,
}


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

    # pred.csv
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

    # validation.npz
    np.savez_compressed(
        output_dir / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction=y_pred,
    )

    # metadata.json
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
    }

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

    num_cols = [
        c for c in features
        if c not in CAT_COLS
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[ID] + features + [TARGET],
    )

    print(
        "train:",
        train.shape,
        "| 피처:",
        len(features),
        f"(범주형 {len(CAT_COLS)}, 수치형 {len(num_cols)})",
    )

    print(
        "시즌:",
        train["season"].min(),
        "~",
        train["season"].max(),
    )

    print(
        f"제구 성공률: {train[TARGET].mean():.4f}"
    )

    # -----------------------------------------------------
    # 전처리
    # -----------------------------------------------------

    preprocessor = ColumnTransformer([
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            CAT_COLS,
        ),
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ),
    ])

    # -----------------------------------------------------
    # 모델
    # -----------------------------------------------------

    estimator = RandomForestClassifier(
        **MODEL_PARAMS
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", estimator),
    ])

    # -----------------------------------------------------
    # 2019~2023 Train / 2024 Validation
    # -----------------------------------------------------

    is_val = train["season"] == 2024

    X_train = train.loc[
        ~is_val,
        features,
    ]

    y_train = train.loc[
        ~is_val,
        TARGET,
    ]

    X_val = train.loc[
        is_val,
        features,
    ]

    y_val = train.loc[
        is_val,
        TARGET,
    ]

    val_row_id = train.loc[
        is_val,
        ID,
    ]

    print(
        "train:",
        len(X_train),
        "| val:",
        len(X_val),
    )

    # -----------------------------------------------------
    # 학습
    # -----------------------------------------------------

    t = time.time()

    model.fit(
        X_train,
        y_train,
    )

    print(
        f"학습 완료 :: {time.time() - t:.1f}s"
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    val_pred = model.predict_proba(
        X_val
    )[:, 1]

    r = y_val.mean()

    brier = (
        (val_pred - y_val.to_numpy()) ** 2
    ).mean()

    baseline_brier = r * (1 - r)

    score = max(
        0,
        100000 * (
            1 - brier / baseline_brier
        ),
    )

    print(
        f"Brier: {brier:.6f}"
        f" | 기준선 r(1-r): {baseline_brier:.6f}"
    )

    print(
        f"Validation Score: {score:.2f}"
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
    )

    # -----------------------------------------------------
    # 전체 데이터 재학습
    # -----------------------------------------------------

    t = time.time()

    model.fit(
        train[features],
        train[TARGET],
    )

    print(
        f"재학습 완료 :: {time.time() - t:.1f}s"
    )

    # -----------------------------------------------------
    # 모델 저장
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        model,
        MODEL_PATH,
        compress=3,
    )

    # 해당 모델을 생성한 train.py snapshot
    shutil.copy2(
        "./train.py",
        MODEL_DIR / "train.py",
    )

    print()
    print("[Model 저장 완료]")
    print(f"- {MODEL_PATH}")
    print(f"- {MODEL_DIR / 'train.py'}")


if __name__ == "__main__":
    main()