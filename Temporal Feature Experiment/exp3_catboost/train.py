# train.py
#
# CatBoost 단독 + EXP3 Temporal Feature 실험
#
# 목적:
#   Feature Engineering 효과를 단일 모델 기준으로 확인하기 위해
#   기존 CatBoost Basic과 동일한 모델 설정을 유지하고
#   EXP3 Temporal Feature 10개만 추가한다.
#
# 비교:
#   CatBoost Basic (Raw 47)
#   vs
#   CatBoost + EXP3 (Raw 47 + Temporal 10 = 57)
#
# Validation:
#   Train      : 2019 ~ 2023
#   Validation : 2024
#
# 이후 Validation에서 선택된 iteration 수로
# 2019~2024 전체 데이터를 재학습한다.

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

EXPERIMENT_NAME = "cb_exp3_v001"

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME


# =========================================================
# CatBoost 범주형 Feature
# 기존 CatBoost Basic과 동일
# =========================================================

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


# =========================================================
# CatBoost 설정
# 기존 CatBoost Basic과 동일
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
# EXP3 Temporal Feature 10개
# =========================================================

TEMPORAL_FEATURES = [
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
    "success_trend_1_3",
    "success_trend_3_5",
    "middle_trend_1_3",
    "middle_trend_3_5",
]


def add_temporal_features(df):
    """
    각 row에 이미 제공된 asof_* 값만 사용해
    EXP3 Temporal Feature 10개를 생성한다.

    원본 asof_* Feature는 제거하지 않는다.
    Test row 간 aggregation / rolling은 사용하지 않는다.
    """

    df = df.copy()

    required_cols = [
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing_cols)
        )

    # -----------------------------------------------------
    # Recent vs Cumulative - Success
    # -----------------------------------------------------

    df["recent1_success_vs_career"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["recent3_success_vs_career"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["recent5_success_vs_career"] = (
        df["asof_pitcher_prev5_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    # -----------------------------------------------------
    # Recent vs Cumulative - Middle
    # -----------------------------------------------------

    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    # -----------------------------------------------------
    # Trend - Success
    # -----------------------------------------------------

    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )

    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    # -----------------------------------------------------
    # Trend - Middle
    # -----------------------------------------------------

    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )

    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


# =========================================================
# CatBoost 전처리
# 기존 CatBoost Basic과 동일
# =========================================================

def prepare_features(df, cat_cols):
    """
    - 수치형 결측값은 CatBoost가 직접 처리
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 str로 통일
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

    pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction": y_pred,
    }).to_csv(
        output_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        output_dir / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction=y_pred,
    )

    metadata = {
        "experiment_name": experiment_name,
        "model_type": type(estimator).__name__,
        "model_params": model_params,
        "feature_mode": "EXP3_TEMPORAL",
        "temporal_features": TEMPORAL_FEATURES,
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
            "기존 폴더를 삭제하거나 EXPERIMENT_NAME을 변경하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"Model experiment already exists: {MODEL_DIR}\n"
            "기존 폴더를 삭제하거나 EXPERIMENT_NAME을 변경하세요."
        )

    # -----------------------------------------------------
    # test.csv 기준 Raw Feature 결정
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        DATA_DIR / "test.csv",
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise ValueError(
            f"test.csv에 ID 컬럼 '{ID}'가 없습니다."
        )

    raw_features = [
        col
        for col in test_cols
        if col != ID
    ]

    # -----------------------------------------------------
    # train.csv 로드
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[
            ID,
            *raw_features,
            TARGET,
        ],
    )

    if "season" not in train.columns:
        raise ValueError(
            "train.csv에 season 컬럼이 없습니다."
        )

    if train[TARGET].isna().any():
        raise ValueError(
            f"타깃 컬럼 '{TARGET}'에 결측값이 있습니다."
        )

    # -----------------------------------------------------
    # EXP3 추가
    # -----------------------------------------------------

    train = add_temporal_features(
        train
    )

    features = (
        raw_features
        + TEMPORAL_FEATURES
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    print("=" * 70)
    print("CatBoost + EXP3")
    print("=" * 70)
    print("Raw Feature      :", len(raw_features))
    print("Temporal Feature :", len(TEMPORAL_FEATURES))
    print("Total Feature    :", len(features))
    print("Categorical      :", len(cat_cols))
    print("Categorical List :", cat_cols)
    print(
        "Season           :",
        train["season"].min(),
        "~",
        train["season"].max(),
    )
    print("=" * 70)

    # =====================================================
    # 2019~2023 Train / 2024 Validation
    # =====================================================

    is_val = (
        train["season"]
        == 2024
    )

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

    print("Train shape      :", X_train.shape)
    print("Validation shape :", X_val.shape)

    # -----------------------------------------------------
    # Validation 모델
    # -----------------------------------------------------

    estimator = CatBoostClassifier(
        **MODEL_PARAMS
    )

    print("=" * 70)
    print("CatBoost + EXP3 Validation 학습 시작")
    print("=" * 70)

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

    val_pred = estimator.predict_proba(
        X_val
    )[:, 1]

    r = float(
        y_val.mean()
    )

    brier = float(
        np.mean(
            (
                val_pred
                - y_val.to_numpy()
            ) ** 2
        )
    )

    baseline_brier = (
        r * (1 - r)
    )

    score = max(
        0.0,
        100000.0
        * (
            1.0
            - brier
            / baseline_brier
        ),
    )

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

    print("=" * 70)
    print("Validation 결과")
    print("=" * 70)
    print(f"Brier Score      : {brier:.8f}")
    print(f"Baseline Brier   : {baseline_brier:.8f}")
    print(f"Validation Score : {score:.2f}")
    print(f"Best iteration   : {best_iteration}")
    print(f"Final iterations : {final_iterations}")
    print(f"Elapsed          : {elapsed:.1f}s")
    print(f"Prediction mean  : {val_pred.mean():.6f}")
    print(f"Prediction std   : {val_pred.std():.6f}")
    print(f"Prediction min   : {val_pred.min():.6f}")
    print(f"Prediction max   : {val_pred.max():.6f}")
    print("=" * 70)

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

    # =====================================================
    # 2019~2024 전체 재학습
    # =====================================================

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

    print("=" * 70)
    print("2019~2024 전체 데이터 재학습 시작")
    print("=" * 70)

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
        Path(__file__),
        MODEL_DIR / "train.py",
    )

    print("=" * 70)
    print("CatBoost + EXP3 모델 저장 완료")
    print(f"- {MODEL_PATH}")
    print(f"- {MODEL_DIR / 'train.py'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
