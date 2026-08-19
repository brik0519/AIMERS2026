# fold3_training_window_ablation.py
#
# LG Aimers - Fold 3 Training Window Ablation
#
# 목적:
# Feature / CatBoost 설정은 완전히 고정하고
# "얼마나 오래된 데이터를 학습에 포함하느냐"만 변경한다.
#
# T1: 2019~2022 -> 2023
# T2: 2020~2022 -> 2023
# T3: 2021~2022 -> 2023
# T4: 2022      -> 2023
#
# 확인 지표:
# - Brier Score
# - T1 대비 개선량
# - ROC-AUC
# - LogLoss
# - Best Iteration
# - Prediction Mean / Std
# - Train Target Mean
# - Train Row 수
# - 학습 시간
#
# 결과:
# validation/fold3_training_window_ablation/
# ├── summary.csv
# ├── T1/pred.csv
# ├── T2/pred.csv
# ├── T3/pred.csv
# └── T4/pred.csv


import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


# =========================================================
# 기본 설정
# =========================================================

DATA_DIR = Path("./data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = (
    Path("./validation")
    / "fold3_training_window_ablation"
)

ID = "row_id"
TARGET = "control_success"

VALIDATION_SEASON = 2023


# =========================================================
# Training Window
# =========================================================

WINDOWS = {
    "T1": {
        "train_seasons": [2019, 2020, 2021, 2022],
        "description": "2019~2022",
    },

    "T2": {
        "train_seasons": [2020, 2021, 2022],
        "description": "2020~2022",
    },

    "T3": {
        "train_seasons": [2021, 2022],
        "description": "2021~2022",
    },

    "T4": {
        "train_seasons": [2022],
        "description": "2022 only",
    },
}


# =========================================================
# 기존 A 모델과 동일한 Categorical Feature
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
# 기존 실험과 동일한 CatBoost 설정
# =========================================================

MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# =========================================================
# 전처리
# =========================================================

def prepare_features(df, cat_cols):

    df = df.copy()

    for col in cat_cols:

        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Metric 계산
# =========================================================

def calculate_metrics(
    y_true,
    pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    brier = float(
        np.mean(
            (pred - y_true) ** 2
        )
    )

    auc = float(
        roc_auc_score(
            y_true,
            pred,
        )
    )

    ll = float(
        log_loss(
            y_true,
            pred,
            labels=[0, 1],
        )
    )

    return {
        "brier": brier,
        "roc_auc": auc,
        "logloss": ll,
        "prediction_mean": float(
            pred.mean()
        ),
        "prediction_std": float(
            pred.std()
        ),
        "prediction_min": float(
            pred.min()
        ),
        "prediction_max": float(
            pred.max()
        ),
    }


# =========================================================
# 단일 Window 실행
# =========================================================

def run_window(
    key,
    config,
    train,
    val_fold,
    features,
):

    train_seasons = (
        config["train_seasons"]
    )

    # -----------------------------------------------------
    # Train Window 구성
    # -----------------------------------------------------

    train_fold = (
        train[
            train["season"].isin(
                train_seasons
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Categorical
    # -----------------------------------------------------

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    # -----------------------------------------------------
    # X / y
    # -----------------------------------------------------

    X_train = prepare_features(
        train_fold[features],
        cat_cols,
    )

    y_train = (
        train_fold[TARGET]
        .to_numpy()
    )

    X_val = prepare_features(
        val_fold[features],
        cat_cols,
    )

    y_val = (
        val_fold[TARGET]
        .to_numpy()
    )

    print()
    print("=" * 90)

    print(
        f"{key} | "
        f"Train {config['description']} "
        f"-> Validation 2023"
    )

    print("=" * 90)

    print(
        f"Train rows        : "
        f"{len(train_fold):,}"
    )

    print(
        f"Train target mean : "
        f"{y_train.mean():.6f}"
    )

    print(
        f"Valid target mean : "
        f"{y_val.mean():.6f}"
    )

    # -----------------------------------------------------
    # 반드시 Window마다 새 모델 생성
    # -----------------------------------------------------

    model = CatBoostClassifier(
        **MODEL_PARAMS
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,

        cat_features=cat_cols,

        eval_set=(
            X_val,
            y_val,
        ),

        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),

        use_best_model=True,
    )

    elapsed = (
        time.time()
        - start
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    pred = (
        model
        .predict_proba(
            X_val
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_val,
        pred,
    )

    best_iteration = (
        model.get_best_iteration()
    )

    # -----------------------------------------------------
    # 출력
    # -----------------------------------------------------

    print()
    print("-" * 90)

    print(
        f"Brier           : "
        f"{metrics['brier']:.8f}"
    )

    print(
        f"ROC-AUC         : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"LogLoss         : "
        f"{metrics['logloss']:.8f}"
    )

    print(
        f"Best iteration  : "
        f"{best_iteration}"
    )

    print(
        f"Prediction mean : "
        f"{metrics['prediction_mean']:.6f}"
    )

    print(
        f"Prediction std  : "
        f"{metrics['prediction_std']:.6f}"
    )

    print(
        f"Elapsed         : "
        f"{elapsed:.1f} sec"
    )

    print("-" * 90)

    # -----------------------------------------------------
    # Prediction 저장
    # -----------------------------------------------------

    window_dir = (
        OUTPUT_DIR
        / key
    )

    window_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df = pd.DataFrame({
        ID: val_fold[ID].to_numpy(),
        TARGET: y_val,
        "prediction": pred,
    })

    pred_df.to_csv(
        window_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 결과 정리
    # -----------------------------------------------------

    result = {
        "experiment": key,

        "train_window": (
            config["description"]
        ),

        "train_seasons": ",".join(
            map(str, train_seasons)
        ),

        "n_train": int(
            len(train_fold)
        ),

        "train_target_mean": float(
            y_train.mean()
        ),

        "valid_target_mean": float(
            y_val.mean()
        ),

        "brier": (
            metrics["brier"]
        ),

        "roc_auc": (
            metrics["roc_auc"]
        ),

        "logloss": (
            metrics["logloss"]
        ),

        "best_iteration": (
            int(best_iteration)
        ),

        "prediction_mean": (
            metrics[
                "prediction_mean"
            ]
        ),

        "prediction_std": (
            metrics[
                "prediction_std"
            ]
        ),

        "elapsed_seconds": float(
            elapsed
        ),
    }

    # -----------------------------------------------------
    # 메모리 정리
    # -----------------------------------------------------

    del model
    del X_train
    del X_val
    del pred

    gc.collect()

    return result


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 90)
    print(
        "LG Aimers - "
        "Fold 3 Training Window Ablation"
    )
    print("=" * 90)

    # -----------------------------------------------------
    # 기존 결과 덮어쓰기 방지
    # -----------------------------------------------------

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "\n이미 결과 폴더가 존재합니다.\n"
            f"{OUTPUT_DIR}\n\n"
            "다시 실행하려면 기존 폴더를 삭제하거나 "
            "OUTPUT_DIR 이름을 변경하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # A와 동일하게 test.csv 기준 전체 Feature 사용
    # -----------------------------------------------------

    test_cols = (
        pd.read_csv(
            TEST_PATH,
            encoding="utf-8-sig",
            nrows=0,
        )
        .columns
        .tolist()
    )

    features = [
        col
        for col in test_cols
        if col != ID
    ]

    print(
        f"Feature 수 : "
        f"{len(features)}"
    )

    # -----------------------------------------------------
    # Train 로드
    # -----------------------------------------------------

    required_cols = list(
        dict.fromkeys(
            [ID]
            + features
            + [TARGET]
        )
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=required_cols,
    )

    # -----------------------------------------------------
    # Validation = 2023 고정
    # -----------------------------------------------------

    val_fold = (
        train[
            train["season"]
            == VALIDATION_SEASON
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Validation rows : "
        f"{len(val_fold):,}"
    )

    print(
        f"2023 Target Mean: "
        f"{val_fold[TARGET].mean():.6f}"
    )

    # -----------------------------------------------------
    # 4개 Training Window 실행
    # -----------------------------------------------------

    results = []

    for key, config in (
        WINDOWS.items()
    ):

        result = run_window(
            key=key,
            config=config,
            train=train,
            val_fold=val_fold,
            features=features,
        )

        results.append(result)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    # T1 기준 Brier
    t1_brier = float(
        summary.loc[
            summary["experiment"]
            == "T1",
            "brier",
        ].iloc[0]
    )

    # 양수 = T1보다 개선
    summary[
        "brier_improvement_vs_T1"
    ] = (
        t1_brier
        - summary["brier"]
    )

    summary[
        "brier_rank"
    ] = (
        summary["brier"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # 저장
    # -----------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 보기 좋은 최종 출력
    # -----------------------------------------------------

    display_cols = [
        "experiment",
        "train_window",
        "n_train",
        "train_target_mean",
        "brier",
        "brier_improvement_vs_T1",
        "roc_auc",
        "logloss",
        "best_iteration",
        "prediction_mean",
        "prediction_std",
        "elapsed_seconds",
    ]

    print()
    print()
    print("=" * 140)

    print(
        "Training Window Ablation 최종 결과"
    )

    print("=" * 140)

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print("=" * 140)

    print()

    print(
        "※ brier_improvement_vs_T1 > 0 "
        "= T1(2019~2022)보다 개선"
    )

    print()

    print(
        "결과 저장:"
    )

    print(
        OUTPUT_DIR
        / "summary.csv"
    )


if __name__ == "__main__":
    main()