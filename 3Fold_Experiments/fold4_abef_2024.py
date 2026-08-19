# fold4_abef_2024.py
#
# LG Aimers - 2024 Generalization Validation
#
# Train      : 2019~2023
# Validation : 2024
#
# 2023 Fold 3에서 유망했던 A / B / E / F를
# 동일한 조건으로 2024에서 재검증한다.
#
# 결과:
# validation/fold4_abef_2024/
# ├── summary.csv
# ├── A/pred.csv
# ├── B/pred.csv
# ├── E/pred.csv
# └── F/pred.csv


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
    / "fold4_abef_2024"
)

ID = "row_id"
TARGET = "control_success"

TRAIN_SEASONS = [
    2019,
    2020,
    2021,
    2022,
    2023,
]

VALIDATION_SEASON = 2024


# =========================================================
# 기존 CatBoost categorical feature
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
# 기존 CatBoost 설정 그대로
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
# Experiment E
#
# PDF에서 지정한 우선 제거 후보
# =========================================================

E_REMOVE_FEATURES = [
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]


# =========================================================
# Experiment F
#
# 주의:
# PDF는 F를
# "경기 상황 + 안정적 투수 통계 중심"
# 으로만 정의하고 정확한 column list는 제시하지 않았다.
#
# 아래는 우리가 Fold 3에서 실제 사용했던
# F Feature Set을 그대로 재사용한다.
# =========================================================

F_FEATURES = [

    # -------------------------
    # 시간 / 경기 기본 정보
    # -------------------------

    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",

    # -------------------------
    # Count
    # -------------------------

    "balls_before",
    "strikes_before",
    "outs_before",

    # -------------------------
    # 점수 상황
    # -------------------------

    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",

    # -------------------------
    # 주자 상황
    # -------------------------

    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",

    # -------------------------
    # 경기 중요도
    # -------------------------

    "home_win_expectancy",
    "away_win_expectancy",
    "li",

    # -------------------------
    # 선수 기본 속성
    # -------------------------

    "pitcher_hand",
    "batter_hand",

    # -------------------------
    # 안정적인 투수 통계
    # -------------------------

    "asof_pitcher_n",
    "asof_pitcher_success_rate",

    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]


# =========================================================
# 전처리
# =========================================================

def prepare_features(
    df,
    cat_cols,
):

    df = df.copy()

    for col in cat_cols:

        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Metric
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
# Experiment 정의
# =========================================================

def build_experiments(
    base_features,
):

    # -----------------------------------------------------
    # E feature 존재 확인
    # -----------------------------------------------------

    missing_e = [
        col
        for col in E_REMOVE_FEATURES
        if col not in base_features
    ]

    if missing_e:

        raise KeyError(
            "Experiment E Feature가 없습니다:\n"
            f"{missing_e}"
        )

    # -----------------------------------------------------
    # F feature 존재 확인
    # -----------------------------------------------------

    missing_f = [
        col
        for col in F_FEATURES
        if col not in base_features
    ]

    if missing_f:

        raise KeyError(
            "Experiment F Feature가 없습니다:\n"
            f"{missing_f}"
        )

    experiments = {

        # =================================================
        # A
        # =================================================

        "A": {
            "description": "전체 Feature",
            "features": base_features.copy(),
        },

        # =================================================
        # B
        # batter_id 제거
        # =================================================

        "B": {
            "description": "batter_id 제거",

            "features": [
                col
                for col in base_features
                if col != "batter_id"
            ],
        },

        # =================================================
        # E
        # 불안정 통계 제거
        # =================================================

        "E": {
            "description": (
                "불안정 middle/batter 통계 제거"
            ),

            "features": [
                col
                for col in base_features
                if col not in E_REMOVE_FEATURES
            ],
        },

        # =================================================
        # F
        # 제한 Feature Model
        # =================================================

        "F": {
            "description": (
                "경기 상황 + 안정적 투수 통계 제한 모델"
            ),

            "features": (
                F_FEATURES.copy()
            ),
        },
    }

    return experiments


# =========================================================
# 단일 Experiment 실행
# =========================================================

def run_experiment(
    key,
    config,
    train_fold,
    val_fold,
):

    features = (
        config["features"]
    )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    print()
    print("=" * 100)

    print(
        f"Experiment {key}"
    )

    print(
        config["description"]
    )

    print("=" * 100)

    print(
        f"Feature 수       : "
        f"{len(features)}"
    )

    print(
        f"Categorical 수   : "
        f"{len(cat_cols)}"
    )

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

    # -----------------------------------------------------
    # 각 실험마다 새 모델
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

    # -----------------------------------------------------
    # Metric
    # -----------------------------------------------------

    metrics = calculate_metrics(
        y_val,
        pred,
    )

    best_iteration = (
        model.get_best_iteration()
    )

    # -----------------------------------------------------
    # 콘솔
    # -----------------------------------------------------

    print()
    print("-" * 100)

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

    print("-" * 100)

    # -----------------------------------------------------
    # Prediction 저장
    # -----------------------------------------------------

    experiment_dir = (
        OUTPUT_DIR / key
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df = pd.DataFrame({
        ID: val_fold[ID].to_numpy(),
        TARGET: y_val,
        "prediction": pred,
    })

    pred_df.to_csv(
        experiment_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    result = {
        "experiment": key,

        "description": (
            config["description"]
        ),

        "n_features": int(
            len(features)
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

        "best_iteration": int(
            best_iteration
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

    print("=" * 100)

    print(
        "LG Aimers - "
        "2024 Generalization Validation"
    )

    print(
        "Train 2019~2023"
    )

    print(
        "Validation 2024"
    )

    print("=" * 100)

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "\n이미 결과 폴더가 존재합니다.\n"
            f"{OUTPUT_DIR}\n\n"
            "기존 폴더를 삭제하거나 "
            "OUTPUT_DIR 이름을 변경하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # 기존 A와 동일하게
    # test.csv 기준 Feature 목록
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

    base_features = [
        col
        for col in test_cols
        if col != ID
    ]

    print(
        f"Base Feature 수 : "
        f"{len(base_features)}"
    )

    # -----------------------------------------------------
    # Train Load
    # -----------------------------------------------------

    required_cols = list(
        dict.fromkeys(
            [ID]
            + base_features
            + [TARGET]
        )
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=required_cols,
    )

    # -----------------------------------------------------
    # Fold 4
    # -----------------------------------------------------

    train_fold = (
        train[
            train["season"].isin(
                TRAIN_SEASONS
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_fold = (
        train[
            train["season"]
            == VALIDATION_SEASON
        ]
        .copy()
        .reset_index(drop=True)
    )

    print()

    print(
        f"Train rows       : "
        f"{len(train_fold):,}"
    )

    print(
        f"Validation rows  : "
        f"{len(val_fold):,}"
    )

    print(
        f"Train target mean: "
        f"{train_fold[TARGET].mean():.6f}"
    )

    print(
        f"Valid target mean: "
        f"{val_fold[TARGET].mean():.6f}"
    )

    # -----------------------------------------------------
    # Experiment
    # -----------------------------------------------------

    experiments = (
        build_experiments(
            base_features
        )
    )

    print()
    print("=" * 100)

    print(
        "실행 Experiment"
    )

    print("=" * 100)

    for key, config in experiments.items():

        print(
            f"{key}: "
            f"{config['description']} "
            f"({len(config['features'])} features)"
        )

    # -----------------------------------------------------
    # A / B / E / F 실행
    # -----------------------------------------------------

    results = []

    for key, config in (
        experiments.items()
    ):

        result = run_experiment(
            key=key,
            config=config,
            train_fold=train_fold,
            val_fold=val_fold,
        )

        results.append(
            result
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # A 기준
    # -----------------------------------------------------

    a_brier = float(
        summary.loc[
            summary["experiment"] == "A",
            "brier",
        ].iloc[0]
    )

    a_auc = float(
        summary.loc[
            summary["experiment"] == "A",
            "roc_auc",
        ].iloc[0]
    )

    # 양수 = 개선
    summary[
        "brier_improvement_vs_A"
    ] = (
        a_brier
        - summary["brier"]
    )

    summary[
        "auc_change_vs_A"
    ] = (
        summary["roc_auc"]
        - a_auc
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
        OUTPUT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 최종 출력
    # -----------------------------------------------------

    display_cols = [
        "experiment",
        "n_features",
        "brier",
        "brier_improvement_vs_A",
        "roc_auc",
        "auc_change_vs_A",
        "logloss",
        "best_iteration",
        "prediction_mean",
        "prediction_std",
        "elapsed_seconds",
    ]

    print()
    print()
    print("=" * 150)

    print(
        "2024 Generalization Validation 최종 결과"
    )

    print("=" * 150)

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print("=" * 150)

    print()

    print(
        "※ brier_improvement_vs_A > 0"
    )

    print(
        "   = 2024에서도 A보다 개선"
    )

    print()

    print(
        "※ auc_change_vs_A > 0"
    )

    print(
        "   = 2024에서도 ranking 개선"
    )

    print()

    print(
        "결과 저장:"
    )

    print(
        OUTPUT_DIR / "summary.csv"
    )


if __name__ == "__main__":
    main()