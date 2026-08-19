# expanding_window_abef_final.py
#
# LG Aimers - A / B / E / F
# Final Expanding Window Generalization Validation
#
# Fold 1 : Train 2019~2020 -> Validation 2021
# Fold 2 : Train 2019~2021 -> Validation 2022
# Fold 3 : Train 2019~2022 -> Validation 2023
# Fold 4 : Train 2019~2023 -> Validation 2024
#
# Experiment
# A : 전체 47개 Feature
# B : batter_id 제거
# E : 불안정 middle/batter 통계 제거
# F : 경기 상황 + 안정적 투수 통계 제한 모델
#
# 최종 기록:
# - Fold별 Brier
# - Fold별 ROC-AUC
# - Fold별 LogLoss
# - Fold별 Best Iteration
# - Prediction Mean / Std
# - 각 Fold에서 A 대비 Brier 개선량
# - Mean Fold Brier
# - Pooled OOF Brier  ★ 최종 비교에서 중요
#
# 결과:
# validation/expanding_abef_final/
# ├── fold_results.csv
# ├── final_summary.csv
# ├── oof_predictions.csv
# └── 각 experiment/fold prediction


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
    / "expanding_abef_final"
)

ID = "row_id"
TARGET = "control_success"


# =========================================================
# Expanding Window
# =========================================================

FOLDS = {

    "Fold1": {
        "train_seasons": [2019, 2020],
        "valid_season": 2021,
    },

    "Fold2": {
        "train_seasons": [2019, 2020, 2021],
        "valid_season": 2022,
    },

    "Fold3": {
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_season": 2023,
    },

    "Fold4": {
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "valid_season": 2024,
    },
}


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
# Experiment E
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
# 앞서 Fold3/Fold4에서 사용한 EXACT SAME feature set
# =========================================================

F_FEATURES = [

    # 시간 / 경기 기본 정보
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",

    # Count
    "balls_before",
    "strikes_before",
    "outs_before",

    # 점수 상황
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",

    # 주자 상황
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",

    # 경기 중요도
    "home_win_expectancy",
    "away_win_expectancy",
    "li",

    # 선수 기본 속성
    "pitcher_hand",
    "batter_hand",

    # 안정적 투수 통계
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]


# =========================================================
# Feature 전처리
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
    }


# =========================================================
# Experiment 구성
# =========================================================

def build_experiments(
    base_features,
):

    missing_e = [
        col
        for col in E_REMOVE_FEATURES
        if col not in base_features
    ]

    if missing_e:
        raise KeyError(
            "E Feature 없음:\n"
            f"{missing_e}"
        )

    missing_f = [
        col
        for col in F_FEATURES
        if col not in base_features
    ]

    if missing_f:
        raise KeyError(
            "F Feature 없음:\n"
            f"{missing_f}"
        )

    return {

        "A": {
            "description": "전체 Feature",
            "features": base_features.copy(),
        },

        "B": {
            "description": "batter_id 제거",

            "features": [
                col
                for col in base_features
                if col != "batter_id"
            ],
        },

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

        "F": {
            "description": (
                "경기 상황 + 안정적 투수 통계 제한"
            ),

            "features": (
                F_FEATURES.copy()
            ),
        },
    }


# =========================================================
# 단일 Fold / Experiment 실행
# =========================================================

def run_one(
    experiment_key,
    experiment_config,
    fold_name,
    fold_config,
    train,
):

    features = (
        experiment_config["features"]
    )

    train_seasons = (
        fold_config["train_seasons"]
    )

    valid_season = (
        fold_config["valid_season"]
    )

    # -----------------------------------------------------
    # Split
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

    valid_fold = (
        train[
            train["season"]
            == valid_season
        ]
        .copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Cat cols
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

    X_valid = prepare_features(
        valid_fold[features],
        cat_cols,
    )

    y_valid = (
        valid_fold[TARGET]
        .to_numpy()
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print()
    print("=" * 100)

    print(
        f"{experiment_key} | "
        f"{fold_name}"
    )

    print(
        f"Train {train_seasons} "
        f"-> Valid {valid_season}"
    )

    print(
        f"Feature 수: {len(features)}"
    )

    print("=" * 100)

    # -----------------------------------------------------
    # 매 Fold마다 새 모델
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
            X_valid,
            y_valid,
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
            X_valid
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_valid,
        pred,
    )

    best_iteration = int(
        model.get_best_iteration()
    )

    # -----------------------------------------------------
    # 출력
    # -----------------------------------------------------

    print(
        f"Brier          : "
        f"{metrics['brier']:.8f}"
    )

    print(
        f"ROC-AUC        : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"LogLoss        : "
        f"{metrics['logloss']:.8f}"
    )

    print(
        f"Best iteration : "
        f"{best_iteration}"
    )

    print(
        f"Pred mean      : "
        f"{metrics['prediction_mean']:.6f}"
    )

    print(
        f"Pred std       : "
        f"{metrics['prediction_std']:.6f}"
    )

    print(
        f"Elapsed        : "
        f"{elapsed:.1f}s"
    )

    # -----------------------------------------------------
    # Prediction 저장
    # -----------------------------------------------------

    pred_dir = (
        OUTPUT_DIR
        / experiment_key
        / fold_name
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df = pd.DataFrame({
        ID: valid_fold[ID].to_numpy(),
        TARGET: y_valid,
        "prediction": pred,
        "experiment": experiment_key,
        "fold": fold_name,
        "validation_season": valid_season,
    })

    pred_df.to_csv(
        pred_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    result = {

        "experiment": (
            experiment_key
        ),

        "fold": (
            fold_name
        ),

        "train_seasons": (
            ",".join(
                map(
                    str,
                    train_seasons,
                )
            )
        ),

        "validation_season": (
            valid_season
        ),

        "n_features": int(
            len(features)
        ),

        "n_train": int(
            len(train_fold)
        ),

        "n_valid": int(
            len(valid_fold)
        ),

        "train_target_mean": float(
            y_train.mean()
        ),

        "valid_target_mean": float(
            y_valid.mean()
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
    del X_valid

    gc.collect()

    return result, pred_df


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 100)

    print(
        "LG Aimers - "
        "Final Expanding Window A/B/E/F"
    )

    print("=" * 100)

    # -----------------------------------------------------
    # 기존 결과 덮어쓰기 방지
    # -----------------------------------------------------

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "\n이미 결과 폴더가 존재합니다.\n"
            f"{OUTPUT_DIR}\n\n"
            "기존 폴더를 삭제하거나 "
            "OUTPUT_DIR을 변경하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Base Feature
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
        f"Base Feature 수: "
        f"{len(base_features)}"
    )

    # -----------------------------------------------------
    # Data
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
    # Experiment
    # -----------------------------------------------------

    experiments = (
        build_experiments(
            base_features
        )
    )

    print()
    print("Experiments")

    for key, config in (
        experiments.items()
    ):

        print(
            f"{key}: "
            f"{config['description']} "
            f"({len(config['features'])} features)"
        )

    # -----------------------------------------------------
    # 전체 실행
    #
    # A Fold1~4
    # B Fold1~4
    # E Fold1~4
    # F Fold1~4
    #
    # 총 16회
    # -----------------------------------------------------

    all_results = []
    all_predictions = []

    for experiment_key, experiment_config in (
        experiments.items()
    ):

        for fold_name, fold_config in (
            FOLDS.items()
        ):

            result, pred_df = run_one(
                experiment_key=experiment_key,
                experiment_config=experiment_config,
                fold_name=fold_name,
                fold_config=fold_config,
                train=train,
            )

            all_results.append(
                result
            )

            all_predictions.append(
                pred_df
            )

    # =====================================================
    # Fold Results
    # =====================================================

    fold_results = pd.DataFrame(
        all_results
    )

    # -----------------------------------------------------
    # 각 Fold의 A Brier 기준
    # -----------------------------------------------------

    a_brier_by_fold = (
        fold_results[
            fold_results[
                "experiment"
            ] == "A"
        ][
            [
                "fold",
                "brier",
            ]
        ]
        .rename(
            columns={
                "brier": "a_brier"
            }
        )
    )

    fold_results = (
        fold_results
        .merge(
            a_brier_by_fold,
            on="fold",
            how="left",
        )
    )

    # 양수 = A보다 개선
    fold_results[
        "brier_improvement_vs_A"
    ] = (
        fold_results["a_brier"]
        - fold_results["brier"]
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # OOF Prediction
    # =====================================================

    oof_all = pd.concat(
        all_predictions,
        axis=0,
        ignore_index=True,
    )

    oof_all.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # Final Summary
    # =====================================================

    final_rows = []

    for experiment_key in (
        experiments.keys()
    ):

        exp_fold = (
            fold_results[
                fold_results[
                    "experiment"
                ]
                == experiment_key
            ]
            .copy()
        )

        exp_oof = (
            oof_all[
                oof_all[
                    "experiment"
                ]
                == experiment_key
            ]
            .copy()
        )

        y_oof = (
            exp_oof[TARGET]
            .to_numpy()
        )

        pred_oof = (
            exp_oof["prediction"]
            .to_numpy()
        )

        # -------------------------------------------------
        # Pooled OOF
        # -------------------------------------------------

        pooled_brier = float(
            np.mean(
                (
                    pred_oof
                    - y_oof
                ) ** 2
            )
        )

        pooled_auc = float(
            roc_auc_score(
                y_oof,
                pred_oof,
            )
        )

        pooled_logloss = float(
            log_loss(
                y_oof,
                pred_oof,
                labels=[0, 1],
            )
        )

        # -------------------------------------------------
        # Fold별 Brier
        # -------------------------------------------------

        fold_briers = {}

        for fold_name in FOLDS:

            value = float(
                exp_fold.loc[
                    exp_fold["fold"]
                    == fold_name,
                    "brier",
                ].iloc[0]
            )

            fold_briers[
                f"{fold_name}_brier"
            ] = value

        # -------------------------------------------------
        # Fold Win Count vs A
        # -------------------------------------------------

        if experiment_key == "A":

            win_count = 0

        else:

            win_count = int(
                (
                    exp_fold[
                        "brier_improvement_vs_A"
                    ] > 0
                ).sum()
            )

        final_rows.append({

            "experiment": (
                experiment_key
            ),

            "n_features": int(
                exp_fold[
                    "n_features"
                ].iloc[0]
            ),

            **fold_briers,

            # Fold 단순 평균
            "mean_fold_brier": float(
                exp_fold[
                    "brier"
                ].mean()
            ),

            # 가장 중요한 pooled OOF
            "pooled_oof_brier": (
                pooled_brier
            ),

            "pooled_oof_auc": (
                pooled_auc
            ),

            "pooled_oof_logloss": (
                pooled_logloss
            ),

            "mean_best_iteration": float(
                exp_fold[
                    "best_iteration"
                ].mean()
            ),

            "wins_vs_A": (
                win_count
            ),
        })

    final_summary = pd.DataFrame(
        final_rows
    )

    # -----------------------------------------------------
    # A 대비 pooled 개선량
    # -----------------------------------------------------

    a_pooled = float(
        final_summary.loc[
            final_summary[
                "experiment"
            ] == "A",
            "pooled_oof_brier",
        ].iloc[0]
    )

    final_summary[
        "pooled_brier_improvement_vs_A"
    ] = (
        a_pooled
        - final_summary[
            "pooled_oof_brier"
        ]
    )

    final_summary[
        "pooled_rank"
    ] = (
        final_summary[
            "pooled_oof_brier"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    final_summary.to_csv(
        OUTPUT_DIR
        / "final_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # Fold 결과 출력
    # =====================================================

    print()
    print()
    print("=" * 150)

    print(
        "Fold별 결과"
    )

    print("=" * 150)

    fold_display_cols = [
        "experiment",
        "fold",
        "validation_season",
        "brier",
        "brier_improvement_vs_A",
        "roc_auc",
        "logloss",
        "best_iteration",
        "prediction_mean",
        "prediction_std",
    ]

    print(
        fold_results[
            fold_display_cols
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # 최종 결과 출력
    # =====================================================

    print()
    print()
    print("=" * 170)

    print(
        "FINAL Expanding Window Summary"
    )

    print("=" * 170)

    final_display_cols = [
        "experiment",
        "n_features",

        "Fold1_brier",
        "Fold2_brier",
        "Fold3_brier",
        "Fold4_brier",

        "mean_fold_brier",

        "pooled_oof_brier",
        "pooled_brier_improvement_vs_A",

        "pooled_oof_auc",
        "pooled_oof_logloss",

        "wins_vs_A",
        "mean_best_iteration",
        "pooled_rank",
    ]

    print(
        final_summary[
            final_display_cols
        ].to_string(
            index=False
        )
    )

    print("=" * 170)

    print()

    print(
        "※ pooled_brier_improvement_vs_A > 0 "
        "= 전체 Expanding Window에서 A보다 개선"
    )

    print(
        "※ wins_vs_A = 4개 Fold 중 A보다 "
        "Brier가 좋았던 Fold 수"
    )

    print(
        "※ 최종 판단에서는 pooled_oof_brier를 "
        "우선 확인"
    )

    print()

    print("저장 완료:")

    print(
        OUTPUT_DIR
        / "fold_results.csv"
    )

    print(
        OUTPUT_DIR
        / "final_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "oof_predictions.csv"
    )


if __name__ == "__main__":
    main()