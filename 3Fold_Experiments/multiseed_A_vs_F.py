# multiseed_A_vs_F.py
#
# LG Aimers
# A vs F Multi-Seed Expanding Window Stability Test
#
# Seeds:
#   7, 42, 123, 999, 2026
#
# Expanding Window:
#   Fold1: 2019~2020 -> 2021
#   Fold2: 2019~2021 -> 2022
#   Fold3: 2019~2022 -> 2023
#   Fold4: 2019~2023 -> 2024
#
# A:
#   기존 전체 47 Feature
#
# F:
#   경기 상황 + 안정적 투수 통계 중심 29 Feature
#
# 핵심 판정:
#   delta = A Brier - F Brier
#
#   delta > 0 : F 승리
#   delta < 0 : A 승리
#
# 저장:
# validation/multiseed_A_vs_F/
# ├── fold_seed_results.csv
# ├── seed_summary.csv
# ├── season_stability.csv
# └── final_summary.csv


import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier


# =========================================================
# 기본 설정
# =========================================================

DATA_DIR = Path("./data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = (
    Path("./validation")
    / "multiseed_A_vs_F"
)

ID = "row_id"
TARGET = "control_success"


# =========================================================
# Multi Seed
# =========================================================

SEEDS = [
    7,
    42,
    123,
    999,
    2026,
]


# =========================================================
# Expanding Window
# =========================================================

FOLDS = {

    "Fold1": {
        "train_seasons": [
            2019,
            2020,
        ],
        "valid_season": 2021,
    },

    "Fold2": {
        "train_seasons": [
            2019,
            2020,
            2021,
        ],
        "valid_season": 2022,
    },

    "Fold3": {
        "train_seasons": [
            2019,
            2020,
            2021,
            2022,
        ],
        "valid_season": 2023,
    },

    "Fold4": {
        "train_seasons": [
            2019,
            2020,
            2021,
            2022,
            2023,
        ],
        "valid_season": 2024,
    },
}


# =========================================================
# 기존 A 모델 Categorical Feature
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
# Experiment F
#
# 앞선 Fold3 / Fold4 / 4-Fold 실험에서 사용한
# EXACT SAME 29 Features
# =========================================================

F_FEATURES = [

    # -----------------------------------------------------
    # 시간 / 경기 기본 정보
    # -----------------------------------------------------

    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",

    # -----------------------------------------------------
    # Count
    # -----------------------------------------------------

    "balls_before",
    "strikes_before",
    "outs_before",

    # -----------------------------------------------------
    # 점수 상황
    # -----------------------------------------------------

    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",

    # -----------------------------------------------------
    # 주자 상황
    # -----------------------------------------------------

    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",

    # -----------------------------------------------------
    # 경기 중요도
    # -----------------------------------------------------

    "home_win_expectancy",
    "away_win_expectancy",
    "li",

    # -----------------------------------------------------
    # 선수 기본 속성
    # -----------------------------------------------------

    "pitcher_hand",
    "batter_hand",

    # -----------------------------------------------------
    # 안정적인 투수 과거 통계
    # -----------------------------------------------------

    "asof_pitcher_n",
    "asof_pitcher_success_rate",

    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]


# =========================================================
# CatBoost 고정 설정
#
# random_seed만 Multi-Seed에서 변경
# =========================================================

BASE_MODEL_PARAMS = {

    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,

    "loss_function": "Logloss",

    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,

    "max_ctr_complexity": 1,

    "thread_count": -1,

    # 출력 최소화
    "verbose": False,

    "allow_writing_files": False,
}


EARLY_STOPPING_ROUNDS = 80


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
# Brier
# =========================================================

def calculate_brier(
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

    return float(
        np.mean(
            (pred - y_true) ** 2
        )
    )


# =========================================================
# 단일 모델 실행
# =========================================================

def train_and_evaluate(
    train_fold,
    valid_fold,
    features,
    seed,
):

    # -----------------------------------------------------
    # 해당 Feature Set에서 categorical만 남김
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
    # Seed만 변경
    # -----------------------------------------------------

    params = (
        BASE_MODEL_PARAMS.copy()
    )

    params[
        "random_seed"
    ] = seed

    # -----------------------------------------------------
    # 매번 완전히 새로운 모델
    # -----------------------------------------------------

    model = CatBoostClassifier(
        **params
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

    # -----------------------------------------------------
    # Metric
    # -----------------------------------------------------

    brier = calculate_brier(
        y_valid,
        pred,
    )

    best_iteration = int(
        model.get_best_iteration()
    )

    pred_mean = float(
        np.mean(pred)
    )

    pred_std = float(
        np.std(pred)
    )

    result = {

        "brier": brier,

        "best_iteration": (
            best_iteration
        ),

        "prediction_mean": (
            pred_mean
        ),

        "prediction_std": (
            pred_std
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
    del pred

    gc.collect()

    return result


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 90)

    print(
        "LG Aimers"
    )

    print(
        "A vs F Multi-Seed "
        "Expanding Window Stability"
    )

    print("=" * 90)

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Total fits: "
        f"{len(SEEDS) * 2 * len(FOLDS)}"
    )

    print()

    # -----------------------------------------------------
    # 기존 결과 덮어쓰기 방지
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

    # =====================================================
    # A Feature = test.csv 전체 Feature
    # =====================================================

    test_cols = (
        pd.read_csv(
            TEST_PATH,
            encoding="utf-8-sig",
            nrows=0,
        )
        .columns
        .tolist()
    )

    A_FEATURES = [
        col
        for col in test_cols
        if col != ID
    ]

    print(
        f"A Features: "
        f"{len(A_FEATURES)}"
    )

    print(
        f"F Features: "
        f"{len(F_FEATURES)}"
    )

    # -----------------------------------------------------
    # F Feature 존재 확인
    # -----------------------------------------------------

    missing_f = [
        col
        for col in F_FEATURES
        if col not in A_FEATURES
    ]

    if missing_f:

        raise KeyError(
            "\nF Feature가 현재 데이터에 없습니다.\n"
            f"{missing_f}"
        )

    # =====================================================
    # Train 데이터 로드
    # =====================================================

    required_cols = list(
        dict.fromkeys(
            [ID]
            + A_FEATURES
            + [TARGET]
        )
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=required_cols,
    )

    # =====================================================
    # 전체 실험
    # =====================================================

    fold_rows = []

    for seed in SEEDS:

        print()
        print()
        print("#" * 90)

        print(
            f"SEED = {seed}"
        )

        print("#" * 90)

        for (
            fold_name,
            fold_config,
        ) in FOLDS.items():

            train_seasons = (
                fold_config[
                    "train_seasons"
                ]
            )

            valid_season = (
                fold_config[
                    "valid_season"
                ]
            )

            # -------------------------------------------------
            # Fold Split
            # -------------------------------------------------

            train_fold = (
                train[
                    train["season"].isin(
                        train_seasons
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            valid_fold = (
                train[
                    train["season"]
                    == valid_season
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            print()
            print("=" * 90)

            print(
                f"Seed {seed} | "
                f"{fold_name}"
            )

            print(
                f"Train {train_seasons} "
                f"-> Valid {valid_season}"
            )

            print("=" * 90)

            # =================================================
            # A
            # =================================================

            print(
                "A training..."
            )

            result_A = (
                train_and_evaluate(

                    train_fold=(
                        train_fold
                    ),

                    valid_fold=(
                        valid_fold
                    ),

                    features=(
                        A_FEATURES
                    ),

                    seed=seed,
                )
            )

            # =================================================
            # F
            # =================================================

            print(
                "F training..."
            )

            result_F = (
                train_and_evaluate(

                    train_fold=(
                        train_fold
                    ),

                    valid_fold=(
                        valid_fold
                    ),

                    features=(
                        F_FEATURES
                    ),

                    seed=seed,
                )
            )

            # =================================================
            # Paired Delta
            #
            # 양수 = F가 좋음
            # =================================================

            delta = (
                result_A["brier"]
                - result_F["brier"]
            )

            winner = (
                "F"
                if delta > 0
                else "A"
            )

            print()

            print(
                f"A Brier : "
                f"{result_A['brier']:.8f}"
            )

            print(
                f"F Brier : "
                f"{result_F['brier']:.8f}"
            )

            print(
                f"Delta   : "
                f"{delta:+.8f}"
            )

            print(
                f"Winner  : "
                f"{winner}"
            )

            # =================================================
            # 기록
            # =================================================

            fold_rows.append({

                "seed": seed,

                "fold": (
                    fold_name
                ),

                "validation_season": (
                    valid_season
                ),

                "n_train": int(
                    len(train_fold)
                ),

                "n_valid": int(
                    len(valid_fold)
                ),

                "A_brier": (
                    result_A["brier"]
                ),

                "F_brier": (
                    result_F["brier"]
                ),

                # + = F improvement
                "delta_A_minus_F": (
                    delta
                ),

                "winner": (
                    winner
                ),

                "A_best_iteration": (
                    result_A[
                        "best_iteration"
                    ]
                ),

                "F_best_iteration": (
                    result_F[
                        "best_iteration"
                    ]
                ),

                "A_prediction_mean": (
                    result_A[
                        "prediction_mean"
                    ]
                ),

                "F_prediction_mean": (
                    result_F[
                        "prediction_mean"
                    ]
                ),

                "A_prediction_std": (
                    result_A[
                        "prediction_std"
                    ]
                ),

                "F_prediction_std": (
                    result_F[
                        "prediction_std"
                    ]
                ),

                "A_elapsed_seconds": (
                    result_A[
                        "elapsed_seconds"
                    ]
                ),

                "F_elapsed_seconds": (
                    result_F[
                        "elapsed_seconds"
                    ]
                ),
            })

            # -------------------------------------------------
            # Fold Data도 정리
            # -------------------------------------------------

            del train_fold
            del valid_fold

            gc.collect()


    # =====================================================
    # Fold × Seed 결과
    # =====================================================

    fold_results = pd.DataFrame(
        fold_rows
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "fold_seed_results.csv",

        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # Seed별 Pooled Brier
    #
    # Fold row 수로 가중평균.
    #
    # 각 validation row가 한 번씩만 등장하므로
    # 전체 OOF pooled Brier와 정확히 동일.
    # =====================================================

    seed_rows = []

    for seed in SEEDS:

        part = (
            fold_results[
                fold_results["seed"]
                == seed
            ]
            .copy()
        )

        weights = (
            part["n_valid"]
            .to_numpy(
                dtype=float
            )
        )

        A_briers = (
            part["A_brier"]
            .to_numpy()
        )

        F_briers = (
            part["F_brier"]
            .to_numpy()
        )

        pooled_A = float(
            np.average(
                A_briers,
                weights=weights,
            )
        )

        pooled_F = float(
            np.average(
                F_briers,
                weights=weights,
            )
        )

        pooled_delta = (
            pooled_A
            - pooled_F
        )

        fold_win_count = int(
            (
                part[
                    "delta_A_minus_F"
                ] > 0
            ).sum()
        )

        seed_rows.append({

            "seed": seed,

            "pooled_A_brier": (
                pooled_A
            ),

            "pooled_F_brier": (
                pooled_F
            ),

            # + = F better
            "pooled_delta_A_minus_F": (
                pooled_delta
            ),

            "winner": (
                "F"
                if pooled_delta > 0
                else "A"
            ),

            "F_fold_wins": (
                fold_win_count
            ),

            "A_fold_wins": (
                len(FOLDS)
                - fold_win_count
            ),

            "A_mean_best_iteration": float(
                part[
                    "A_best_iteration"
                ].mean()
            ),

            "F_mean_best_iteration": float(
                part[
                    "F_best_iteration"
                ].mean()
            ),
        })


    seed_summary = pd.DataFrame(
        seed_rows
    )

    seed_summary.to_csv(
        OUTPUT_DIR
        / "seed_summary.csv",

        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # Season Stability
    #
    # 각 Validation year별로
    # 5 Seed delta 평균 / 표준편차 / 승수 계산
    # =====================================================

    season_rows = []

    for (
        fold_name,
        fold_config,
    ) in FOLDS.items():

        valid_season = (
            fold_config[
                "valid_season"
            ]
        )

        part = (
            fold_results[
                fold_results["fold"]
                == fold_name
            ]
            .copy()
        )

        deltas = (
            part[
                "delta_A_minus_F"
            ]
            .to_numpy()
        )

        season_rows.append({

            "fold": (
                fold_name
            ),

            "validation_season": (
                valid_season
            ),

            "mean_delta_A_minus_F": float(
                np.mean(deltas)
            ),

            "std_delta": float(
                np.std(
                    deltas,
                    ddof=1,
                )
            ),

            "F_seed_wins": int(
                np.sum(
                    deltas > 0
                )
            ),

            "A_seed_wins": int(
                np.sum(
                    deltas < 0
                )
            ),

            "mean_A_brier": float(
                part[
                    "A_brier"
                ].mean()
            ),

            "mean_F_brier": float(
                part[
                    "F_brier"
                ].mean()
            ),
        })


    season_stability = (
        pd.DataFrame(
            season_rows
        )
    )

    season_stability.to_csv(
        OUTPUT_DIR
        / "season_stability.csv",

        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 최종 Multi-Seed Summary
    # =====================================================

    pooled_deltas = (
        seed_summary[
            "pooled_delta_A_minus_F"
        ]
        .to_numpy()
    )

    final_summary = pd.DataFrame([{

        "n_seeds": int(
            len(SEEDS)
        ),

        "mean_A_pooled_brier": float(
            seed_summary[
                "pooled_A_brier"
            ].mean()
        ),

        "mean_F_pooled_brier": float(
            seed_summary[
                "pooled_F_brier"
            ].mean()
        ),

        # + = F better
        "mean_delta_A_minus_F": float(
            np.mean(
                pooled_deltas
            )
        ),

        "delta_std": float(
            np.std(
                pooled_deltas,
                ddof=1,
            )
        ),

        "delta_min": float(
            np.min(
                pooled_deltas
            )
        ),

        "delta_max": float(
            np.max(
                pooled_deltas
            )
        ),

        "F_seed_wins": int(
            np.sum(
                pooled_deltas > 0
            )
        ),

        "A_seed_wins": int(
            np.sum(
                pooled_deltas < 0
            )
        ),

        "F_seed_win_rate": float(
            np.mean(
                pooled_deltas > 0
            )
        ),
    }])


    final_summary.to_csv(
        OUTPUT_DIR
        / "final_summary.csv",

        index=False,
        encoding="utf-8-sig",
    )


    # =====================================================
    # 출력
    # =====================================================

    print()
    print()
    print("=" * 120)

    print(
        "SEED별 POOLED 결과"
    )

    print("=" * 120)

    print(
        seed_summary[
            [
                "seed",
                "pooled_A_brier",
                "pooled_F_brier",
                "pooled_delta_A_minus_F",
                "winner",
                "F_fold_wins",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print()
    print("=" * 120)

    print(
        "SEASON STABILITY"
    )

    print("=" * 120)

    print(
        season_stability.to_string(
            index=False
        )
    )


    print()
    print()
    print("=" * 120)

    print(
        "FINAL MULTI-SEED SUMMARY"
    )

    print("=" * 120)

    print(
        final_summary.to_string(
            index=False
        )
    )

    print("=" * 120)

    print()

    print(
        "※ delta = A Brier - F Brier"
    )

    print(
        "※ delta > 0 : F 개선"
    )

    print(
        "※ delta < 0 : A 개선"
    )

    print()

    print(
        "결과 저장:"
    )

    print(
        OUTPUT_DIR
        / "fold_seed_results.csv"
    )

    print(
        OUTPUT_DIR
        / "seed_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "season_stability.csv"
    )

    print(
        OUTPUT_DIR
        / "final_summary.csv"
    )


if __name__ == "__main__":
    main()