# fold3_f_restricted.py
#
# LG Aimers - Fold 3 원인 분석
# Experiment F
#
# Train      : 2019~2022
# Validation : 2023
#
# 목적:
# 경기 상황 + 시간적으로 비교적 안정적인 투수 성공률 통계만 남겼을 때
# Fold 3의 best_iteration=0 붕괴 현상이 사라지는지 확인한다.

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

ID = "row_id"
TARGET = "control_success"

TRAIN_SEASONS = [
    2019,
    2020,
    2021,
    2022,
]

VALIDATION_SEASON = 2023


# =========================================================
# Experiment A 결과
#
# 방금 수행한 기준 모델 결과.
# F와 비교하기 위해 사용한다.
# =========================================================

A_BRIER = 0.25001575
A_AUC = 0.515434
A_BEST_ITERATION = 0


# =========================================================
# Experiment F Feature
#
# "경기 상황 + 안정적 투수 통계 중심"
#
# 의도적으로 제외:
# - pitcher_id / batter_id
# - team_id
# - batter 누적 통계
# - middle 계열
# - ball / strike
# - pitchmix 계열
# - reverse_rate
#
# 안정성이 상대적으로 확인된 pitcher success 계열만 유지.
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
    # 투구 직전 Count
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
    # 선수의 비교적 안정적인 속성
    #
    # 선수 ID 자체는 제거
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
# Categorical Feature
# =========================================================

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]


# =========================================================
# 기존 cb_v002와 동일한 CatBoost 설정
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
# Feature 전처리
# =========================================================

def prepare_features(df):

    df = df.copy()

    for col in CAT_COLS:

        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 80)
    print("Experiment F")
    print("경기 상황 + 안정적 투수 통계 제한 모델")
    print("=" * 80)

    print(
        f"Feature 수: {len(F_FEATURES)}"
    )

    print()

    for feature in F_FEATURES:
        print(f"- {feature}")

    print()

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------

    required_cols = list(
        dict.fromkeys(
            [ID]
            + F_FEATURES
            + [TARGET]
        )
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=required_cols,
    )

    # -----------------------------------------------------
    # Feature 존재 확인
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in F_FEATURES
        if feature not in train.columns
    ]

    if missing_features:

        raise KeyError(
            "F Feature가 데이터에 없습니다:\n"
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # Fold 3
    # -----------------------------------------------------

    train_mask = (
        train["season"]
        .isin(TRAIN_SEASONS)
    )

    val_mask = (
        train["season"]
        == VALIDATION_SEASON
    )

    train_fold = (
        train.loc[train_mask]
        .copy()
    )

    val_fold = (
        train.loc[val_mask]
        .copy()
    )

    print("=" * 80)

    print(
        "Train seasons :",
        TRAIN_SEASONS,
    )

    print(
        "Validation    :",
        VALIDATION_SEASON,
    )

    print(
        "Train rows    :",
        len(train_fold),
    )

    print(
        "Valid rows    :",
        len(val_fold),
    )

    print(
        "Train target mean :",
        f"{train_fold[TARGET].mean():.6f}",
    )

    print(
        "Valid target mean :",
        f"{val_fold[TARGET].mean():.6f}",
    )

    print("=" * 80)

    # -----------------------------------------------------
    # X / y
    # -----------------------------------------------------

    X_train = prepare_features(
        train_fold[
            F_FEATURES
        ]
    )

    y_train = (
        train_fold[TARGET]
        .to_numpy()
    )

    X_val = prepare_features(
        val_fold[
            F_FEATURES
        ]
    )

    y_val = (
        val_fold[TARGET]
        .to_numpy()
    )

    # -----------------------------------------------------
    # 새 모델 생성
    # -----------------------------------------------------

    model = CatBoostClassifier(
        **MODEL_PARAMS
    )

    # -----------------------------------------------------
    # 학습
    # -----------------------------------------------------

    start = time.time()

    model.fit(
        X_train,
        y_train,

        cat_features=CAT_COLS,

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

    brier = float(
        np.mean(
            (
                pred
                - y_val
            ) ** 2
        )
    )

    auc = float(
        roc_auc_score(
            y_val,
            pred,
        )
    )

    ll = float(
        log_loss(
            y_val,
            pred,
            labels=[0, 1],
        )
    )

    best_iteration = (
        model
        .get_best_iteration()
    )

    pred_mean = float(
        pred.mean()
    )

    pred_std = float(
        pred.std()
    )

    # -----------------------------------------------------
    # A 대비 변화
    # 양수면 F가 개선
    # -----------------------------------------------------

    brier_improvement = (
        A_BRIER
        - brier
    )

    auc_improvement = (
        auc
        - A_AUC
    )

    # -----------------------------------------------------
    # 결과 출력
    # -----------------------------------------------------

    print()
    print("=" * 80)

    print("Fold 3 Experiment F 결과")

    print("=" * 80)

    print(
        f"Brier             : "
        f"{brier:.8f}"
    )

    print(
        f"A 대비 Brier 개선 : "
        f"{brier_improvement:+.8f}"
    )

    print(
        f"ROC-AUC           : "
        f"{auc:.6f}"
    )

    print(
        f"A 대비 AUC 변화   : "
        f"{auc_improvement:+.6f}"
    )

    print(
        f"LogLoss           : "
        f"{ll:.8f}"
    )

    print(
        f"Best iteration    : "
        f"{best_iteration}"
    )

    print(
        f"Prediction mean   : "
        f"{pred_mean:.6f}"
    )

    print(
        f"Prediction std    : "
        f"{pred_std:.6f}"
    )

    print(
        f"Elapsed           : "
        f"{elapsed:.1f} sec"
    )

    print()

    print("-" * 80)

    print("기준 A")

    print(
        f"Brier          : "
        f"{A_BRIER:.8f}"
    )

    print(
        f"AUC            : "
        f"{A_AUC:.6f}"
    )

    print(
        f"Best iteration : "
        f"{A_BEST_ITERATION}"
    )

    print("-" * 80)

    # -----------------------------------------------------
    # 자동 간단 판정
    # -----------------------------------------------------

    print()

    if best_iteration >= 5:

        print(
            "[판정] "
            "best_iteration이 확실히 증가했습니다."
        )

        print(
            "→ 기존의 불안정한 Feature들이 "
            "Fold 3 붕괴에 크게 관여했을 가능성이 있습니다."
        )

    elif best_iteration > 0:

        print(
            "[판정] "
            "best_iteration이 약간 증가했습니다."
        )

        print(
            "→ Feature instability 영향은 있으나 "
            "Fold 3 붕괴가 완전히 해결된 것은 아닙니다."
        )

    else:

        print(
            "[판정] "
            "F에서도 best_iteration=0입니다."
        )

        print(
            "→ 일부 Feature 제거만으로는 설명하기 어렵고,"
        )

        print(
            "   2023 전체의 regime / base-rate / "
            "calibration shift 가능성이 더 강합니다."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()