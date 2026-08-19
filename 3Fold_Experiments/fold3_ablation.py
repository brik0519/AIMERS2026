# fold3_ablation.py
#
# LG Aimers - 2023 Validation 구조 변화 원인 분석
#
# 현재 cb_v002 CatBoost 설정을 그대로 유지하면서
# Fold 3 (Train 2019~2022 -> Validation 2023)만 대상으로
# Feature Ablation A~E를 연속 실행한다.
#
# ---------------------------------------------------------
# Experiment
# ---------------------------------------------------------
# A: 전체 Feature
# B: batter_id 제거
# C: batter_id + asof_batter_* 제거
# D: pitcher_id + batter_id 제거
# E: 불안정한 middle / batter 통계 제거
#
# ---------------------------------------------------------
# 기록 지표
# ---------------------------------------------------------
# - Brier Score
# - A 대비 Brier 개선량
# - Best Iteration
# - Prediction Mean / Std
# - ROC-AUC
# - LogLoss
# - 학습 시간
# - Seen / Unseen ID 그룹별 Brier
#
# 결과:
# validation/fold3_ablation_2023/
# ├── summary.csv
# ├── summary.json
# ├── A/
# │   └── pred.csv
# ├── B/
# │   └── pred.csv
# ├── C/
# │   └── pred.csv
# ├── D/
# │   └── pred.csv
# └── E/
#     └── pred.csv


import gc
import json
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
# 실험 설정
# =========================================================

EXPERIMENT_NAME = "fold3_ablation_2023"

DATA_DIR = Path("./data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

VALIDATION_ROOT = Path("./validation")
OUTPUT_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

ID = "row_id"
TARGET = "control_success"


# =========================================================
# Fold 3 고정
# =========================================================

TRAIN_SEASONS = [
    2019,
    2020,
    2021,
    2022,
]

VALIDATION_SEASON = 2023


# =========================================================
# 기존 cb_v002와 동일한 Categorical Feature
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
# Experiment E 제거 Feature
# PDF에서 지정한 우선 제거 후보 그대로 사용
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
# Feature 전처리
# 기존 cb_v002와 동일
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
# Metric
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    전체 Validation 지표 계산.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    positive_rate = float(
        np.mean(y_true)
    )

    baseline_brier = float(
        positive_rate
        * (1.0 - positive_rate)
    )

    auc = float(
        roc_auc_score(
            y_true,
            y_pred,
        )
    )

    ll = float(
        log_loss(
            y_true,
            y_pred,
            labels=[0, 1],
        )
    )

    return {
        "positive_rate": positive_rate,
        "baseline_brier": baseline_brier,
        "brier": brier,
        "roc_auc": auc,
        "logloss": ll,
        "prediction_mean": float(
            np.mean(y_pred)
        ),
        "prediction_std": float(
            np.std(y_pred)
        ),
        "prediction_min": float(
            np.min(y_pred)
        ),
        "prediction_max": float(
            np.max(y_pred)
        ),
    }


# =========================================================
# ID 비교용 normalization
# =========================================================

def normalize_id(series):
    """
    Train / Validation ID membership 비교를 안정적으로 수행하기 위해
    문자열 형태로 통일한다.
    """

    return (
        series
        .fillna("__MISSING__")
        .astype(str)
    )


# =========================================================
# Seen / Unseen 그룹 생성
# =========================================================

def create_seen_unseen_groups(
    train_df,
    val_df,
):
    """
    Fold 3 Train에서 pitcher_id / batter_id가 존재했는지 기준으로
    2023 Validation을 4개 그룹으로 나눈다.

    1. both_seen
    2. pitcher_unseen_only
    3. batter_unseen_only
    4. both_unseen
    """

    required = [
        "pitcher_id",
        "batter_id",
    ]

    for col in required:
        if col not in train_df.columns:
            raise KeyError(
                f"Seen/Unseen 분석에 필요한 "
                f"컬럼이 없습니다: {col}"
            )

    train_pitchers = set(
        normalize_id(
            train_df["pitcher_id"]
        )
    )

    train_batters = set(
        normalize_id(
            train_df["batter_id"]
        )
    )

    val_pitcher = normalize_id(
        val_df["pitcher_id"]
    )

    val_batter = normalize_id(
        val_df["batter_id"]
    )

    pitcher_seen = (
        val_pitcher.isin(
            train_pitchers
        )
    )

    batter_seen = (
        val_batter.isin(
            train_batters
        )
    )

    groups = np.full(
        len(val_df),
        "both_unseen",
        dtype=object,
    )

    groups[
        pitcher_seen
        & batter_seen
    ] = "both_seen"

    groups[
        (~pitcher_seen)
        & batter_seen
    ] = "pitcher_unseen_only"

    groups[
        pitcher_seen
        & (~batter_seen)
    ] = "batter_unseen_only"

    return pd.Series(
        groups,
        index=val_df.index,
        name="id_group",
    )


# =========================================================
# 그룹별 Brier
# =========================================================

def calculate_group_brier(
    y_true,
    y_pred,
    groups,
):
    """
    Seen / Unseen 그룹별 Brier Score 계산.
    """

    temp = pd.DataFrame({
        "target": np.asarray(y_true),
        "prediction": np.asarray(y_pred),
        "group": np.asarray(groups),
    })

    results = {}

    group_names = [
        "both_seen",
        "pitcher_unseen_only",
        "batter_unseen_only",
        "both_unseen",
    ]

    for group in group_names:

        part = temp[
            temp["group"] == group
        ]

        n = len(part)

        if n == 0:
            results[
                f"{group}_n"
            ] = 0

            results[
                f"{group}_brier"
            ] = np.nan

            continue

        brier = float(
            np.mean(
                (
                    part["prediction"]
                    - part["target"]
                ) ** 2
            )
        )

        results[
            f"{group}_n"
        ] = int(n)

        results[
            f"{group}_brier"
        ] = brier

    # 전체 unseen이 하나라도 있는 경우도 참고용으로 기록
    any_unseen = temp[
        temp["group"]
        != "both_seen"
    ]

    if len(any_unseen) > 0:

        results[
            "any_unseen_n"
        ] = int(
            len(any_unseen)
        )

        results[
            "any_unseen_brier"
        ] = float(
            np.mean(
                (
                    any_unseen["prediction"]
                    - any_unseen["target"]
                ) ** 2
            )
        )

    else:
        results["any_unseen_n"] = 0
        results["any_unseen_brier"] = np.nan

    return results


# =========================================================
# Ablation Experiment 정의
# =========================================================

def build_experiments(
    base_features,
):
    """
    실제 test.csv Feature 기준으로
    A~E의 제거 Feature를 정의한다.
    """

    # -----------------------------------------------------
    # C:
    # batter_id + 모든 asof_batter_* 제거
    # -----------------------------------------------------

    asof_batter_features = [
        col
        for col in base_features
        if col.startswith(
            "asof_batter_"
        )
    ]

    # -----------------------------------------------------
    # E Feature 존재 여부 확인
    # typo나 데이터 버전 차이를 조기에 발견하기 위함
    # -----------------------------------------------------

    missing_e = [
        col
        for col in E_REMOVE_FEATURES
        if col not in base_features
    ]

    if missing_e:
        raise KeyError(
            "\nExperiment E에서 제거해야 할 "
            "Feature가 현재 데이터에 없습니다.\n"
            f"Missing: {missing_e}\n"
            "컬럼명을 확인하세요."
        )

    experiments = {
        "A": {
            "description": (
                "현재 전체 Feature 기준 모델"
            ),
            "remove": [],
        },

        "B": {
            "description": (
                "batter_id 제거"
            ),
            "remove": [
                "batter_id",
            ],
        },

        "C": {
            "description": (
                "batter_id + asof_batter_* 제거"
            ),
            "remove": (
                ["batter_id"]
                + asof_batter_features
            ),
        },

        "D": {
            "description": (
                "pitcher_id + batter_id 제거"
            ),
            "remove": [
                "pitcher_id",
                "batter_id",
            ],
        },

        "E": {
            "description": (
                "불안정한 middle/batter 통계 제거"
            ),
            "remove": (
                E_REMOVE_FEATURES.copy()
            ),
        },
    }

    return experiments


# =========================================================
# 단일 Experiment 실행
# =========================================================

def run_experiment(
    experiment_key,
    config,
    train_fold,
    val_fold,
    base_features,
    id_groups,
):
    """
    A~E 중 하나를 실행한다.
    """

    remove_features = set(
        config["remove"]
    )

    features = [
        col
        for col in base_features
        if col not in remove_features
    ]

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    print()
    print("=" * 80)
    print(
        f"Experiment {experiment_key}"
    )
    print(
        config["description"]
    )
    print("=" * 80)

    print(
        f"Feature 수       : "
        f"{len(features)}"
    )

    print(
        f"제거 Feature 수  : "
        f"{len(remove_features)}"
    )

    if remove_features:
        print("제거 Feature:")

        for col in config["remove"]:
            print(f"  - {col}")

    else:
        print(
            "제거 Feature      : 없음"
        )

    print(
        f"Categorical      : "
        f"{cat_cols}"
    )

    # -----------------------------------------------------
    # Data
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
    # Model
    # -----------------------------------------------------

    estimator = CatBoostClassifier(
        **MODEL_PARAMS
    )

    start_time = time.time()

    estimator.fit(
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
        - start_time
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    pred = (
        estimator
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
        estimator
        .get_best_iteration()
    )

    group_metrics = (
        calculate_group_brier(
            y_true=y_val,
            y_pred=pred,
            groups=id_groups,
        )
    )

    # -----------------------------------------------------
    # 결과 저장
    # -----------------------------------------------------

    experiment_dir = (
        OUTPUT_DIR
        / experiment_key
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df = pd.DataFrame({
        "row_id": val_fold[
            ID
        ].to_numpy(),

        "target": y_val,

        "prediction": pred,

        "id_group": (
            id_groups
            .to_numpy()
        ),
    })

    pred_df.to_csv(
        experiment_dir
        / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "experiment": (
            experiment_key
        ),

        "description": (
            config["description"]
        ),

        "n_features": int(
            len(features)
        ),

        "n_removed_features": int(
            len(remove_features)
        ),

        "removed_features": (
            ", ".join(
                config["remove"]
            )
        ),

        "brier": (
            metrics["brier"]
        ),

        "baseline_brier": (
            metrics[
                "baseline_brier"
            ]
        ),

        "roc_auc": (
            metrics["roc_auc"]
        ),

        "logloss": (
            metrics["logloss"]
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

        "prediction_min": (
            metrics[
                "prediction_min"
            ]
        ),

        "prediction_max": (
            metrics[
                "prediction_max"
            ]
        ),

        "best_iteration": (
            int(best_iteration)
            if best_iteration
            is not None
            else None
        ),

        "elapsed_seconds": float(
            elapsed
        ),

        **group_metrics,
    }

    # -----------------------------------------------------
    # 콘솔 출력
    # -----------------------------------------------------

    print()
    print("-" * 80)

    print(
        f"Brier Score      : "
        f"{result['brier']:.8f}"
    )

    print(
        f"Baseline Brier   : "
        f"{result['baseline_brier']:.8f}"
    )

    print(
        f"ROC-AUC          : "
        f"{result['roc_auc']:.6f}"
    )

    print(
        f"LogLoss          : "
        f"{result['logloss']:.8f}"
    )

    print(
        f"Prediction mean  : "
        f"{result['prediction_mean']:.6f}"
    )

    print(
        f"Prediction std   : "
        f"{result['prediction_std']:.6f}"
    )

    print(
        f"Best iteration   : "
        f"{result['best_iteration']}"
    )

    print(
        f"학습 시간         : "
        f"{elapsed:.1f}초"
    )

    print()
    print("[Seen / Unseen Brier]")

    for group in [
        "both_seen",
        "pitcher_unseen_only",
        "batter_unseen_only",
        "both_unseen",
    ]:

        n = result[
            f"{group}_n"
        ]

        brier = result[
            f"{group}_brier"
        ]

        if np.isnan(brier):
            brier_text = "N/A"
        else:
            brier_text = (
                f"{brier:.8f}"
            )

        print(
            f"{group:22s} "
            f"N={n:7d} | "
            f"Brier={brier_text}"
        )

    print("-" * 80)

    # -----------------------------------------------------
    # 메모리 정리
    # -----------------------------------------------------

    del estimator
    del X_train
    del X_val
    del pred

    gc.collect()

    return result


# =========================================================
# 결과 해석용 출력
# =========================================================

def print_final_summary(
    summary,
):
    """
    최종 비교표 출력.
    """

    display_cols = [
        "experiment",
        "n_features",
        "brier",
        "brier_improvement_vs_A",
        "roc_auc",
        "logloss",
        "best_iteration",
        "prediction_mean",
        "prediction_std",
        "elapsed_seconds",
    ]

    display = (
        summary[
            display_cols
        ]
        .copy()
    )

    for col in [
        "brier",
        "brier_improvement_vs_A",
        "logloss",
    ]:
        display[col] = (
            display[col]
            .map(
                lambda x: (
                    f"{x:.8f}"
                )
            )
        )

    for col in [
        "roc_auc",
        "prediction_mean",
        "prediction_std",
    ]:
        display[col] = (
            display[col]
            .map(
                lambda x: (
                    f"{x:.6f}"
                )
            )
        )

    display[
        "elapsed_seconds"
    ] = (
        display[
            "elapsed_seconds"
        ]
        .map(
            lambda x: (
                f"{x:.1f}"
            )
        )
    )

    print()
    print()
    print("=" * 120)
    print(
        "Fold 3 Ablation 최종 결과"
    )
    print("=" * 120)

    print(
        display.to_string(
            index=False
        )
    )

    print("=" * 120)

    print()
    print(
        "※ brier_improvement_vs_A:"
    )

    print(
        "   양수 = A보다 개선"
    )

    print(
        "   음수 = A보다 악화"
    )

    print()


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 80)
    print(
        "LG Aimers "
        "Fold 3 Ablation Experiment"
    )
    print(
        "Train: 2019~2022"
    )
    print(
        "Validation: 2023"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # 기존 결과 덮어쓰기 방지
    # -----------------------------------------------------

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "\n이미 동일한 실험 결과가 존재합니다.\n"
            f"{OUTPUT_DIR}\n\n"
            "기존 결과를 삭제하거나 "
            "EXPERIMENT_NAME을 변경하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Test 기준 Feature 목록
    #
    # 기존 cb_v002와 동일하게
    # test.csv의 row_id를 제외한 컬럼을 Feature로 사용
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

    # -----------------------------------------------------
    # 필요한 train 컬럼
    #
    # Seen / Unseen 분석용 ID도 반드시 포함
    # -----------------------------------------------------

    required_train_cols = list(
        dict.fromkeys(
            [ID]
            + base_features
            + [TARGET]
            + [
                "pitcher_id",
                "batter_id",
            ]
        )
    )

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=required_train_cols,
    )

    print()
    print(
        f"전체 Train shape : "
        f"{train.shape}"
    )

    print(
        f"기본 Feature 수   : "
        f"{len(base_features)}"
    )

    print(
        f"전체 시즌         : "
        f"{train['season'].min()}"
        f" ~ "
        f"{train['season'].max()}"
    )

    # -----------------------------------------------------
    # Fold 3 Split
    # -----------------------------------------------------

    train_fold = (
        train[
            train["season"].isin(
                TRAIN_SEASONS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    val_fold = (
        train[
            train["season"]
            == VALIDATION_SEASON
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 80)

    print(
        f"Fold 3 Train      : "
        f"{train_fold.shape}"
    )

    print(
        f"Fold 3 Validation : "
        f"{val_fold.shape}"
    )

    print(
        f"Train Target Mean : "
        f"{train_fold[TARGET].mean():.6f}"
    )

    print(
        f"Valid Target Mean : "
        f"{val_fold[TARGET].mean():.6f}"
    )

    print("=" * 80)

    # -----------------------------------------------------
    # Seen / Unseen ID 그룹
    # -----------------------------------------------------

    id_groups = (
        create_seen_unseen_groups(
            train_df=train_fold,
            val_df=val_fold,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "2023 Seen / Unseen ID 분포"
    )

    group_counts = (
        id_groups
        .value_counts()
    )

    for group, count in (
        group_counts.items()
    ):
        ratio = (
            count
            / len(id_groups)
        )

        print(
            f"- {group:22s}: "
            f"{count:7d} "
            f"({ratio:.2%})"
        )

    # -----------------------------------------------------
    # Experiment 구성
    # -----------------------------------------------------

    experiments = (
        build_experiments(
            base_features
        )
    )

    print()
    print("=" * 80)
    print("실행 Experiment")
    print("=" * 80)

    for key, config in (
        experiments.items()
    ):

        print(
            f"{key}: "
            f"{config['description']}"
        )

    # -----------------------------------------------------
    # A~E 실행
    # -----------------------------------------------------

    results = []

    for key, config in (
        experiments.items()
    ):

        result = run_experiment(
            experiment_key=key,
            config=config,
            train_fold=train_fold,
            val_fold=val_fold,
            base_features=base_features,
            id_groups=id_groups,
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

    # A Brier
    baseline_a = float(
        summary.loc[
            summary[
                "experiment"
            ] == "A",
            "brier",
        ].iloc[0]
    )

    # -----------------------------------------------------
    # 개선량 정의
    #
    # A Brier - Experiment Brier
    #
    # 양수 = 개선
    # 음수 = 악화
    # -----------------------------------------------------

    summary[
        "brier_improvement_vs_A"
    ] = (
        baseline_a
        - summary["brier"]
    )

    # 보기 편하도록 Brier 기준 정렬한 별도 rank
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
    # CSV 저장
    # -----------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # JSON 저장
    # -----------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "train_seasons": (
            TRAIN_SEASONS
        ),

        "validation_season": (
            VALIDATION_SEASON
        ),

        "model_params": (
            MODEL_PARAMS
        ),

        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),

        "base_feature_count": (
            len(base_features)
        ),

        "experiments": {
            key: {
                "description": (
                    value[
                        "description"
                    ]
                ),
                "remove": (
                    value["remove"]
                ),
            }
            for key, value
            in experiments.items()
        },

        "results": (
            summary
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # 최종 결과 출력
    # -----------------------------------------------------

    print_final_summary(
        summary
    )

    print(
        "결과 저장 완료:"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'summary.csv'}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'summary.json'}"
    )

    print()

    print(
        "각 Experiment Prediction:"
    )

    for key in experiments:

        print(
            f"- "
            f"{OUTPUT_DIR / key / 'pred.csv'}"
        )


if __name__ == "__main__":
    main()