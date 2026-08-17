# train.py: XGBoost xgb_v002
# xgb_v001의 모델/전처리 구조를 유지하면서
# 4-Fold Expanding Window Validation을 적용한다.

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


# =========================================================
# 실험 설정
# =========================================================

EXPERIMENT_NAME = "xgb_v002"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID = "row_id"
TARGET = "control_success"


# =========================================================
# 학습 설정
# =========================================================

MAX_ITERATIONS = 5000
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 42

# Hidden future와 가장 가까운
# 2019~2023 -> 2024 Fold의 Brier best iteration을
# Final model iteration으로 사용한다.
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# Expanding Window 설정
# =========================================================

EXPANDING_FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2019, 2020],
        "validation_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "validation_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "validation_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "validation_season": 2024,
    },
]


# =========================================================
# Feature 정책
# =========================================================

# 저카디널리티 명목형만 categorical encoding
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]

# 고카디널리티 ID는 categorical encoding하지 않고
# numeric 값으로 유지
ID_LIKE_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]


# =========================================================
# XGBoost 모델 파라미터
# =========================================================

MODEL_PARAMS = {
    "objective": "binary:logistic",

    "learning_rate": 0.03,

    "max_depth": 8,
    "min_child_weight": 10.0,

    "subsample": 0.85,
    "colsample_bytree": 0.85,

    "reg_alpha": 1.0,
    "reg_lambda": 10.0,

    "tree_method": "hist",
    "device": "cpu",

    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =========================================================
# Brier Score
# =========================================================

def brier_metric(
    y_true,
    y_pred,
):
    """
    XGBoost validation용 Brier Score.

    callable eval_metric은 cost function으로 사용되며
    값이 작을수록 좋다.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


# =========================================================
# Raw Feature 준비
# =========================================================

def prepare_raw_features(
    df,
    features,
    id_like_cols,
):
    """
    XGBoost 전처리 직전 DataFrame을 생성한다.

    1. 학습 Feature 순서 고정
    2. ID 계열은 numeric으로 강제 변환
    3. 나머지 numeric NaN은 그대로 유지
       → XGBoost가 직접 처리
    """

    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "입력 데이터에 학습 Feature가 없습니다: "
            + ", ".join(
                missing_features
            )
        )

    X = df[
        features
    ].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


# =========================================================
# 전처리 생성
# =========================================================

def create_preprocessor(
    cat_cols,
    num_cols,
):
    """
    새로운 OrdinalEncoder를 포함하는
    XGBoost 전처리 객체를 매번 생성한다.

    각 Expanding Fold와 Final model이 서로 독립된
    fitted encoder를 갖도록 한다.

    categorical:
        unseen category → -1
        missing category → -1

    numeric:
        그대로 통과
        NaN도 그대로 XGBoost에 전달
    """

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# Validation 모델 생성
# =========================================================

def create_validation_model():
    """
    Brier Score 기반 early stopping을 사용하는
    Validation용 XGBClassifier를 생성한다.
    """

    return XGBClassifier(
        n_estimators=MAX_ITERATIONS,

        eval_metric=brier_metric,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,

        **MODEL_PARAMS,
    )


# =========================================================
# Final 모델 생성
# =========================================================

def create_final_model(
    n_estimators,
):
    """
    Validation에서 결정한 iteration만큼
    전체 데이터로 다시 학습할 Final 모델.

    Final 모델에는 eval_metric callable과
    early stopping 설정을 넣지 않는다.

    이렇게 해야 저장된 model.pkl이
    script.py의 함수 정의에 의존하지 않는다.
    """

    return XGBClassifier(
        n_estimators=n_estimators,
        **MODEL_PARAMS,
    )


# =========================================================
# Metric 계산
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    positive_rate = float(
        np.mean(y_true)
    )

    brier = float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    baseline_brier = float(
        positive_rate
        * (1.0 - positive_rate)
    )

    if baseline_brier <= 0.0:
        raise RuntimeError(
            "Baseline Brier Score가 "
            "0 이하입니다."
        )

    bss_raw = float(
        100000.0
        * (
            1.0
            - brier / baseline_brier
        )
    )

    bss = max(
        0.0,
        bss_raw,
    )

    return {
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": bss,
    }


# =========================================================
# Validation Prediction 검증
# =========================================================

def validate_predictions(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    if len(y_pred) != len(y_true):
        raise RuntimeError(
            "Validation 예측 수와 "
            "정답 수가 다릅니다."
        )

    if not np.isfinite(
        y_pred
    ).all():
        raise RuntimeError(
            "Validation 예측에 "
            "NaN 또는 무한대가 있습니다."
        )

    if not (
        (y_pred >= 0.0)
        & (y_pred <= 1.0)
    ).all():
        raise RuntimeError(
            "Validation 예측이 "
            "0~1 범위를 벗어났습니다."
        )


# =========================================================
# Fold Validation 결과 저장
# =========================================================

def save_fold_validation(
    experiment_name,
    fold,
    train_seasons,
    validation_season,
    estimator,
    preprocessor,
    model_params,
    features,
    cat_cols,
    id_like_cols,
    num_cols,
    row_ids,
    y_true,
    y_pred,
    metrics,
    output_root,
    model_path,
    n_train,
    elapsed_seconds,
    best_iteration,
    final_iterations,
):
    row_ids = np.asarray(
        row_ids
    )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    assert (
        len(row_ids)
        == len(y_true)
        == len(y_pred)
    )

    output_dir = (
        Path(output_root)
        / experiment_name
        / f"fold_{validation_season}"
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
        "validation_season": (
            validation_season
        ),
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
        validation_season=np.full(
            len(y_true),
            validation_season,
        ),
    )

    # -----------------------------------------------------
    # metadata.json
    # -----------------------------------------------------

    metadata = {
        "experiment_name": experiment_name,
        "fold": int(fold),

        "train_seasons": [
            int(season)
            for season in train_seasons
        ],

        "validation_season": int(
            validation_season
        ),

        "n_train": int(n_train),
        "n_validation": int(
            len(y_true)
        ),

        "model_type": type(
            estimator
        ).__name__,

        "model_params": model_params,

        "max_iterations": (
            MAX_ITERATIONS
        ),

        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),

        "best_iteration": int(
            best_iteration
        ),

        "final_iterations": int(
            final_iterations
        ),

        "positive_rate": float(
            metrics["positive_rate"]
        ),

        "brier": float(
            metrics["brier"]
        ),

        "baseline_brier": float(
            metrics["baseline_brier"]
        ),

        "bss_raw": float(
            metrics["bss_raw"]
        ),

        "bss": float(
            metrics["bss"]
        ),

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

        "elapsed_seconds": float(
            elapsed_seconds
        ),

        "model_path": str(
            model_path
        ),

        "features": features,
        "categorical_features": (
            cat_cols
        ),
        "id_like_features": (
            id_like_cols
        ),
        "numeric_features": num_cols,

        "xgboost_version": (
            xgb.__version__
        ),
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

    # Fold checkpoint는 연구/재현용이다.
    # Final submission에서는 사용하지 않는다.
    fold_artifact = {
        "model": estimator,
        "preprocessor": preprocessor,

        "features": features,
        "cat_cols": cat_cols,
        "id_like_cols": id_like_cols,
        "num_cols": num_cols,

        "best_iteration": int(
            best_iteration
        ),

        "final_iterations": int(
            final_iterations
        ),

        "validation_brier": float(
            metrics["brier"]
        ),

        "validation_score": float(
            metrics["bss"]
        ),

        "validation_bss_raw": float(
            metrics["bss_raw"]
        ),

        "validation_season": int(
            validation_season
        ),

        "model_name": (
            f"{experiment_name}"
            f"_fold_{validation_season}"
        ),

        "xgboost_version": (
            xgb.__version__
        ),
    }

    joblib.dump(
        fold_artifact,
        model_path,
        compress=3,
    )

    return (
        pred_df,
        metadata,
    )


# =========================================================
# Expanding Window Validation
# =========================================================

def run_expanding_validation(
    train,
    features,
    cat_cols,
    id_like_cols,
    num_cols,
):
    fold_results = []
    oof_frames = []

    for fold_config in EXPANDING_FOLDS:
        fold = fold_config["fold"]

        train_seasons = (
            fold_config[
                "train_seasons"
            ]
        )

        validation_season = (
            fold_config[
                "validation_season"
            ]
        )

        train_mask = (
            train["season"]
            .isin(train_seasons)
        )

        val_mask = (
            train["season"]
            == validation_season
        )

        if not train_mask.any():
            raise ValueError(
                f"Fold {fold}의 학습 데이터가 "
                "없습니다."
            )

        if not val_mask.any():
            raise ValueError(
                f"Fold {fold}의 Validation "
                "데이터가 없습니다."
            )

        train_df = train.loc[
            train_mask
        ].copy()

        val_df = train.loc[
            val_mask
        ].copy()

        y_train = train_df[
            TARGET
        ].copy()

        y_val = val_df[
            TARGET
        ].copy()

        val_row_id = val_df[
            ID
        ].copy()

        # -------------------------------------------------
        # Fold별 Raw Feature
        # -------------------------------------------------

        X_train_raw = (
            prepare_raw_features(
                df=train_df,
                features=features,
                id_like_cols=(
                    id_like_cols
                ),
            )
        )

        X_val_raw = (
            prepare_raw_features(
                df=val_df,
                features=features,
                id_like_cols=(
                    id_like_cols
                ),
            )
        )

        # -------------------------------------------------
        # Fold별 독립 Encoder
        #
        # 미래 Validation 시즌의 범주 정보를
        # fit에 사용하지 않는다.
        # -------------------------------------------------

        validation_preprocessor = (
            create_preprocessor(
                cat_cols=cat_cols,
                num_cols=num_cols,
            )
        )

        X_train = (
            validation_preprocessor
            .fit_transform(
                X_train_raw
            )
        )

        X_val = (
            validation_preprocessor
            .transform(
                X_val_raw
            )
        )

        print()
        print("=" * 70)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print("=" * 70)

        print(
            "학습 데이터 shape     :",
            X_train.shape,
        )

        print(
            "검증 데이터 shape     :",
            X_val.shape,
        )

        print(
            f"학습 Target 평균      : "
            f"{y_train.mean():.6f}"
        )

        print(
            f"검증 Target 평균      : "
            f"{y_val.mean():.6f}"
        )

        # -------------------------------------------------
        # Fold Validation 모델
        # -------------------------------------------------

        validation_model = (
            create_validation_model()
        )

        validation_start = (
            time.time()
        )

        validation_model.fit(
            X_train,
            y_train,

            eval_set=[
                (
                    X_val,
                    y_val,
                ),
            ],

            verbose=100,
        )

        validation_elapsed = (
            time.time()
            - validation_start
        )

        # -------------------------------------------------
        # Best iteration
        # -------------------------------------------------

        best_iteration = getattr(
            validation_model,
            "best_iteration",
            None,
        )

        if (
            best_iteration is None
            or best_iteration < 0
        ):
            best_iteration = (
                MAX_ITERATIONS - 1
            )

        # XGBoost best_iteration은 0-based
        final_iterations = (
            best_iteration + 1
        )

        # -------------------------------------------------
        # Validation prediction
        # -------------------------------------------------

        val_pred = (
            validation_model
            .predict_proba(
                X_val
            )[:, 1]
        )

        validate_predictions(
            y_true=y_val,
            y_pred=val_pred,
        )

        metrics = calculate_metrics(
            y_true=y_val,
            y_pred=val_pred,
        )

        # -------------------------------------------------
        # Fold model 저장 경로
        # -------------------------------------------------

        fold_model_dir = (
            MODEL_DIR
            / f"fold_{validation_season}"
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_model_path = (
            fold_model_dir
            / (
                f"{EXPERIMENT_NAME}"
                f"_fold_{validation_season}.pkl"
            )
        )

        # -------------------------------------------------
        # Fold Validation 저장
        # -------------------------------------------------

        (
            fold_pred_df,
            fold_metadata,
        ) = save_fold_validation(
            experiment_name=(
                EXPERIMENT_NAME
            ),
            fold=fold,
            train_seasons=(
                train_seasons
            ),
            validation_season=(
                validation_season
            ),
            estimator=(
                validation_model
            ),
            preprocessor=(
                validation_preprocessor
            ),
            model_params=(
                MODEL_PARAMS
            ),
            features=features,
            cat_cols=cat_cols,
            id_like_cols=(
                id_like_cols
            ),
            num_cols=num_cols,
            row_ids=val_row_id,
            y_true=y_val,
            y_pred=val_pred,
            metrics=metrics,
            output_root=(
                VALIDATION_ROOT
            ),
            model_path=(
                fold_model_path
            ),
            n_train=len(
                train_df
            ),
            elapsed_seconds=(
                validation_elapsed
            ),
            best_iteration=(
                best_iteration
            ),
            final_iterations=(
                final_iterations
            ),
        )

        fold_results.append(
            fold_metadata
        )

        oof_frames.append(
            fold_pred_df
        )

        print()
        print(
            f"Brier Score          : "
            f"{metrics['brier']:.10f}"
        )

        print(
            f"Baseline Brier       : "
            f"{metrics['baseline_brier']:.10f}"
        )

        print(
            f"BSS Raw              : "
            f"{metrics['bss_raw']:.4f}"
        )

        print(
            f"Validation Score     : "
            f"{metrics['bss']:.4f}"
        )

        print(
            f"Prediction mean      : "
            f"{val_pred.mean():.6f}"
        )

        print(
            f"Prediction std       : "
            f"{val_pred.std():.6f}"
        )

        print(
            f"Prediction min       : "
            f"{val_pred.min():.6f}"
        )

        print(
            f"Prediction max       : "
            f"{val_pred.max():.6f}"
        )

        print(
            f"Best iteration       : "
            f"{best_iteration}"
        )

        print(
            f"Final iterations     : "
            f"{final_iterations}"
        )

        print(
            f"학습 시간            : "
            f"{validation_elapsed:.1f}초"
        )

        print(
            f"Fold model           : "
            f"{fold_model_path}"
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    return (
        fold_results,
        oof,
    )


# =========================================================
# Validation Summary 저장
# =========================================================

def save_validation_summary(
    features,
    cat_cols,
    id_like_cols,
    num_cols,
    fold_results,
    oof,
):
    briers = np.asarray(
        [
            fold["brier"]
            for fold in fold_results
        ],
        dtype=float,
    )

    bss_values = np.asarray(
        [
            fold["bss"]
            for fold in fold_results
        ],
        dtype=float,
    )

    bss_raw_values = np.asarray(
        [
            fold["bss_raw"]
            for fold in fold_results
        ],
        dtype=float,
    )

    fold_sizes = np.asarray(
        [
            fold["n_validation"]
            for fold in fold_results
        ],
        dtype=float,
    )

    recent_briers = [
        fold["brier"]
        for fold in fold_results
        if fold[
            "validation_season"
        ] in (2023, 2024)
    ]

    oof_metrics = (
        calculate_metrics(
            y_true=oof["target"],
            y_pred=oof[
                "prediction"
            ],
        )
    )

    summary = {
        "n_oof": int(
            len(oof)
        ),

        "mean_fold_brier": float(
            np.mean(briers)
        ),

        "std_fold_brier": float(
            np.std(briers)
        ),

        "weighted_fold_brier": float(
            np.average(
                briers,
                weights=fold_sizes,
            )
        ),

        "best_fold_brier": float(
            np.min(briers)
        ),

        "worst_fold_brier": float(
            np.max(briers)
        ),

        "fold_brier_range": float(
            np.max(briers)
            - np.min(briers)
        ),

        "recent_2023_2024_mean_brier": (
            float(
                np.mean(
                    recent_briers
                )
            )
        ),

        "mean_fold_bss_raw": float(
            np.mean(
                bss_raw_values
            )
        ),

        "weighted_fold_bss_raw": float(
            np.average(
                bss_raw_values,
                weights=fold_sizes,
            )
        ),

        "mean_fold_bss": float(
            np.mean(
                bss_values
            )
        ),

        "weighted_fold_bss": float(
            np.average(
                bss_values,
                weights=fold_sizes,
            )
        ),

        "pooled_oof_brier": float(
            oof_metrics["brier"]
        ),

        "pooled_oof_baseline_brier": float(
            oof_metrics[
                "baseline_brier"
            ]
        ),

        "pooled_oof_bss_raw": float(
            oof_metrics["bss_raw"]
        ),

        "pooled_oof_bss": float(
            oof_metrics["bss"]
        ),
    }

    # -----------------------------------------------------
    # OOF 저장
    # -----------------------------------------------------

    oof.to_csv(
        VALIDATION_DIR
        / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR
        / "oof_validation.npz",
        row_id=oof[
            "row_id"
        ].to_numpy(),
        target=oof[
            "target"
        ].to_numpy(),
        prediction=oof[
            "prediction"
        ].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    # -----------------------------------------------------
    # 전체 metadata 저장
    # -----------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "validation_method": (
            "expanding_window"
        ),

        "model_params": (
            MODEL_PARAMS
        ),

        "max_iterations": (
            MAX_ITERATIONS
        ),

        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),

        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),

        "features": features,

        "categorical_features": (
            cat_cols
        ),

        "id_like_features": (
            id_like_cols
        ),

        "numeric_features": (
            num_cols
        ),

        "folds": fold_results,

        "summary": summary,

        "xgboost_version": (
            xgb.__version__
        ),
    }

    with open(
        VALIDATION_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return summary


# =========================================================
# Final iteration 선택
# =========================================================

def select_final_iterations(
    fold_results,
):
    if (
        FINAL_ITERATION_STRATEGY
        == "latest_fold"
    ):
        latest_fold = max(
            fold_results,
            key=lambda fold: (
                fold[
                    "validation_season"
                ]
            ),
        )

        return int(
            latest_fold[
                "final_iterations"
            ]
        )

    raise ValueError(
        "지원하지 않는 "
        "FINAL_ITERATION_STRATEGY입니다: "
        f"{FINAL_ITERATION_STRATEGY}"
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 덮어쓰기 방지
    # -----------------------------------------------------

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            "Validation experiment already exists: "
            f"{VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            "Model experiment already exists: "
            f"{MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # 데이터 파일 검증
    # -----------------------------------------------------

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "test.csv를 찾을 수 없습니다: "
            f"{TEST_PATH}"
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            "train.csv를 찾을 수 없습니다: "
            f"{TRAIN_PATH}"
        )

    # -----------------------------------------------------
    # test.csv 기준 Feature 결정
    #
    # test의 값/분포/다른 행 정보는 사용하지 않고
    # 공식 입력 schema의 컬럼명만 사용한다.
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID not in test_cols:
        raise ValueError(
            f"test.csv에 ID 컬럼 "
            f"'{ID}'가 없습니다."
        )

    features = [
        col
        for col in test_cols
        if col != ID
    ]

    cat_cols = [
        col
        for col in CAT_COLS
        if col in features
    ]

    id_like_cols = [
        col
        for col in ID_LIKE_COLS
        if col in features
    ]

    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]

    # -----------------------------------------------------
    # 학습 데이터 로드
    # -----------------------------------------------------

    train = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            *features,
            TARGET,
        ],
    )

    required_columns = {
        ID,
        "season",
        TARGET,
    }

    missing_columns = (
        required_columns
        - set(train.columns)
    )

    if missing_columns:
        raise ValueError(
            "train.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if train[
        TARGET
    ].isna().any():
        raise ValueError(
            f"타깃 컬럼 '{TARGET}'에 "
            "결측값이 있습니다."
        )

    print("=" * 70)
    print(
        "XGBoost Expanding Window "
        "데이터 로드 완료"
    )
    print("=" * 70)

    print(
        "Experiment           :",
        EXPERIMENT_NAME,
    )

    print(
        "XGBoost version      :",
        xgb.__version__,
    )

    print(
        "Train shape          :",
        train.shape,
    )

    print(
        "Feature 수           :",
        len(features),
    )

    print(
        "Categorical 수       :",
        len(cat_cols),
    )

    print(
        "Numeric 수           :",
        len(num_cols),
    )

    print(
        "Categorical Feature  :",
        cat_cols,
    )

    print(
        "Numeric ID Feature   :",
        id_like_cols,
    )

    print(
        "시즌 범위            :",
        train[
            "season"
        ].min(),
        "~",
        train[
            "season"
        ].max(),
    )

    print(
        f"전체 제구 성공률      : "
        f"{train[TARGET].mean():.6f}"
    )

    print(
        "Early stopping metric: "
        "Brier Score"
    )

    print(
        "Device               : cpu"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # 출력 디렉터리 생성
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Expanding Window Validation
    # -----------------------------------------------------

    (
        fold_results,
        oof,
    ) = run_expanding_validation(
        train=train,
        features=features,
        cat_cols=cat_cols,
        id_like_cols=id_like_cols,
        num_cols=num_cols,
    )

    summary = (
        save_validation_summary(
            features=features,
            cat_cols=cat_cols,
            id_like_cols=id_like_cols,
            num_cols=num_cols,
            fold_results=(
                fold_results
            ),
            oof=oof,
        )
    )

    print()
    print("=" * 70)
    print(
        "XGBoost Expanding Window "
        "결과 요약"
    )
    print("=" * 70)

    for fold in fold_results:
        print(
            f"Valid "
            f"{fold['validation_season']}: "
            f"Brier="
            f"{fold['brier']:.10f}, "
            f"BSS="
            f"{fold['bss']:.4f}, "
            f"iter="
            f"{fold['final_iterations']}"
        )

    print(
        "Fold 단순 평균 Brier : "
        f"{summary['mean_fold_brier']:.10f}"
    )

    print(
        "Fold 가중 평균 Brier : "
        f"{summary['weighted_fold_brier']:.10f}"
    )

    print(
        "Fold Brier 표준편차  : "
        f"{summary['std_fold_brier']:.10f}"
    )

    print(
        "Fold Brier 범위      : "
        f"{summary['fold_brier_range']:.10f}"
    )

    print(
        "2023~2024 평균 Brier : "
        f"{summary['recent_2023_2024_mean_brier']:.10f}"
    )

    print(
        "OOF pooled Brier     : "
        f"{summary['pooled_oof_brier']:.10f}"
    )

    print(
        "OOF pooled BSS       : "
        f"{summary['pooled_oof_bss']:.4f}"
    )

    print(
        "Validation 저장      : "
        f"{VALIDATION_DIR}"
    )

    print(
        "Fold 모델 저장       : "
        f"{MODEL_DIR}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Final iteration 선택
    # -----------------------------------------------------

    final_iterations = (
        select_final_iterations(
            fold_results
        )
    )

    latest_fold = max(
        fold_results,
        key=lambda fold: (
            fold[
                "validation_season"
            ]
        ),
    )

    print()
    print("=" * 70)
    print("Final iteration 선택")
    print("=" * 70)

    print(
        "Strategy             : "
        f"{FINAL_ITERATION_STRATEGY}"
    )

    print(
        "기준 Validation      : "
        f"{latest_fold['validation_season']}"
    )

    print(
        "Final iterations     : "
        f"{final_iterations}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # 전체 데이터용 Raw Feature
    # -----------------------------------------------------

    X_full_raw = (
        prepare_raw_features(
            df=train,
            features=features,
            id_like_cols=(
                id_like_cols
            ),
        )
    )

    y_full = train[
        TARGET
    ].copy()

    # -----------------------------------------------------
    # Final encoder
    #
    # Fold Validation encoder를 재사용하지 않는다.
    # 공식 train 전체만으로 새로 fit한다.
    # -----------------------------------------------------

    final_preprocessor = (
        create_preprocessor(
            cat_cols=cat_cols,
            num_cols=num_cols,
        )
    )

    X_full = (
        final_preprocessor
        .fit_transform(
            X_full_raw
        )
    )

    # -----------------------------------------------------
    # Final model
    #
    # Fold Validation model을 재사용하지 않는다.
    # -----------------------------------------------------

    final_model = (
        create_final_model(
            n_estimators=(
                final_iterations
            ),
        )
    )

    print()
    print("=" * 70)
    print(
        "XGBoost 전체 데이터 "
        "재학습 시작"
    )

    print(
        f"사용 iteration       : "
        f"{final_iterations}"
    )

    print(
        "학습 시즌           : "
        f"{int(train['season'].min())}"
        "~"
        f"{int(train['season'].max())}"
    )

    print(
        "Device               : CPU"
    )

    print("=" * 70)

    final_start = (
        time.time()
    )

    final_model.fit(
        X_full,
        y_full,
        verbose=False,
    )

    final_elapsed = (
        time.time()
        - final_start
    )

    print(
        f"전체 데이터 재학습 완료: "
        f"{final_elapsed:.1f}초"
    )

    # -----------------------------------------------------
    # 최종 Model artifact
    #
    # xgb_v001과 동일한 핵심 key를 유지한다.
    # -----------------------------------------------------

    artifact = {
        "model": final_model,
        "preprocessor": (
            final_preprocessor
        ),

        "features": features,
        "cat_cols": cat_cols,
        "id_like_cols": (
            id_like_cols
        ),
        "num_cols": num_cols,

        # package/script 호환을 위해
        # 기존 key를 그대로 유지한다.
        "best_iteration": int(
            latest_fold[
                "best_iteration"
            ]
        ),

        "final_iterations": int(
            final_iterations
        ),

        "validation_brier": float(
            latest_fold["brier"]
        ),

        "validation_score": float(
            latest_fold["bss"]
        ),

        "model_name": (
            EXPERIMENT_NAME
        ),

        "xgboost_version": (
            xgb.__version__
        ),

        # v002 추가 metadata
        "validation_method": (
            "expanding_window"
        ),

        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),

        "expanding_summary": (
            summary
        ),
    }

    # -----------------------------------------------------
    # 모델 저장
    # -----------------------------------------------------

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    # -----------------------------------------------------
    # Final metadata 저장
    # -----------------------------------------------------

    final_metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),

        "model_type": (
            type(
                final_model
            ).__name__
        ),

        "training_seasons": sorted(
            [
                int(season)
                for season in train[
                    "season"
                ].unique()
            ]
        ),

        "n_train": int(
            len(train)
        ),

        "features": features,

        "categorical_features": (
            cat_cols
        ),

        "id_like_features": (
            id_like_cols
        ),

        "numeric_features": (
            num_cols
        ),

        "model_params": (
            MODEL_PARAMS
        ),

        "final_iterations": int(
            final_iterations
        ),

        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),

        "reference_validation_season": int(
            latest_fold[
                "validation_season"
            ]
        ),

        "reference_validation_brier": float(
            latest_fold[
                "brier"
            ]
        ),

        "reference_validation_bss": float(
            latest_fold[
                "bss"
            ]
        ),

        "elapsed_seconds": float(
            final_elapsed
        ),

        "model_path": str(
            MODEL_PATH
        ),

        "xgboost_version": (
            xgb.__version__
        ),
    }

    with open(
        MODEL_DIR
        / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # train.py snapshot
    # -----------------------------------------------------

    source_path = Path(
        __file__
    ).resolve()

    shutil.copy2(
        source_path,
        MODEL_DIR / "train.py",
    )

    print()
    print("=" * 70)
    print(
        "XGBoost xgb_v002 모델 저장 완료"
    )

    print(
        f"저장 경로            : "
        f"{MODEL_PATH}"
    )

    print(
        f"Final iterations     : "
        f"{final_iterations}"
    )

    print(
        f"Reference Brier      : "
        f"{latest_fold['brier']:.10f}"
    )

    print(
        f"Reference Score      : "
        f"{latest_fold['bss']:.4f}"
    )

    print(
        f"Validation metadata  : "
        f"{VALIDATION_DIR / 'metadata.json'}"
    )

    print(
        f"Final metadata       : "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )

    print(
        f"train.py snapshot    : "
        f"{MODEL_DIR / 'train.py'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
