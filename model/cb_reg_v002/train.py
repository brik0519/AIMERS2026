# train.py: CatBoost Regressor cb_reg_v002
#
# cb_v002의 Expanding Window Validation 구조를 유지하면서
# CatBoostClassifier(Logloss) -> CatBoostRegressor(RMSE) 변경 효과를 검증한다.
#
# 핵심 원칙
# - 평가 데이터(test.csv)의 다른 행/전체 분포를 Feature 생성에 사용하지 않는다.
# - Feature 목록 확인을 위해 test.csv의 header만 사용한다.
# - Validation은 train.csv 내부 시즌만으로 Expanding Window 방식으로 수행한다.
#
# 추가 저장 항목
# - fold별 raw prediction / clipped prediction
# - raw / clipped Brier Score 및 BSS
# - [0, 1] 범위 이탈 개수/비율
# - pooled OOF raw / clipped metric
# - prediction 분포 통계

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_reg_v002"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

# Fold 모델은 최종 모델과 분리하여 동일 experiment 하위에 저장
FOLD_MODEL_DIR = MODEL_DIR / "folds"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 데이터 설정
# =========================================================

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
# CatBoost 설정
# =========================================================

# cb_v002와 capacity 관련 파라미터를 동일하게 유지한다.
# 핵심 변경점:
#   CatBoostClassifier + Logloss
#       ->
#   CatBoostRegressor + RMSE
MODEL_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "RMSE",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "random_seed": 42,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80

# Final Model의 iteration은 hidden future와 가장 가까운
# 2019~2023 -> 2024 Fold의 best iteration을 사용한다.
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# Feature 전처리
# =========================================================

def prepare_features(df, cat_cols):
    """
    cb_v002와 동일한 전처리.

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

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(
        positive_rate * (1.0 - positive_rate)
    )

    if baseline_brier <= 0.0:
        bss_raw = 0.0
        bss = 0.0
    else:
        bss_raw = float(
            100000.0
            * (
                1.0
                - brier / baseline_brier
            )
        )
        bss = max(0.0, bss_raw)

    return {
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss_raw": bss_raw,
        "bss": bss,
    }


def calculate_range_diagnostics(y_pred_raw):
    y_pred_raw = np.asarray(
        y_pred_raw,
        dtype=np.float64,
    )

    below_zero = y_pred_raw < 0.0
    above_one = y_pred_raw > 1.0
    outside_range = below_zero | above_one

    return {
        "n_prediction_below_0": int(
            np.sum(below_zero)
        ),
        "ratio_prediction_below_0": float(
            np.mean(below_zero)
        ),
        "n_prediction_above_1": int(
            np.sum(above_one)
        ),
        "ratio_prediction_above_1": float(
            np.mean(above_one)
        ),
        "n_prediction_outside_0_1": int(
            np.sum(outside_range)
        ),
        "ratio_prediction_outside_0_1": float(
            np.mean(outside_range)
        ),
    }


# =========================================================
# Fold Validation 결과 저장
# =========================================================

def save_fold_validation(
    experiment_name,
    fold,
    train_seasons,
    validation_season,
    estimator,
    model_params,
    row_ids,
    y_true,
    y_pred_raw,
    y_pred_clip,
    raw_metrics,
    clipped_metrics,
    output_root,
    model_path,
    n_train,
    elapsed_seconds,
    best_iteration=None,
    final_iterations=None,
):
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred_raw = np.asarray(y_pred_raw)
    y_pred_clip = np.asarray(y_pred_clip)

    assert (
        len(row_ids)
        == len(y_true)
        == len(y_pred_raw)
        == len(y_pred_clip)
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

    range_diag = calculate_range_diagnostics(
        y_pred_raw
    )

    # -----------------------------------------------------
    # pred.csv
    # -----------------------------------------------------

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction_raw": y_pred_raw,
        "prediction_clip": y_pred_clip,
        # 기존 cb_v002 분석 코드와의 호환성을 위해
        # 최종 평가용 prediction은 clipped prediction으로 둔다.
        "prediction": y_pred_clip,
        "validation_season": validation_season,
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
        prediction_raw=y_pred_raw,
        prediction_clip=y_pred_clip,
        prediction=y_pred_clip,
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
            int(s)
            for s in train_seasons
        ],
        "validation_season": int(
            validation_season
        ),
        "n_train": int(n_train),
        "n_validation": int(len(y_true)),
        "model_type": type(estimator).__name__,
        "model_params": model_params,
        "positive_rate": float(
            raw_metrics["positive_rate"]
        ),

        # raw 평가
        "raw_brier": float(
            raw_metrics["brier"]
        ),
        "raw_bss_raw": float(
            raw_metrics["bss_raw"]
        ),
        "raw_bss": float(
            raw_metrics["bss"]
        ),

        # clipped 평가
        "clipped_brier": float(
            clipped_metrics["brier"]
        ),
        "clipped_bss_raw": float(
            clipped_metrics["bss_raw"]
        ),
        "clipped_bss": float(
            clipped_metrics["bss"]
        ),

        # cb_v002와 summary schema 호환용
        "brier": float(
            clipped_metrics["brier"]
        ),
        "baseline_brier": float(
            clipped_metrics["baseline_brier"]
        ),
        "bss_raw": float(
            clipped_metrics["bss_raw"]
        ),
        "bss": float(
            clipped_metrics["bss"]
        ),

        # raw prediction 분포
        "prediction_raw_mean": float(
            np.mean(y_pred_raw)
        ),
        "prediction_raw_std": float(
            np.std(y_pred_raw)
        ),
        "prediction_raw_min": float(
            np.min(y_pred_raw)
        ),
        "prediction_raw_max": float(
            np.max(y_pred_raw)
        ),

        # clipped prediction 분포
        "prediction_clip_mean": float(
            np.mean(y_pred_clip)
        ),
        "prediction_clip_std": float(
            np.std(y_pred_clip)
        ),
        "prediction_clip_min": float(
            np.min(y_pred_clip)
        ),
        "prediction_clip_max": float(
            np.max(y_pred_clip)
        ),

        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "model_path": str(model_path),
        **range_diag,
    }

    if best_iteration is not None:
        metadata["best_iteration"] = int(
            best_iteration
        )

    if final_iterations is not None:
        metadata["final_iterations"] = int(
            final_iterations
        )

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

    return pred_df, metadata


# =========================================================
# Expanding Window Validation
# =========================================================

def run_expanding_validation(
    train,
    features,
    cat_cols,
):
    fold_results = []
    oof_frames = []

    for fold_config in EXPANDING_FOLDS:
        fold = fold_config["fold"]
        train_seasons = (
            fold_config["train_seasons"]
        )
        validation_season = (
            fold_config["validation_season"]
        )

        train_mask = train["season"].isin(
            train_seasons
        )

        val_mask = (
            train["season"]
            == validation_season
        )

        X_train = prepare_features(
            train.loc[
                train_mask,
                features,
            ],
            cat_cols,
        )

        y_train = (
            train.loc[
                train_mask,
                TARGET,
            ]
            .astype(np.float64)
        )

        X_val = prepare_features(
            train.loc[
                val_mask,
                features,
            ],
            cat_cols,
        )

        y_val = (
            train.loc[
                val_mask,
                TARGET,
            ]
            .astype(np.float64)
        )

        val_row_id = train.loc[
            val_mask,
            ID,
        ]

        print()
        print("=" * 70)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}"
            f"~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print("=" * 70)
        print("학습 데이터:", X_train.shape)
        print("검증 데이터:", X_val.shape)

        estimator = CatBoostRegressor(
            **MODEL_PARAMS
        )

        start_time = time.time()

        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            use_best_model=True,
        )

        elapsed = (
            time.time()
            - start_time
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        val_pred_raw = estimator.predict(
            X_val
        )

        val_pred_clip = np.clip(
            val_pred_raw,
            0.0,
            1.0,
        )

        raw_metrics = calculate_metrics(
            y_val,
            val_pred_raw,
        )

        clipped_metrics = calculate_metrics(
            y_val,
            val_pred_clip,
        )

        range_diag = (
            calculate_range_diagnostics(
                val_pred_raw
            )
        )

        # -------------------------------------------------
        # Best iteration
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Fold model 저장
        # -------------------------------------------------

        fold_model_dir = (
            FOLD_MODEL_DIR
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

        joblib.dump(
            estimator,
            fold_model_path,
            compress=3,
        )

        pred_df, metadata = (
            save_fold_validation(
                experiment_name=EXPERIMENT_NAME,
                fold=fold,
                train_seasons=train_seasons,
                validation_season=validation_season,
                estimator=estimator,
                model_params=MODEL_PARAMS,
                row_ids=val_row_id,
                y_true=y_val,
                y_pred_raw=val_pred_raw,
                y_pred_clip=val_pred_clip,
                raw_metrics=raw_metrics,
                clipped_metrics=clipped_metrics,
                output_root=VALIDATION_ROOT,
                model_path=fold_model_path,
                n_train=len(y_train),
                elapsed_seconds=elapsed,
                best_iteration=best_iteration,
                final_iterations=final_iterations,
            )
        )

        fold_results.append(metadata)
        oof_frames.append(pred_df)

        print("[Raw prediction]")
        print(
            f"Brier Score      : "
            f"{raw_metrics['brier']:.8f}"
        )
        print(
            f"BSS              : "
            f"{raw_metrics['bss']:.2f}"
        )

        print()

        print("[Clipped prediction]")
        print(
            f"Brier Score      : "
            f"{clipped_metrics['brier']:.8f}"
        )
        print(
            f"Baseline Brier   : "
            f"{clipped_metrics['baseline_brier']:.8f}"
        )
        print(
            f"BSS Raw          : "
            f"{clipped_metrics['bss_raw']:.2f}"
        )
        print(
            f"BSS              : "
            f"{clipped_metrics['bss']:.2f}"
        )

        print()

        print("[Prediction distribution]")
        print(
            f"Raw mean         : "
            f"{val_pred_raw.mean():.6f}"
        )
        print(
            f"Raw std          : "
            f"{val_pred_raw.std():.6f}"
        )
        print(
            f"Raw min          : "
            f"{val_pred_raw.min():.6f}"
        )
        print(
            f"Raw max          : "
            f"{val_pred_raw.max():.6f}"
        )

        print()

        print("[Range diagnostics]")
        print(
            f"Prediction < 0   : "
            f"{range_diag['n_prediction_below_0']} "
            f"("
            f"{range_diag['ratio_prediction_below_0']:.6%}"
            f")"
        )
        print(
            f"Prediction > 1   : "
            f"{range_diag['n_prediction_above_1']} "
            f"("
            f"{range_diag['ratio_prediction_above_1']:.6%}"
            f")"
        )
        print(
            f"Outside [0, 1]   : "
            f"{range_diag['n_prediction_outside_0_1']} "
            f"("
            f"{range_diag['ratio_prediction_outside_0_1']:.6%}"
            f")"
        )

        print()
        print(
            f"Best iteration   : "
            f"{best_iteration}"
        )
        print(
            f"Final iterations : "
            f"{final_iterations}"
        )
        print(
            f"학습 시간         : "
            f"{elapsed:.1f}초"
        )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    return fold_results, oof


# =========================================================
# Validation Summary 저장
# =========================================================

def save_validation_summary(
    train,
    features,
    cat_cols,
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

    fold_sizes = np.asarray(
        [
            fold["n_validation"]
            for fold in fold_results
        ],
        dtype=float,
    )

    weighted_fold_brier = float(
        np.average(
            briers,
            weights=fold_sizes,
        )
    )

    weighted_fold_bss = float(
        np.average(
            bss_values,
            weights=fold_sizes,
        )
    )

    recent_briers = [
        fold["brier"]
        for fold in fold_results
        if fold["validation_season"]
        in (2023, 2024)
    ]

    raw_oof_metrics = calculate_metrics(
        oof["target"],
        oof["prediction_raw"],
    )

    clipped_oof_metrics = (
        calculate_metrics(
            oof["target"],
            oof["prediction_clip"],
        )
    )

    raw_range_diag = (
        calculate_range_diagnostics(
            oof["prediction_raw"]
        )
    )

    summary = {
        "n_oof": int(len(oof)),

        # Fold summary는 clipped prediction 기준
        "mean_fold_brier": float(
            np.mean(briers)
        ),
        "std_fold_brier": float(
            np.std(briers)
        ),
        "weighted_fold_brier": (
            weighted_fold_brier
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
            float(np.mean(recent_briers))
        ),
        "mean_fold_bss": float(
            np.mean(bss_values)
        ),
        "weighted_fold_bss": (
            weighted_fold_bss
        ),

        # pooled OOF raw
        "pooled_oof_raw_brier": float(
            raw_oof_metrics["brier"]
        ),
        "pooled_oof_raw_bss_raw": float(
            raw_oof_metrics["bss_raw"]
        ),
        "pooled_oof_raw_bss": float(
            raw_oof_metrics["bss"]
        ),

        # pooled OOF clipped
        "pooled_oof_clipped_brier": float(
            clipped_oof_metrics["brier"]
        ),
        "pooled_oof_clipped_baseline_brier": float(
            clipped_oof_metrics[
                "baseline_brier"
            ]
        ),
        "pooled_oof_clipped_bss_raw": float(
            clipped_oof_metrics["bss_raw"]
        ),
        "pooled_oof_clipped_bss": float(
            clipped_oof_metrics["bss"]
        ),

        # cb_v002 summary key와 호환
        "pooled_oof_brier": float(
            clipped_oof_metrics["brier"]
        ),
        "pooled_oof_baseline_brier": float(
            clipped_oof_metrics[
                "baseline_brier"
            ]
        ),
        "pooled_oof_bss_raw": float(
            clipped_oof_metrics["bss_raw"]
        ),
        "pooled_oof_bss": float(
            clipped_oof_metrics["bss"]
        ),

        **raw_range_diag,
    }

    # -----------------------------------------------------
    # OOF 저장
    # -----------------------------------------------------

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR
        / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction_raw=oof[
            "prediction_raw"
        ].to_numpy(),
        prediction_clip=oof[
            "prediction_clip"
        ].to_numpy(),
        prediction=oof[
            "prediction"
        ].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "validation_method": (
            "expanding_window"
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "prediction_for_evaluation": (
            "clipped"
        ),
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "features": features,
        "categorical_features": cat_cols,
        "folds": fold_results,
        "summary": summary,
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
            key=lambda x: x[
                "validation_season"
            ],
        )

        return int(
            latest_fold[
                "final_iterations"
            ]
        )

    raise ValueError(
        "Unsupported "
        "FINAL_ITERATION_STRATEGY: "
        f"{FINAL_ITERATION_STRATEGY}"
    )


# =========================================================
# Final Model 학습
# =========================================================

def train_final_model(
    train,
    features,
    cat_cols,
    final_iterations,
):
    X_full = prepare_features(
        train[features],
        cat_cols,
    )

    y_full = (
        train[TARGET]
        .astype(np.float64)
    )

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    final_model = CatBoostRegressor(
        **final_params
    )

    print()
    print("=" * 70)
    print(
        "전체 데이터 "
        "CatBoostRegressor 재학습 시작"
    )
    print(
        f"Final iterations: "
        f"{final_iterations}"
    )
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

    return (
        final_model,
        final_params,
        elapsed,
    )


# =========================================================
# Final metadata 저장
# =========================================================

def save_final_metadata(
    train,
    features,
    cat_cols,
    final_params,
    final_iterations,
    elapsed_seconds,
):
    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "model_type": (
            "CatBoostRegressor"
        ),
        "objective": "RMSE",
        "training_seasons": sorted(
            [
                int(x)
                for x
                in train[
                    "season"
                ].unique()
            ]
        ),
        "n_train": int(len(train)),
        "features": features,
        "categorical_features": (
            cat_cols
        ),
        "model_params": final_params,
        "final_iterations": int(
            final_iterations
        ),
        "final_iteration_strategy": (
            FINAL_ITERATION_STRATEGY
        ),
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "inference_note": (
            "CatBoostRegressor predict() output "
            "may fall outside [0, 1]. "
            "Use np.clip(pred, 0.0, 1.0) "
            "for submission if Validation "
            "policy remains clipped."
        ),
    }

    with open(
        MODEL_DIR
        / "final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
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
    # 데이터
    # -----------------------------------------------------

    # test.csv의 다른 평가 행/분포는 사용하지 않는다.
    # 여기서는 feature schema 확인을 위해 header만 읽는다.
    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    features = [
        c
        for c in test_cols
        if c != ID
    ]

    cat_cols = [
        c
        for c in CAT_COLS
        if c in features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + features
            + [TARGET]
        ),
    )

    print("=" * 70)
    print("데이터 로드 완료")
    print(
        "train shape:",
        train.shape,
    )
    print(
        "Feature 수:",
        len(features),
    )
    print(
        "범주형 Feature 수:",
        len(cat_cols),
    )
    print(
        "범주형 Feature:",
        cat_cols,
    )
    print(
        "시즌:",
        train["season"].min(),
        "~",
        train["season"].max(),
    )
    print(
        f"제구 성공률: "
        f"{train[TARGET].mean():.6f}"
    )
    print(
        "Training objective    : RMSE"
    )
    print(
        "Validation prediction : "
        "raw + clipped"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # 디렉터리 생성
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    FOLD_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Expanding Window Validation
    # -----------------------------------------------------

    fold_results, oof = (
        run_expanding_validation(
            train=train,
            features=features,
            cat_cols=cat_cols,
        )
    )

    summary = save_validation_summary(
        train=train,
        features=features,
        cat_cols=cat_cols,
        fold_results=fold_results,
        oof=oof,
    )

    print()
    print("=" * 70)
    print(
        "Expanding Window 결과 요약"
    )
    print("=" * 70)

    for fold in fold_results:
        print(
            f"Valid "
            f"{fold['validation_season']}: "
            f"Brier="
            f"{fold['brier']:.8f}, "
            f"BSS="
            f"{fold['bss']:.2f}, "
            f"iter="
            f"{fold['final_iterations']}, "
            f"outside="
            f"{fold['ratio_prediction_outside_0_1']:.6%}"
        )

    print(
        "Fold 단순 평균 Brier : "
        f"{summary['mean_fold_brier']:.8f}"
    )
    print(
        "Fold 가중 평균 Brier : "
        f"{summary['weighted_fold_brier']:.8f}"
    )
    print(
        "Fold Brier 표준편차  : "
        f"{summary['std_fold_brier']:.8f}"
    )
    print(
        "Fold Brier 범위       : "
        f"{summary['fold_brier_range']:.8f}"
    )
    print(
        "2023~2024 평균 Brier : "
        f"{summary['recent_2023_2024_mean_brier']:.8f}"
    )
    print(
        "OOF raw Brier        : "
        f"{summary['pooled_oof_raw_brier']:.8f}"
    )
    print(
        "OOF clipped Brier    : "
        f"{summary['pooled_oof_clipped_brier']:.8f}"
    )
    print(
        "OOF clipped BSS      : "
        f"{summary['pooled_oof_clipped_bss']:.2f}"
    )
    print(
        "OOF outside [0,1]    : "
        f"{summary['ratio_prediction_outside_0_1']:.6%}"
    )
    print(
        "Validation 저장      : "
        f"{VALIDATION_DIR}"
    )
    print(
        "Fold 모델 저장       : "
        f"{FOLD_MODEL_DIR}"
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

    print()
    print(
        "Final iteration strategy : "
        f"{FINAL_ITERATION_STRATEGY}"
    )
    print(
        "Final iterations         : "
        f"{final_iterations}"
    )

    # -----------------------------------------------------
    # 전체 데이터 재학습
    # -----------------------------------------------------

    (
        final_model,
        final_params,
        final_elapsed,
    ) = train_final_model(
        train=train,
        features=features,
        cat_cols=cat_cols,
        final_iterations=final_iterations,
    )

    # -----------------------------------------------------
    # 모델 저장
    # -----------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_PATH,
        compress=3,
    )

    save_final_metadata(
        train=train,
        features=features,
        cat_cols=cat_cols,
        final_params=final_params,
        final_iterations=final_iterations,
        elapsed_seconds=final_elapsed,
    )

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
        "Final Model 저장 완료"
    )
    print(
        f"- {MODEL_PATH}"
    )
    print(
        f"- "
        f"{MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        f"- "
        f"{MODEL_DIR / 'train.py'}"
    )
    print("=" * 70)

    print()
    print("[중요]")
    print(
        "CatBoostRegressor의 predict() 출력은 "
        "[0, 1] 범위를 벗어날 수 있습니다."
    )
    print(
        "현재 Validation 정책은 "
        "prediction_clip = np.clip("
        "prediction_raw, 0.0, 1.0)입니다."
    )
    print(
        "script.py에서도 동일한 후처리를 "
        "적용해야 합니다."
    )


if __name__ == "__main__":
    main()
