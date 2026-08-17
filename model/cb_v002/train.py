# train.py: CatBoost cb_v002
# cb_v001의 구조를 유지하면서 Expanding Window Validation을 적용한다.

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

EXPERIMENT_NAME = "cb_v002"

DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

FOLD_MODEL_DIR = MODEL_DIR / EXPERIMENT_NAME

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

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

# Final Model의 iteration은 hidden future와 가장 가까운
# 2019~2023 -> 2024 Fold의 best iteration을 사용한다.
FINAL_ITERATION_STRATEGY = "latest_fold"


# =========================================================
# Feature 전처리
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

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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
            100000.0 * (
                1.0 - brier / baseline_brier
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
    y_pred,
    metrics,
    output_root,
    model_path,
    n_train,
    elapsed_seconds,
    best_iteration=None,
    final_iterations=None,
):
    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert len(row_ids) == len(y_true) == len(y_pred)

    output_dir = (
        Path(output_root)
        / experiment_name
        / f"fold_{validation_season}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction": y_pred,
        "validation_season": validation_season,
    })

    pred_df.to_csv(
        output_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

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

    metadata = {
        "experiment_name": experiment_name,
        "fold": int(fold),
        "train_seasons": [
            int(s) for s in train_seasons
        ],
        "validation_season": int(validation_season),
        "n_train": int(n_train),
        "n_validation": int(len(y_true)),
        "model_type": type(estimator).__name__,
        "model_params": model_params,
        "positive_rate": float(
            metrics["positive_rate"]
        ),
        "brier": float(metrics["brier"]),
        "baseline_brier": float(
            metrics["baseline_brier"]
        ),
        "bss_raw": float(metrics["bss_raw"]),
        "bss": float(metrics["bss"]),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "elapsed_seconds": float(elapsed_seconds),
        "model_path": str(model_path),
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
        train_seasons = fold_config["train_seasons"]
        validation_season = (
            fold_config["validation_season"]
        )

        train_mask = train["season"].isin(
            train_seasons
        )
        val_mask = (
            train["season"] == validation_season
        )

        X_train = prepare_features(
            train.loc[
                train_mask,
                features,
            ],
            cat_cols,
        )

        y_train = train.loc[
            train_mask,
            TARGET,
        ]

        X_val = prepare_features(
            train.loc[
                val_mask,
                features,
            ],
            cat_cols,
        )

        y_val = train.loc[
            val_mask,
            TARGET,
        ]

        val_row_id = train.loc[
            val_mask,
            ID,
        ]

        print()
        print("=" * 70)
        print(
            f"Fold {fold}: "
            f"{train_seasons[0]}~{train_seasons[-1]} "
            f"-> {validation_season}"
        )
        print("=" * 70)
        print("학습 데이터:", X_train.shape)
        print("검증 데이터:", X_val.shape)

        estimator = CatBoostClassifier(
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

        elapsed = time.time() - start_time

        val_pred = estimator.predict_proba(
            X_val
        )[:, 1]

        metrics = calculate_metrics(
            y_val,
            val_pred,
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

        pred_df, metadata = save_fold_validation(
            experiment_name=EXPERIMENT_NAME,
            fold=fold,
            train_seasons=train_seasons,
            validation_season=validation_season,
            estimator=estimator,
            model_params=MODEL_PARAMS,
            row_ids=val_row_id,
            y_true=y_val,
            y_pred=val_pred,
            metrics=metrics,
            output_root=VALIDATION_ROOT,
            model_path=fold_model_path,
            n_train=len(y_train),
            elapsed_seconds=elapsed,
            best_iteration=best_iteration,
            final_iterations=final_iterations,
        )

        fold_results.append(metadata)
        oof_frames.append(pred_df)

        print(
            f"Brier Score      : "
            f"{metrics['brier']:.8f}"
        )
        print(
            f"Baseline Brier   : "
            f"{metrics['baseline_brier']:.8f}"
        )
        print(
            f"BSS Raw          : "
            f"{metrics['bss_raw']:.2f}"
        )
        print(
            f"BSS              : "
            f"{metrics['bss']:.2f}"
        )
        print(
            f"Prediction mean  : "
            f"{val_pred.mean():.6f}"
        )
        print(
            f"Prediction std   : "
            f"{val_pred.std():.6f}"
        )
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

    oof_metrics = calculate_metrics(
        oof["target"],
        oof["prediction"],
    )

    summary = {
        "n_oof": int(len(oof)),
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
            np.max(briers) - np.min(briers)
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
        "pooled_oof_brier": float(
            oof_metrics["brier"]
        ),
        "pooled_oof_baseline_brier": float(
            oof_metrics["baseline_brier"]
        ),
        "pooled_oof_bss_raw": float(
            oof_metrics["bss_raw"]
        ),
        "pooled_oof_bss": float(
            oof_metrics["bss"]
        ),
    }

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VALIDATION_DIR / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction=oof[
            "prediction"
        ].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "validation_method": (
            "expanding_window"
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
        VALIDATION_DIR / "metadata.json",
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
    if FINAL_ITERATION_STRATEGY == "latest_fold":
        latest_fold = max(
            fold_results,
            key=lambda x: x[
                "validation_season"
            ],
        )
        return int(
            latest_fold["final_iterations"]
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

    y_full = train[TARGET]

    final_params = {
        **MODEL_PARAMS,
        "iterations": int(
            final_iterations
        ),
    }

    final_model = CatBoostClassifier(
        **final_params
    )

    print()
    print("=" * 70)
    print("전체 데이터 재학습 시작")
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

    elapsed = time.time() - start_time

    print(
        f"전체 데이터 재학습 완료: "
        f"{elapsed:.1f}초"
    )

    return final_model, final_params, elapsed


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
        "experiment_name": EXPERIMENT_NAME,
        "model_type": "CatBoostClassifier",
        "training_seasons": sorted(
            [
                int(x)
                for x in train[
                    "season"
                ].unique()
            ]
        ),
        "n_train": int(len(train)),
        "features": features,
        "categorical_features": cat_cols,
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
        "model_path": str(MODEL_PATH),
    }

    with open(
        FOLD_MODEL_DIR
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

    if FOLD_MODEL_DIR.exists():
        raise FileExistsError(
            "Fold model experiment already exists: "
            f"{FOLD_MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_PATH.exists():
        raise FileExistsError(
            "Final model already exists: "
            f"{MODEL_PATH}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # 데이터
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    features = [
        c for c in test_cols
        if c != ID
    ]

    cat_cols = [
        c for c in CAT_COLS
        if c in features
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[ID] + features + [TARGET],
    )

    print("=" * 70)
    print("데이터 로드 완료")
    print("train shape:", train.shape)
    print("Feature 수:", len(features))
    print("범주형 Feature 수:", len(cat_cols))
    print("범주형 Feature:", cat_cols)
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
    print("=" * 70)

    # -----------------------------------------------------
    # 디렉터리 생성
    # -----------------------------------------------------

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    FOLD_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
    print("Expanding Window 결과 요약")
    print("=" * 70)

    for fold in fold_results:
        print(
            f"Valid {fold['validation_season']}: "
            f"Brier={fold['brier']:.8f}, "
            f"BSS={fold['bss']:.2f}, "
            f"iter={fold['final_iterations']}"
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
        "2023~2024 평균 Brier: "
        f"{summary['recent_2023_2024_mean_brier']:.8f}"
    )
    print(
        "OOF pooled Brier    : "
        f"{summary['pooled_oof_brier']:.8f}"
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

    source_path = Path(__file__).resolve()

    shutil.copy2(
        source_path,
        FOLD_MODEL_DIR / "train.py",
    )

    print()
    print("=" * 70)
    print("Final Model 저장 완료")
    print(f"- {MODEL_PATH}")
    print(
        f"- "
        f"{FOLD_MODEL_DIR / 'final_metadata.json'}"
    )
    print(
        f"- {FOLD_MODEL_DIR / 'train.py'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
