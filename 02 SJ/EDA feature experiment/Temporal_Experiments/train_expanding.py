"""CatBoost v001 with four-fold expanding-window validation.

The script evaluates temporal stability only. It does not train a final
2019-2024 production model or create submission.csv.
"""

import gc
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

EXPERIMENT_NAME = "cb_v001_ctr1_expanding"


MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2019, 2020],
        "valid_season": 2021,
    },
    {
        "fold": 2,
        "train_seasons": [2019, 2020, 2021],
        "valid_season": 2022,
    },
    {
        "fold": 3,
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_season": 2023,
    },
    {
        "fold": 4,
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "valid_season": 2024,
    },
]


# =========================================================
# CatBoost 설정: v001과 동일
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

def prepare_features(df, cat_cols):
    """Apply the same feature preparation as the original v001 model."""
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


def calculate_metrics(y_true, y_pred):
    """Return Brier, fold baseline Brier, and scaled Brier Skill Score."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))
    bss = float(
        max(
            0.0,
            100000.0 * (1.0 - brier / baseline_brier),
        )
    )

    return {
        "positive_rate": positive_rate,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "bss": bss,
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


def save_fold_validation(
    fold_dir,
    row_ids,
    y_true,
    y_pred,
    metadata,
):
    """Save predictions and metadata for one validation season."""
    fold_dir.mkdir(parents=True, exist_ok=False)

    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pred_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "target": y_true,
            "prediction": y_pred,
        }
    )
    pred_df.to_csv(
        fold_dir / "pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        fold_dir / "validation.npz",
        row_id=row_ids,
        target=y_true,
        prediction=y_pred,
    )

    with (fold_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# =========================================================
# Main
# =========================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation experiment already exists: {VALIDATION_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    if MODEL_DIR.exists():
        raise FileExistsError(
            f"Model experiment already exists: {MODEL_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )

    test_cols = pd.read_csv(
        DATA_DIR / "test.csv",
        encoding="utf-8-sig",
        nrows=0,
    ).columns

    features = [c for c in test_cols if c != ID]
    cat_cols = [c for c in CAT_COLS if c in features]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[ID] + features + [TARGET],
    )

    available_seasons = sorted(
        int(season) for season in train["season"].unique()
    )
    required_seasons = sorted(
        {
            season
            for fold in FOLDS
            for season in (
                fold["train_seasons"] + [fold["valid_season"]]
            )
        }
    )
    missing_seasons = sorted(set(required_seasons) - set(available_seasons))
    if missing_seasons:
        raise ValueError(f"데이터에 없는 시즌: {missing_seasons}")

    print("=" * 70)
    print("Expanding Window 데이터 로드 완료")
    print("train shape:", train.shape)
    print("Feature 수:", len(features))
    print("범주형 Feature 수:", len(cat_cols))
    print("시즌:", available_seasons)
    print("전체 제구 성공률:", f"{train[TARGET].mean():.6f}")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)

    shutil.copy2(
        Path(__file__).resolve(),
        MODEL_DIR / "train_expanding.py",
    )

    fold_results = []
    oof_frames = []

    for fold_config in FOLDS:
        fold_number = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        valid_season = fold_config["valid_season"]

        is_train = train["season"].isin(train_seasons)
        is_valid = train["season"] == valid_season

        if not is_train.any() or not is_valid.any():
            raise ValueError(
                f"Fold {fold_number}의 학습 또는 검증 데이터가 비었습니다."
            )

        X_train = prepare_features(
            train.loc[is_train, features],
            cat_cols,
        )
        y_train = train.loc[is_train, TARGET]

        X_valid = prepare_features(
            train.loc[is_valid, features],
            cat_cols,
        )
        y_valid = train.loc[is_valid, TARGET]
        valid_row_ids = train.loc[is_valid, ID]

        estimator = CatBoostClassifier(**MODEL_PARAMS)

        print()
        print("=" * 70)
        print(
            f"Fold {fold_number}: {train_seasons[0]}~{train_seasons[-1]} "
            f"Train -> {valid_season} Validation"
        )
        print("학습 데이터:", X_train.shape)
        print("검증 데이터:", X_valid.shape)
        print("=" * 70)

        start_time = time.time()
        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        elapsed_seconds = float(time.time() - start_time)

        valid_pred = estimator.predict_proba(X_valid)[:, 1]
        metrics = calculate_metrics(y_valid.to_numpy(), valid_pred)

        best_iteration = estimator.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            final_iterations = MODEL_PARAMS["iterations"]
        else:
            final_iterations = int(best_iteration + 1)

        fold_name = f"fold_{valid_season}"
        fold_model_dir = MODEL_DIR / fold_name
        fold_validation_dir = VALIDATION_DIR / fold_name

        fold_model_dir.mkdir(parents=True, exist_ok=False)
        model_path = (
            fold_model_dir
            / f"{EXPERIMENT_NAME}_{fold_name}.pkl"
        )
        joblib.dump(estimator, model_path, compress=3)

        fold_metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "fold": int(fold_number),
            "train_seasons": [int(x) for x in train_seasons],
            "validation_season": int(valid_season),
            "n_train": int(is_train.sum()),
            "n_validation": int(is_valid.sum()),
            "model_type": type(estimator).__name__,
            "model_params": MODEL_PARAMS,
            **metrics,
            "best_iteration": int(best_iteration),
            "final_iterations": int(final_iterations),
            "elapsed_seconds": elapsed_seconds,
            "model_path": str(model_path),
        }

        save_fold_validation(
            fold_validation_dir,
            valid_row_ids,
            y_valid,
            valid_pred,
            fold_metadata,
        )

        fold_results.append(fold_metadata)
        oof_frames.append(
            pd.DataFrame(
                {
                    "row_id": valid_row_ids.to_numpy(),
                    "season": int(valid_season),
                    "target": y_valid.to_numpy(),
                    "prediction": valid_pred,
                    "fold": int(fold_number),
                }
            )
        )

        print(f"Brier Score      : {metrics['brier']:.8f}")
        print(f"Baseline Brier   : {metrics['baseline_brier']:.8f}")
        print(f"Validation Score : {metrics['bss']:.2f}")
        print(f"Best iteration   : {best_iteration}")
        print(f"학습 시간          : {elapsed_seconds:.1f}초")
        print(f"Model            : {model_path}")

        del X_train, y_train, X_valid, y_valid, estimator
        gc.collect()

    oof = pd.concat(oof_frames, ignore_index=True)
    oof_metrics = calculate_metrics(
        oof["target"].to_numpy(),
        oof["prediction"].to_numpy(),
    )

    fold_briers = np.asarray(
        [fold["brier"] for fold in fold_results],
        dtype=np.float64,
    )
    fold_bss = np.asarray(
        [fold["bss"] for fold in fold_results],
        dtype=np.float64,
    )
    fold_sizes = np.asarray(
        [fold["n_validation"] for fold in fold_results],
        dtype=np.float64,
    )
    recent_briers = np.asarray(
        [
            fold["brier"]
            for fold in fold_results
            if fold["validation_season"] in (2023, 2024)
        ],
        dtype=np.float64,
    )

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "validation_method": "expanding_window",
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "features": features,
        "categorical_features": cat_cols,
        "folds": fold_results,
        "summary": {
            "n_oof": int(len(oof)),
            "mean_fold_brier": float(np.mean(fold_briers)),
            "std_fold_brier": float(np.std(fold_briers)),
            "weighted_fold_brier": float(
                np.average(fold_briers, weights=fold_sizes)
            ),
            "recent_2023_2024_mean_brier": float(
                np.mean(recent_briers)
            ),
            "mean_fold_bss": float(np.mean(fold_bss)),
            "weighted_fold_bss": float(
                np.average(fold_bss, weights=fold_sizes)
            ),
            "pooled_oof_brier": oof_metrics["brier"],
            "pooled_oof_baseline_brier": oof_metrics[
                "baseline_brier"
            ],
            "pooled_oof_bss": oof_metrics["bss"],
        },
    }

    oof.to_csv(
        VALIDATION_DIR / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        VALIDATION_DIR / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(),
        season=oof["season"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction=oof["prediction"].to_numpy(),
        fold=oof["fold"].to_numpy(),
    )
    with (VALIDATION_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

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
    print(f"Fold 단순 평균 Brier : {np.mean(fold_briers):.8f}")
    print(
        "Fold 가중 평균 Brier : "
        f"{np.average(fold_briers, weights=fold_sizes):.8f}"
    )
    print(f"Fold Brier 표준편차  : {np.std(fold_briers):.8f}")
    print(f"2023~2024 평균 Brier: {np.mean(recent_briers):.8f}")
    print(f"OOF pooled Brier    : {oof_metrics['brier']:.8f}")
    print(f"Validation 저장      : {VALIDATION_DIR}")
    print(f"Fold 모델 저장       : {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
