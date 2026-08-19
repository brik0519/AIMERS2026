# train_expanding_exp3.py
"""
LG Aimers Phase2 - CatBoost EXP3 Temporal Feature Expanding Window Validation

목적
----
현재 검증 기준인 CatBoost cb_v001_ctr1_expanding과 모델 조건을 동일하게 유지하고,
EXP3 Temporal Feature 10개만 추가하여 4-Fold Expanding Window Validation을 수행한다.

중요
----
- Feature만 변경한다.
- CatBoost 하이퍼파라미터, seed, Early Stopping, Fold 구성은 Basic과 동일하다.
- max_ctr_complexity=1을 유지한다.
- Test 내부 다른 row를 사용한 groupby/rolling/lag/집계를 수행하지 않는다.
- Temporal 10개는 각 row에 이미 제공된 공식 asof_* 값끼리의 row-wise 연산만 사용한다.
- 최종 2019~2024 전체 재학습/제출 모델 생성은 하지 않는다.
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
# 경로
# =========================================================

EXPERIMENT_NAME = "cb_v001_ctr1_expanding_exp3"

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

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
# CatBoost 설정
# Basic cb_v001_ctr1_expanding과 동일
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
# EXP3 Temporal Feature
# =========================================================

TEMPORAL_FEATURES = [
    # Recent vs Cumulative - Success
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",

    # Recent vs Cumulative - Middle
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",

    # Success Trend
    "success_trend_1_3",
    "success_trend_3_5",

    # Middle Trend
    "middle_trend_1_3",
    "middle_trend_3_5",
]


def add_temporal_features(df):
    """
    EXP3 Temporal Feature 10개를 생성한다.

    모든 계산은 동일 row의 공식 asof_* 값끼리 수행한다.
    따라서 test의 다른 row를 사용하지 않는다.

    변수명에는 기존 실험과의 호환성을 위해 *_vs_career를 유지하지만,
    의미는 최종 career가 아니라 '해당 시점까지의 cumulative(as-of) 값'이다.
    """
    df = df.copy()

    required_cols = [
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Temporal Feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing_cols)
        )

    # Recent vs Cumulative - Success
    df["recent1_success_vs_career"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )
    df["recent3_success_vs_career"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )
    df["recent5_success_vs_career"] = (
        df["asof_pitcher_prev5_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    # Recent vs Cumulative - Middle
    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )
    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )
    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    # Success Trend
    df["success_trend_1_3"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev3_game_success_rate"]
    )
    df["success_trend_3_5"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )

    # Middle Trend
    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )
    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


# =========================================================
# Data QA
# =========================================================

def validate_temporal_features(before_df, after_df, raw_features):
    """
    Feature 생성 과정이 의도대로 동작하는지 검증한다.

    검사:
    - row 수/row_id 순서 보존
    - row_id unique
    - Target 결측 없음
    - Raw 47 + Temporal 10 = 57
    - 10개 수식 정확성
    - inf/-inf 없음
    - NaN 개수 기록
    """
    assert len(before_df) == len(after_df), "Feature 생성 후 row 수가 변했습니다."
    assert before_df[ID_COL].equals(after_df[ID_COL]), "row_id 순서가 변했습니다."
    assert after_df[ID_COL].is_unique, "row_id 중복이 있습니다."
    assert after_df[TARGET_COL].notna().all(), "Target에 결측이 있습니다."

    if len(raw_features) != 47:
        raise ValueError(
            f"Raw Feature 수가 예상(47)과 다릅니다: {len(raw_features)}"
        )

    total_features = list(raw_features) + TEMPORAL_FEATURES
    if len(total_features) != 57:
        raise ValueError(
            f"EXP3 전체 Feature 수가 예상(57)과 다릅니다: {len(total_features)}"
        )

    expected_map = {
        "recent1_success_vs_career": (
            after_df["asof_pitcher_prev1_game_success_rate"]
            - after_df["asof_pitcher_success_rate"]
        ),
        "recent3_success_vs_career": (
            after_df["asof_pitcher_prev3_game_success_rate"]
            - after_df["asof_pitcher_success_rate"]
        ),
        "recent5_success_vs_career": (
            after_df["asof_pitcher_prev5_game_success_rate"]
            - after_df["asof_pitcher_success_rate"]
        ),
        "recent1_middle_vs_career": (
            after_df["asof_pitcher_prev1_game_middle_rate"]
            - after_df["asof_pitcher_middle_rate"]
        ),
        "recent3_middle_vs_career": (
            after_df["asof_pitcher_prev3_game_middle_rate"]
            - after_df["asof_pitcher_middle_rate"]
        ),
        "recent5_middle_vs_career": (
            after_df["asof_pitcher_prev5_game_middle_rate"]
            - after_df["asof_pitcher_middle_rate"]
        ),
        "success_trend_1_3": (
            after_df["asof_pitcher_prev1_game_success_rate"]
            - after_df["asof_pitcher_prev3_game_success_rate"]
        ),
        "success_trend_3_5": (
            after_df["asof_pitcher_prev3_game_success_rate"]
            - after_df["asof_pitcher_prev5_game_success_rate"]
        ),
        "middle_trend_1_3": (
            after_df["asof_pitcher_prev1_game_middle_rate"]
            - after_df["asof_pitcher_prev3_game_middle_rate"]
        ),
        "middle_trend_3_5": (
            after_df["asof_pitcher_prev3_game_middle_rate"]
            - after_df["asof_pitcher_prev5_game_middle_rate"]
        ),
    }

    for feature, expected in expected_map.items():
        actual = after_df[feature]
        if not np.allclose(
            expected.to_numpy(dtype=np.float64),
            actual.to_numpy(dtype=np.float64),
            equal_nan=True,
        ):
            raise AssertionError(f"Temporal Feature 수식 불일치: {feature}")

    temporal_array = after_df[TEMPORAL_FEATURES].to_numpy(dtype=np.float64)
    inf_counts = dict(
        zip(
            TEMPORAL_FEATURES,
            np.isinf(temporal_array).sum(axis=0).astype(int).tolist(),
        )
    )
    if any(count > 0 for count in inf_counts.values()):
        raise ValueError(f"Temporal Feature에 inf/-inf가 있습니다: {inf_counts}")

    nan_counts = {
        feature: int(after_df[feature].isna().sum())
        for feature in TEMPORAL_FEATURES
    }

    qa = {
        "n_rows_before": int(len(before_df)),
        "n_rows_after": int(len(after_df)),
        "row_id_unique": bool(after_df[ID_COL].is_unique),
        "raw_feature_count": int(len(raw_features)),
        "temporal_feature_count": int(len(TEMPORAL_FEATURES)),
        "total_feature_count": int(len(total_features)),
        "temporal_nan_counts": nan_counts,
        "temporal_inf_counts": inf_counts,
    }

    return qa


# =========================================================
# CatBoost 전처리 / Metric / 저장
# =========================================================

def prepare_features(df, cat_cols):
    """Basic과 동일하게 범주형 결측만 문자열로 처리한다."""
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


def calculate_metrics(y_true, y_pred):
    """Brier, fold baseline Brier, Brier Skill Score를 계산한다."""
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
    """한 Fold의 OOF prediction과 metadata를 저장한다."""
    fold_dir.mkdir(parents=True, exist_ok=False)

    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert len(row_ids) == len(y_true) == len(y_pred)

    pd.DataFrame(
        {
            "row_id": row_ids,
            "target": y_true,
            "prediction": y_pred,
        }
    ).to_csv(
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
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# Main
# =========================================================

def main():
    experiment_model_dir = MODEL_PATH.parent

    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"Validation experiment already exists: {VALIDATION_DIR}\n"
            "기존 결과를 보존하고 새 EXPERIMENT_NAME을 사용하세요."
        )

    if experiment_model_dir.exists():
        raise FileExistsError(
            f"Model experiment already exists: {experiment_model_dir}\n"
            "기존 결과를 보존하고 새 EXPERIMENT_NAME을 사용하세요."
        )

    # -----------------------------------------------------
    # Raw Feature는 공식 test.csv 구조로 결정
    # -----------------------------------------------------

    test_cols = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    if ID_COL not in test_cols:
        raise ValueError(f"test.csv에 '{ID_COL}' 컬럼이 없습니다.")

    raw_features = [
        col for col in test_cols
        if col != ID_COL
    ]

    # -----------------------------------------------------
    # Train 로드
    # -----------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[ID_COL] + raw_features + [TARGET_COL],
    )

    if "season" not in train.columns:
        raise ValueError("train.csv에 season 컬럼이 없습니다.")

    before = train.copy()
    train = add_temporal_features(train)
    feature_qa = validate_temporal_features(
        before_df=before,
        after_df=train,
        raw_features=raw_features,
    )
    del before
    gc.collect()

    features = raw_features + TEMPORAL_FEATURES
    cat_cols = [col for col in CAT_COLS if col in features]

    available_seasons = sorted(
        int(season)
        for season in train["season"].unique()
    )
    required_seasons = sorted(
        {
            season
            for fold in FOLDS
            for season in (
                fold["train_seasons"]
                + [fold["valid_season"]]
            )
        }
    )
    missing_seasons = sorted(
        set(required_seasons)
        - set(available_seasons)
    )
    if missing_seasons:
        raise ValueError(f"데이터에 없는 시즌: {missing_seasons}")

    print("=" * 70)
    print("CatBoost EXP3 Expanding Window")
    print("=" * 70)
    print("train shape        :", train.shape)
    print("Raw Feature        :", len(raw_features))
    print("Temporal Feature   :", len(TEMPORAL_FEATURES))
    print("Total Feature      :", len(features))
    print("Categorical Feature:", len(cat_cols))
    print("Season             :", available_seasons)
    print("전체 제구 성공률     :", f"{train[TARGET_COL].mean():.6f}")
    print("Temporal NaN counts:", feature_qa["temporal_nan_counts"])
    print("=" * 70)

    experiment_model_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        experiment_model_dir / "train_expanding_exp3.py",
    )

    fold_results = []
    oof_frames = []

    for fold_config in FOLDS:
        fold_number = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        valid_season = fold_config["valid_season"]

        # 시간 순서 QA
        assert max(train_seasons) < valid_season

        is_train = train["season"].isin(train_seasons)
        is_valid = train["season"] == valid_season

        if not is_train.any() or not is_valid.any():
            raise ValueError(
                f"Fold {fold_number}의 학습 또는 검증 데이터가 비었습니다."
            )

        train_row_ids = train.loc[is_train, ID_COL]
        valid_row_ids = train.loc[is_valid, ID_COL]

        if not set(train_row_ids).isdisjoint(set(valid_row_ids)):
            raise AssertionError(
                f"Fold {fold_number}: Train/Validation row_id가 겹칩니다."
            )

        X_train = prepare_features(
            train.loc[is_train, features],
            cat_cols,
        )
        y_train = train.loc[is_train, TARGET_COL]

        X_valid = prepare_features(
            train.loc[is_valid, features],
            cat_cols,
        )
        y_valid = train.loc[is_valid, TARGET_COL]

        estimator = CatBoostClassifier(**MODEL_PARAMS)

        print()
        print("=" * 70)
        print(
            f"Fold {fold_number}: "
            f"{train_seasons[0]}~{train_seasons[-1]} Train "
            f"-> {valid_season} Validation"
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

        elapsed_seconds = float(
            time.time()
            - start_time
        )

        valid_pred = estimator.predict_proba(
            X_valid
        )[:, 1]

        metrics = calculate_metrics(
            y_valid.to_numpy(),
            valid_pred,
        )

        best_iteration = estimator.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            final_iterations = MODEL_PARAMS["iterations"]
        else:
            final_iterations = int(best_iteration + 1)

        fold_name = f"fold_{valid_season}"

        fold_model_dir = (
            experiment_model_dir
            / fold_name
        )
        fold_validation_dir = (
            VALIDATION_DIR
            / fold_name
        )

        fold_model_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_model_path = (
            fold_model_dir
            / f"{EXPERIMENT_NAME}_{fold_name}.pkl"
        )

        joblib.dump(
            estimator,
            fold_model_path,
            compress=3,
        )

        fold_metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "feature_mode": "EXP3_TEMPORAL",
            "temporal_features": TEMPORAL_FEATURES,
            "fold": int(fold_number),
            "train_seasons": [int(x) for x in train_seasons],
            "validation_season": int(valid_season),
            "n_train": int(is_train.sum()),
            "n_validation": int(is_valid.sum()),
            "n_features": int(len(features)),
            "model_type": type(estimator).__name__,
            "model_params": MODEL_PARAMS,
            **metrics,
            "best_iteration": int(best_iteration),
            "final_iterations": int(final_iterations),
            "elapsed_seconds": elapsed_seconds,
            "model_path": str(fold_model_path),
        }

        save_fold_validation(
            fold_dir=fold_validation_dir,
            row_ids=valid_row_ids,
            y_true=y_valid,
            y_pred=valid_pred,
            metadata=fold_metadata,
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
        print(f"Model            : {fold_model_path}")

        del (
            X_train,
            y_train,
            X_valid,
            y_valid,
            estimator,
            train_row_ids,
            valid_row_ids,
        )
        gc.collect()

    # -----------------------------------------------------
    # OOF 통합
    # -----------------------------------------------------

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    if not oof[ID_COL].is_unique:
        raise AssertionError("OOF row_id가 중복되었습니다.")

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
        "feature_mode": "EXP3_TEMPORAL",
        "model_params": MODEL_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "features": features,
        "categorical_features": cat_cols,
        "temporal_features": TEMPORAL_FEATURES,
        "feature_qa": feature_qa,
        "folds": fold_results,
        "summary": {
            "n_oof": int(len(oof)),
            "mean_fold_brier": float(np.mean(fold_briers)),
            "std_fold_brier": float(np.std(fold_briers)),
            "weighted_fold_brier": float(
                np.average(
                    fold_briers,
                    weights=fold_sizes,
                )
            ),
            "recent_2023_2024_mean_brier": float(
                np.mean(recent_briers)
            ),
            "mean_fold_bss": float(np.mean(fold_bss)),
            "weighted_fold_bss": float(
                np.average(
                    fold_bss,
                    weights=fold_sizes,
                )
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
        row_id=oof[ID_COL].to_numpy(),
        season=oof["season"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction=oof["prediction"].to_numpy(),
        fold=oof["fold"].to_numpy(),
    )

    with (
        VALIDATION_DIR / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("EXP3 Expanding Window 결과 요약")
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
    print(f"Fold 모델 저장       : {experiment_model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
