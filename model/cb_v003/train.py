# train.py: CatBoost cb_v003
# HB-EB Rate 피처를 적용하고, 추론에 필요한 모든 객체를 하나의 모델 파일로 통합한다.
# 프로젝트 표준 경로 선언 규칙을 엄격히 준수한다.

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
EXPERIMENT_NAME = "cb_v004"

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

# Fold별 모델 저장을 위한 파생 경로 (표준 경로 기반 조합)
FOLD_MODEL_DIR = MODEL_DIR / EXPERIMENT_NAME


# =========================================================
# 기본 설정
# =========================================================
ID = "row_id"
TARGET = "control_success"

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_id", 
    "batter_id", "pitcher_team_id", "batter_team_id", 
    "pitcher_hand", "batter_hand",
]

EXPANDING_FOLDS = [
    {"fold": 1, "train_seasons": [2019, 2020], "validation_season": 2021},
    {"fold": 2, "train_seasons": [2019, 2020, 2021], "validation_season": 2022},
    {"fold": 3, "train_seasons": [2019, 2020, 2021, 2022], "validation_season": 2023},
    {"fold": 4, "train_seasons": [2019, 2020, 2021, 2022, 2023], "validation_season": 2024},
]

# =========================================================
# HB-EB 교체 설정
# =========================================================
HBEB_CONFIGS = [
    {
        "group_cols": ["pitcher_hand"],
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "new_col": "hbeb_pitcher_success_rate",
    },
    {
        "group_cols": ["batter_hand"],
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "new_col": "hbeb_batter_success_rate",
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
FINAL_ITERATION_STRATEGY = "latest_fold"

# =========================================================
# HB-EB (계층적 베이즈) 기능 구현
# =========================================================
def compute_hbeb_priors(df_train, group_cols, rate_col):
    # 1. Global Stats
    g_mu = df_train[rate_col].mean()
    g_var = df_train[rate_col].var()
    
    if pd.isna(g_var) or g_var >= g_mu * (1 - g_mu):
        g_var = g_mu * (1 - g_mu) * 0.5 
        
    g_m = (g_mu * (1 - g_mu) / g_var) - 1
    g_alpha = g_mu * g_m
    g_beta = (1 - g_mu) * g_m
    
    # 2. Group Stats
    stats = df_train.groupby(group_cols)[rate_col].agg(['mean', 'var', 'count']).reset_index()
    valid_mask = (stats['var'].notna()) & (stats['var'] < stats['mean'] * (1 - stats['mean'])) & (stats['count'] >= 30)
    
    stats['alpha'] = g_alpha
    stats['beta'] = g_beta
    
    if valid_mask.any():
        mu = stats.loc[valid_mask, 'mean']
        var = stats.loc[valid_mask, 'var']
        m = (mu * (1 - mu) / var) - 1
        stats.loc[valid_mask, 'alpha'] = mu * m
        stats.loc[valid_mask, 'beta'] = (1 - mu) * m
        
    prior_df = stats[group_cols + ['alpha', 'beta']].copy()
    return prior_df, (g_alpha, g_beta)

def apply_hbeb(df, group_cols, n_col, rate_col, prior_df, global_prior):
    df_merged = df.merge(prior_df, on=group_cols, how='left')
    
    alpha = df_merged['alpha'].fillna(global_prior[0]).values
    beta = df_merged['beta'].fillna(global_prior[1]).values
    
    S = df[n_col].values * df[rate_col].fillna(0).values
    N = df[n_col].values
    
    eb_rate = (S + alpha) / (N + alpha + beta)
    return eb_rate

# =========================================================
# 기본 기능 (전처리 및 평가지표)
# =========================================================
def prepare_features(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna("__MISSING__").astype(str)
    return df

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    positive_rate = float(np.mean(y_true))
    brier = float(np.mean((y_pred - y_true) ** 2))
    baseline_brier = float(positive_rate * (1.0 - positive_rate))

    if baseline_brier <= 0.0:
        bss_raw, bss = 0.0, 0.0
    else:
        bss_raw = float(100000.0 * (1.0 - brier / baseline_brier))
        bss = max(0.0, bss_raw)

    return {"positive_rate": positive_rate, "brier": brier, "baseline_brier": baseline_brier, "bss_raw": bss_raw, "bss": bss}


# =========================================================
# Validation 결과 저장
# =========================================================
def save_fold_validation(
    validation_root,
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
    elapsed_seconds,
    best_iteration,
    final_iterations,
):
    """
    기존 학습/검증 로직에는 영향을 주지 않고,
    fold별 Validation 결과만 별도 저장한다.
    """

    fold_dir = (
        Path(validation_root)
        / experiment_name
        / f"fold_{validation_season}"
    )
    fold_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    row_ids = np.asarray(row_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if not (
        len(row_ids)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "row_id, target, prediction 길이가 일치하지 않습니다."
        )

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "target": y_true,
        "prediction": y_pred,
        "validation_season": validation_season,
    })

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
        validation_season=np.full(
            len(y_true),
            validation_season,
        ),
    )

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
        "n_validation": int(
            len(y_true)
        ),
        "model_type": type(estimator).__name__,
        "model_params": model_params,
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
        "best_iteration": int(
            best_iteration
        ),
        "final_iterations": int(
            final_iterations
        ),
    }

    with open(
        fold_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata


def save_validation_summary(
    validation_dir,
    experiment_name,
    fold_results,
    oof,
    oof_metrics,
):
    """
    fold 결과와 pooled OOF를 Validation experiment 단위로 저장한다.
    """

    validation_dir = Path(
        validation_dir
    )

    oof.to_csv(
        validation_dir / "oof_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        validation_dir / "oof_validation.npz",
        row_id=oof["row_id"].to_numpy(),
        target=oof["target"].to_numpy(),
        prediction=oof["prediction"].to_numpy(),
        validation_season=oof[
            "validation_season"
        ].to_numpy(),
    )

    fold_briers = np.asarray(
        [
            x["brier"]
            for x in fold_results
        ],
        dtype=float,
    )

    fold_bss = np.asarray(
        [
            x["bss"]
            for x in fold_results
        ],
        dtype=float,
    )

    summary = {
        "experiment_name": experiment_name,
        "validation_method": "expanding_window",
        "n_oof": int(len(oof)),
        "folds": fold_results,
        "summary": {
            "mean_fold_brier": float(
                np.mean(fold_briers)
            ),
            "std_fold_brier": float(
                np.std(fold_briers)
            ),
            "best_fold_brier": float(
                np.min(fold_briers)
            ),
            "worst_fold_brier": float(
                np.max(fold_briers)
            ),
            "mean_fold_bss": float(
                np.mean(fold_bss)
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
        },
    }

    with open(
        validation_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

# =========================================================
# Expanding Window Validation
# =========================================================
def run_expanding_validation(train, base_features, cat_cols):
    fold_results, oof_frames = [], []

    for fold_config in EXPANDING_FOLDS:
        fold = fold_config["fold"]
        train_seasons = fold_config["train_seasons"]
        validation_season = fold_config["validation_season"]

        train_mask = train["season"].isin(train_seasons)
        val_mask = train["season"] == validation_season

        df_train = train.loc[train_mask].copy()
        df_val = train.loc[val_mask].copy()
        
        current_features = base_features.copy()
        for config in HBEB_CONFIGS:
            prior_df, global_prior = compute_hbeb_priors(df_train, config["group_cols"], config["rate_col"])
            
            df_train[config["new_col"]] = apply_hbeb(df_train, config["group_cols"], config["n_col"], config["rate_col"], prior_df, global_prior)
            df_val[config["new_col"]] = apply_hbeb(df_val, config["group_cols"], config["n_col"], config["rate_col"], prior_df, global_prior)
            
            if config["rate_col"] in current_features:
                current_features.remove(config["rate_col"])
            current_features.append(config["new_col"])
            
        X_train = prepare_features(df_train[current_features], cat_cols)
        y_train = df_train[TARGET]
        X_val = prepare_features(df_val[current_features], cat_cols)
        y_val = df_val[TARGET]
        val_row_id = df_val[ID]

        print(f"\n{'=' * 70}\nFold {fold}: {train_seasons[0]}~{train_seasons[-1]} -> {validation_season}")
        
        estimator = CatBoostClassifier(**MODEL_PARAMS)
        start_time = time.time()
        
        estimator.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True)
        
        elapsed = time.time() - start_time
        val_pred = estimator.predict_proba(X_val)[:, 1]
        metrics = calculate_metrics(y_val, val_pred)
        best_iteration = estimator.get_best_iteration()
        final_iterations = best_iteration + 1 if best_iteration and best_iteration >= 0 else MODEL_PARAMS["iterations"]

        pred_df = pd.DataFrame({"row_id": val_row_id, "target": y_val, "prediction": val_pred, "validation_season": validation_season})

        fold_metadata = save_fold_validation(
            validation_root=VALIDATION_ROOT,
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
            elapsed_seconds=elapsed,
            best_iteration=best_iteration,
            final_iterations=final_iterations,
        )

        fold_results.append({
            **fold_metadata,
        })
        oof_frames.append(pred_df)

        print(f"Brier Score      : {metrics['brier']:.8f}")
        print(f"BSS              : {metrics['bss']:.2f}")

    return fold_results, pd.concat(oof_frames, ignore_index=True), current_features

# =========================================================
# Main
# =========================================================
def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(f"{VALIDATION_DIR} exists. Change EXPERIMENT_NAME.")

    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
    base_features = [c for c in test_cols if c != ID]
    cat_cols = [c for c in CAT_COLS if c in base_features]
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=[ID] + base_features + [TARGET])

    VALIDATION_DIR.mkdir(parents=True, exist_ok=False)
    FOLD_MODEL_DIR.mkdir(parents=True, exist_ok=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Expanding Window Validation
    fold_results, oof, final_features = run_expanding_validation(train, base_features, cat_cols)
    
    oof_metrics = calculate_metrics(oof["target"], oof["prediction"])
    print(f"\n{'=' * 70}\nOOF pooled Brier: {oof_metrics['brier']:.8f}\nOOF BSS         : {oof_metrics['bss']:.2f}\n{'=' * 70}")

    save_validation_summary(
        validation_dir=VALIDATION_DIR,
        experiment_name=EXPERIMENT_NAME,
        fold_results=fold_results,
        oof=oof,
        oof_metrics=oof_metrics,
    )

    print("\nValidation 결과 저장 완료:")
    print(f"- {VALIDATION_DIR / 'oof_pred.csv'}")
    print(f"- {VALIDATION_DIR / 'oof_validation.npz'}")
    print(f"- {VALIDATION_DIR / 'metadata.json'}")

    final_iterations = max(fold_results, key=lambda x: x["validation_season"])["final_iterations"]

    # 2. 전체 데이터 재학습 및 Prior 산출 (최종 Inference용)
    print("\n최종 모델 및 Prior 학습 시작...")
    df_full = train.copy()
    final_priors_dict = {}
    
    for config in HBEB_CONFIGS:
        prior_df, global_prior = compute_hbeb_priors(df_full, config["group_cols"], config["rate_col"])
        df_full[config["new_col"]] = apply_hbeb(df_full, config["group_cols"], config["n_col"], config["rate_col"], prior_df, global_prior)
        
        final_priors_dict[config["new_col"]] = {
            "prior_df": prior_df, 
            "global_prior": global_prior,
            "group_cols": config["group_cols"],
            "n_col": config["n_col"],
            "rate_col": config["rate_col"]
        }

    X_full = prepare_features(df_full[final_features], cat_cols)
    y_full = df_full[TARGET]

    final_params = {**MODEL_PARAMS, "iterations": int(final_iterations)}
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_full, y_full, cat_features=cat_cols)

    # 3. 모델 및 사전 분포(Prior), 피처 리스트를 단일 Artifact로 통합하여 저장
    final_artifact = {
        "model": final_model,
        "hbeb_priors": final_priors_dict,
        "features": final_features,
        "cat_cols": cat_cols
    }
    
    joblib.dump(final_artifact, MODEL_PATH, compress=3)
    
    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, FOLD_MODEL_DIR / "train.py")
    
    print(f"\n최종 Artifact 통합 저장 완료:\n- {MODEL_PATH}")

if __name__ == "__main__":
    main()