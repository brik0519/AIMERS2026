from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_group_es_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"


# =========================================================
# 경로
# =========================================================
DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# CatBoost 설정
# =========================================================
# 현재 기준 모델 cb_v002와 동일하게 유지한다.
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

CATBOOST_PARAMS = {
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
# Expanding Window
# =========================================================
FOLDS = [
    {
        "fold": 1,
        "train_start": 2019,
        "train_end": 2020,
        "valid_season": 2021,
    },
    {
        "fold": 2,
        "train_start": 2019,
        "train_end": 2021,
        "valid_season": 2022,
    },
    {
        "fold": 3,
        "train_start": 2019,
        "train_end": 2022,
        "valid_season": 2023,
    },
    {
        "fold": 4,
        "train_start": 2019,
        "train_end": 2023,
        "valid_season": 2024,
    },
]


# =========================================================
# 유틸
# =========================================================
def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    현재 cb_v002와 동일한 categorical preprocessing.

    - 범주형 결측값 -> "__MISSING__"
    - 문자열 변환
    - numeric feature의 NaN은 CatBoost가 직접 처리
    """
    df = df.copy()

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return df


def calc_brier(y_true, y_pred):
    """Brier Score."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean((y_pred - y_true) ** 2))


def calc_brier_skill_score(y_true, y_pred):
    """
    대회 방식의 Brier Skill Score.

    Score =
    max(0, 100000 * (1 - Brier / r(1-r)))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r = float(np.mean(y_true))
    brier = calc_brier(y_true, y_pred)
    baseline_brier = r * (1.0 - r)

    if baseline_brier <= 0:
        return 0.0, brier, baseline_brier

    score = max(
        0.0,
        100000.0 * (1.0 - brier / baseline_brier)
    )

    return score, brier, baseline_brier


def get_best_tree_count(model):
    """
    CatBoost best_iteration은 0-based.

    best_iteration = 0
    -> 실제 사용 tree 수 = 1
    """
    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        return CATBOOST_PARAMS["iterations"]

    return int(best_iteration) + 1


def create_model():
    """현재 cb_v002와 동일한 CatBoostClassifier."""
    return CatBoostClassifier(**CATBOOST_PARAMS)


def print_group_distribution(valid_df):
    """Validation의 R/F 분포와 target rate 출력."""
    print("\n[Validation game_type distribution]")

    for group in [R_GROUP, F_GROUP]:
        part = valid_df[valid_df[GROUP_COL] == group]

        if len(part) == 0:
            print(f"{group}: 0 rows")
            continue

        print(
            f"{group}: "
            f"n={len(part):,} | "
            f"target_rate={part[TARGET_COL].mean():.6f}"
        )


# =========================================================
# 데이터 로드
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)

print("\nLoad feature schema...")

test_columns = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
    nrows=0,
).columns.tolist()

FEATURES = [
    col
    for col in test_columns
    if col != ID_COL
]

print(f"Features: {len(FEATURES)}")
print(f"Categorical: {len(CAT_COLS)}")


print("\nLoad train data...")

train = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
    usecols=[ID_COL] + FEATURES + [TARGET_COL],
)

print(f"Train shape: {train.shape}")
print(
    f"Season: "
    f"{train[SEASON_COL].min()} ~ "
    f"{train[SEASON_COL].max()}"
)

train = prepare_categorical(train)


# =========================================================
# game_type 확인
# =========================================================
game_types = sorted(
    train[GROUP_COL]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(f"\ngame_type values: {game_types}")

unexpected_groups = set(game_types) - {R_GROUP, F_GROUP}

if unexpected_groups:
    raise ValueError(
        "이번 실험은 game_type R/F만을 대상으로 합니다. "
        f"예상하지 못한 값 발견: {unexpected_groups}"
    )


# =========================================================
# 저장 폴더
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_MODEL_DIR = MODEL_DIR / EXPERIMENT_NAME

FOLD_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# OOF 저장용
# =========================================================
all_oof = []
summary_rows = []


# =========================================================
# Expanding Window Validation
# =========================================================
for fold_info in FOLDS:

    fold = fold_info["fold"]
    train_start = fold_info["train_start"]
    train_end = fold_info["train_end"]
    valid_season = fold_info["valid_season"]

    print("\n")
    print("=" * 80)
    print(
        f"Fold {fold} | "
        f"Train {train_start}~{train_end} "
        f"-> Valid {valid_season}"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # Fold split
    # -----------------------------------------------------
    train_mask = (
        (train[SEASON_COL] >= train_start)
        & (train[SEASON_COL] <= train_end)
    )

    valid_mask = (
        train[SEASON_COL] == valid_season
    )

    fold_train = train.loc[train_mask].copy()
    fold_valid = train.loc[valid_mask].copy()

    X_train = fold_train[FEATURES]
    y_train = fold_train[TARGET_COL]

    X_valid = fold_valid[FEATURES]
    y_valid = fold_valid[TARGET_COL]

    print(
        f"\nTrain rows: {len(fold_train):,}"
    )
    print(
        f"Valid rows: {len(fold_valid):,}"
    )

    print_group_distribution(fold_valid)

    # -----------------------------------------------------
    # Validation group mask
    # -----------------------------------------------------
    valid_r_mask = (
        fold_valid[GROUP_COL] == R_GROUP
    )

    valid_f_mask = (
        fold_valid[GROUP_COL] == F_GROUP
    )

    if valid_r_mask.sum() == 0:
        raise ValueError(
            f"Fold {fold}: R validation row가 없습니다."
        )

    if valid_f_mask.sum() == 0:
        raise ValueError(
            f"Fold {fold}: F validation row가 없습니다."
        )


    # =====================================================
    # EXP 0
    # 기존 cb_v002 Global Baseline
    #
    # Train = R + F
    # Valid = R + F
    # =====================================================
    print("\n")
    print("-" * 80)
    print("[EXP 0] Global Baseline")
    print("-" * 80)

    baseline_model = create_model()

    start = time.time()

    baseline_model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    baseline_time = time.time() - start

    baseline_pred = baseline_model.predict_proba(
        X_valid
    )[:, 1]

    baseline_brier = calc_brier(
        y_valid,
        baseline_pred,
    )

    baseline_r_brier = calc_brier(
        y_valid.loc[valid_r_mask],
        baseline_pred[valid_r_mask.to_numpy()],
    )

    baseline_f_brier = calc_brier(
        y_valid.loc[valid_f_mask],
        baseline_pred[valid_f_mask.to_numpy()],
    )

    baseline_iter = get_best_tree_count(
        baseline_model
    )

    print(
        f"\nBaseline Brier : {baseline_brier:.8f}"
    )
    print(
        f"R Brier        : {baseline_r_brier:.8f}"
    )
    print(
        f"F Brier        : {baseline_f_brier:.8f}"
    )
    print(
        f"Best Trees     : {baseline_iter}"
    )
    print(
        f"Time           : {baseline_time:.1f}s"
    )


    # =====================================================
    # EXP 1-A
    # Global-R Model
    #
    # Train = R + F 전체
    # Valid = R만
    # =====================================================
    print("\n")
    print("-" * 80)
    print("[EXP 1-A] Global-R")
    print("Train = R + F | Early Stopping = R")
    print("-" * 80)

    global_r_model = create_model()

    X_valid_r = X_valid.loc[valid_r_mask]
    y_valid_r = y_valid.loc[valid_r_mask]

    start = time.time()

    global_r_model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid_r, y_valid_r),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    global_r_time = time.time() - start

    global_r_pred = global_r_model.predict_proba(
        X_valid_r
    )[:, 1]

    global_r_brier = calc_brier(
        y_valid_r,
        global_r_pred,
    )

    global_r_iter = get_best_tree_count(
        global_r_model
    )

    print(
        f"\nR Brier    : {global_r_brier:.8f}"
    )
    print(
        f"Best Trees : {global_r_iter}"
    )
    print(
        f"Pred Mean  : {global_r_pred.mean():.6f}"
    )
    print(
        f"True Rate  : {y_valid_r.mean():.6f}"
    )
    print(
        f"Time       : {global_r_time:.1f}s"
    )


    # =====================================================
    # EXP 1-B
    # Global-F Model
    #
    # Train = R + F 전체
    # Valid = F만
    # =====================================================
    print("\n")
    print("-" * 80)
    print("[EXP 1-B] Global-F")
    print("Train = R + F | Early Stopping = F")
    print("-" * 80)

    global_f_model = create_model()

    X_valid_f = X_valid.loc[valid_f_mask]
    y_valid_f = y_valid.loc[valid_f_mask]

    start = time.time()

    global_f_model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(X_valid_f, y_valid_f),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    global_f_time = time.time() - start

    global_f_pred = global_f_model.predict_proba(
        X_valid_f
    )[:, 1]

    global_f_brier = calc_brier(
        y_valid_f,
        global_f_pred,
    )

    global_f_iter = get_best_tree_count(
        global_f_model
    )

    print(
        f"\nF Brier    : {global_f_brier:.8f}"
    )
    print(
        f"Best Trees : {global_f_iter}"
    )
    print(
        f"Pred Mean  : {global_f_pred.mean():.6f}"
    )
    print(
        f"True Rate  : {y_valid_f.mean():.6f}"
    )
    print(
        f"Time       : {global_f_time:.1f}s"
    )


    # =====================================================
    # R/F prediction 결합
    # =====================================================
    group_es_pred = np.empty(
        len(fold_valid),
        dtype=float,
    )

    group_es_pred[
        valid_r_mask.to_numpy()
    ] = global_r_pred

    group_es_pred[
        valid_f_mask.to_numpy()
    ] = global_f_pred


    # =====================================================
    # Fold 전체 성능
    # =====================================================
    group_es_brier = calc_brier(
        y_valid,
        group_es_pred,
    )

    improvement = (
        baseline_brier
        - group_es_brier
    )

    baseline_score, _, _ = (
        calc_brier_skill_score(
            y_valid,
            baseline_pred,
        )
    )

    group_es_score, _, _ = (
        calc_brier_skill_score(
            y_valid,
            group_es_pred,
        )
    )

    print("\n")
    print("=" * 80)
    print(f"[Fold {fold} Result]")
    print("=" * 80)

    print(
        f"Baseline Brier : {baseline_brier:.8f}"
    )

    print(
        f"Group-ES Brier : {group_es_brier:.8f}"
    )

    print(
        f"Improvement    : {improvement:+.8f}"
    )

    print(
        f"Baseline BSS   : {baseline_score:.2f}"
    )

    print(
        f"Group-ES BSS   : {group_es_score:.2f}"
    )

    print("\n[Best Trees]")

    print(
        f"Baseline : {baseline_iter}"
    )

    print(
        f"Global-R : {global_r_iter}"
    )

    print(
        f"Global-F : {global_f_iter}"
    )


    # =====================================================
    # Prediction 저장
    # =====================================================
    fold_result = pd.DataFrame({
        ID_COL: fold_valid[ID_COL].values,
        SEASON_COL: fold_valid[SEASON_COL].values,
        GROUP_COL: fold_valid[GROUP_COL].values,
        "y_true": y_valid.values,
        "baseline_pred": baseline_pred,
        "group_es_pred": group_es_pred,
    })

    fold_result["baseline_sq_error"] = (
        fold_result["baseline_pred"]
        - fold_result["y_true"]
    ) ** 2

    fold_result["group_es_sq_error"] = (
        fold_result["group_es_pred"]
        - fold_result["y_true"]
    ) ** 2

    fold_result.to_csv(
        VALIDATION_DIR
        / f"fold{fold}_{valid_season}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_oof.append(
        fold_result
    )


    # =====================================================
    # Fold summary 저장
    # =====================================================
    summary_rows.append({
        "fold": fold,
        "train_period": f"{train_start}-{train_end}",
        "valid_season": valid_season,

        "n_train": len(fold_train),
        "n_valid": len(fold_valid),
        "n_valid_R": int(valid_r_mask.sum()),
        "n_valid_F": int(valid_f_mask.sum()),

        "R_true_rate": float(y_valid_r.mean()),
        "F_true_rate": float(y_valid_f.mean()),

        "baseline_brier": baseline_brier,
        "baseline_R_brier": baseline_r_brier,
        "baseline_F_brier": baseline_f_brier,

        "group_es_brier": group_es_brier,
        "group_es_R_brier": global_r_brier,
        "group_es_F_brier": global_f_brier,

        "delta_baseline_minus_group_es": improvement,

        "baseline_best_trees": baseline_iter,
        "global_R_best_trees": global_r_iter,
        "global_F_best_trees": global_f_iter,

        "global_R_pred_mean": float(
            global_r_pred.mean()
        ),

        "global_F_pred_mean": float(
            global_f_pred.mean()
        ),
    })


    # =====================================================
    # Fold model 저장
    # =====================================================
    joblib.dump(
        baseline_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_baseline.pkl",
        compress=3,
    )

    joblib.dump(
        global_r_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_global_R.pkl",
        compress=3,
    )

    joblib.dump(
        global_f_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_global_F.pkl",
        compress=3,
    )


# =========================================================
# 전체 OOF 평가
# =========================================================
oof = pd.concat(
    all_oof,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)

baseline_oof_brier = calc_brier(
    oof["y_true"],
    oof["baseline_pred"],
)

group_es_oof_brier = calc_brier(
    oof["y_true"],
    oof["group_es_pred"],
)

oof_improvement = (
    baseline_oof_brier
    - group_es_oof_brier
)


baseline_oof_score, _, baseline_ref = (
    calc_brier_skill_score(
        oof["y_true"],
        oof["baseline_pred"],
    )
)

group_es_oof_score, _, _ = (
    calc_brier_skill_score(
        oof["y_true"],
        oof["group_es_pred"],
    )
)


# =========================================================
# Group별 pooled OOF
# =========================================================
oof_r = oof[
    oof[GROUP_COL] == R_GROUP
]

oof_f = oof[
    oof[GROUP_COL] == F_GROUP
]


baseline_r_oof = calc_brier(
    oof_r["y_true"],
    oof_r["baseline_pred"],
)

group_es_r_oof = calc_brier(
    oof_r["y_true"],
    oof_r["group_es_pred"],
)


baseline_f_oof = calc_brier(
    oof_f["y_true"],
    oof_f["baseline_pred"],
)

group_es_f_oof = calc_brier(
    oof_f["y_true"],
    oof_f["group_es_pred"],
)


# =========================================================
# 결과 출력
# =========================================================
print("\n")
print("#" * 80)
print("FINAL OOF RESULT")
print("#" * 80)

print("\n[Fold Summary]")

display_cols = [
    "fold",
    "valid_season",
    "baseline_brier",
    "group_es_brier",
    "delta_baseline_minus_group_es",
    "baseline_best_trees",
    "global_R_best_trees",
    "global_F_best_trees",
]

print(
    summary[display_cols]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


print("\n[POOLED OOF]")

print(
    f"Baseline Brier : "
    f"{baseline_oof_brier:.8f}"
)

print(
    f"Group-ES Brier : "
    f"{group_es_oof_brier:.8f}"
)

print(
    f"Improvement    : "
    f"{oof_improvement:+.8f}"
)

print(
    f"Baseline BSS   : "
    f"{baseline_oof_score:.2f}"
)

print(
    f"Group-ES BSS   : "
    f"{group_es_oof_score:.2f}"
)

print(
    f"Reference      : "
    f"{baseline_ref:.8f}"
)


print("\n[R Pooled OOF]")

print(
    f"Baseline : {baseline_r_oof:.8f}"
)

print(
    f"Group-ES : {group_es_r_oof:.8f}"
)

print(
    f"Delta    : "
    f"{baseline_r_oof - group_es_r_oof:+.8f}"
)


print("\n[F Pooled OOF]")

print(
    f"Baseline : {baseline_f_oof:.8f}"
)

print(
    f"Group-ES : {group_es_f_oof:.8f}"
)

print(
    f"Delta    : "
    f"{baseline_f_oof - group_es_f_oof:+.8f}"
)


# =========================================================
# 결과 파일 저장
# =========================================================
oof.to_csv(
    VALIDATION_DIR / "oof_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

summary.to_csv(
    VALIDATION_DIR / "fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


metadata = {
    "experiment_name": EXPERIMENT_NAME,

    "reference_model": "cb_v002",

    "experiment": (
        "Same training data, "
        "group-specific early stopping "
        "for game_type R/F"
    ),

    "feature_count": len(FEATURES),

    "categorical_features": CAT_COLS,

    "early_stopping_rounds":
        EARLY_STOPPING_ROUNDS,

    "catboost_params":
        CATBOOST_PARAMS,

    "baseline_oof_brier":
        baseline_oof_brier,

    "group_es_oof_brier":
        group_es_oof_brier,

    "improvement":
        oof_improvement,

    "baseline_R_oof_brier":
        baseline_r_oof,

    "group_es_R_oof_brier":
        group_es_r_oof,

    "baseline_F_oof_brier":
        baseline_f_oof,

    "group_es_F_oof_brier":
        group_es_f_oof,
}

with open(
    VALIDATION_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\nSaved:")
print(
    VALIDATION_DIR / "oof_predictions.csv"
)
print(
    VALIDATION_DIR / "fold_summary.csv"
)
print(
    VALIDATION_DIR / "metadata.json"
)

print("\nExperiment complete.")