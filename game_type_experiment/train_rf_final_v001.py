from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_final_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"

FINAL_SEED = 42

SEEDS = [7, 42, 123, 999, 2026]

R_GLOBAL_WEIGHT = 0.49
R_ONLY_WEIGHT = 0.51

F_GLOBAL_WEIGHT = 0.64
F_ONLY_WEIGHT = 0.36


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
# CatBoost
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

BASE_CATBOOST_PARAMS = {
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "allow_writing_files": False,
}

MAX_ITERATIONS = 800
EARLY_STOPPING_ROUNDS = 80


# =========================================================
# 유틸
# =========================================================
def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    행 단위 전처리만 수행.
    다른 row의 값/통계/분포는 사용하지 않는다.
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


def create_es_model(seed: int):
    params = BASE_CATBOOST_PARAMS.copy()

    params.update({
        "iterations": MAX_ITERATIONS,
        "random_seed": seed,
        "verbose": 0,
    })

    return CatBoostClassifier(**params)


def create_final_model(
    seed: int,
    iterations: int,
):
    params = BASE_CATBOOST_PARAMS.copy()

    params.update({
        "iterations": int(iterations),
        "random_seed": seed,
        "verbose": 100,
    })

    return CatBoostClassifier(**params)


def get_best_tree_count(model):
    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        return MAX_ITERATIONS

    return int(best_iteration) + 1


# =========================================================
# 데이터
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)

print("\nLoad train.csv...")

train = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
)

if ID_COL not in train.columns:
    raise ValueError(f"{ID_COL} not found.")

if TARGET_COL not in train.columns:
    raise ValueError(f"{TARGET_COL} not found.")


# ---------------------------------------------------------
# Feature schema는 오직 공식 train.csv에서 결정
#
# test.csv의 값/분포/통계는 사용하지 않는다.
# ---------------------------------------------------------
FEATURES = [
    col
    for col in train.columns
    if col not in [
        ID_COL,
        TARGET_COL,
    ]
]

print(f"Train shape : {train.shape}")
print(f"Features    : {len(FEATURES)}")
print(f"Season      : {train[SEASON_COL].min()} ~ {train[SEASON_COL].max()}")


train = prepare_categorical(train)


# =========================================================
# game_type 검증
# =========================================================
game_types = set(
    train[GROUP_COL]
    .dropna()
    .astype(str)
    .unique()
)

unexpected = game_types - {
    R_GROUP,
    F_GROUP,
}

if unexpected:
    raise ValueError(
        f"Unexpected game_type: {unexpected}"
    )


# =========================================================
# Step 1
# 최종 iteration 선택
#
# Train = 2019~2023
# Validation = 2024
#
# 5 seeds에서 각 모델 best tree를 구한 뒤
# median을 final iteration으로 사용한다.
# =========================================================
print("\n")
print("#" * 80)
print("STEP 1. FINAL ITERATION SELECTION")
print("#" * 80)


selection_train = train[
    train[SEASON_COL] <= 2023
].copy()

selection_valid = train[
    train[SEASON_COL] == 2024
].copy()


selection_train_r = selection_train[
    selection_train[GROUP_COL] == R_GROUP
].copy()

selection_train_f = selection_train[
    selection_train[GROUP_COL] == F_GROUP
].copy()

selection_valid_r = selection_valid[
    selection_valid[GROUP_COL] == R_GROUP
].copy()

selection_valid_f = selection_valid[
    selection_valid[GROUP_COL] == F_GROUP
].copy()


X_train_all = selection_train[FEATURES]
y_train_all = selection_train[TARGET_COL]

X_train_r = selection_train_r[FEATURES]
y_train_r = selection_train_r[TARGET_COL]

X_train_f = selection_train_f[FEATURES]
y_train_f = selection_train_f[TARGET_COL]

X_valid_r = selection_valid_r[FEATURES]
y_valid_r = selection_valid_r[TARGET_COL]

X_valid_f = selection_valid_f[FEATURES]
y_valid_f = selection_valid_f[TARGET_COL]


iteration_rows = []


for seed in SEEDS:

    print(f"\nSeed {seed}")

    # -----------------------------------------------------
    # Global-R
    # Train = R + F
    # Valid = R
    # -----------------------------------------------------
    global_r = create_es_model(seed)

    global_r.fit(
        X_train_all,
        y_train_all,
        cat_features=CAT_COLS,
        eval_set=(X_valid_r, y_valid_r),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    global_r_trees = get_best_tree_count(
        global_r
    )


    # -----------------------------------------------------
    # R-only
    # -----------------------------------------------------
    r_only = create_es_model(seed)

    r_only.fit(
        X_train_r,
        y_train_r,
        cat_features=CAT_COLS,
        eval_set=(X_valid_r, y_valid_r),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    r_only_trees = get_best_tree_count(
        r_only
    )


    # -----------------------------------------------------
    # Global-F
    # Train = R + F
    # Valid = F
    # -----------------------------------------------------
    global_f = create_es_model(seed)

    global_f.fit(
        X_train_all,
        y_train_all,
        cat_features=CAT_COLS,
        eval_set=(X_valid_f, y_valid_f),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    global_f_trees = get_best_tree_count(
        global_f
    )


    # -----------------------------------------------------
    # F-only
    # -----------------------------------------------------
    f_only = create_es_model(seed)

    f_only.fit(
        X_train_f,
        y_train_f,
        cat_features=CAT_COLS,
        eval_set=(X_valid_f, y_valid_f),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    f_only_trees = get_best_tree_count(
        f_only
    )


    iteration_rows.append({
        "seed": seed,
        "global_R": global_r_trees,
        "R_only": r_only_trees,
        "global_F": global_f_trees,
        "F_only": f_only_trees,
    })


    print(
        f"Global-R={global_r_trees} | "
        f"R-only={r_only_trees} | "
        f"Global-F={global_f_trees} | "
        f"F-only={f_only_trees}"
    )


iteration_df = pd.DataFrame(
    iteration_rows
)


# =========================================================
# 중앙값
# =========================================================
FINAL_GLOBAL_R_ITER = int(
    np.median(
        iteration_df["global_R"]
    )
)

FINAL_R_ONLY_ITER = int(
    np.median(
        iteration_df["R_only"]
    )
)

FINAL_GLOBAL_F_ITER = int(
    np.median(
        iteration_df["global_F"]
    )
)

FINAL_F_ONLY_ITER = int(
    np.median(
        iteration_df["F_only"]
    )
)


print("\n[Iteration Summary]")

print(
    iteration_df.to_string(
        index=False
    )
)


print("\n[Selected Final Iterations]")

print(
    f"Global-R : {FINAL_GLOBAL_R_ITER}"
)

print(
    f"R-only   : {FINAL_R_ONLY_ITER}"
)

print(
    f"Global-F : {FINAL_GLOBAL_F_ITER}"
)

print(
    f"F-only   : {FINAL_F_ONLY_ITER}"
)


# =========================================================
# Step 2
# 2019~2024 전체 학습
# =========================================================
print("\n")
print("#" * 80)
print("STEP 2. RETRAIN WITH FULL 2019~2024 DATA")
print("#" * 80)


full_r = train[
    train[GROUP_COL] == R_GROUP
].copy()

full_f = train[
    train[GROUP_COL] == F_GROUP
].copy()


X_full = train[FEATURES]
y_full = train[TARGET_COL]

X_full_r = full_r[FEATURES]
y_full_r = full_r[TARGET_COL]

X_full_f = full_f[FEATURES]
y_full_f = full_f[TARGET_COL]


# ---------------------------------------------------------
# Final Global-R
# ---------------------------------------------------------
print("\nTrain Final Global-R...")

final_global_r = create_final_model(
    seed=FINAL_SEED,
    iterations=FINAL_GLOBAL_R_ITER,
)

final_global_r.fit(
    X_full,
    y_full,
    cat_features=CAT_COLS,
)


# ---------------------------------------------------------
# Final R-only
# ---------------------------------------------------------
print("\nTrain Final R-only...")

final_r_only = create_final_model(
    seed=FINAL_SEED,
    iterations=FINAL_R_ONLY_ITER,
)

final_r_only.fit(
    X_full_r,
    y_full_r,
    cat_features=CAT_COLS,
)


# ---------------------------------------------------------
# Final Global-F
# ---------------------------------------------------------
print("\nTrain Final Global-F...")

final_global_f = create_final_model(
    seed=FINAL_SEED,
    iterations=FINAL_GLOBAL_F_ITER,
)

final_global_f.fit(
    X_full,
    y_full,
    cat_features=CAT_COLS,
)


# ---------------------------------------------------------
# Final F-only
# ---------------------------------------------------------
print("\nTrain Final F-only...")

final_f_only = create_final_model(
    seed=FINAL_SEED,
    iterations=FINAL_F_ONLY_ITER,
)

final_f_only.fit(
    X_full_f,
    y_full_f,
    cat_features=CAT_COLS,
)


# =========================================================
# Bundle 저장
# =========================================================
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


bundle = {
    "experiment_name": EXPERIMENT_NAME,

    "features": FEATURES,
    "cat_cols": CAT_COLS,

    "group_col": GROUP_COL,

    "R_group": R_GROUP,
    "F_group": F_GROUP,

    "R_global_weight": R_GLOBAL_WEIGHT,
    "R_only_weight": R_ONLY_WEIGHT,

    "F_global_weight": F_GLOBAL_WEIGHT,
    "F_only_weight": F_ONLY_WEIGHT,

    "final_seed": FINAL_SEED,

    "iterations": {
        "global_R": FINAL_GLOBAL_R_ITER,
        "R_only": FINAL_R_ONLY_ITER,
        "global_F": FINAL_GLOBAL_F_ITER,
        "F_only": FINAL_F_ONLY_ITER,
    },

    "global_R_model": final_global_r,
    "R_only_model": final_r_only,

    "global_F_model": final_global_f,
    "F_only_model": final_f_only,
}


joblib.dump(
    bundle,
    MODEL_PATH,
    compress=3,
)


iteration_df.to_csv(
    VALIDATION_DIR
    / "final_iteration_selection.csv",
    index=False,
    encoding="utf-8-sig",
)


metadata = {
    "experiment_name": EXPERIMENT_NAME,

    "final_seed": FINAL_SEED,

    "weights": {
        "R_global": R_GLOBAL_WEIGHT,
        "R_only": R_ONLY_WEIGHT,
        "F_global": F_GLOBAL_WEIGHT,
        "F_only": F_ONLY_WEIGHT,
    },

    "iterations": {
        "global_R": FINAL_GLOBAL_R_ITER,
        "R_only": FINAL_R_ONLY_ITER,
        "global_F": FINAL_GLOBAL_F_ITER,
        "F_only": FINAL_F_ONLY_ITER,
    },

    "iteration_selection": (
        "Median best tree count across "
        "5 seeds using 2019-2023 train "
        "and 2024 group-specific validation."
    ),
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


print("\n")
print("=" * 80)
print("FINAL MODEL SAVED")
print("=" * 80)

print(f"MODEL_PATH = {MODEL_PATH}")
print(f"Features   = {len(FEATURES)}")

print("\nDone.")