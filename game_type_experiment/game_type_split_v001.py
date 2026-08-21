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
EXPERIMENT_NAME = "cb_v002_game_type_split_v001"

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
# 직전 Group-Specific Early Stopping 실험과 완전히 동일하게 유지
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
    기존 cb_v002 / Group-ES와 동일한 categorical 처리.
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean((y_pred - y_true) ** 2)
    )


def calc_brier_skill_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r = float(np.mean(y_true))
    brier = calc_brier(y_true, y_pred)

    baseline_brier = r * (1.0 - r)

    if baseline_brier <= 0:
        return 0.0, brier, baseline_brier

    score = max(
        0.0,
        100000.0
        * (1.0 - brier / baseline_brier),
    )

    return score, brier, baseline_brier


def get_best_tree_count(model):
    """
    CatBoost get_best_iteration()은 0-based.
    """
    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        return CATBOOST_PARAMS["iterations"]

    return int(best_iteration) + 1


def create_model():
    return CatBoostClassifier(
        **CATBOOST_PARAMS
    )


def print_distribution(df, name):
    print(f"\n[{name}]")

    for group in [R_GROUP, F_GROUP]:
        part = df[
            df[GROUP_COL] == group
        ]

        if len(part) == 0:
            print(f"{group}: 0 rows")
            continue

        print(
            f"{group}: "
            f"n={len(part):,} | "
            f"target_rate="
            f"{part[TARGET_COL].mean():.6f}"
        )


# =========================================================
# 데이터 로드
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)


# ---------------------------------------------------------
# test.csv는 오직 입력 feature schema 확인용
#
# test 전체 통계/분포/집계는 절대 사용하지 않음.
# ---------------------------------------------------------
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
print(
    f"Categorical: {len(CAT_COLS)}"
)


# ---------------------------------------------------------
# 학습 데이터
# ---------------------------------------------------------
print("\nLoad train data...")

train = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
    usecols=[
        ID_COL,
        *FEATURES,
        TARGET_COL,
    ],
)

print(
    f"Train shape: {train.shape}"
)

print(
    f"Season: "
    f"{train[SEASON_COL].min()} ~ "
    f"{train[SEASON_COL].max()}"
)


train = prepare_categorical(
    train
)


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

print(
    f"\ngame_type values: "
    f"{game_types}"
)


unexpected_groups = (
    set(game_types)
    - {R_GROUP, F_GROUP}
)

if unexpected_groups:
    raise ValueError(
        "이번 실험은 game_type R/F만 "
        "대상으로 합니다. "
        f"예상하지 못한 값: "
        f"{unexpected_groups}"
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

FOLD_MODEL_DIR = (
    MODEL_DIR / EXPERIMENT_NAME
)

FOLD_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 결과 저장용
# =========================================================
all_oof = []
summary_rows = []


# =========================================================
# Expanding Window
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


    # =====================================================
    # Fold Split
    # =====================================================
    train_mask = (
        (train[SEASON_COL] >= train_start)
        & (train[SEASON_COL] <= train_end)
    )

    valid_mask = (
        train[SEASON_COL]
        == valid_season
    )

    fold_train = (
        train.loc[train_mask]
        .copy()
    )

    fold_valid = (
        train.loc[valid_mask]
        .copy()
    )


    print(
        f"\nTrain rows: "
        f"{len(fold_train):,}"
    )

    print(
        f"Valid rows: "
        f"{len(fold_valid):,}"
    )


    print_distribution(
        fold_train,
        "Train game_type distribution",
    )

    print_distribution(
        fold_valid,
        "Validation game_type distribution",
    )


    # =====================================================
    # R/F 데이터 분리
    # =====================================================
    train_r = fold_train[
        fold_train[GROUP_COL] == R_GROUP
    ].copy()

    train_f = fold_train[
        fold_train[GROUP_COL] == F_GROUP
    ].copy()

    valid_r = fold_valid[
        fold_valid[GROUP_COL] == R_GROUP
    ].copy()

    valid_f = fold_valid[
        fold_valid[GROUP_COL] == F_GROUP
    ].copy()


    if len(train_r) == 0:
        raise ValueError(
            f"Fold {fold}: "
            "R training data 없음"
        )

    if len(train_f) == 0:
        raise ValueError(
            f"Fold {fold}: "
            "F training data 없음"
        )

    if len(valid_r) == 0:
        raise ValueError(
            f"Fold {fold}: "
            "R validation data 없음"
        )

    if len(valid_f) == 0:
        raise ValueError(
            f"Fold {fold}: "
            "F validation data 없음"
        )


    # -----------------------------------------------------
    # 전체 학습 데이터
    # -----------------------------------------------------
    X_train_all = (
        fold_train[FEATURES]
    )

    y_train_all = (
        fold_train[TARGET_COL]
    )


    # -----------------------------------------------------
    # R
    # -----------------------------------------------------
    X_train_r = train_r[FEATURES]
    y_train_r = train_r[TARGET_COL]

    X_valid_r = valid_r[FEATURES]
    y_valid_r = valid_r[TARGET_COL]


    # -----------------------------------------------------
    # F
    # -----------------------------------------------------
    X_train_f = train_f[FEATURES]
    y_train_f = train_f[TARGET_COL]

    X_valid_f = valid_f[FEATURES]
    y_valid_f = valid_f[TARGET_COL]


    # =====================================================
    # EXP 1-A
    #
    # Global-R
    # Train = R + F
    # Valid = R
    # =====================================================
    print("\n")
    print("-" * 80)

    print("[GLOBAL-R]")

    print(
        "Train = R + F | "
        "Early Stopping = R"
    )

    print("-" * 80)


    global_r_model = create_model()

    start = time.time()

    global_r_model.fit(
        X_train_all,
        y_train_all,
        cat_features=CAT_COLS,
        eval_set=(
            X_valid_r,
            y_valid_r,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    global_r_time = (
        time.time() - start
    )


    global_r_pred = (
        global_r_model
        .predict_proba(
            X_valid_r
        )[:, 1]
    )

    global_r_brier = calc_brier(
        y_valid_r,
        global_r_pred,
    )

    global_r_iter = (
        get_best_tree_count(
            global_r_model
        )
    )


    print(
        f"\nR Brier    : "
        f"{global_r_brier:.8f}"
    )

    print(
        f"Best Trees : "
        f"{global_r_iter}"
    )

    print(
        f"Pred Mean  : "
        f"{global_r_pred.mean():.6f}"
    )

    print(
        f"True Rate  : "
        f"{y_valid_r.mean():.6f}"
    )

    print(
        f"Time       : "
        f"{global_r_time:.1f}s"
    )


    # =====================================================
    # EXP 2-A
    #
    # R-only
    # Train = R
    # Valid = R
    # =====================================================
    print("\n")
    print("-" * 80)

    print("[R-ONLY]")

    print(
        "Train = R | "
        "Early Stopping = R"
    )

    print("-" * 80)


    r_only_model = create_model()

    start = time.time()

    r_only_model.fit(
        X_train_r,
        y_train_r,
        cat_features=CAT_COLS,
        eval_set=(
            X_valid_r,
            y_valid_r,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    r_only_time = (
        time.time() - start
    )


    r_only_pred = (
        r_only_model
        .predict_proba(
            X_valid_r
        )[:, 1]
    )

    r_only_brier = calc_brier(
        y_valid_r,
        r_only_pred,
    )

    r_only_iter = (
        get_best_tree_count(
            r_only_model
        )
    )


    print(
        f"\nR Brier    : "
        f"{r_only_brier:.8f}"
    )

    print(
        f"Best Trees : "
        f"{r_only_iter}"
    )

    print(
        f"Pred Mean  : "
        f"{r_only_pred.mean():.6f}"
    )

    print(
        f"True Rate  : "
        f"{y_valid_r.mean():.6f}"
    )

    print(
        f"Time       : "
        f"{r_only_time:.1f}s"
    )


    # =====================================================
    # EXP 1-B
    #
    # Global-F
    # Train = R + F
    # Valid = F
    # =====================================================
    print("\n")
    print("-" * 80)

    print("[GLOBAL-F]")

    print(
        "Train = R + F | "
        "Early Stopping = F"
    )

    print("-" * 80)


    global_f_model = create_model()

    start = time.time()

    global_f_model.fit(
        X_train_all,
        y_train_all,
        cat_features=CAT_COLS,
        eval_set=(
            X_valid_f,
            y_valid_f,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    global_f_time = (
        time.time() - start
    )


    global_f_pred = (
        global_f_model
        .predict_proba(
            X_valid_f
        )[:, 1]
    )

    global_f_brier = calc_brier(
        y_valid_f,
        global_f_pred,
    )

    global_f_iter = (
        get_best_tree_count(
            global_f_model
        )
    )


    print(
        f"\nF Brier    : "
        f"{global_f_brier:.8f}"
    )

    print(
        f"Best Trees : "
        f"{global_f_iter}"
    )

    print(
        f"Pred Mean  : "
        f"{global_f_pred.mean():.6f}"
    )

    print(
        f"True Rate  : "
        f"{y_valid_f.mean():.6f}"
    )

    print(
        f"Time       : "
        f"{global_f_time:.1f}s"
    )


    # =====================================================
    # EXP 2-B
    #
    # F-only
    # Train = F
    # Valid = F
    # =====================================================
    print("\n")
    print("-" * 80)

    print("[F-ONLY]")

    print(
        "Train = F | "
        "Early Stopping = F"
    )

    print("-" * 80)


    f_only_model = create_model()

    start = time.time()

    f_only_model.fit(
        X_train_f,
        y_train_f,
        cat_features=CAT_COLS,
        eval_set=(
            X_valid_f,
            y_valid_f,
        ),
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
        use_best_model=True,
    )

    f_only_time = (
        time.time() - start
    )


    f_only_pred = (
        f_only_model
        .predict_proba(
            X_valid_f
        )[:, 1]
    )

    f_only_brier = calc_brier(
        y_valid_f,
        f_only_pred,
    )

    f_only_iter = (
        get_best_tree_count(
            f_only_model
        )
    )


    print(
        f"\nF Brier    : "
        f"{f_only_brier:.8f}"
    )

    print(
        f"Best Trees : "
        f"{f_only_iter}"
    )

    print(
        f"Pred Mean  : "
        f"{f_only_pred.mean():.6f}"
    )

    print(
        f"True Rate  : "
        f"{y_valid_f.mean():.6f}"
    )

    print(
        f"Time       : "
        f"{f_only_time:.1f}s"
    )


    # =====================================================
    # Global Group-ES 결합
    # =====================================================
    global_group_pred = np.empty(
        len(fold_valid),
        dtype=float,
    )


    valid_r_mask = (
        fold_valid[GROUP_COL]
        == R_GROUP
    ).to_numpy()

    valid_f_mask = (
        fold_valid[GROUP_COL]
        == F_GROUP
    ).to_numpy()


    global_group_pred[
        valid_r_mask
    ] = global_r_pred

    global_group_pred[
        valid_f_mask
    ] = global_f_pred


    # =====================================================
    # Split-only 결합
    # =====================================================
    split_group_pred = np.empty(
        len(fold_valid),
        dtype=float,
    )


    split_group_pred[
        valid_r_mask
    ] = r_only_pred

    split_group_pred[
        valid_f_mask
    ] = f_only_pred


    # =====================================================
    # Fold 전체 Brier
    # =====================================================
    y_valid = (
        fold_valid[TARGET_COL]
        .to_numpy()
    )


    global_group_brier = calc_brier(
        y_valid,
        global_group_pred,
    )

    split_group_brier = calc_brier(
        y_valid,
        split_group_pred,
    )


    delta = (
        global_group_brier
        - split_group_brier
    )


    global_score, _, _ = (
        calc_brier_skill_score(
            y_valid,
            global_group_pred,
        )
    )

    split_score, _, _ = (
        calc_brier_skill_score(
            y_valid,
            split_group_pred,
        )
    )


    # =====================================================
    # Fold 결과
    # =====================================================
    print("\n")
    print("=" * 80)

    print(
        f"[Fold {fold} Result]"
    )

    print("=" * 80)


    print(
        f"Global Group-ES Brier : "
        f"{global_group_brier:.8f}"
    )

    print(
        f"Split-only Brier      : "
        f"{split_group_brier:.8f}"
    )

    print(
        f"Improvement           : "
        f"{delta:+.8f}"
    )


    print(
        f"\nGlobal Group-ES BSS : "
        f"{global_score:.2f}"
    )

    print(
        f"Split-only BSS      : "
        f"{split_score:.2f}"
    )


    print("\n[R]")

    print(
        f"Global-R : "
        f"{global_r_brier:.8f} "
        f"({global_r_iter} trees)"
    )

    print(
        f"R-only   : "
        f"{r_only_brier:.8f} "
        f"({r_only_iter} trees)"
    )

    print(
        f"Delta    : "
        f"{global_r_brier - r_only_brier:+.8f}"
    )


    print("\n[F]")

    print(
        f"Global-F : "
        f"{global_f_brier:.8f} "
        f"({global_f_iter} trees)"
    )

    print(
        f"F-only   : "
        f"{f_only_brier:.8f} "
        f"({f_only_iter} trees)"
    )

    print(
        f"Delta    : "
        f"{global_f_brier - f_only_brier:+.8f}"
    )


    # =====================================================
    # OOF prediction 저장
    # =====================================================
    fold_result = pd.DataFrame({
        ID_COL:
            fold_valid[ID_COL].values,

        SEASON_COL:
            fold_valid[SEASON_COL].values,

        GROUP_COL:
            fold_valid[GROUP_COL].values,

        "y_true":
            fold_valid[TARGET_COL].values,

        "global_group_pred":
            global_group_pred,

        "split_group_pred":
            split_group_pred,
    })


    # 개별 R/F 모델 prediction도 보존
    fold_result[
        "global_R_pred"
    ] = np.nan

    fold_result[
        "R_only_pred"
    ] = np.nan

    fold_result[
        "global_F_pred"
    ] = np.nan

    fold_result[
        "F_only_pred"
    ] = np.nan


    fold_result.loc[
        valid_r_mask,
        "global_R_pred",
    ] = global_r_pred

    fold_result.loc[
        valid_r_mask,
        "R_only_pred",
    ] = r_only_pred


    fold_result.loc[
        valid_f_mask,
        "global_F_pred",
    ] = global_f_pred

    fold_result.loc[
        valid_f_mask,
        "F_only_pred",
    ] = f_only_pred


    fold_result.to_csv(
        VALIDATION_DIR
        / (
            f"fold{fold}_"
            f"{valid_season}_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )


    all_oof.append(
        fold_result
    )


    # =====================================================
    # Fold summary
    # =====================================================
    summary_rows.append({

        "fold": fold,

        "train_period":
            f"{train_start}-{train_end}",

        "valid_season":
            valid_season,

        "n_train_all":
            len(fold_train),

        "n_train_R":
            len(train_r),

        "n_train_F":
            len(train_f),

        "n_valid_R":
            len(valid_r),

        "n_valid_F":
            len(valid_f),

        "R_true_rate":
            float(y_valid_r.mean()),

        "F_true_rate":
            float(y_valid_f.mean()),


        "global_R_brier":
            global_r_brier,

        "R_only_brier":
            r_only_brier,

        "R_delta":
            (
                global_r_brier
                - r_only_brier
            ),


        "global_F_brier":
            global_f_brier,

        "F_only_brier":
            f_only_brier,

        "F_delta":
            (
                global_f_brier
                - f_only_brier
            ),


        "global_group_brier":
            global_group_brier,

        "split_group_brier":
            split_group_brier,

        "overall_delta":
            delta,


        "global_R_best_trees":
            global_r_iter,

        "R_only_best_trees":
            r_only_iter,

        "global_F_best_trees":
            global_f_iter,

        "F_only_best_trees":
            f_only_iter,


        "global_R_pred_mean":
            float(global_r_pred.mean()),

        "R_only_pred_mean":
            float(r_only_pred.mean()),

        "global_F_pred_mean":
            float(global_f_pred.mean()),

        "F_only_pred_mean":
            float(f_only_pred.mean()),
    })


    # =====================================================
    # 모델 저장
    # =====================================================
    joblib.dump(
        global_r_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_global_R.pkl",
        compress=3,
    )

    joblib.dump(
        r_only_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_R_only.pkl",
        compress=3,
    )

    joblib.dump(
        global_f_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_global_F.pkl",
        compress=3,
    )

    joblib.dump(
        f_only_model,
        FOLD_MODEL_DIR
        / f"fold{fold}_F_only.pkl",
        compress=3,
    )


# =========================================================
# Pooled OOF
# =========================================================
oof = pd.concat(
    all_oof,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)


global_oof_brier = calc_brier(
    oof["y_true"],
    oof["global_group_pred"],
)

split_oof_brier = calc_brier(
    oof["y_true"],
    oof["split_group_pred"],
)


overall_improvement = (
    global_oof_brier
    - split_oof_brier
)


global_oof_score, _, reference = (
    calc_brier_skill_score(
        oof["y_true"],
        oof["global_group_pred"],
    )
)

split_oof_score, _, _ = (
    calc_brier_skill_score(
        oof["y_true"],
        oof["split_group_pred"],
    )
)


# =========================================================
# R pooled
# =========================================================
oof_r = oof[
    oof[GROUP_COL] == R_GROUP
].copy()


global_r_oof = calc_brier(
    oof_r["y_true"],
    oof_r["global_R_pred"],
)

r_only_oof = calc_brier(
    oof_r["y_true"],
    oof_r["R_only_pred"],
)


# =========================================================
# F pooled
# =========================================================
oof_f = oof[
    oof[GROUP_COL] == F_GROUP
].copy()


global_f_oof = calc_brier(
    oof_f["y_true"],
    oof_f["global_F_pred"],
)

f_only_oof = calc_brier(
    oof_f["y_true"],
    oof_f["F_only_pred"],
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("#" * 80)
print("FINAL OOF RESULT")
print("#" * 80)


print("\n[Fold Summary]")

display_cols = [
    "fold",
    "valid_season",
    "global_group_brier",
    "split_group_brier",
    "overall_delta",
    "global_R_best_trees",
    "R_only_best_trees",
    "global_F_best_trees",
    "F_only_best_trees",
]


print(
    summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[POOLED OOF]")

print(
    f"Global Group-ES : "
    f"{global_oof_brier:.8f}"
)

print(
    f"Split-only      : "
    f"{split_oof_brier:.8f}"
)

print(
    f"Improvement     : "
    f"{overall_improvement:+.8f}"
)

print(
    f"Global BSS      : "
    f"{global_oof_score:.2f}"
)

print(
    f"Split BSS       : "
    f"{split_oof_score:.2f}"
)

print(
    f"Reference       : "
    f"{reference:.8f}"
)


print("\n[R POOLED OOF]")

print(
    f"Global-R : "
    f"{global_r_oof:.8f}"
)

print(
    f"R-only   : "
    f"{r_only_oof:.8f}"
)

print(
    f"Delta    : "
    f"{global_r_oof - r_only_oof:+.8f}"
)


print("\n[F POOLED OOF]")

print(
    f"Global-F : "
    f"{global_f_oof:.8f}"
)

print(
    f"F-only   : "
    f"{f_only_oof:.8f}"
)

print(
    f"Delta    : "
    f"{global_f_oof - f_only_oof:+.8f}"
)


# =========================================================
# 저장
# =========================================================
oof.to_csv(
    VALIDATION_DIR
    / "oof_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

summary.to_csv(
    VALIDATION_DIR
    / "fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


metadata = {

    "experiment_name":
        EXPERIMENT_NAME,

    "reference_model":
        "cb_v002_game_type_group_es_v001",

    "experiment": (
        "Compare global group-specific "
        "early stopping models against "
        "R-only / F-only models."
    ),

    "feature_count":
        len(FEATURES),

    "categorical_features":
        CAT_COLS,

    "early_stopping_rounds":
        EARLY_STOPPING_ROUNDS,

    "catboost_params":
        CATBOOST_PARAMS,

    "global_group_oof_brier":
        global_oof_brier,

    "split_group_oof_brier":
        split_oof_brier,

    "overall_improvement":
        overall_improvement,

    "global_R_oof_brier":
        global_r_oof,

    "R_only_oof_brier":
        r_only_oof,

    "global_F_oof_brier":
        global_f_oof,

    "F_only_oof_brier":
        f_only_oof,
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
        ensure_ascii=False,
        indent=2,
    )


print("\nSaved:")

print(
    VALIDATION_DIR
    / "oof_predictions.csv"
)

print(
    VALIDATION_DIR
    / "fold_summary.csv"
)

print(
    VALIDATION_DIR
    / "metadata.json"
)


print("\nExperiment complete.")