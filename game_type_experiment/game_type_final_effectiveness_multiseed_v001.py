from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_final_effectiveness_multiseed_v001"

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
# Multi-Seed
# =========================================================
SEEDS = [
    7,
    42,
    123,
    999,
    2026,
]


# =========================================================
# R/F Classifier Ensemble Weight
#
# 기존 R/F 실험에서 정한 weight 그대로 고정.
# 이번 실험에서는 weight 재탐색하지 않는다.
# =========================================================
R_GLOBAL_WEIGHT = 0.49
R_ONLY_WEIGHT = 0.51

F_GLOBAL_WEIGHT = 0.64
F_ONLY_WEIGHT = 0.36


# =========================================================
# R RMSE Ensemble Weight
#
# Multi-Seed 검증을 통과한 fixed weight.
# =========================================================
R_CLASSIFIER_WEIGHT = 0.73
R_RMSE_WEIGHT = 0.27


# =========================================================
# Categorical Features
# =========================================================
BASE_CAT_COLS = [
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

ADD_CAT_FEATURES = [
    "count_state",
    "hand_matchup",
]

CAT_COLS = (
    BASE_CAT_COLS
    + ADD_CAT_FEATURES
)


# =========================================================
# CatBoost 공통 설정
# =========================================================
BASE_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 0,
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
# PDF에서 사용한 15개 Feature
# =========================================================
ADD_FEATURES = [

    # History / reliability
    "pitcher_no_history",
    "batter_no_history",
    "pitcher_no_prev_game",

    # Sample size
    "pitcher_log_n",
    "batter_log_n",

    # Recent vs career
    "pitcher_recent1_success_delta",
    "pitcher_recent3_success_delta",
    "pitcher_recent5_success_delta",
    "pitcher_recent3_middle_delta",

    # Situation
    "count_state",
    "hand_matchup",

    # Score context
    "abs_score_diff",
    "pitcher_team_leading",
    "pitcher_team_trailing",
    "tie_game",
]


# =========================================================
# Feature Engineering
# =========================================================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    15개 Feature 생성.

    모든 Feature는 현재 row의 값 또는
    공식 asof_* 정보만 사용한다.

    evaluation 데이터 내 다른 row의
    통계 / 순서 / 집계를 사용하지 않는다.
    """

    df = df.copy()

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------
    df["pitcher_no_history"] = (
        df["asof_pitcher_n"] == 0
    ).astype(int)

    df["batter_no_history"] = (
        df["asof_batter_n"] == 0
    ).astype(int)

    df["pitcher_no_prev_game"] = (
        df[
            "asof_pitcher_prev1_game_success_rate"
        ].isna()
    ).astype(int)

    # -----------------------------------------------------
    # Sample size
    # -----------------------------------------------------
    df["pitcher_log_n"] = np.log1p(
        df["asof_pitcher_n"]
    )

    df["batter_log_n"] = np.log1p(
        df["asof_batter_n"]
    )

    # -----------------------------------------------------
    # Recent vs Career
    # -----------------------------------------------------
    df["pitcher_recent1_success_delta"] = (
        df[
            "asof_pitcher_prev1_game_success_rate"
        ]
        - df["asof_pitcher_success_rate"]
    )

    df["pitcher_recent3_success_delta"] = (
        df[
            "asof_pitcher_prev3_game_success_rate"
        ]
        - df["asof_pitcher_success_rate"]
    )

    df["pitcher_recent5_success_delta"] = (
        df[
            "asof_pitcher_prev5_game_success_rate"
        ]
        - df["asof_pitcher_success_rate"]
    )

    df["pitcher_recent3_middle_delta"] = (
        df[
            "asof_pitcher_prev3_game_middle_rate"
        ]
        - df["asof_pitcher_middle_rate"]
    )

    # -----------------------------------------------------
    # Count State
    # -----------------------------------------------------
    df["count_state"] = (
        df["balls_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
        + "_"
        + df["strikes_before"]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # Hand Matchup
    # -----------------------------------------------------
    df["hand_matchup"] = (
        df["pitcher_hand"]
        .fillna("__MISSING__")
        .astype(str)
        + "_"
        + df["batter_hand"]
        .fillna("__MISSING__")
        .astype(str)
    )

    # -----------------------------------------------------
    # Score Context
    # -----------------------------------------------------
    df["abs_score_diff"] = (
        df["score_diff_pitcher_team"].abs()
    )

    df["pitcher_team_leading"] = (
        df["score_diff_pitcher_team"] > 0
    ).astype(int)

    df["pitcher_team_trailing"] = (
        df["score_diff_pitcher_team"] < 0
    ).astype(int)

    df["tie_game"] = (
        df["score_diff_pitcher_team"] == 0
    ).astype(int)

    return df


# =========================================================
# Categorical 전처리
# =========================================================
def prepare_categorical(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for col in CAT_COLS:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return df


# =========================================================
# 평가 함수
# =========================================================
def calc_brier(
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

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


# =========================================================
# 모델 생성
# =========================================================
def create_classifier(seed):

    params = BASE_PARAMS.copy()

    params.update({
        "loss_function": "Logloss",
        "random_seed": int(seed),
    })

    return CatBoostClassifier(
        **params
    )


def create_rmse(seed):

    params = BASE_PARAMS.copy()

    params.update({
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": int(seed),
    })

    return CatBoostRegressor(
        **params
    )


def get_best_tree_count(model):

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        return BASE_PARAMS["iterations"]

    return int(best_iteration) + 1


# =========================================================
# 데이터 로드
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)

print("\nLoad train.csv...")

train_raw = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
)


# =========================================================
# Raw Feature
# =========================================================
BASE_FEATURES = [
    col
    for col in train_raw.columns
    if col not in [
        ID_COL,
        TARGET_COL,
    ]
]


print(
    f"Raw Features   : "
    f"{len(BASE_FEATURES)}"
)


# =========================================================
# 62FE 생성
# =========================================================
train = make_features(
    train_raw
)


FE_FEATURES = (
    BASE_FEATURES.copy()
)


for col in ADD_FEATURES:

    if col not in FE_FEATURES:

        FE_FEATURES.append(
            col
        )


print(
    f"Added Features : "
    f"{len(ADD_FEATURES)}"
)

print(
    f"Final Features : "
    f"{len(FE_FEATURES)}"
)


if len(FE_FEATURES) != 62:

    print(
        "WARNING: "
        f"Feature count = "
        f"{len(FE_FEATURES)}"
    )


# =========================================================
# Feature 검증
# =========================================================
missing_features = [
    col
    for col in FE_FEATURES
    if col not in train.columns
]

if missing_features:

    raise ValueError(
        "Missing features: "
        f"{missing_features}"
    )


train = prepare_categorical(
    train
)


# =========================================================
# game_type 검증
# =========================================================
game_types = set(
    train[GROUP_COL]
    .dropna()
    .astype(str)
    .unique()
)

unexpected = (
    game_types
    - {R_GROUP, F_GROUP}
)

if unexpected:

    raise ValueError(
        "Unexpected game_type: "
        f"{unexpected}"
    )


# =========================================================
# 저장 폴더
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 결과 저장
# =========================================================
all_seed_oof = []

seed_summary_rows = []
season_summary_rows = []
tree_summary_rows = []


# =========================================================
# Multi-Seed
# =========================================================
for seed in SEEDS:

    print("\n")
    print("#" * 80)
    print(f"SEED {seed}")
    print("#" * 80)

    seed_start = time.time()

    seed_fold_parts = []


    # =====================================================
    # Fold
    # =====================================================
    for fold_info in FOLDS:

        fold = fold_info["fold"]

        train_start = (
            fold_info["train_start"]
        )

        train_end = (
            fold_info["train_end"]
        )

        valid_season = (
            fold_info["valid_season"]
        )


        print("\n")
        print("=" * 80)

        print(
            f"Seed {seed} | "
            f"Fold {fold} | "
            f"Train {train_start}~{train_end} "
            f"-> Valid {valid_season}"
        )

        print("=" * 80)


        # =================================================
        # Fold split
        # =================================================
        train_mask = (
            (
                train[SEASON_COL]
                >= train_start
            )
            &
            (
                train[SEASON_COL]
                <= train_end
            )
        )

        valid_mask = (
            train[SEASON_COL]
            == valid_season
        )


        fold_train = (
            train.loc[
                train_mask
            ]
            .copy()
        )

        fold_valid = (
            train.loc[
                valid_mask
            ]
            .copy()
        )


        train_r = (
            fold_train[
                fold_train[GROUP_COL]
                == R_GROUP
            ]
            .copy()
        )

        train_f = (
            fold_train[
                fold_train[GROUP_COL]
                == F_GROUP
            ]
            .copy()
        )

        valid_r = (
            fold_valid[
                fold_valid[GROUP_COL]
                == R_GROUP
            ]
            .copy()
        )

        valid_f = (
            fold_valid[
                fold_valid[GROUP_COL]
                == F_GROUP
            ]
            .copy()
        )


        # =================================================
        # Data
        # =================================================
        X_train_all = (
            fold_train[
                FE_FEATURES
            ]
        )

        y_train_all = (
            fold_train[
                TARGET_COL
            ]
        )


        X_train_r = (
            train_r[
                FE_FEATURES
            ]
        )

        y_train_r = (
            train_r[
                TARGET_COL
            ]
        )


        X_train_f = (
            train_f[
                FE_FEATURES
            ]
        )

        y_train_f = (
            train_f[
                TARGET_COL
            ]
        )


        X_valid_all = (
            fold_valid[
                FE_FEATURES
            ]
        )

        y_valid_all = (
            fold_valid[
                TARGET_COL
            ]
        )


        X_valid_r = (
            valid_r[
                FE_FEATURES
            ]
        )

        y_valid_r = (
            valid_r[
                TARGET_COL
            ]
        )


        X_valid_f = (
            valid_f[
                FE_FEATURES
            ]
        )

        y_valid_f = (
            valid_f[
                TARGET_COL
            ]
        )


        # =================================================
        # A. Standard CatBoost 62FE
        #
        # Train = 전체
        # Valid = 전체
        #
        # R/F 처리 없음
        # =================================================
        print("\n[A] Train Standard 62FE...")

        standard_model = (
            create_classifier(seed)
        )

        standard_model.fit(
            X_train_all,
            y_train_all,

            cat_features=CAT_COLS,

            eval_set=(
                X_valid_all,
                y_valid_all,
            ),

            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),

            use_best_model=True,
        )


        standard_pred = (
            standard_model
            .predict_proba(
                X_valid_all
            )[:, 1]
        )


        standard_trees = (
            get_best_tree_count(
                standard_model
            )
        )


        # =================================================
        # B-1. Global-R
        #
        # Train 전체
        # ES Validation R
        # =================================================
        print("[B] Train Global-R...")

        global_r_model = (
            create_classifier(seed)
        )

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


        global_r_pred = (
            global_r_model
            .predict_proba(
                X_valid_r
            )[:, 1]
        )


        global_r_trees = (
            get_best_tree_count(
                global_r_model
            )
        )


        # =================================================
        # B-2. Global-F
        # =================================================
        print("[B] Train Global-F...")

        global_f_model = (
            create_classifier(seed)
        )

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


        global_f_pred = (
            global_f_model
            .predict_proba(
                X_valid_f
            )[:, 1]
        )


        global_f_trees = (
            get_best_tree_count(
                global_f_model
            )
        )


        # =================================================
        # C-1. R-only
        # =================================================
        print("[C] Train R-only...")

        r_only_model = (
            create_classifier(seed)
        )

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


        r_only_pred = (
            r_only_model
            .predict_proba(
                X_valid_r
            )[:, 1]
        )


        r_only_trees = (
            get_best_tree_count(
                r_only_model
            )
        )


        # =================================================
        # C-2. F-only
        # =================================================
        print("[C] Train F-only...")

        f_only_model = (
            create_classifier(seed)
        )

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


        f_only_pred = (
            f_only_model
            .predict_proba(
                X_valid_f
            )[:, 1]
        )


        f_only_trees = (
            get_best_tree_count(
                f_only_model
            )
        )


        # =================================================
        # D. Global-R RMSE
        #
        # Train 전체
        # ES Validation R
        # =================================================
        print("[D] Train Global-R RMSE...")

        rmse_r_model = (
            create_rmse(seed)
        )

        rmse_r_model.fit(
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


        rmse_r_pred_raw = (
            rmse_r_model.predict(
                X_valid_r
            )
        )


        rmse_r_pred = np.clip(
            rmse_r_pred_raw,
            0.0,
            1.0,
        )


        rmse_r_trees = (
            get_best_tree_count(
                rmse_r_model
            )
        )


        # =================================================
        # Validation mask
        # =================================================
        valid_r_mask_np = (
            fold_valid[GROUP_COL]
            == R_GROUP
        ).to_numpy()

        valid_f_mask_np = (
            fold_valid[GROUP_COL]
            == F_GROUP
        ).to_numpy()


        # =================================================
        # B. R/F Group-specific ES
        #
        # R = Global-R
        # F = Global-F
        # =================================================
        group_es_pred = np.empty(
            len(fold_valid),
            dtype=float,
        )


        group_es_pred[
            valid_r_mask_np
        ] = global_r_pred


        group_es_pred[
            valid_f_mask_np
        ] = global_f_pred


        # =================================================
        # C. R/F Classifier Ensemble
        # =================================================
        classifier_r_pred = (

            R_GLOBAL_WEIGHT
            * global_r_pred

            +

            R_ONLY_WEIGHT
            * r_only_pred
        )


        classifier_f_pred = (

            F_GLOBAL_WEIGHT
            * global_f_pred

            +

            F_ONLY_WEIGHT
            * f_only_pred
        )


        rf_classifier_pred = np.empty(
            len(fold_valid),
            dtype=float,
        )


        rf_classifier_pred[
            valid_r_mask_np
        ] = classifier_r_pred


        rf_classifier_pred[
            valid_f_mask_np
        ] = classifier_f_pred


        # =================================================
        # D. Final R/F + RMSE
        #
        # R =
        # 73% R classifier ensemble
        # +27% RMSE
        #
        # F = 기존 classifier ensemble
        # =================================================
        final_r_pred = (

            R_CLASSIFIER_WEIGHT
            * classifier_r_pred

            +

            R_RMSE_WEIGHT
            * rmse_r_pred
        )


        final_pred = np.empty(
            len(fold_valid),
            dtype=float,
        )


        final_pred[
            valid_r_mask_np
        ] = final_r_pred


        final_pred[
            valid_f_mask_np
        ] = classifier_f_pred


        # =================================================
        # Fold 평가
        # =================================================
        y_valid = (
            y_valid_all
            .to_numpy(
                dtype=float
            )
        )


        standard_brier = calc_brier(
            y_valid,
            standard_pred,
        )


        group_es_brier = calc_brier(
            y_valid,
            group_es_pred,
        )


        rf_classifier_brier = calc_brier(
            y_valid,
            rf_classifier_pred,
        )


        final_brier = calc_brier(
            y_valid,
            final_pred,
        )


        print("\n[Fold Result]")

        print(
            f"A Standard 62FE      : "
            f"{standard_brier:.8f}"
        )

        print(
            f"B R/F Group-ES       : "
            f"{group_es_brier:.8f} "
            f"| A→B "
            f"{standard_brier - group_es_brier:+.8f}"
        )

        print(
            f"C R/F Classifier     : "
            f"{rf_classifier_brier:.8f} "
            f"| B→C "
            f"{group_es_brier - rf_classifier_brier:+.8f}"
        )

        print(
            f"D R/F + RMSE         : "
            f"{final_brier:.8f} "
            f"| C→D "
            f"{rf_classifier_brier - final_brier:+.8f}"
        )

        print(
            f"A→D Total            : "
            f"{standard_brier - final_brier:+.8f}"
        )


        print(
            "\nTrees | "
            f"Standard={standard_trees} | "
            f"Global-R={global_r_trees} | "
            f"R-only={r_only_trees} | "
            f"Global-F={global_f_trees} | "
            f"F-only={f_only_trees} | "
            f"RMSE-R={rmse_r_trees}"
        )


        # =================================================
        # OOF 저장
        # =================================================
        fold_result = pd.DataFrame({

            ID_COL:
                fold_valid[
                    ID_COL
                ].values,

            SEASON_COL:
                fold_valid[
                    SEASON_COL
                ].values,

            GROUP_COL:
                fold_valid[
                    GROUP_COL
                ].values,

            "y_true":
                y_valid,

            "standard_62FE_pred":
                standard_pred,

            "group_es_pred":
                group_es_pred,

            "rf_classifier_pred":
                rf_classifier_pred,

            "rf_rmse_final_pred":
                final_pred,

            "seed":
                seed,

            "fold":
                fold,
        })


        seed_fold_parts.append(
            fold_result
        )


        tree_summary_rows.append({

            "seed":
                seed,

            "fold":
                fold,

            "valid_season":
                valid_season,

            "standard_trees":
                standard_trees,

            "global_R_trees":
                global_r_trees,

            "R_only_trees":
                r_only_trees,

            "global_F_trees":
                global_f_trees,

            "F_only_trees":
                f_only_trees,

            "RMSE_R_trees":
                rmse_r_trees,
        })


    # =====================================================
    # Seed OOF
    # =====================================================
    seed_oof = pd.concat(
        seed_fold_parts,
        ignore_index=True,
    )


    standard_oof = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "standard_62FE_pred"
        ],
    )


    group_es_oof = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "group_es_pred"
        ],
    )


    classifier_oof = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "rf_classifier_pred"
        ],
    )


    final_oof = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "rf_rmse_final_pred"
        ],
    )


    # =====================================================
    # 효과 분해
    # =====================================================
    group_es_delta = (
        standard_oof
        - group_es_oof
    )

    classifier_delta = (
        group_es_oof
        - classifier_oof
    )

    rmse_delta = (
        classifier_oof
        - final_oof
    )

    total_delta = (
        standard_oof
        - final_oof
    )


    elapsed = (
        time.time()
        - seed_start
    )


    seed_summary_rows.append({

        "seed":
            seed,

        "standard_62FE":
            standard_oof,

        "group_es":
            group_es_oof,

        "rf_classifier":
            classifier_oof,

        "rf_rmse_final":
            final_oof,

        "A_to_B_group_es_delta":
            group_es_delta,

        "B_to_C_classifier_delta":
            classifier_delta,

        "C_to_D_rmse_delta":
            rmse_delta,

        "A_to_D_total_delta":
            total_delta,

        "elapsed_sec":
            elapsed,
    })


    # =====================================================
    # 시즌별
    # =====================================================
    for season in sorted(
        seed_oof[
            SEASON_COL
        ].unique()
    ):

        season_df = (
            seed_oof[
                seed_oof[
                    SEASON_COL
                ]
                == season
            ]
            .copy()
        )


        season_standard = calc_brier(
            season_df["y_true"],
            season_df[
                "standard_62FE_pred"
            ],
        )


        season_group_es = calc_brier(
            season_df["y_true"],
            season_df[
                "group_es_pred"
            ],
        )


        season_classifier = calc_brier(
            season_df["y_true"],
            season_df[
                "rf_classifier_pred"
            ],
        )


        season_final = calc_brier(
            season_df["y_true"],
            season_df[
                "rf_rmse_final_pred"
            ],
        )


        season_summary_rows.append({

            "seed":
                seed,

            "season":
                int(season),

            "standard_62FE":
                season_standard,

            "group_es":
                season_group_es,

            "rf_classifier":
                season_classifier,

            "rf_rmse_final":
                season_final,

            "A_to_B_delta":
                (
                    season_standard
                    - season_group_es
                ),

            "B_to_C_delta":
                (
                    season_group_es
                    - season_classifier
                ),

            "C_to_D_delta":
                (
                    season_classifier
                    - season_final
                ),

            "A_to_D_delta":
                (
                    season_standard
                    - season_final
                ),
        })


    all_seed_oof.append(
        seed_oof
    )


    print("\n")
    print("-" * 80)
    print(f"[SEED {seed} FINAL]")
    print("-" * 80)

    print(
        f"A Standard       : "
        f"{standard_oof:.8f}"
    )

    print(
        f"B Group-ES       : "
        f"{group_es_oof:.8f} "
        f"| Δ {group_es_delta:+.8f}"
    )

    print(
        f"C R/F Classifier : "
        f"{classifier_oof:.8f} "
        f"| Δ {classifier_delta:+.8f}"
    )

    print(
        f"D R/F + RMSE     : "
        f"{final_oof:.8f} "
        f"| Δ {rmse_delta:+.8f}"
    )

    print(
        f"A→D Total Delta  : "
        f"{total_delta:+.8f}"
    )

    print(
        f"Time             : "
        f"{elapsed:.1f}s"
    )


# =========================================================
# 결과 DataFrame
# =========================================================
seed_summary = pd.DataFrame(
    seed_summary_rows
)

season_summary = pd.DataFrame(
    season_summary_rows
)

tree_summary = pd.DataFrame(
    tree_summary_rows
)

all_oof = pd.concat(
    all_seed_oof,
    ignore_index=True,
)


# =========================================================
# Multi-Seed Summary 함수
# =========================================================
def summarize_delta(
    df,
    column,
):

    values = (
        df[column]
        .to_numpy(
            dtype=float
        )
    )

    return {
        "mean":
            float(
                np.mean(values)
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "wins":
            int(
                np.sum(
                    values > 0
                )
            ),
    }


# =========================================================
# 전체 seed 효과
# =========================================================
ab_stats = summarize_delta(
    seed_summary,
    "A_to_B_group_es_delta",
)

bc_stats = summarize_delta(
    seed_summary,
    "B_to_C_classifier_delta",
)

cd_stats = summarize_delta(
    seed_summary,
    "C_to_D_rmse_delta",
)

ad_stats = summarize_delta(
    seed_summary,
    "A_to_D_total_delta",
)


# =========================================================
# Mean model Brier
# =========================================================
mean_standard = float(
    seed_summary[
        "standard_62FE"
    ].mean()
)

mean_group_es = float(
    seed_summary[
        "group_es"
    ].mean()
)

mean_classifier = float(
    seed_summary[
        "rf_classifier"
    ].mean()
)

mean_final = float(
    seed_summary[
        "rf_rmse_final"
    ].mean()
)


# =========================================================
# 시즌별 Multi-Seed 평균
# =========================================================
season_mean = (

    season_summary

    .groupby(
        "season",
        as_index=False,
    )

    .agg(

        standard_mean=(
            "standard_62FE",
            "mean",
        ),

        group_es_mean=(
            "group_es",
            "mean",
        ),

        rf_classifier_mean=(
            "rf_classifier",
            "mean",
        ),

        final_mean=(
            "rf_rmse_final",
            "mean",
        ),

        A_to_B_mean=(
            "A_to_B_delta",
            "mean",
        ),

        A_to_B_std=(
            "A_to_B_delta",
            "std",
        ),

        A_to_B_wins=(
            "A_to_B_delta",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        B_to_C_mean=(
            "B_to_C_delta",
            "mean",
        ),

        B_to_C_wins=(
            "B_to_C_delta",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        C_to_D_mean=(
            "C_to_D_delta",
            "mean",
        ),

        C_to_D_wins=(
            "C_to_D_delta",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        A_to_D_mean=(
            "A_to_D_delta",
            "mean",
        ),

        A_to_D_std=(
            "A_to_D_delta",
            "std",
        ),

        A_to_D_wins=(
            "A_to_D_delta",
            lambda x: int(
                (x > 0).sum()
            ),
        ),
    )
)


# =========================================================
# 2024만 별도
# =========================================================
season_2024 = (
    season_summary[
        season_summary[
            "season"
        ]
        == 2024
    ]
    .copy()
)


ab_2024 = summarize_delta(
    season_2024,
    "A_to_B_delta",
)

ad_2024 = summarize_delta(
    season_2024,
    "A_to_D_delta",
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("#" * 80)
print("R/F FINAL EFFECTIVENESS RESULT")
print("#" * 80)


print("\n[Seed Summary]")

print(
    seed_summary[
        [
            "seed",
            "standard_62FE",
            "group_es",
            "rf_classifier",
            "rf_rmse_final",
            "A_to_B_group_es_delta",
            "B_to_C_classifier_delta",
            "C_to_D_rmse_delta",
            "A_to_D_total_delta",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[Mean Brier Across 5 Seeds]")

print(
    f"A Standard 62FE      : "
    f"{mean_standard:.8f}"
)

print(
    f"B R/F Group-ES       : "
    f"{mean_group_es:.8f}"
)

print(
    f"C R/F Classifier     : "
    f"{mean_classifier:.8f}"
)

print(
    f"D R/F + RMSE         : "
    f"{mean_final:.8f}"
)


print("\n[Effect Decomposition]")

print(
    "A → B | Group-specific ES"
)

print(
    f"Mean Delta : "
    f"{ab_stats['mean']:+.8f}"
)

print(
    f"Delta Std  : "
    f"{ab_stats['std']:.8f}"
)

print(
    f"Win Count  : "
    f"{ab_stats['wins']}/5"
)


print(
    "\nB → C | R-only/F-only Ensemble"
)

print(
    f"Mean Delta : "
    f"{bc_stats['mean']:+.8f}"
)

print(
    f"Delta Std  : "
    f"{bc_stats['std']:.8f}"
)

print(
    f"Win Count  : "
    f"{bc_stats['wins']}/5"
)


print(
    "\nC → D | R-RMSE 27%"
)

print(
    f"Mean Delta : "
    f"{cd_stats['mean']:+.8f}"
)

print(
    f"Delta Std  : "
    f"{cd_stats['std']:.8f}"
)

print(
    f"Win Count  : "
    f"{cd_stats['wins']}/5"
)


print(
    "\nA → D | Final R/F Total Effect"
)

print(
    f"Mean Delta : "
    f"{ad_stats['mean']:+.8f}"
)

print(
    f"Delta Std  : "
    f"{ad_stats['std']:.8f}"
)

print(
    f"Win Count  : "
    f"{ad_stats['wins']}/5"
)


print("\n[Season Mean Across Seeds]")

print(
    season_mean.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[2024 R/F Stability]")

print(
    "A → B Group-ES"
)

print(
    f"Mean Delta : "
    f"{ab_2024['mean']:+.8f}"
)

print(
    f"Delta Std  : "
    f"{ab_2024['std']:.8f}"
)

print(
    f"Win Count  : "
    f"{ab_2024['wins']}/5"
)


print(
    "\nA → D Final R/F"
)

print(
    f"Mean Delta : "
    f"{ad_2024['mean']:+.8f}"
)

print(
    f"Delta Std  : "
    f"{ad_2024['std']:.8f}"
)

print(
    f"Win Count  : "
    f"{ad_2024['wins']}/5"
)


# =========================================================
# 자동 판정
# =========================================================
print("\n")
print("=" * 80)
print("R/F EFFECTIVENESS CHECK")
print("=" * 80)


if ad_stats["mean"] > 0:
    print(
        "Final R/F mean delta     : PASS"
    )
else:
    print(
        "Final R/F mean delta     : FAIL"
    )


if ad_stats["wins"] >= 4:
    print(
        "Final R/F pooled wins    : PASS"
    )
else:
    print(
        "Final R/F pooled wins    : FAIL"
    )


if ad_2024["mean"] > 0:
    print(
        "Final R/F 2024 mean      : PASS"
    )
else:
    print(
        "Final R/F 2024 mean      : FAIL"
    )


if ad_2024["wins"] >= 4:
    print(
        "Final R/F 2024 wins      : PASS"
    )
else:
    print(
        "Final R/F 2024 wins      : FAIL"
    )


# =========================================================
# 해석 보조
# =========================================================
print("\n")
print("=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)

print(
    """
A → B:
  game_type별 Early Stopping 자체 효과

B → C:
  R-only / F-only 모델을 추가한 효과

C → D:
  R에 RMSE 27%를 추가한 효과

A → D:
  동일한 62 Features를 사용했을 때
  R/F 최종 구조 자체가 Standard CatBoost보다
  실제로 추가 가치를 만드는지 확인하는 핵심 비교
"""
)


# =========================================================
# 저장
# =========================================================
seed_summary.to_csv(
    VALIDATION_DIR
    / "seed_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


season_summary.to_csv(
    VALIDATION_DIR
    / "seed_season_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


season_mean.to_csv(
    VALIDATION_DIR
    / "season_mean_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


tree_summary.to_csv(
    VALIDATION_DIR
    / "tree_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


all_oof.to_csv(
    VALIDATION_DIR
    / "all_seed_oof_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


metadata = {

    "experiment_name":
        EXPERIMENT_NAME,

    "seeds":
        SEEDS,

    "feature_count":
        len(FE_FEATURES),

    "models": {
        "A":
            "Standard CatBoost 62FE",

        "B":
            "R/F Group-specific Early Stopping",

        "C":
            "R/F Classifier Ensemble",

        "D":
            "R/F Classifier Ensemble + R RMSE 27%",
    },

    "weights": {

        "R_global":
            R_GLOBAL_WEIGHT,

        "R_only":
            R_ONLY_WEIGHT,

        "F_global":
            F_GLOBAL_WEIGHT,

        "F_only":
            F_ONLY_WEIGHT,

        "R_classifier":
            R_CLASSIFIER_WEIGHT,

        "R_rmse":
            R_RMSE_WEIGHT,
    },

    "mean_brier": {

        "standard_62FE":
            mean_standard,

        "group_es":
            mean_group_es,

        "rf_classifier":
            mean_classifier,

        "rf_rmse_final":
            mean_final,
    },

    "A_to_B": {
        "mean_delta":
            ab_stats["mean"],

        "delta_std":
            ab_stats["std"],

        "win_count":
            ab_stats["wins"],
    },

    "B_to_C": {
        "mean_delta":
            bc_stats["mean"],

        "delta_std":
            bc_stats["std"],

        "win_count":
            bc_stats["wins"],
    },

    "C_to_D": {
        "mean_delta":
            cd_stats["mean"],

        "delta_std":
            cd_stats["std"],

        "win_count":
            cd_stats["wins"],
    },

    "A_to_D": {
        "mean_delta":
            ad_stats["mean"],

        "delta_std":
            ad_stats["std"],

        "win_count":
            ad_stats["wins"],
    },

    "2024_A_to_D": {
        "mean_delta":
            ad_2024["mean"],

        "delta_std":
            ad_2024["std"],

        "win_count":
            ad_2024["wins"],
    },
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
    / "seed_summary.csv"
)

print(
    VALIDATION_DIR
    / "seed_season_summary.csv"
)

print(
    VALIDATION_DIR
    / "season_mean_summary.csv"
)

print(
    VALIDATION_DIR
    / "tree_summary.csv"
)

print(
    VALIDATION_DIR
    / "all_seed_oof_predictions.csv"
)

print(
    VALIDATION_DIR
    / "metadata.json"
)


print("\nExperiment complete.")