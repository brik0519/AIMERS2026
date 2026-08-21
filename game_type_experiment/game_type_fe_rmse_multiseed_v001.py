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
EXPERIMENT_NAME = "cb_v002_game_type_fe_rmse_multiseed_v001"

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
# Seed
# =========================================================
SEEDS = [
    7,
    42,
    123,
    999,
    2026,
]

# RMSE 27% weight를 찾은 seed
TUNING_SEED = 42


# =========================================================
# 기존 62FE R/F Classifier Ensemble Weight
# =========================================================
R_GLOBAL_WEIGHT = 0.49
R_ONLY_WEIGHT = 0.51

F_GLOBAL_WEIGHT = 0.64
F_ONLY_WEIGHT = 0.36


# =========================================================
# 이번에 검증할 Fixed RMSE Weight
#
# R:
# 73% = 기존 R classifier ensemble
# 27% = Global-R RMSE
#
# 이번 Primary 검증에서는 절대 다시 튜닝하지 않는다.
# =========================================================
CURRENT_R_ENSEMBLE_WEIGHT = 0.73
RMSE_R_WEIGHT = 0.27


# =========================================================
# Secondary 분석용 RMSE Weight Search
# =========================================================
WEIGHT_STEP = 0.01

RMSE_WEIGHTS = np.round(
    np.arange(
        0.0,
        1.0 + WEIGHT_STEP,
        WEIGHT_STEP,
    ),
    2,
)


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
# 15개 추가 Feature
# =========================================================
ADD_FEATURES = [
    # History
    "pitcher_no_history",
    "batter_no_history",
    "pitcher_no_prev_game",

    # Sample size
    "pitcher_log_n",
    "batter_log_n",

    # Recent form
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
def make_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    모든 파생변수는 현재 row에 존재하는 값과
    공식 asof_* 값만 이용한다.

    다른 evaluation row의 통계/분포/순서/집계는
    사용하지 않는다.
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
    # Count state
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
    # Hand matchup
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
    # Score context
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
# Categorical 처리
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
# 평가
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
# 모델
# =========================================================
def create_classifier(
    seed,
):

    params = BASE_PARAMS.copy()

    params.update({
        "loss_function": "Logloss",
        "random_seed": int(seed),
    })

    return CatBoostClassifier(
        **params
    )


def create_rmse(
    seed,
):

    params = BASE_PARAMS.copy()

    params.update({
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": int(seed),
    })

    return CatBoostRegressor(
        **params
    )


def get_best_tree_count(
    model,
):

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        return BASE_PARAMS[
            "iterations"
        ]

    return int(
        best_iteration
    ) + 1


# =========================================================
# Secondary RMSE Weight Search
# =========================================================
def search_best_rmse_weight(
    y_true,
    classifier_pred,
    rmse_pred,
):

    rows = []

    for rmse_weight in RMSE_WEIGHTS:

        classifier_weight = (
            1.0 - rmse_weight
        )

        pred = (
            classifier_weight
            * classifier_pred
            +
            rmse_weight
            * rmse_pred
        )

        brier = calc_brier(
            y_true,
            pred,
        )

        rows.append({
            "rmse_weight":
                float(rmse_weight),

            "classifier_weight":
                float(classifier_weight),

            "brier":
                brier,
        })

    result = pd.DataFrame(
        rows
    )

    best_idx = (
        result["brier"]
        .idxmin()
    )

    best = result.loc[
        best_idx
    ]

    return (
        float(
            best["rmse_weight"]
        ),
        float(
            best["brier"]
        ),
    )


# =========================================================
# 데이터 로드
#
# 이번 실험은 train.csv만 사용.
# test.csv는 읽지 않는다.
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)

print("\nLoad train.csv...")

train_raw = pd.read_csv(
    DATA_DIR / "train.csv",
    encoding="utf-8-sig",
)


BASE_FEATURES = [
    col
    for col in train_raw.columns
    if col not in [
        ID_COL,
        TARGET_COL,
    ]
]


print(
    f"Raw Features : "
    f"{len(BASE_FEATURES)}"
)


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
# 저장
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 결과 저장용
# =========================================================
all_seed_oof = []

seed_summary_rows = []
season_summary_rows = []
weight_summary_rows = []
tree_summary_rows = []


# =========================================================
# Multi-Seed
# =========================================================
for seed in SEEDS:

    print("\n")
    print("#" * 80)

    print(
        f"SEED {seed}"
    )

    print("#" * 80)

    seed_start = time.time()

    seed_fold_parts = []


    # =====================================================
    # Fold
    # =====================================================
    for fold_info in FOLDS:

        fold = fold_info[
            "fold"
        ]

        train_start = fold_info[
            "train_start"
        ]

        train_end = fold_info[
            "train_end"
        ]

        valid_season = fold_info[
            "valid_season"
        ]


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
        # Expanding Split
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
        # 1. Global-R Classifier
        # =================================================
        global_r_model = (
            create_classifier(
                seed
            )
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
        # 2. R-only Classifier
        # =================================================
        r_only_model = (
            create_classifier(
                seed
            )
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
        # 3. Global-F Classifier
        # =================================================
        global_f_model = (
            create_classifier(
                seed
            )
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
        # 4. F-only Classifier
        # =================================================
        f_only_model = (
            create_classifier(
                seed
            )
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
        # 5. Global-R RMSE
        #
        # Train = R + F 전체
        # Validation = R
        # =================================================
        rmse_r_model = (
            create_rmse(
                seed
            )
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
        # 기존 62FE R/F Classifier Ensemble
        # =================================================
        current_r_pred = (

            R_GLOBAL_WEIGHT
            * global_r_pred

            +

            R_ONLY_WEIGHT
            * r_only_pred
        )


        current_f_pred = (

            F_GLOBAL_WEIGHT
            * global_f_pred

            +

            F_ONLY_WEIGHT
            * f_only_pred
        )


        current_pred = np.empty(
            len(fold_valid),
            dtype=float,
        )


        valid_r_mask_np = (
            fold_valid[GROUP_COL]
            == R_GROUP
        ).to_numpy()

        valid_f_mask_np = (
            fold_valid[GROUP_COL]
            == F_GROUP
        ).to_numpy()


        current_pred[
            valid_r_mask_np
        ] = current_r_pred


        current_pred[
            valid_f_mask_np
        ] = current_f_pred


        # =================================================
        # Fixed RMSE Ensemble
        #
        # R:
        # 73% current classifier
        # 27% RMSE
        #
        # F:
        # 기존 classifier 그대로
        # =================================================
        fixed_r_pred = (

            CURRENT_R_ENSEMBLE_WEIGHT
            * current_r_pred

            +

            RMSE_R_WEIGHT
            * rmse_r_pred
        )


        fixed_pred = np.empty(
            len(fold_valid),
            dtype=float,
        )


        fixed_pred[
            valid_r_mask_np
        ] = fixed_r_pred


        fixed_pred[
            valid_f_mask_np
        ] = current_f_pred


        # =================================================
        # Fold 평가
        # =================================================
        y_valid = (
            fold_valid[
                TARGET_COL
            ]
            .to_numpy(
                dtype=float
            )
        )


        current_brier = calc_brier(
            y_valid,
            current_pred,
        )


        fixed_brier = calc_brier(
            y_valid,
            fixed_pred,
        )


        delta = (
            current_brier
            - fixed_brier
        )


        print(
            f"\nCurrent 62FE : "
            f"{current_brier:.8f}"
        )

        print(
            f"Fixed RMSE   : "
            f"{fixed_brier:.8f}"
        )

        print(
            f"Delta        : "
            f"{delta:+.8f}"
        )


        print(
            "\nTrees | "
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

            "current_62FE_pred":
                current_pred,

            "fixed_rmse_pred":
                fixed_pred,

            "seed":
                seed,

            "fold":
                fold,
        })


        # 개별 R prediction 보존
        fold_result[
            "current_R_pred"
        ] = np.nan

        fold_result[
            "RMSE_R_pred"
        ] = np.nan


        fold_result.loc[
            valid_r_mask_np,
            "current_R_pred",
        ] = current_r_pred


        fold_result.loc[
            valid_r_mask_np,
            "RMSE_R_pred",
        ] = rmse_r_pred


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


    current_oof = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "current_62FE_pred"
        ],
    )


    fixed_oof = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "fixed_rmse_pred"
        ],
    )


    pooled_delta = (
        current_oof
        - fixed_oof
    )


    # =====================================================
    # R만
    # =====================================================
    seed_r = (
        seed_oof[
            seed_oof[GROUP_COL]
            == R_GROUP
        ]
        .copy()
    )


    current_r_brier = calc_brier(
        seed_r["y_true"],
        seed_r[
            "current_R_pred"
        ],
    )


    rmse_r_brier = calc_brier(
        seed_r["y_true"],
        seed_r[
            "RMSE_R_pred"
        ],
    )


    fixed_r_pred = (

        CURRENT_R_ENSEMBLE_WEIGHT
        * seed_r[
            "current_R_pred"
        ].to_numpy()

        +

        RMSE_R_WEIGHT
        * seed_r[
            "RMSE_R_pred"
        ].to_numpy()
    )


    fixed_r_brier = calc_brier(
        seed_r["y_true"],
        fixed_r_pred,
    )


    # =====================================================
    # Secondary:
    # seed별 최적 RMSE weight
    # =====================================================
    (
        optimal_rmse_weight,
        optimal_r_brier,
    ) = search_best_rmse_weight(

        y_true=
            seed_r["y_true"],

        classifier_pred=
            seed_r[
                "current_R_pred"
            ].to_numpy(),

        rmse_pred=
            seed_r[
                "RMSE_R_pred"
            ].to_numpy(),
    )


    weight_summary_rows.append({

        "seed":
            seed,

        "optimal_classifier_weight":
            1.0
            - optimal_rmse_weight,

        "optimal_rmse_weight":
            optimal_rmse_weight,

        "optimal_R_brier":
            optimal_r_brier,
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


        season_current = calc_brier(
            season_df[
                "y_true"
            ],
            season_df[
                "current_62FE_pred"
            ],
        )


        season_fixed = calc_brier(
            season_df[
                "y_true"
            ],
            season_df[
                "fixed_rmse_pred"
            ],
        )


        season_delta = (
            season_current
            - season_fixed
        )


        # ---------------------------------------------
        # R만
        # ---------------------------------------------
        season_r = (
            season_df[
                season_df[
                    GROUP_COL
                ]
                == R_GROUP
            ]
        )


        season_current_r = calc_brier(
            season_r["y_true"],
            season_r[
                "current_R_pred"
            ],
        )


        season_rmse_r = calc_brier(
            season_r["y_true"],
            season_r[
                "RMSE_R_pred"
            ],
        )


        season_summary_rows.append({

            "seed":
                seed,

            "season":
                int(season),

            "current_62FE_brier":
                season_current,

            "fixed_rmse_brier":
                season_fixed,

            "delta":
                season_delta,

            "current_R_brier":
                season_current_r,

            "RMSE_R_brier":
                season_rmse_r,

            "RMSE_R_delta":
                (
                    season_current_r
                    - season_rmse_r
                ),
        })


    # =====================================================
    # Seed Summary
    # =====================================================
    elapsed = (
        time.time()
        - seed_start
    )


    seed_summary_rows.append({

        "seed":
            seed,

        "current_62FE_oof":
            current_oof,

        "fixed_rmse_oof":
            fixed_oof,

        "delta":
            pooled_delta,

        "current_R_oof":
            current_r_brier,

        "RMSE_R_oof":
            rmse_r_brier,

        "fixed_R_oof":
            fixed_r_brier,

        "R_fixed_delta":
            (
                current_r_brier
                - fixed_r_brier
            ),

        "elapsed_sec":
            elapsed,
    })


    all_seed_oof.append(
        seed_oof
    )


    print("\n")
    print("-" * 80)

    print(
        f"[SEED {seed} FINAL]"
    )

    print("-" * 80)

    print(
        f"Current 62FE : "
        f"{current_oof:.8f}"
    )

    print(
        f"Fixed RMSE   : "
        f"{fixed_oof:.8f}"
    )

    print(
        f"Delta        : "
        f"{pooled_delta:+.8f}"
    )

    print(
        f"R Fixed Delta: "
        f"{current_r_brier - fixed_r_brier:+.8f}"
    )

    print(
        f"Optimal RMSE weight "
        f"(secondary): "
        f"{optimal_rmse_weight:.2f}"
    )

    print(
        f"Time: "
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

weight_summary = pd.DataFrame(
    weight_summary_rows
)

tree_summary = pd.DataFrame(
    tree_summary_rows
)

all_oof = pd.concat(
    all_seed_oof,
    ignore_index=True,
)


# =========================================================
# 5 Seed 전체
# =========================================================
mean_current = float(
    seed_summary[
        "current_62FE_oof"
    ].mean()
)

mean_fixed = float(
    seed_summary[
        "fixed_rmse_oof"
    ].mean()
)

mean_delta = float(
    seed_summary[
        "delta"
    ].mean()
)

std_delta = float(
    seed_summary[
        "delta"
    ].std(
        ddof=1
    )
)

win_count = int(
    (
        seed_summary[
            "delta"
        ]
        > 0
    ).sum()
)


# =========================================================
# Seed 42 제외
# =========================================================
unseen = (
    seed_summary[
        seed_summary["seed"]
        != TUNING_SEED
    ]
    .copy()
)


unseen_mean_delta = float(
    unseen[
        "delta"
    ].mean()
)

unseen_std_delta = float(
    unseen[
        "delta"
    ].std(
        ddof=1
    )
)

unseen_win_count = int(
    (
        unseen[
            "delta"
        ]
        > 0
    ).sum()
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

        current_mean=(
            "current_62FE_brier",
            "mean",
        ),

        fixed_rmse_mean=(
            "fixed_rmse_brier",
            "mean",
        ),

        mean_delta=(
            "delta",
            "mean",
        ),

        delta_std=(
            "delta",
            "std",
        ),

        win_count=(
            "delta",
            lambda x: int(
                (x > 0).sum()
            ),
        ),

        RMSE_R_mean_delta=(
            "RMSE_R_delta",
            "mean",
        ),

        RMSE_R_win_count=(
            "RMSE_R_delta",
            lambda x: int(
                (x > 0).sum()
            ),
        ),
    )
)


# =========================================================
# 2024 별도
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


delta_2024_mean = float(
    season_2024[
        "delta"
    ].mean()
)

delta_2024_std = float(
    season_2024[
        "delta"
    ].std(
        ddof=1
    )
)

win_2024 = int(
    (
        season_2024[
            "delta"
        ]
        > 0
    ).sum()
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("#" * 80)
print("RMSE MULTI-SEED FINAL RESULT")
print("#" * 80)


print("\n[Seed Summary]")

print(
    seed_summary[
        [
            "seed",
            "current_62FE_oof",
            "fixed_rmse_oof",
            "delta",
            "current_R_oof",
            "RMSE_R_oof",
            "fixed_R_oof",
            "R_fixed_delta",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[ALL 5 SEEDS]")

print(
    f"Mean Current 62FE : "
    f"{mean_current:.8f}"
)

print(
    f"Mean Fixed RMSE   : "
    f"{mean_fixed:.8f}"
)

print(
    f"Mean Delta        : "
    f"{mean_delta:+.8f}"
)

print(
    f"Delta Std         : "
    f"{std_delta:.8f}"
)

print(
    f"Win Count         : "
    f"{win_count}/5"
)


print(
    "\n[EXCLUDING TUNING SEED 42]"
)

print(
    f"Mean Delta : "
    f"{unseen_mean_delta:+.8f}"
)

print(
    f"Delta Std  : "
    f"{unseen_std_delta:.8f}"
)

print(
    f"Win Count  : "
    f"{unseen_win_count}/4"
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


print("\n[2024 Stability]")

print(
    f"2024 Mean Delta : "
    f"{delta_2024_mean:+.8f}"
)

print(
    f"2024 Delta Std  : "
    f"{delta_2024_std:.8f}"
)

print(
    f"2024 Win Count  : "
    f"{win_2024}/5"
)


print(
    "\n[Optimal RMSE Weights - Secondary]"
)

print(
    weight_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =========================================================
# 판정
# =========================================================
print("\n")
print("=" * 80)
print("RMSE STABILITY CHECK")
print("=" * 80)


if mean_delta > 0:
    print(
        "Mean pooled delta       : PASS"
    )
else:
    print(
        "Mean pooled delta       : FAIL"
    )


if win_count >= 3:
    print(
        "Pooled win >= 3/5       : PASS"
    )
else:
    print(
        "Pooled win >= 3/5       : FAIL"
    )


if unseen_mean_delta > 0:
    print(
        "Exclude seed42 mean     : PASS"
    )
else:
    print(
        "Exclude seed42 mean     : FAIL"
    )


if win_2024 >= 3:
    print(
        "2024 win >= 3/5         : PASS"
    )
else:
    print(
        "2024 win >= 3/5         : FAIL"
    )


if delta_2024_mean > 0:
    print(
        "2024 mean delta         : PASS"
    )
else:
    print(
        "2024 mean delta         : FAIL"
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


weight_summary.to_csv(
    VALIDATION_DIR
    / "optimal_rmse_weight_summary.csv",
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

    "tuning_seed":
        TUNING_SEED,

    "feature_count":
        len(FE_FEATURES),

    "classifier_weights": {

        "R_global":
            R_GLOBAL_WEIGHT,

        "R_only":
            R_ONLY_WEIGHT,

        "F_global":
            F_GLOBAL_WEIGHT,

        "F_only":
            F_ONLY_WEIGHT,
    },

    "fixed_rmse_mix": {

        "current_R_classifier":
            CURRENT_R_ENSEMBLE_WEIGHT,

        "RMSE_R":
            RMSE_R_WEIGHT,
    },

    "mean_current_oof":
        mean_current,

    "mean_fixed_rmse_oof":
        mean_fixed,

    "mean_delta":
        mean_delta,

    "delta_std":
        std_delta,

    "win_count":
        win_count,

    "unseen_mean_delta":
        unseen_mean_delta,

    "unseen_win_count":
        unseen_win_count,

    "2024_mean_delta":
        delta_2024_mean,

    "2024_win_count":
        win_2024,

    "note": (
        "Primary evaluation uses fixed "
        "RMSE weight 0.27 selected on "
        "seed 42. Per-seed optimal RMSE "
        "weights are secondary diagnostics."
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
    / "optimal_rmse_weight_summary.csv"
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