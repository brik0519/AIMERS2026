from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from catboost import (
    CatBoostRegressor,
)


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_fe_rmse_v001"

SOURCE_EXPERIMENT_NAME = "cb_v002_game_type_fe_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"

SEED = 42


# =========================================================
# 현재 62FE R/F Ensemble Weight
# =========================================================
CURRENT_R_GLOBAL_WEIGHT = 0.49
CURRENT_R_ONLY_WEIGHT = 0.51

CURRENT_F_GLOBAL_WEIGHT = 0.64
CURRENT_F_ONLY_WEIGHT = 0.36


# =========================================================
# PDF 최종 구조에서 사용한 Weight
#
# R:
# Global Logloss 30%
# R-only Logloss 20%
# Global RMSE 50%
#
# F:
# Global Logloss 60%
# F-only Logloss 40%
# =========================================================
PDF_R_GLOBAL_WEIGHT = 0.30
PDF_R_ONLY_WEIGHT = 0.20
PDF_R_RMSE_WEIGHT = 0.50

PDF_F_GLOBAL_WEIGHT = 0.60
PDF_F_ONLY_WEIGHT = 0.40


# =========================================================
# RMSE 추가 비중 탐색
#
# 현재 R Ensemble:
# 0.49 Global + 0.51 R-only
#
# 여기에 RMSE를 얼마나 섞는지 탐색.
#
# 최종:
# (1-w) * 현재 R Ensemble
# + w * RMSE
#
# 이 탐색은 Secondary 분석용.
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
# 경로
# =========================================================
DATA_DIR = Path("./data")

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

SOURCE_VALIDATION_DIR = (
    VALIDATION_ROOT
    / SOURCE_EXPERIMENT_NAME
)

SOURCE_OOF_PATH = (
    SOURCE_VALIDATION_DIR
    / "oof_predictions.csv"
)

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# CatBoost 설정
#
# 이전 실험과 동일한 구조.
# loss만 RMSE로 변경.
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


RMSE_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,

    "loss_function": "RMSE",
    "eval_metric": "RMSE",

    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,

    "random_seed": SEED,

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
# 이전과 동일한 15개 Feature
# =========================================================
ADD_FEATURES = [

    # History
    "pitcher_no_history",
    "batter_no_history",
    "pitcher_no_prev_game",

    # 표본 수
    "pitcher_log_n",
    "batter_log_n",

    # 최근 폼
    "pitcher_recent1_success_delta",
    "pitcher_recent3_success_delta",
    "pitcher_recent5_success_delta",
    "pitcher_recent3_middle_delta",

    # 상황
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
    이전 62FE 실험과 동일.

    모든 파생변수는 현재 row와
    공식 asof_* 정보만 이용한다.

    다른 test row / validation row의
    통계나 집계는 사용하지 않는다.
    """

    df = df.copy()


    # =====================================================
    # History
    # =====================================================
    df["pitcher_no_history"] = (
        df["asof_pitcher_n"] == 0
    ).astype(int)

    df["batter_no_history"] = (
        df["asof_batter_n"] == 0
    ).astype(int)

    df["pitcher_no_prev_game"] = (
        df[
            "asof_pitcher_prev1_game_success_rate"
        ]
        .isna()
    ).astype(int)


    # =====================================================
    # 표본량
    # =====================================================
    df["pitcher_log_n"] = np.log1p(
        df["asof_pitcher_n"]
    )

    df["batter_log_n"] = np.log1p(
        df["asof_batter_n"]
    )


    # =====================================================
    # Recent vs Career
    # =====================================================
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


    # =====================================================
    # Count interaction
    # =====================================================
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


    # =====================================================
    # Hand matchup
    # =====================================================
    df["hand_matchup"] = (
        df["pitcher_hand"]
        .fillna("__MISSING__")
        .astype(str)
        + "_"
        + df["batter_hand"]
        .fillna("__MISSING__")
        .astype(str)
    )


    # =====================================================
    # Score context
    # =====================================================
    df["abs_score_diff"] = (
        df["score_diff_pitcher_team"]
        .abs()
    )

    df["pitcher_team_leading"] = (
        df["score_diff_pitcher_team"]
        > 0
    ).astype(int)

    df["pitcher_team_trailing"] = (
        df["score_diff_pitcher_team"]
        < 0
    ).astype(int)

    df["tie_game"] = (
        df["score_diff_pitcher_team"]
        == 0
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


def create_rmse_model():

    return CatBoostRegressor(
        **RMSE_PARAMS
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
        return RMSE_PARAMS[
            "iterations"
        ]

    return int(
        best_iteration
    ) + 1


# =========================================================
# 데이터 로드
#
# validation 실험이므로 test.csv는 읽지 않는다.
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


train = prepare_categorical(
    train
)


# =========================================================
# 기존 62FE classifier OOF 로드
# =========================================================
print("\nLoad previous 62FE OOF...")

if not SOURCE_OOF_PATH.exists():

    raise FileNotFoundError(
        "기존 62FE OOF가 없습니다:\n"
        f"{SOURCE_OOF_PATH}"
    )


source_oof = pd.read_csv(
    SOURCE_OOF_PATH,
    encoding="utf-8-sig",
)


REQUIRED_SOURCE_COLS = [

    ID_COL,
    SEASON_COL,
    GROUP_COL,

    "y_true",

    "fe_weighted_pred",

    "global_R_pred",
    "R_only_pred",

    "global_F_pred",
    "F_only_pred",
]


missing_source = [
    col
    for col in REQUIRED_SOURCE_COLS
    if col not in source_oof.columns
]


if missing_source:

    raise ValueError(
        "기존 OOF에 필요한 컬럼 없음:\n"
        f"{missing_source}"
    )


# =========================================================
# RMSE OOF
#
# 중요:
#
# Train = R + F 전체
# Validation = R만
#
# 즉 PDF의 Global RMSE R 구조를 따라간다.
# =========================================================
rmse_oof_parts = []

fold_rows = []


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
        f"Fold {fold} | "
        f"Train {train_start}~{train_end} "
        f"-> Valid {valid_season}"
    )

    print("=" * 80)


    # =====================================================
    # Expanding split
    # =====================================================
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

    valid_r_mask = (
        (train[SEASON_COL] == valid_season)
        &
        (train[GROUP_COL] == R_GROUP)
    )


    fold_train = (
        train.loc[
            train_mask
        ]
        .copy()
    )

    fold_valid_r = (
        train.loc[
            valid_r_mask
        ]
        .copy()
    )


    print(
        f"Train rows   : "
        f"{len(fold_train):,}"
    )

    print(
        f"Valid R rows : "
        f"{len(fold_valid_r):,}"
    )


    X_train = (
        fold_train[
            FE_FEATURES
        ]
    )

    y_train = (
        fold_train[
            TARGET_COL
        ]
    )


    X_valid_r = (
        fold_valid_r[
            FE_FEATURES
        ]
    )

    y_valid_r = (
        fold_valid_r[
            TARGET_COL
        ]
    )


    # =====================================================
    # RMSE 학습
    # =====================================================
    print("\nTrain Global-R RMSE...")

    start = time.time()


    rmse_model = (
        create_rmse_model()
    )


    rmse_model.fit(

        X_train,
        y_train,

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


    elapsed = (
        time.time()
        - start
    )


    # =====================================================
    # Prediction
    # =====================================================
    rmse_pred_raw = (
        rmse_model.predict(
            X_valid_r
        )
    )


    # -----------------------------------------------------
    # 제출 확률은 [0,1]이어야 하므로 clip.
    #
    # Binary target에 대한 squared error에서는
    # [0,1] 바깥 값을 경계로 자르는 것이
    # Brier를 악화시키지 않는다.
    # -----------------------------------------------------
    rmse_pred = np.clip(
        rmse_pred_raw,
        0.0,
        1.0,
    )


    rmse_brier = calc_brier(
        y_valid_r,
        rmse_pred,
    )


    rmse_best_trees = (
        get_best_tree_count(
            rmse_model
        )
    )


    out_of_range = int(
        (
            (rmse_pred_raw < 0.0)
            |
            (rmse_pred_raw > 1.0)
        ).sum()
    )


    print(
        f"\nRMSE R Brier : "
        f"{rmse_brier:.8f}"
    )

    print(
        f"Best Trees   : "
        f"{rmse_best_trees}"
    )

    print(
        f"Pred Mean    : "
        f"{rmse_pred.mean():.6f}"
    )

    print(
        f"True Rate    : "
        f"{y_valid_r.mean():.6f}"
    )

    print(
        f"Pred Range   : "
        f"{rmse_pred_raw.min():.6f} "
        f"~ "
        f"{rmse_pred_raw.max():.6f}"
    )

    print(
        f"Out of Range : "
        f"{out_of_range}"
    )


    rmse_part = pd.DataFrame({

        ID_COL:
            fold_valid_r[
                ID_COL
            ].values,

        SEASON_COL:
            fold_valid_r[
                SEASON_COL
            ].values,

        "y_true_rmse":
            y_valid_r.values,

        "rmse_R_pred_raw":
            rmse_pred_raw,

        "rmse_R_pred":
            rmse_pred,
    })


    rmse_oof_parts.append(
        rmse_part
    )


    fold_rows.append({

        "fold":
            fold,

        "valid_season":
            valid_season,

        "rmse_R_brier":
            rmse_brier,

        "rmse_R_best_trees":
            rmse_best_trees,

        "rmse_pred_mean":
            float(
                rmse_pred.mean()
            ),

        "R_true_rate":
            float(
                y_valid_r.mean()
            ),

        "out_of_range":
            out_of_range,

        "elapsed_sec":
            elapsed,
    })


# =========================================================
# RMSE OOF 결합
# =========================================================
rmse_oof = pd.concat(
    rmse_oof_parts,
    ignore_index=True,
)


fold_summary = pd.DataFrame(
    fold_rows
)


# =========================================================
# 기존 OOF + RMSE merge
# =========================================================
compare = (
    source_oof
    .merge(
        rmse_oof[
            [
                ID_COL,
                "rmse_R_pred_raw",
                "rmse_R_pred",
            ]
        ],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
)


# =========================================================
# R/F mask
# =========================================================
r_mask = (
    compare[GROUP_COL]
    == R_GROUP
)

f_mask = (
    compare[GROUP_COL]
    == F_GROUP
)


if (
    compare.loc[
        r_mask,
        "rmse_R_pred",
    ]
    .isna()
    .any()
):

    raise ValueError(
        "R row에 RMSE prediction이 "
        "누락되었습니다."
    )


# =========================================================
# 기존 62FE R/F Ensemble
# =========================================================
current_pred = (
    compare[
        "fe_weighted_pred"
    ]
    .to_numpy(
        dtype=float
    )
)


current_oof_brier = calc_brier(
    compare["y_true"],
    current_pred,
)


# =========================================================
# 현재 R ensemble 성능
# =========================================================
current_r_brier = calc_brier(
    compare.loc[
        r_mask,
        "y_true",
    ],
    compare.loc[
        r_mask,
        "fe_weighted_pred",
    ],
)


# =========================================================
# RMSE R 단독 성능
# =========================================================
rmse_r_brier = calc_brier(
    compare.loc[
        r_mask,
        "y_true",
    ],
    compare.loc[
        r_mask,
        "rmse_R_pred",
    ],
)


# =========================================================
# PDF-style R
#
# R =
# 30% Global Logloss
# +20% R-only Logloss
# +50% RMSE
#
# F는 현재 64/36 그대로 유지
#
# → RMSE가 R에 추가되는 효과를
#   가장 직접적으로 확인.
# =========================================================
pdf_r_pred = (

    PDF_R_GLOBAL_WEIGHT
    * compare.loc[
        r_mask,
        "global_R_pred",
    ].to_numpy()

    +

    PDF_R_ONLY_WEIGHT
    * compare.loc[
        r_mask,
        "R_only_pred",
    ].to_numpy()

    +

    PDF_R_RMSE_WEIGHT
    * compare.loc[
        r_mask,
        "rmse_R_pred",
    ].to_numpy()
)


pdf_r_current_f_pred = (
    current_pred.copy()
)


pdf_r_current_f_pred[
    r_mask.to_numpy()
] = pdf_r_pred


pdf_r_current_f_brier = (
    calc_brier(
        compare["y_true"],
        pdf_r_current_f_pred,
    )
)


# =========================================================
# PDF 전체 Weight 구조
#
# R = 30 / 20 / 50
# F = 60 / 40
# =========================================================
pdf_full_pred = np.empty(
    len(compare),
    dtype=float,
)


pdf_full_pred[
    r_mask.to_numpy()
] = pdf_r_pred


pdf_f_pred = (

    PDF_F_GLOBAL_WEIGHT
    * compare.loc[
        f_mask,
        "global_F_pred",
    ].to_numpy()

    +

    PDF_F_ONLY_WEIGHT
    * compare.loc[
        f_mask,
        "F_only_pred",
    ].to_numpy()
)


pdf_full_pred[
    f_mask.to_numpy()
] = pdf_f_pred


pdf_full_brier = calc_brier(
    compare["y_true"],
    pdf_full_pred,
)


# =========================================================
# Secondary:
# 현재 R ensemble + RMSE weight search
#
# R =
# (1-w) * Current R
# + w * RMSE
#
# F는 현재 64/36 고정
# =========================================================
weight_rows = []


r_y = (
    compare.loc[
        r_mask,
        "y_true",
    ]
    .to_numpy(
        dtype=float
    )
)


current_r_pred = (
    compare.loc[
        r_mask,
        "fe_weighted_pred",
    ]
    .to_numpy(
        dtype=float
    )
)


rmse_r_pred = (
    compare.loc[
        r_mask,
        "rmse_R_pred",
    ]
    .to_numpy(
        dtype=float
    )
)


for rmse_weight in RMSE_WEIGHTS:

    classifier_weight = (
        1.0
        - rmse_weight
    )


    mixed_r_pred = (

        classifier_weight
        * current_r_pred

        +

        rmse_weight
        * rmse_r_pred
    )


    brier = calc_brier(
        r_y,
        mixed_r_pred,
    )


    weight_rows.append({

        "rmse_weight":
            float(
                rmse_weight
            ),

        "current_classifier_weight":
            float(
                classifier_weight
            ),

        "R_brier":
            brier,
    })


weight_search = pd.DataFrame(
    weight_rows
)


best_idx = (
    weight_search[
        "R_brier"
    ]
    .idxmin()
)


best_row = (
    weight_search.loc[
        best_idx
    ]
)


BEST_RMSE_WEIGHT = float(
    best_row[
        "rmse_weight"
    ]
)


BEST_CURRENT_WEIGHT = (
    1.0
    - BEST_RMSE_WEIGHT
)


BEST_R_BRIER = float(
    best_row[
        "R_brier"
    ]
)


# =========================================================
# 최적 RMSE mix 전체 OOF
# =========================================================
best_mix_pred = (
    current_pred.copy()
)


best_r_pred = (

    BEST_CURRENT_WEIGHT
    * current_r_pred

    +

    BEST_RMSE_WEIGHT
    * rmse_r_pred
)


best_mix_pred[
    r_mask.to_numpy()
] = best_r_pred


best_mix_oof_brier = calc_brier(
    compare["y_true"],
    best_mix_pred,
)


# =========================================================
# 결과 컬럼 저장
# =========================================================
compare[
    "current_62FE_pred"
] = current_pred

compare[
    "pdf_R_current_F_pred"
] = pdf_r_current_f_pred

compare[
    "pdf_full_pred"
] = pdf_full_pred

compare[
    "best_rmse_mix_pred"
] = best_mix_pred


# =========================================================
# 시즌별 비교
# =========================================================
season_rows = []


for season in sorted(
    compare[
        SEASON_COL
    ].unique()
):

    season_df = (
        compare[
            compare[
                SEASON_COL
            ]
            == season
        ]
        .copy()
    )


    current_brier = calc_brier(
        season_df["y_true"],
        season_df[
            "current_62FE_pred"
        ],
    )


    pdf_r_brier = calc_brier(
        season_df["y_true"],
        season_df[
            "pdf_R_current_F_pred"
        ],
    )


    pdf_full_season_brier = (
        calc_brier(
            season_df["y_true"],
            season_df[
                "pdf_full_pred"
            ],
        )
    )


    best_mix_brier = calc_brier(
        season_df["y_true"],
        season_df[
            "best_rmse_mix_pred"
        ],
    )


    # -----------------------------------------------------
    # R만 별도
    # -----------------------------------------------------
    season_r = (
        season_df[
            season_df[
                GROUP_COL
            ]
            == R_GROUP
        ]
    )


    current_R = calc_brier(
        season_r["y_true"],
        season_r[
            "fe_weighted_pred"
        ],
    )


    rmse_R = calc_brier(
        season_r["y_true"],
        season_r[
            "rmse_R_pred"
        ],
    )


    season_rows.append({

        "season":
            int(season),

        "current_62FE":
            current_brier,

        "pdf_R_current_F":
            pdf_r_brier,

        "pdf_full":
            pdf_full_season_brier,

        "best_rmse_mix":
            best_mix_brier,

        "pdf_R_delta_vs_current":
            (
                current_brier
                - pdf_r_brier
            ),

        "best_mix_delta_vs_current":
            (
                current_brier
                - best_mix_brier
            ),

        "current_R":
            current_R,

        "RMSE_R":
            rmse_R,

        "RMSE_R_delta":
            (
                current_R
                - rmse_R
            ),
    })


season_summary = pd.DataFrame(
    season_rows
)


# =========================================================
# 최적 weight 주변 안정성
# =========================================================
lower = max(
    0.0,
    BEST_RMSE_WEIGHT - 0.10,
)

upper = min(
    1.0,
    BEST_RMSE_WEIGHT + 0.10,
)


weight_neighborhood = (
    weight_search[
        (
            weight_search[
                "rmse_weight"
            ]
            >= lower
        )
        &
        (
            weight_search[
                "rmse_weight"
            ]
            <= upper
        )
    ]
    .copy()
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("#" * 80)
print("RMSE EXPERIMENT RESULT")
print("#" * 80)


print("\n[RMSE Fold Summary]")

print(
    fold_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[R POOLED]")

print(
    f"Current 62FE R Ensemble : "
    f"{current_r_brier:.8f}"
)

print(
    f"Global RMSE R           : "
    f"{rmse_r_brier:.8f}"
)

print(
    f"RMSE standalone Delta   : "
    f"{current_r_brier - rmse_r_brier:+.8f}"
)


print("\n[POOLED OOF]")

print(
    f"Current 62FE R/F        : "
    f"{current_oof_brier:.8f}"
)

print(
    f"PDF R + Current F       : "
    f"{pdf_r_current_f_brier:.8f}"
)

print(
    f"PDF Full Weights        : "
    f"{pdf_full_brier:.8f}"
)

print(
    f"Best RMSE Mix           : "
    f"{best_mix_oof_brier:.8f}"
)


print(
    f"\nPDF-R Improvement       : "
    f"{current_oof_brier - pdf_r_current_f_brier:+.8f}"
)

print(
    f"PDF-Full Improvement    : "
    f"{current_oof_brier - pdf_full_brier:+.8f}"
)

print(
    f"Best-Mix Improvement    : "
    f"{current_oof_brier - best_mix_oof_brier:+.8f}"
)


print("\n[RMSE Weight Search - Secondary]")

print(
    f"Best Current R Weight : "
    f"{BEST_CURRENT_WEIGHT:.2f}"
)

print(
    f"Best RMSE Weight      : "
    f"{BEST_RMSE_WEIGHT:.2f}"
)

print(
    f"Best R Brier          : "
    f"{BEST_R_BRIER:.8f}"
)


print("\n[Weight Neighborhood]")

print(
    weight_neighborhood.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[Season Comparison]")

print(
    season_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =========================================================
# 자동 판정
# =========================================================
pdf_season_wins = int(
    (
        season_summary[
            "pdf_R_delta_vs_current"
        ]
        > 0
    ).sum()
)


best_mix_season_wins = int(
    (
        season_summary[
            "best_mix_delta_vs_current"
        ]
        > 0
    ).sum()
)


rmse_r_season_wins = int(
    (
        season_summary[
            "RMSE_R_delta"
        ]
        > 0
    ).sum()
)


print("\n")
print("=" * 80)
print("RMSE CHECK")
print("=" * 80)


print(
    f"RMSE standalone R wins : "
    f"{rmse_r_season_wins}/4"
)

print(
    f"PDF-R overall wins     : "
    f"{pdf_season_wins}/4"
)

print(
    f"Best mix overall wins  : "
    f"{best_mix_season_wins}/4"
)


season_2024 = (
    season_summary[
        season_summary[
            "season"
        ]
        == 2024
    ]
)


if len(season_2024) == 1:

    row_2024 = (
        season_2024.iloc[0]
    )

    print(
        f"\n2024 PDF-R Delta : "
        f"{row_2024['pdf_R_delta_vs_current']:+.8f}"
    )

    print(
        f"2024 Best Delta  : "
        f"{row_2024['best_mix_delta_vs_current']:+.8f}"
    )

    print(
        f"2024 RMSE R Delta: "
        f"{row_2024['RMSE_R_delta']:+.8f}"
    )


# =========================================================
# 저장
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


compare.to_csv(
    VALIDATION_DIR
    / "oof_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


fold_summary.to_csv(
    VALIDATION_DIR
    / "rmse_fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


season_summary.to_csv(
    VALIDATION_DIR
    / "season_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


weight_search.to_csv(
    VALIDATION_DIR
    / "rmse_weight_search.csv",
    index=False,
    encoding="utf-8-sig",
)


metadata = {

    "experiment_name":
        EXPERIMENT_NAME,

    "source_experiment":
        SOURCE_EXPERIMENT_NAME,

    "seed":
        SEED,

    "feature_count":
        len(FE_FEATURES),

    "current_oof_brier":
        current_oof_brier,

    "rmse_R_oof_brier":
        rmse_r_brier,

    "pdf_R_current_F_oof_brier":
        pdf_r_current_f_brier,

    "pdf_full_oof_brier":
        pdf_full_brier,

    "best_rmse_mix_oof_brier":
        best_mix_oof_brier,

    "best_rmse_weight":
        BEST_RMSE_WEIGHT,

    "best_current_R_weight":
        BEST_CURRENT_WEIGHT,

    "pdf_weights": {

        "R_global":
            PDF_R_GLOBAL_WEIGHT,

        "R_only":
            PDF_R_ONLY_WEIGHT,

        "R_RMSE":
            PDF_R_RMSE_WEIGHT,

        "F_global":
            PDF_F_GLOBAL_WEIGHT,

        "F_only":
            PDF_F_ONLY_WEIGHT,
    },

    "note": (
        "RMSE weight search is secondary. "
        "Primary comparison includes fixed "
        "PDF-style R 30/20/50 weights."
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
    / "oof_predictions.csv"
)

print(
    VALIDATION_DIR
    / "rmse_fold_summary.csv"
)

print(
    VALIDATION_DIR
    / "season_summary.csv"
)

print(
    VALIDATION_DIR
    / "rmse_weight_search.csv"
)

print(
    VALIDATION_DIR
    / "metadata.json"
)


print("\nExperiment complete.")