from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_recency_weight_2024_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"

SEED = 42


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
# Validation
#
# 지금 목적은:
# 2019~2023을 이용해서 2024를 얼마나 잘 예측하는가?
#
# 따라서 Fold4 하나만 빠르게 테스트
# =========================================================
TRAIN_START = 2019
TRAIN_END = 2023
VALID_SEASON = 2024


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
# 추가 15 Features
# =========================================================
ADD_FEATURES = [
    "pitcher_no_history",
    "batter_no_history",
    "pitcher_no_prev_game",

    "pitcher_log_n",
    "batter_log_n",

    "pitcher_recent1_success_delta",
    "pitcher_recent3_success_delta",
    "pitcher_recent5_success_delta",
    "pitcher_recent3_middle_delta",

    "count_state",
    "hand_matchup",

    "abs_score_diff",
    "pitcher_team_leading",
    "pitcher_team_trailing",
    "tie_game",
]


# =========================================================
# CatBoost
# =========================================================
CATBOOST_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
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
# Weight Schemes
# =========================================================
WEIGHT_SCHEMES = {

    # ---------------------------------------------
    # A. 기존 방식
    # ---------------------------------------------
    "uniform": {
        2019: 1.00,
        2020: 1.00,
        2021: 1.00,
        2022: 1.00,
        2023: 1.00,
    },

    # ---------------------------------------------
    # B. 과거 데이터 약간 down-weight
    # ---------------------------------------------
    "old_075": {
        2019: 0.75,
        2020: 0.75,
        2021: 0.75,
        2022: 0.75,
        2023: 1.00,
    },

    # ---------------------------------------------
    # C. 과거 데이터 절반
    # ---------------------------------------------
    "old_050": {
        2019: 0.50,
        2020: 0.50,
        2021: 0.50,
        2022: 0.50,
        2023: 1.00,
    },

    # ---------------------------------------------
    # D. 최근 시즌일수록 점진적으로 증가
    # ---------------------------------------------
    "linear": {
        2019: 0.50,
        2020: 0.60,
        2021: 0.70,
        2022: 0.85,
        2023: 1.00,
    },

    # ---------------------------------------------
    # E. 강한 Recency
    # ---------------------------------------------
    "strong_recency": {
        2019: 0.25,
        2020: 0.40,
        2021: 0.60,
        2022: 0.80,
        2023: 1.00,
    },
}


# =========================================================
# Feature Engineering
# =========================================================
def make_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # History
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

    # Sample size
    df["pitcher_log_n"] = np.log1p(
        df["asof_pitcher_n"]
    )

    df["batter_log_n"] = np.log1p(
        df["asof_batter_n"]
    )

    # Recent vs Career
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

    # Count state
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

    # Hand matchup
    df["hand_matchup"] = (
        df["pitcher_hand"]
        .fillna("__MISSING__")
        .astype(str)
        + "_"
        + df["batter_hand"]
        .fillna("__MISSING__")
        .astype(str)
    )

    # Score context
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
# Brier
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
# Sample Weight 생성
# =========================================================
def make_sample_weight(
    season_series,
    weight_map,
):
    """
    season별 상대 weight 생성 후
    전체 평균이 1이 되도록 normalize.

    따라서 각 실험에서 중요한 것은
    절대 weight가 아니라 연도 간 상대 비율임.
    """

    weights = (
        season_series
        .map(weight_map)
        .astype(float)
        .to_numpy()
    )

    if np.isnan(weights).any():

        raise ValueError(
            "Weight가 정의되지 않은 season이 있습니다."
        )

    weights = (
        weights
        / weights.mean()
    )

    return weights


# =========================================================
# 데이터
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
# 기존 47 Features
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
    f"Raw Features   : {len(BASE_FEATURES)}"
)


# =========================================================
# 62FE
# =========================================================
train = make_features(
    train_raw
)


FEATURES = (
    BASE_FEATURES.copy()
)


for col in ADD_FEATURES:

    if col not in FEATURES:

        FEATURES.append(
            col
        )


print(
    f"Added Features : {len(ADD_FEATURES)}"
)

print(
    f"Final Features : {len(FEATURES)}"
)


if len(FEATURES) != 62:

    raise ValueError(
        f"Expected 62 features, got {len(FEATURES)}"
    )


train = prepare_categorical(
    train
)


# =========================================================
# Fold4
# =========================================================
train_mask = (
    (train[SEASON_COL] >= TRAIN_START)
    &
    (train[SEASON_COL] <= TRAIN_END)
)

valid_mask = (
    train[SEASON_COL]
    == VALID_SEASON
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


print("\n[Dataset]")

print(
    f"Train {TRAIN_START}~{TRAIN_END}: "
    f"{len(fold_train):,}"
)

print(
    f"Valid {VALID_SEASON}: "
    f"{len(fold_valid):,}"
)


X_train = (
    fold_train[
        FEATURES
    ]
)

y_train = (
    fold_train[
        TARGET_COL
    ]
)


X_valid = (
    fold_valid[
        FEATURES
    ]
)

y_valid = (
    fold_valid[
        TARGET_COL
    ]
)


# =========================================================
# 실험
# =========================================================
result_rows = []
prediction_df = pd.DataFrame({
    ID_COL:
        fold_valid[
            ID_COL
        ].values,

    "y_true":
        y_valid.values,
})


for scheme_name, weight_map in WEIGHT_SCHEMES.items():

    print("\n")
    print("=" * 80)

    print(
        f"Weight Scheme: {scheme_name}"
    )

    print("=" * 80)


    # -----------------------------------------------------
    # Weight
    # -----------------------------------------------------
    sample_weight = (
        make_sample_weight(
            fold_train[
                SEASON_COL
            ],
            weight_map,
        )
    )


    print("\nRelative weights:")

    for year in sorted(
        weight_map.keys()
    ):

        print(
            f"  {year}: "
            f"{weight_map[year]:.2f}"
        )


    print(
        f"Normalized weight mean: "
        f"{sample_weight.mean():.6f}"
    )


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = CatBoostClassifier(
        **CATBOOST_PARAMS
    )


    model.fit(
        X_train,
        y_train,

        sample_weight=
            sample_weight,

        cat_features=
            CAT_COLS,

        eval_set=(
            X_valid,
            y_valid,
        ),

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,

        use_best_model=True,
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    pred = (
        model
        .predict_proba(
            X_valid
        )[:, 1]
    )


    brier = calc_brier(
        y_valid,
        pred,
    )


    best_iteration = (
        model.get_best_iteration()
    )


    if (
        best_iteration is None
        or best_iteration < 0
    ):

        best_trees = (
            CATBOOST_PARAMS[
                "iterations"
            ]
        )

    else:

        best_trees = (
            int(best_iteration)
            + 1
        )


    print(
        f"\n2024 Brier : "
        f"{brier:.8f}"
    )

    print(
        f"Best Trees : "
        f"{best_trees}"
    )


    result_rows.append({

        "scheme":
            scheme_name,

        "brier_2024":
            brier,

        "best_trees":
            best_trees,
    })


    prediction_df[
        scheme_name
    ] = pred


# =========================================================
# 결과
# =========================================================
results = pd.DataFrame(
    result_rows
)


baseline_brier = float(
    results.loc[
        results["scheme"]
        == "uniform",
        "brier_2024",
    ].iloc[0]
)


results[
    "delta_vs_uniform"
] = (
    baseline_brier
    - results[
        "brier_2024"
    ]
)


results = (
    results
    .sort_values(
        "brier_2024"
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 출력
# =========================================================
print("\n")
print("#" * 80)
print("RECENCY WEIGHT RESULT")
print("#" * 80)


print(
    results.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


best_row = (
    results.iloc[0]
)


print("\n[Best]")

print(
    f"Scheme : "
    f"{best_row['scheme']}"
)

print(
    f"Brier  : "
    f"{best_row['brier_2024']:.8f}"
)

print(
    f"Delta  : "
    f"{best_row['delta_vs_uniform']:+.8f}"
)


# =========================================================
# 저장
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    VALIDATION_DIR
    / "weight_results.csv",
    index=False,
    encoding="utf-8-sig",
)


prediction_df.to_csv(
    VALIDATION_DIR
    / "oof_predictions_2024.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\nSaved:")

print(
    VALIDATION_DIR
    / "weight_results.csv"
)

print(
    VALIDATION_DIR
    / "oof_predictions_2024.csv"
)


print("\nExperiment complete.")