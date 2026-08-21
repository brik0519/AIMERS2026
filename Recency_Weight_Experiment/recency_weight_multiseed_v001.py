from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 기본 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_recency_weight_multiseed_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"

# 기존 seed=42는 이미 실험했으므로 제외
SEEDS = [7, 123, 999]


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
# 이번 실험
#
# Train : 2019 ~ 2023
# Valid : 2024
#
# A. uniform
#    2019~2023 모두 동일 weight
#
# B. old_050
#    2019~2022 = 0.5
#    2023      = 1.0
# =========================================================
TRAIN_START = 2019
TRAIN_END = 2023
VALID_SEASON = 2024


WEIGHT_SCHEMES = {

    "uniform": {
        2019: 1.0,
        2020: 1.0,
        2021: 1.0,
        2022: 1.0,
        2023: 1.0,
    },

    "old_050": {
        2019: 0.5,
        2020: 0.5,
        2021: 0.5,
        2022: 0.5,
        2023: 1.0,
    },
}


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
# CatBoost 공통 설정
# =========================================================
BASE_CATBOOST_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.07,
    "depth": 8,
    "loss_function": "Logloss",
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "max_ctr_complexity": 1,
    "thread_count": -1,
    "verbose": 0,
    "allow_writing_files": False,
}

EARLY_STOPPING_ROUNDS = 80


# =========================================================
# Feature Engineering
# =========================================================
def make_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # -----------------------------------------------------
    # History 존재 여부
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
    # Sample Size
    # -----------------------------------------------------
    df["pitcher_log_n"] = np.log1p(
        df["asof_pitcher_n"]
    )

    df["batter_log_n"] = np.log1p(
        df["asof_batter_n"]
    )


    # -----------------------------------------------------
    # Recent - Career
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
# Categorical 처리
# =========================================================
def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:

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
# Brier Score
# =========================================================
def calc_brier(y_true, y_pred):

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
# Sample Weight
# =========================================================
def make_sample_weight(
    season_series,
    weight_map,
):

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

    # 평균 weight = 1로 정규화
    weights = (
        weights
        / weights.mean()
    )

    return weights


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
# 기존 Feature
# =========================================================
BASE_FEATURES = [
    col
    for col in train_raw.columns
    if col not in [
        ID_COL,
        TARGET_COL,
    ]
]


# =========================================================
# 62 Feature 생성
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
    f"Raw Features   : {len(BASE_FEATURES)}"
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
    f"Train : {TRAIN_START}~{TRAIN_END}"
    f" ({len(fold_train):,})"
)

print(
    f"Valid : {VALID_SEASON}"
    f" ({len(fold_valid):,})"
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
# Multi-Seed 실험
# =========================================================
result_rows = []


for seed in SEEDS:

    print("\n")
    print("#" * 80)
    print(f"SEED = {seed}")
    print("#" * 80)


    seed_results = {}


    for scheme_name, weight_map in WEIGHT_SCHEMES.items():

        print("\n" + "-" * 70)

        print(
            f"Seed={seed} | "
            f"Scheme={scheme_name}"
        )

        print("-" * 70)


        # -------------------------------------------------
        # Weight
        # -------------------------------------------------
        sample_weight = (
            make_sample_weight(
                fold_train[
                    SEASON_COL
                ],
                weight_map,
            )
        )


        # -------------------------------------------------
        # Model
        # -------------------------------------------------
        params = (
            BASE_CATBOOST_PARAMS.copy()
        )

        params[
            "random_seed"
        ] = seed


        model = CatBoostClassifier(
            **params
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


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------
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
                BASE_CATBOOST_PARAMS[
                    "iterations"
                ]
            )

        else:

            best_trees = (
                int(best_iteration)
                + 1
            )


        seed_results[
            scheme_name
        ] = brier


        result_rows.append({

            "seed":
                seed,

            "scheme":
                scheme_name,

            "brier_2024":
                brier,

            "best_trees":
                best_trees,
        })


        print(
            f"Brier      : "
            f"{brier:.8f}"
        )

        print(
            f"Best Trees : "
            f"{best_trees}"
        )


    # -----------------------------------------------------
    # Seed별 A/B 비교
    # delta > 0 이면 old_050 승
    # -----------------------------------------------------
    uniform_brier = (
        seed_results[
            "uniform"
        ]
    )

    old050_brier = (
        seed_results[
            "old_050"
        ]
    )

    delta = (
        uniform_brier
        - old050_brier
    )


    print("\n[Seed Comparison]")

    print(
        f"Uniform : "
        f"{uniform_brier:.8f}"
    )

    print(
        f"Old050  : "
        f"{old050_brier:.8f}"
    )

    print(
        f"Delta   : "
        f"{delta:+.8f}"
    )


    if delta > 0:

        print(
            "Winner  : old_050"
        )

    elif delta < 0:

        print(
            "Winner  : uniform"
        )

    else:

        print(
            "Winner  : tie"
        )


# =========================================================
# 전체 결과 DataFrame
# =========================================================
results = pd.DataFrame(
    result_rows
)


# =========================================================
# Seed별 비교표
# =========================================================
pivot = (
    results
    .pivot(
        index="seed",
        columns="scheme",
        values="brier_2024",
    )
    .reset_index()
)


pivot[
    "delta_uniform_minus_old050"
] = (
    pivot["uniform"]
    - pivot["old_050"]
)


pivot[
    "winner"
] = np.where(
    pivot[
        "delta_uniform_minus_old050"
    ] > 0,
    "old_050",
    "uniform",
)


# =========================================================
# 통계
# =========================================================
deltas = (
    pivot[
        "delta_uniform_minus_old050"
    ]
    .to_numpy()
)


win_count = int(
    np.sum(
        deltas > 0
    )
)

loss_count = int(
    np.sum(
        deltas < 0
    )
)

mean_delta = float(
    np.mean(
        deltas
    )
)

std_delta = float(
    np.std(
        deltas,
        ddof=1,
    )
)


mean_uniform = float(
    pivot["uniform"].mean()
)

mean_old050 = float(
    pivot["old_050"].mean()
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("=" * 80)
print("MULTI-SEED RESULT")
print("=" * 80)


print(
    pivot.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[Summary]")

print(
    f"Uniform Mean : "
    f"{mean_uniform:.8f}"
)

print(
    f"Old050 Mean  : "
    f"{mean_old050:.8f}"
)

print(
    f"Mean Delta   : "
    f"{mean_delta:+.8f}"
)

print(
    f"Delta Std    : "
    f"{std_delta:.8f}"
)

print(
    f"Old050 Wins  : "
    f"{win_count}/{len(SEEDS)}"
)

print(
    f"Uniform Wins : "
    f"{loss_count}/{len(SEEDS)}"
)


# =========================================================
# 아주 간단한 자동 판단
# =========================================================
print("\n[Quick Interpretation]")


if (
    win_count == len(SEEDS)
    and mean_delta > 0
):

    print(
        "old_050이 모든 새 seed에서 승리했습니다."
    )

    print(
        "→ recency weighting 효과가 "
        "seed 42의 우연일 가능성이 낮아졌습니다."
    )


elif (
    win_count >= 2
    and mean_delta > 0
):

    print(
        "old_050이 다수 seed에서 승리했고 "
        "평균 개선도 양수입니다."
    )

    print(
        "→ 약한 긍정 신호입니다."
    )


else:

    print(
        "old_050 효과가 seed에 따라 흔들립니다."
    )

    print(
        "→ recency weighting을 "
        "최종 모델에 바로 적용하기에는 근거가 약합니다."
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
    / "multiseed_raw_results.csv",
    index=False,
    encoding="utf-8-sig",
)


pivot.to_csv(
    VALIDATION_DIR
    / "multiseed_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\nSaved:")

print(
    VALIDATION_DIR
    / "multiseed_raw_results.csv"
)

print(
    VALIDATION_DIR
    / "multiseed_comparison.csv"
)


print("\nExperiment complete.")