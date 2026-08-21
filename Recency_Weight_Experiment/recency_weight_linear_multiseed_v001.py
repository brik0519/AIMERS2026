from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 기본 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_recency_weight_linear_multiseed_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"

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
# Train / Validation
# =========================================================
TRAIN_START = 2019
TRAIN_END = 2023
VALID_SEASON = 2024


# =========================================================
# 이번에 확인할 Linear Weight
#
# 최근 연도일수록 점진적으로 중요도를 높임
# =========================================================
LINEAR_WEIGHTS = {
    2019: 0.50,
    2020: 0.60,
    2021: 0.70,
    2022: 0.85,
    2023: 1.00,
}


# =========================================================
# 이미 확보한 기존 실험 결과
# 비교표 출력을 위해서만 사용
# =========================================================
PREVIOUS_RESULTS = {
    7: {
        "uniform": 0.24799672,
        "old_050": 0.24799707,
    },
    123: {
        "uniform": 0.24802805,
        "old_050": 0.24798505,
    },
    999: {
        "uniform": 0.24798616,
        "old_050": 0.24796549,
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

CAT_COLS = BASE_CAT_COLS + ADD_CAT_FEATURES


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

    # History
    df["pitcher_no_history"] = (
        df["asof_pitcher_n"] == 0
    ).astype(int)

    df["batter_no_history"] = (
        df["asof_batter_n"] == 0
    ).astype(int)

    df["pitcher_no_prev_game"] = (
        df["asof_pitcher_prev1_game_success_rate"].isna()
    ).astype(int)

    # Sample size
    df["pitcher_log_n"] = np.log1p(
        df["asof_pitcher_n"]
    )

    df["batter_log_n"] = np.log1p(
        df["asof_batter_n"]
    )

    # Recent - Career
    df["pitcher_recent1_success_delta"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["pitcher_recent3_success_delta"] = (
        df["asof_pitcher_prev3_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["pitcher_recent5_success_delta"] = (
        df["asof_pitcher_prev5_game_success_rate"]
        - df["asof_pitcher_success_rate"]
    )

    df["pitcher_recent3_middle_delta"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    # Count
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

    # Score
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
# Brier
# =========================================================
def calc_brier(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


# =========================================================
# Weight
# =========================================================
def make_sample_weight(season_series):

    weights = (
        season_series
        .map(LINEAR_WEIGHTS)
        .astype(float)
        .to_numpy()
    )

    if np.isnan(weights).any():
        raise ValueError(
            "Weight가 정의되지 않은 season이 있습니다."
        )

    # 평균 weight = 1
    weights = weights / weights.mean()

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
# 기존 Features
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
# 62FE
# =========================================================
train = make_features(train_raw)

FEATURES = BASE_FEATURES.copy()

for col in ADD_FEATURES:
    if col not in FEATURES:
        FEATURES.append(col)


print(f"Raw Features   : {len(BASE_FEATURES)}")
print(f"Added Features : {len(ADD_FEATURES)}")
print(f"Final Features : {len(FEATURES)}")


if len(FEATURES) != 62:
    raise ValueError(
        f"Expected 62 features, got {len(FEATURES)}"
    )


train = prepare_categorical(train)


# =========================================================
# Fold4
# =========================================================
train_mask = (
    (train[SEASON_COL] >= TRAIN_START)
    &
    (train[SEASON_COL] <= TRAIN_END)
)

valid_mask = (
    train[SEASON_COL] == VALID_SEASON
)


fold_train = train.loc[train_mask].copy()
fold_valid = train.loc[valid_mask].copy()


print("\n[Dataset]")
print(
    f"Train {TRAIN_START}~{TRAIN_END}: "
    f"{len(fold_train):,}"
)
print(
    f"Valid {VALID_SEASON}: "
    f"{len(fold_valid):,}"
)


X_train = fold_train[FEATURES]
y_train = fold_train[TARGET_COL]

X_valid = fold_valid[FEATURES]
y_valid = fold_valid[TARGET_COL]


# =========================================================
# Linear sample weight
# =========================================================
sample_weight = make_sample_weight(
    fold_train[SEASON_COL]
)


# =========================================================
# 3 Seed 실험
# =========================================================
result_rows = []


for seed in SEEDS:

    print("\n" + "=" * 80)
    print(f"SEED = {seed}")
    print("=" * 80)

    params = BASE_CATBOOST_PARAMS.copy()
    params["random_seed"] = seed

    model = CatBoostClassifier(
        **params
    )

    model.fit(
        X_train,
        y_train,

        sample_weight=sample_weight,

        cat_features=CAT_COLS,

        eval_set=(
            X_valid,
            y_valid,
        ),

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,

        use_best_model=True,
    )

    pred = model.predict_proba(
        X_valid
    )[:, 1]

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
            BASE_CATBOOST_PARAMS["iterations"]
        )
    else:
        best_trees = (
            int(best_iteration) + 1
        )


    print(
        f"Linear Brier : {brier:.8f}"
    )
    print(
        f"Best Trees   : {best_trees}"
    )


    result_rows.append({
        "seed": seed,
        "linear": brier,
        "best_trees": best_trees,
    })


# =========================================================
# 결과 정리
# =========================================================
linear_df = pd.DataFrame(
    result_rows
)


comparison_rows = []


for _, row in linear_df.iterrows():

    seed = int(row["seed"])

    uniform = (
        PREVIOUS_RESULTS[seed]["uniform"]
    )

    old050 = (
        PREVIOUS_RESULTS[seed]["old_050"]
    )

    linear = float(
        row["linear"]
    )

    scores = {
        "uniform": uniform,
        "old_050": old050,
        "linear": linear,
    }

    winner = min(
        scores,
        key=scores.get,
    )


    comparison_rows.append({

        "seed": seed,

        "uniform": uniform,

        "old_050": old050,

        "linear": linear,

        "best_trees_linear":
            int(row["best_trees"]),

        "winner": winner,
    })


comparison = pd.DataFrame(
    comparison_rows
)


# =========================================================
# 평균
# =========================================================
mean_uniform = comparison["uniform"].mean()
mean_old050 = comparison["old_050"].mean()
mean_linear = comparison["linear"].mean()


mean_scores = {
    "uniform": mean_uniform,
    "old_050": mean_old050,
    "linear": mean_linear,
}


best_mean_scheme = min(
    mean_scores,
    key=mean_scores.get,
)


# =========================================================
# 출력
# =========================================================
print("\n")
print("#" * 90)
print("UNIFORM vs OLD_050 vs LINEAR")
print("#" * 90)


print(
    comparison.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[Mean Brier]")

print(
    f"Uniform : {mean_uniform:.8f}"
)

print(
    f"Old050  : {mean_old050:.8f}"
)

print(
    f"Linear  : {mean_linear:.8f}"
)


print("\n[Delta vs Uniform]")

print(
    f"Old050 : "
    f"{mean_uniform - mean_old050:+.8f}"
)

print(
    f"Linear : "
    f"{mean_uniform - mean_linear:+.8f}"
)


print("\n[Best Mean Scheme]")

print(
    best_mean_scheme
)


# =========================================================
# 저장
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


comparison.to_csv(
    VALIDATION_DIR
    / "linear_multiseed_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\nSaved:")

print(
    VALIDATION_DIR
    / "linear_multiseed_comparison.csv"
)

print("\nExperiment complete.")