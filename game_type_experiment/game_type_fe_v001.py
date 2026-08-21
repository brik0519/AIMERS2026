from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_fe_v001"

# 비교 대상: 네가 직접 돌린 Raw 47 R/F Ensemble
SOURCE_EXPERIMENT_NAME = "cb_v002_game_type_ensemble_v001"

ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
GROUP_COL = "game_type"

R_GROUP = "R"
F_GROUP = "F"

SEED = 42


# =========================================================
# 기존 R/F Ensemble Weight
#
# 중요:
# 이번 Primary 실험에서는 weight까지 바꾸지 않는다.
# Feature 효과만 보기 위해 기존 weight를 그대로 고정.
# =========================================================
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

SOURCE_VALIDATION_DIR = (
    VALIDATION_ROOT / SOURCE_EXPERIMENT_NAME
)

SOURCE_OOF_PATH = (
    SOURCE_VALIDATION_DIR / "oof_predictions.csv"
)

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# CatBoost 설정
#
# 이전 R/F 실험과 완전히 동일
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
# PDF/notebook에서 실제 최종 사용한 15개 Feature
# =========================================================
ADD_FEATURES = [

    # ---------------------------------------------
    # History / 신규 선수
    # ---------------------------------------------
    "pitcher_no_history",
    "batter_no_history",
    "pitcher_no_prev_game",

    # ---------------------------------------------
    # 표본 수
    # ---------------------------------------------
    "pitcher_log_n",
    "batter_log_n",

    # ---------------------------------------------
    # 최근 폼
    # ---------------------------------------------
    "pitcher_recent1_success_delta",
    "pitcher_recent3_success_delta",
    "pitcher_recent5_success_delta",
    "pitcher_recent3_middle_delta",

    # ---------------------------------------------
    # 경기 상황
    # ---------------------------------------------
    "count_state",
    "hand_matchup",

    # ---------------------------------------------
    # 점수 상황
    # ---------------------------------------------
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
    PDF/notebook의 15개 Feature를 동일하게 생성.

    모든 Feature는:
    - 현재 행 자체의 값
    - 주최 측이 제공한 공식 asof_* 값

    만 사용한다.

    다른 test/train row의 통계나 집계는 사용하지 않는다.
    """

    df = df.copy()


    # =====================================================
    # 1. 결측 / 신규 선수
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
    # 2. 표본 수
    # =====================================================
    df["pitcher_log_n"] = np.log1p(
        df["asof_pitcher_n"]
    )

    df["batter_log_n"] = np.log1p(
        df["asof_batter_n"]
    )


    # =====================================================
    # 3. 최근 폼
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
    # 4. 볼카운트
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
    # 5. 투수 / 타자 손 조합
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
    # 6. 점수 상황
    # =====================================================
    df["abs_score_diff"] = (
        df["score_diff_pitcher_team"]
        .abs()
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


def create_model():

    return CatBoostClassifier(
        **CATBOOST_PARAMS
    )


def get_best_tree_count(model):

    best_iteration = (
        model.get_best_iteration()
    )

    if (
        best_iteration is None
        or best_iteration < 0
    ):
        return CATBOOST_PARAMS[
            "iterations"
        ]

    return int(
        best_iteration
    ) + 1


# =========================================================
# 데이터 로드
#
# Feature schema도 train.csv에서만 결정.
# test.csv는 이번 validation 실험에서 읽지 않는다.
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


if len(BASE_FEATURES) != 47:

    print(
        "WARNING: "
        f"Raw Feature 수가 47이 아닙니다: "
        f"{len(BASE_FEATURES)}"
    )


# =========================================================
# 62 Feature 데이터 생성
# =========================================================
train = make_features(
    train_raw
)


FE_FEATURES = (
    BASE_FEATURES.copy()
)


for col in ADD_FEATURES:

    if col not in FE_FEATURES:
        FE_FEATURES.append(col)


print(
    f"Added Features : "
    f"{len(ADD_FEATURES)}"
)

print(
    f"FE Features    : "
    f"{len(FE_FEATURES)}"
)


# =========================================================
# Feature 존재 검증
# =========================================================
missing_features = [
    col
    for col in FE_FEATURES
    if col not in train.columns
]


if missing_features:

    raise ValueError(
        "생성되지 않은 Feature 발견: "
        f"{missing_features}"
    )


missing_cat_cols = [
    col
    for col in CAT_COLS
    if col not in FE_FEATURES
]


if missing_cat_cols:

    raise ValueError(
        "Categorical Feature가 "
        "FE_FEATURES에 없습니다: "
        f"{missing_cat_cols}"
    )


train = prepare_categorical(
    train
)


# =========================================================
# R/F 확인
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
# 결과
# =========================================================
all_oof = []

fold_summary_rows = []


# =========================================================
# Expanding Window
# =========================================================
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
    # Fold split
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


    # =====================================================
    # R / F
    # =====================================================
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


    # =====================================================
    # Data
    # =====================================================
    X_train_all = (
        fold_train[FE_FEATURES]
    )

    y_train_all = (
        fold_train[TARGET_COL]
    )


    X_train_r = (
        train_r[FE_FEATURES]
    )

    y_train_r = (
        train_r[TARGET_COL]
    )


    X_train_f = (
        train_f[FE_FEATURES]
    )

    y_train_f = (
        train_f[TARGET_COL]
    )


    X_valid_r = (
        valid_r[FE_FEATURES]
    )

    y_valid_r = (
        valid_r[TARGET_COL]
    )


    X_valid_f = (
        valid_f[FE_FEATURES]
    )

    y_valid_f = (
        valid_f[TARGET_COL]
    )


    # =====================================================
    # 1. Global-R
    #
    # Train = R + F
    # Validation = R
    # =====================================================
    print("\nTrain Global-R...")

    start = time.time()

    global_r_model = create_model()

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


    global_r_iter = (
        get_best_tree_count(
            global_r_model
        )
    )


    # =====================================================
    # 2. R-only
    # =====================================================
    print("Train R-only...")

    r_only_model = create_model()

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


    r_only_iter = (
        get_best_tree_count(
            r_only_model
        )
    )


    # =====================================================
    # 3. Global-F
    # =====================================================
    print("Train Global-F...")

    global_f_model = create_model()

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


    global_f_iter = (
        get_best_tree_count(
            global_f_model
        )
    )


    # =====================================================
    # 4. F-only
    # =====================================================
    print("Train F-only...")

    f_only_model = create_model()

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


    f_only_iter = (
        get_best_tree_count(
            f_only_model
        )
    )


    # =====================================================
    # Prediction 결합
    # =====================================================
    valid_r_mask = (
        fold_valid[GROUP_COL]
        == R_GROUP
    ).to_numpy()

    valid_f_mask = (
        fold_valid[GROUP_COL]
        == F_GROUP
    ).to_numpy()


    # -----------------------------------------------------
    # FE Group-ES
    #
    # R = Global-R
    # F = Global-F
    # -----------------------------------------------------
    fe_group_es_pred = np.empty(
        len(fold_valid),
        dtype=float,
    )


    fe_group_es_pred[
        valid_r_mask
    ] = global_r_pred

    fe_group_es_pred[
        valid_f_mask
    ] = global_f_pred


    # -----------------------------------------------------
    # FE Fixed Weighted Ensemble
    # -----------------------------------------------------
    r_weighted_pred = (
        R_GLOBAL_WEIGHT
        * global_r_pred
        +
        R_ONLY_WEIGHT
        * r_only_pred
    )


    f_weighted_pred = (
        F_GLOBAL_WEIGHT
        * global_f_pred
        +
        F_ONLY_WEIGHT
        * f_only_pred
    )


    fe_weighted_pred = np.empty(
        len(fold_valid),
        dtype=float,
    )


    fe_weighted_pred[
        valid_r_mask
    ] = r_weighted_pred

    fe_weighted_pred[
        valid_f_mask
    ] = f_weighted_pred


    # =====================================================
    # Fold Brier
    # =====================================================
    y_valid = (
        fold_valid[TARGET_COL]
        .to_numpy(dtype=float)
    )


    fe_group_es_brier = calc_brier(
        y_valid,
        fe_group_es_pred,
    )


    fe_weighted_brier = calc_brier(
        y_valid,
        fe_weighted_pred,
    )


    print(
        f"\nFE Group-ES Brier : "
        f"{fe_group_es_brier:.8f}"
    )

    print(
        f"FE Weighted Brier : "
        f"{fe_weighted_brier:.8f}"
    )


    print(
        "\nTrees | "
        f"Global-R={global_r_iter} | "
        f"R-only={r_only_iter} | "
        f"Global-F={global_f_iter} | "
        f"F-only={f_only_iter}"
    )


    elapsed = (
        time.time() - start
    )


    # =====================================================
    # 저장용
    # =====================================================
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

        "fe_group_es_pred":
            fe_group_es_pred,

        "fe_weighted_pred":
            fe_weighted_pred,
    })


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


    all_oof.append(
        fold_result
    )


    fold_summary_rows.append({

        "fold":
            fold,

        "valid_season":
            valid_season,

        "fe_group_es_brier":
            fe_group_es_brier,

        "fe_weighted_brier":
            fe_weighted_brier,

        "global_R_best_trees":
            global_r_iter,

        "R_only_best_trees":
            r_only_iter,

        "global_F_best_trees":
            global_f_iter,

        "F_only_best_trees":
            f_only_iter,

        "elapsed_sec":
            elapsed,
    })


# =========================================================
# FE OOF
# =========================================================
oof = pd.concat(
    all_oof,
    ignore_index=True,
)


fold_summary = pd.DataFrame(
    fold_summary_rows
)


fe_group_es_oof = calc_brier(
    oof["y_true"],
    oof["fe_group_es_pred"],
)


fe_weighted_oof = calc_brier(
    oof["y_true"],
    oof["fe_weighted_pred"],
)


# =========================================================
# 기존 Raw R/F OOF 로드
# =========================================================
print("\nLoad Raw 47 R/F OOF...")


if not SOURCE_OOF_PATH.exists():

    raise FileNotFoundError(
        "기존 R/F OOF가 없습니다:\n"
        f"{SOURCE_OOF_PATH}"
    )


raw_oof = pd.read_csv(
    SOURCE_OOF_PATH,
    encoding="utf-8-sig",
)


if "ensemble_pred" not in raw_oof.columns:

    raise ValueError(
        "기존 OOF에 ensemble_pred "
        "컬럼이 없습니다."
    )


# =========================================================
# row_id 기준 비교
# =========================================================
compare = (
    oof[
        [
            ID_COL,
            SEASON_COL,
            GROUP_COL,
            "y_true",
            "fe_group_es_pred",
            "fe_weighted_pred",
        ]
    ]
    .merge(
        raw_oof[
            [
                ID_COL,
                "ensemble_pred",
            ]
        ],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
)


if len(compare) != len(oof):

    raise ValueError(
        "Raw OOF와 FE OOF의 row_id가 "
        "완전히 일치하지 않습니다."
    )


raw_weighted_oof = calc_brier(
    compare["y_true"],
    compare["ensemble_pred"],
)


feature_delta = (
    raw_weighted_oof
    - fe_weighted_oof
)


# =========================================================
# 시즌별 비교
# =========================================================
season_rows = []


for season in sorted(
    compare[SEASON_COL]
    .unique()
):

    season_df = compare[
        compare[SEASON_COL]
        == season
    ]


    raw_brier = calc_brier(
        season_df["y_true"],
        season_df["ensemble_pred"],
    )


    fe_brier = calc_brier(
        season_df["y_true"],
        season_df["fe_weighted_pred"],
    )


    season_rows.append({

        "season":
            int(season),

        "raw_47_R_F":
            raw_brier,

        "fe_62_R_F":
            fe_brier,

        "delta":
            raw_brier - fe_brier,
    })


season_summary = pd.DataFrame(
    season_rows
)


# =========================================================
# R/F 별 비교
# =========================================================
group_rows = []


for group in [
    R_GROUP,
    F_GROUP,
]:

    group_df = compare[
        compare[GROUP_COL]
        == group
    ]


    raw_brier = calc_brier(
        group_df["y_true"],
        group_df["ensemble_pred"],
    )


    fe_brier = calc_brier(
        group_df["y_true"],
        group_df["fe_weighted_pred"],
    )


    group_rows.append({

        "game_type":
            group,

        "raw_47_R_F":
            raw_brier,

        "fe_62_R_F":
            fe_brier,

        "delta":
            raw_brier - fe_brier,
    })


group_summary = pd.DataFrame(
    group_rows
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("#" * 80)
print("FINAL FEATURE EXPERIMENT RESULT")
print("#" * 80)


print("\n[Fold Summary]")

print(
    fold_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


print("\n[POOLED OOF]")

print(
    f"Raw 47 R/F Ensemble : "
    f"{raw_weighted_oof:.8f}"
)

print(
    f"FE 62 Group-ES      : "
    f"{fe_group_es_oof:.8f}"
)

print(
    f"FE 62 R/F Ensemble  : "
    f"{fe_weighted_oof:.8f}"
)

print(
    f"Feature Improvement : "
    f"{feature_delta:+.8f}"
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


print("\n[Game Type Comparison]")

print(
    group_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =========================================================
# 간단 판정
# =========================================================
season_win_count = int(
    (
        season_summary["delta"]
        > 0
    ).sum()
)


print("\n")
print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)


if feature_delta > 0:

    print(
        "Pooled OOF : PASS"
    )

else:

    print(
        "Pooled OOF : FAIL"
    )


print(
    f"Season Wins: "
    f"{season_win_count}/"
    f"{len(season_summary)}"
)


season_2024 = (
    season_summary[
        season_summary["season"]
        == 2024
    ]
)


if len(season_2024) == 1:

    delta_2024 = float(
        season_2024.iloc[0][
            "delta"
        ]
    )

    print(
        f"2024 Delta: "
        f"{delta_2024:+.8f}"
    )

    if delta_2024 > 0:

        print(
            "2024      : PASS"
        )

    else:

        print(
            "2024      : FAIL"
        )


# =========================================================
# 결과 저장
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


oof.to_csv(
    VALIDATION_DIR
    / "oof_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


fold_summary.to_csv(
    VALIDATION_DIR
    / "fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


season_summary.to_csv(
    VALIDATION_DIR
    / "season_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


group_summary.to_csv(
    VALIDATION_DIR
    / "group_summary.csv",
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

    "base_feature_count":
        len(BASE_FEATURES),

    "added_feature_count":
        len(ADD_FEATURES),

    "final_feature_count":
        len(FE_FEATURES),

    "added_features":
        ADD_FEATURES,

    "fixed_weights": {

        "R_global":
            R_GLOBAL_WEIGHT,

        "R_only":
            R_ONLY_WEIGHT,

        "F_global":
            F_GLOBAL_WEIGHT,

        "F_only":
            F_ONLY_WEIGHT,
    },

    "raw_R_F_oof_brier":
        raw_weighted_oof,

    "fe_R_F_oof_brier":
        fe_weighted_oof,

    "feature_improvement":
        feature_delta,

    "season_win_count":
        season_win_count,
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
    / "season_summary.csv"
)

print(
    VALIDATION_DIR
    / "group_summary.csv"
)

print(
    VALIDATION_DIR
    / "metadata.json"
)


print("\nExperiment complete.")