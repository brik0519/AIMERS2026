from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_ensemble_multiseed_v001"

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

# 기존 weight를 찾았던 seed.
TUNING_SEED = 42


# =========================================================
# Fixed Ensemble Weight
#
# 이전 seed=42 OOF에서 확정한 weight.
#
# 이번 실험의 Primary 평가는 이 값을 절대 다시 최적화하지 않고
# 그대로 다른 seed에 적용했을 때의 성능이다.
# =========================================================
R_GLOBAL_WEIGHT = 0.49
R_ONLY_WEIGHT = 0.51

F_GLOBAL_WEIGHT = 0.64
F_ONLY_WEIGHT = 0.36


# =========================================================
# Secondary 분석용 Weight Search
#
# seed별 optimal weight는 "분석용"일 뿐,
# Primary 성능으로 사용하지 않는다.
# =========================================================
WEIGHT_STEP = 0.01

WEIGHTS = np.round(
    np.arange(
        0.0,
        1.0 + WEIGHT_STEP,
        WEIGHT_STEP,
    ),
    2,
)


# =========================================================
# CatBoost 기본 설정
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
    기존 cb_v002와 동일한 categorical 처리.
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
    """
    Brier Score.
    낮을수록 좋음.
    """

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


def create_model(seed):
    """
    seed만 변경하고 나머지 설정은 전부 동일하게 유지.
    """

    params = BASE_CATBOOST_PARAMS.copy()
    params["random_seed"] = int(seed)

    return CatBoostClassifier(
        **params
    )


def get_best_tree_count(model):
    """
    CatBoost get_best_iteration()은 0-based.
    """

    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        return BASE_CATBOOST_PARAMS[
            "iterations"
        ]

    return int(best_iteration) + 1


def ensemble_prediction(
    global_pred,
    group_only_pred,
    global_weight,
):
    """
    prediction
    =
    global_weight * global
    +
    (1-global_weight) * group-only
    """

    global_pred = np.asarray(
        global_pred,
        dtype=float,
    )

    group_only_pred = np.asarray(
        group_only_pred,
        dtype=float,
    )

    return (
        global_weight * global_pred
        + (1.0 - global_weight)
        * group_only_pred
    )


def search_best_weight(
    y_true,
    global_pred,
    group_only_pred,
):
    """
    Secondary 분석용.

    해당 seed OOF에서 최적 weight를 탐색.
    이 결과는 Primary 성능으로 사용하지 않는다.
    """

    rows = []

    for weight in WEIGHTS:

        pred = ensemble_prediction(
            global_pred=global_pred,
            group_only_pred=group_only_pred,
            global_weight=weight,
        )

        brier = calc_brier(
            y_true,
            pred,
        )

        rows.append({
            "global_weight":
                float(weight),

            "group_only_weight":
                float(1.0 - weight),

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
        float(best["global_weight"]),
        float(best["brier"]),
        result,
    )


# =========================================================
# 데이터 로드
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)


# ---------------------------------------------------------
# test.csv는 컬럼 schema 확인용으로만 사용.
#
# test 행 값 / 통계 / 분포 / 빈도 / 집계는
# 절대 사용하지 않는다.
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


print(
    f"Features: {len(FEATURES)}"
)

print(
    f"Categorical: {len(CAT_COLS)}"
)


# ---------------------------------------------------------
# train.csv 로드
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
# game_type 검증
# =========================================================
game_types = sorted(
    train[GROUP_COL]
    .dropna()
    .unique()
    .tolist()
)

print(
    f"game_type values: "
    f"{game_types}"
)


unexpected_groups = (
    set(game_types)
    - {R_GROUP, F_GROUP}
)

if unexpected_groups:
    raise ValueError(
        "예상하지 못한 game_type 값 발견: "
        f"{unexpected_groups}"
    )


# =========================================================
# 저장 폴더
# =========================================================
VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 전체 결과 저장용
# =========================================================
all_seed_oof = []

seed_summary_rows = []
seed_season_rows = []
seed_weight_rows = []


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

    seed_fold_predictions = []


    # =====================================================
    # Fold
    # =====================================================
    for fold_info in FOLDS:

        fold = fold_info["fold"]
        train_start = fold_info["train_start"]
        train_end = fold_info["train_end"]
        valid_season = fold_info[
            "valid_season"
        ]


        print("\n")
        print("=" * 80)

        print(
            f"Seed {seed} | "
            f"Fold {fold} | "
            f"Train {train_start}~"
            f"{train_end} "
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


        fold_train = train.loc[
            train_mask
        ].copy()

        fold_valid = train.loc[
            valid_mask
        ].copy()


        train_r = fold_train[
            fold_train[GROUP_COL]
            == R_GROUP
        ].copy()

        train_f = fold_train[
            fold_train[GROUP_COL]
            == F_GROUP
        ].copy()

        valid_r = fold_valid[
            fold_valid[GROUP_COL]
            == R_GROUP
        ].copy()

        valid_f = fold_valid[
            fold_valid[GROUP_COL]
            == F_GROUP
        ].copy()


        if (
            len(train_r) == 0
            or len(train_f) == 0
            or len(valid_r) == 0
            or len(valid_f) == 0
        ):
            raise ValueError(
                f"Seed {seed}, Fold {fold}: "
                "R/F 데이터가 비어 있습니다."
            )


        # =================================================
        # Data
        # =================================================
        X_train_all = (
            fold_train[FEATURES]
        )

        y_train_all = (
            fold_train[TARGET_COL]
        )


        X_train_r = (
            train_r[FEATURES]
        )

        y_train_r = (
            train_r[TARGET_COL]
        )


        X_train_f = (
            train_f[FEATURES]
        )

        y_train_f = (
            train_f[TARGET_COL]
        )


        X_valid_r = (
            valid_r[FEATURES]
        )

        y_valid_r = (
            valid_r[TARGET_COL]
        )


        X_valid_f = (
            valid_f[FEATURES]
        )

        y_valid_f = (
            valid_f[TARGET_COL]
        )


        # =================================================
        # 1. Global-R
        #
        # Train = R + F
        # Validation = R
        # =================================================
        global_r_model = create_model(
            seed
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

        global_r_iter = (
            get_best_tree_count(
                global_r_model
            )
        )


        # =================================================
        # 2. R-only
        #
        # Train = R
        # Validation = R
        # =================================================
        r_only_model = create_model(
            seed
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

        r_only_iter = (
            get_best_tree_count(
                r_only_model
            )
        )


        # =================================================
        # 3. Global-F
        #
        # Train = R + F
        # Validation = F
        # =================================================
        global_f_model = create_model(
            seed
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

        global_f_iter = (
            get_best_tree_count(
                global_f_model
            )
        )


        # =================================================
        # 4. F-only
        #
        # Train = F
        # Validation = F
        # =================================================
        f_only_model = create_model(
            seed
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

        f_only_iter = (
            get_best_tree_count(
                f_only_model
            )
        )


        # =================================================
        # Group-ES prediction
        #
        # R -> Global-R
        # F -> Global-F
        # =================================================
        group_es_pred = np.empty(
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


        group_es_pred[
            valid_r_mask
        ] = global_r_pred

        group_es_pred[
            valid_f_mask
        ] = global_f_pred


        # =================================================
        # Fixed Ensemble prediction
        #
        # R = 0.49 Global + 0.51 R-only
        # F = 0.64 Global + 0.36 F-only
        # =================================================
        r_fixed_pred = (
            R_GLOBAL_WEIGHT
            * global_r_pred
            +
            R_ONLY_WEIGHT
            * r_only_pred
        )


        f_fixed_pred = (
            F_GLOBAL_WEIGHT
            * global_f_pred
            +
            F_ONLY_WEIGHT
            * f_only_pred
        )


        fixed_ensemble_pred = np.empty(
            len(fold_valid),
            dtype=float,
        )


        fixed_ensemble_pred[
            valid_r_mask
        ] = r_fixed_pred

        fixed_ensemble_pred[
            valid_f_mask
        ] = f_fixed_pred


        # =================================================
        # Fold Brier
        # =================================================
        y_valid = (
            fold_valid[TARGET_COL]
            .to_numpy(dtype=float)
        )


        group_es_brier = calc_brier(
            y_valid,
            group_es_pred,
        )

        fixed_brier = calc_brier(
            y_valid,
            fixed_ensemble_pred,
        )


        delta = (
            group_es_brier
            - fixed_brier
        )


        print(
            f"\nGroup-ES : "
            f"{group_es_brier:.8f}"
        )

        print(
            f"Fixed Ens: "
            f"{fixed_brier:.8f}"
        )

        print(
            f"Delta    : "
            f"{delta:+.8f}"
        )


        print(
            "\nTrees | "
            f"Global-R={global_r_iter} | "
            f"R-only={r_only_iter} | "
            f"Global-F={global_f_iter} | "
            f"F-only={f_only_iter}"
        )


        # =================================================
        # Fold prediction 저장
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

            "group_es_pred":
                group_es_pred,

            "fixed_ensemble_pred":
                fixed_ensemble_pred,
        })


        # ---------------------------------------------
        # R/F 개별 prediction
        # ---------------------------------------------
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


        fold_result["seed"] = seed
        fold_result["fold"] = fold


        seed_fold_predictions.append(
            fold_result
        )


    # =====================================================
    # Seed OOF
    # =====================================================
    seed_oof = pd.concat(
        seed_fold_predictions,
        ignore_index=True,
    )


    group_es_oof_brier = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "group_es_pred"
        ],
    )


    fixed_oof_brier = calc_brier(
        seed_oof["y_true"],
        seed_oof[
            "fixed_ensemble_pred"
        ],
    )


    pooled_delta = (
        group_es_oof_brier
        - fixed_oof_brier
    )


    # =====================================================
    # R pooled
    # =====================================================
    seed_r = seed_oof[
        seed_oof[GROUP_COL]
        == R_GROUP
    ].copy()


    group_r_brier = calc_brier(
        seed_r["y_true"],
        seed_r["global_R_pred"],
    )


    fixed_r_pred = (
        R_GLOBAL_WEIGHT
        * seed_r["global_R_pred"]
        +
        R_ONLY_WEIGHT
        * seed_r["R_only_pred"]
    )


    fixed_r_brier = calc_brier(
        seed_r["y_true"],
        fixed_r_pred,
    )


    # =====================================================
    # F pooled
    # =====================================================
    seed_f = seed_oof[
        seed_oof[GROUP_COL]
        == F_GROUP
    ].copy()


    group_f_brier = calc_brier(
        seed_f["y_true"],
        seed_f["global_F_pred"],
    )


    fixed_f_pred = (
        F_GLOBAL_WEIGHT
        * seed_f["global_F_pred"]
        +
        F_ONLY_WEIGHT
        * seed_f["F_only_pred"]
    )


    fixed_f_brier = calc_brier(
        seed_f["y_true"],
        fixed_f_pred,
    )


    # =====================================================
    # Secondary:
    # seed별 optimal weight
    # =====================================================
    (
        optimal_r_global_weight,
        optimal_r_brier,
        _
    ) = search_best_weight(
        y_true=seed_r["y_true"],
        global_pred=seed_r[
            "global_R_pred"
        ],
        group_only_pred=seed_r[
            "R_only_pred"
        ],
    )


    (
        optimal_f_global_weight,
        optimal_f_brier,
        _
    ) = search_best_weight(
        y_true=seed_f["y_true"],
        global_pred=seed_f[
            "global_F_pred"
        ],
        group_only_pred=seed_f[
            "F_only_pred"
        ],
    )


    seed_weight_rows.append({

        "seed":
            seed,

        "optimal_R_global_weight":
            optimal_r_global_weight,

        "optimal_R_only_weight":
            (
                1.0
                - optimal_r_global_weight
            ),

        "optimal_R_brier":
            optimal_r_brier,

        "optimal_F_global_weight":
            optimal_f_global_weight,

        "optimal_F_only_weight":
            (
                1.0
                - optimal_f_global_weight
            ),

        "optimal_F_brier":
            optimal_f_brier,
    })


    # =====================================================
    # 시즌별 결과
    # =====================================================
    for season in sorted(
        seed_oof[SEASON_COL]
        .unique()
    ):

        season_df = seed_oof[
            seed_oof[SEASON_COL]
            == season
        ].copy()


        season_group_brier = (
            calc_brier(
                season_df["y_true"],
                season_df[
                    "group_es_pred"
                ],
            )
        )


        season_fixed_brier = (
            calc_brier(
                season_df["y_true"],
                season_df[
                    "fixed_ensemble_pred"
                ],
            )
        )


        season_delta = (
            season_group_brier
            - season_fixed_brier
        )


        seed_season_rows.append({

            "seed":
                seed,

            "season":
                int(season),

            "group_es_brier":
                season_group_brier,

            "fixed_ensemble_brier":
                season_fixed_brier,

            "delta":
                season_delta,
        })


    # =====================================================
    # Seed summary
    # =====================================================
    seed_elapsed = (
        time.time() - seed_start
    )


    seed_summary_rows.append({

        "seed":
            seed,

        "group_es_oof_brier":
            group_es_oof_brier,

        "fixed_ensemble_oof_brier":
            fixed_oof_brier,

        "delta":
            pooled_delta,

        "R_group_brier":
            group_r_brier,

        "R_fixed_brier":
            fixed_r_brier,

        "R_delta":
            (
                group_r_brier
                - fixed_r_brier
            ),

        "F_group_brier":
            group_f_brier,

        "F_fixed_brier":
            fixed_f_brier,

        "F_delta":
            (
                group_f_brier
                - fixed_f_brier
            ),

        "elapsed_sec":
            seed_elapsed,
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
        f"Group-ES OOF : "
        f"{group_es_oof_brier:.8f}"
    )

    print(
        f"Fixed Ens OOF: "
        f"{fixed_oof_brier:.8f}"
    )

    print(
        f"Delta        : "
        f"{pooled_delta:+.8f}"
    )

    print(
        f"R Delta      : "
        f"{group_r_brier - fixed_r_brier:+.8f}"
    )

    print(
        f"F Delta      : "
        f"{group_f_brier - fixed_f_brier:+.8f}"
    )

    print(
        f"Optimal R Global Weight "
        f"(secondary): "
        f"{optimal_r_global_weight:.2f}"
    )

    print(
        f"Optimal F Global Weight "
        f"(secondary): "
        f"{optimal_f_global_weight:.2f}"
    )

    print(
        f"Time: "
        f"{seed_elapsed:.1f}s"
    )


# =========================================================
# 전체 Summary
# =========================================================
seed_summary = pd.DataFrame(
    seed_summary_rows
)

season_summary = pd.DataFrame(
    seed_season_rows
)

weight_summary = pd.DataFrame(
    seed_weight_rows
)

all_oof = pd.concat(
    all_seed_oof,
    ignore_index=True,
)


# =========================================================
# 전체 5 Seeds 통계
# =========================================================
mean_delta = float(
    seed_summary["delta"].mean()
)

std_delta = float(
    seed_summary["delta"].std(
        ddof=1
    )
)

win_count = int(
    (
        seed_summary["delta"] > 0
    ).sum()
)


mean_group_brier = float(
    seed_summary[
        "group_es_oof_brier"
    ].mean()
)

mean_fixed_brier = float(
    seed_summary[
        "fixed_ensemble_oof_brier"
    ].mean()
)


# =========================================================
# Tuning Seed=42 제외 결과
#
# fixed weight가 seed 42에서 선택됐기 때문에
# 다른 4개 seed 결과를 별도로 보는 것이 중요.
# =========================================================
unseen_seed_summary = (
    seed_summary[
        seed_summary["seed"]
        != TUNING_SEED
    ]
    .copy()
)


unseen_mean_delta = float(
    unseen_seed_summary[
        "delta"
    ].mean()
)

unseen_std_delta = float(
    unseen_seed_summary[
        "delta"
    ].std(
        ddof=1
    )
)

unseen_win_count = int(
    (
        unseen_seed_summary[
            "delta"
        ]
        > 0
    ).sum()
)


# =========================================================
# 시즌별 평균
# =========================================================
season_mean = (
    season_summary
    .groupby(
        "season",
        as_index=False,
    )
    .agg(
        group_es_mean=(
            "group_es_brier",
            "mean",
        ),
        fixed_ensemble_mean=(
            "fixed_ensemble_brier",
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
    )
)


# =========================================================
# 최종 출력
# =========================================================
print("\n")
print("#" * 80)
print("MULTI-SEED FINAL RESULT")
print("#" * 80)


print("\n[Seed Summary]")

print(
    seed_summary[
        [
            "seed",
            "group_es_oof_brier",
            "fixed_ensemble_oof_brier",
            "delta",
            "R_delta",
            "F_delta",
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
    f"Mean Group-ES Brier : "
    f"{mean_group_brier:.8f}"
)

print(
    f"Mean Fixed Ens Brier: "
    f"{mean_fixed_brier:.8f}"
)

print(
    f"Mean Delta          : "
    f"{mean_delta:+.8f}"
)

print(
    f"Delta Std           : "
    f"{std_delta:.8f}"
)

print(
    f"Win Count           : "
    f"{win_count}/{len(SEEDS)}"
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
    f"{unseen_win_count}/"
    f"{len(unseen_seed_summary)}"
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


print("\n[Optimal Weights - Secondary Analysis]")

print(
    weight_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =========================================================
# 간단 자동 판정
# =========================================================
print("\n")
print("=" * 80)
print("STABILITY CHECK")
print("=" * 80)


if mean_delta > 0:
    print(
        "ALL seeds mean delta : PASS"
    )
else:
    print(
        "ALL seeds mean delta : FAIL"
    )


if win_count >= 3:
    print(
        "Win count >= 3/5     : PASS"
    )
else:
    print(
        "Win count >= 3/5     : FAIL"
    )


if unseen_mean_delta > 0:
    print(
        "Excluding seed42 mean: PASS"
    )
else:
    print(
        "Excluding seed42 mean: FAIL"
    )


if unseen_win_count >= 2:
    print(
        "Unseen seed wins     : PASS"
    )
else:
    print(
        "Unseen seed wins     : FAIL"
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
    / "optimal_weight_summary.csv",
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

    "primary_fixed_weights": {
        "R_global":
            R_GLOBAL_WEIGHT,

        "R_only":
            R_ONLY_WEIGHT,

        "F_global":
            F_GLOBAL_WEIGHT,

        "F_only":
            F_ONLY_WEIGHT,
    },

    "mean_group_es_brier":
        mean_group_brier,

    "mean_fixed_ensemble_brier":
        mean_fixed_brier,

    "mean_delta":
        mean_delta,

    "delta_std":
        std_delta,

    "win_count":
        win_count,

    "unseen_seed_mean_delta":
        unseen_mean_delta,

    "unseen_seed_delta_std":
        unseen_std_delta,

    "unseen_seed_win_count":
        unseen_win_count,

    "note": (
        "Primary evaluation uses fixed "
        "weights selected on seed 42. "
        "Per-seed optimal weights are "
        "secondary diagnostics only."
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
    / "optimal_weight_summary.csv"
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