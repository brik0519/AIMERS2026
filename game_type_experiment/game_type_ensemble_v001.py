from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# 실험 설정
# =========================================================
EXPERIMENT_NAME = "cb_v002_game_type_ensemble_v001"

SOURCE_EXPERIMENT_NAME = "cb_v002_game_type_split_v001"

ID_COL = "row_id"
TARGET_COL = "y_true"
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

SOURCE_VALIDATION_DIR = (
    VALIDATION_ROOT / SOURCE_EXPERIMENT_NAME
)

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# Source OOF
# =========================================================
SOURCE_OOF_PATH = (
    SOURCE_VALIDATION_DIR
    / "oof_predictions.csv"
)


# =========================================================
# Weight Search
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
# 유틸
# =========================================================
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


def weighted_prediction(
    global_pred,
    split_pred,
    global_weight,
):
    """
    global_weight:
        1.0 -> Global 100%
        0.0 -> Split 100%
    """
    return (
        global_weight * global_pred
        + (1.0 - global_weight)
        * split_pred
    )


def search_best_weight(
    df,
    global_col,
    split_col,
):
    """
    주어진 데이터에서
    Global / Split prediction의
    최적 Global weight를 탐색.
    """

    y = (
        df[TARGET_COL]
        .to_numpy(dtype=float)
    )

    global_pred = (
        df[global_col]
        .to_numpy(dtype=float)
    )

    split_pred = (
        df[split_col]
        .to_numpy(dtype=float)
    )

    results = []

    for weight in WEIGHTS:

        pred = weighted_prediction(
            global_pred=global_pred,
            split_pred=split_pred,
            global_weight=weight,
        )

        brier = calc_brier(
            y,
            pred,
        )

        results.append({
            "global_weight": float(weight),
            "split_weight": float(
                1.0 - weight
            ),
            "brier": brier,
        })

    result_df = pd.DataFrame(
        results
    )

    best_idx = (
        result_df["brier"]
        .idxmin()
    )

    best_row = (
        result_df
        .loc[best_idx]
    )

    return (
        float(
            best_row["global_weight"]
        ),
        float(
            best_row["brier"]
        ),
        result_df,
    )


def print_neighborhood(
    search_df,
    best_weight,
    name,
):
    """
    최적 weight 주변 ±0.10 출력.
    한 점에만 날카롭게 최적화됐는지 확인.
    """

    lower = max(
        0.0,
        best_weight - 0.10,
    )

    upper = min(
        1.0,
        best_weight + 0.10,
    )

    neighborhood = search_df[
        (
            search_df["global_weight"]
            >= lower
        )
        &
        (
            search_df["global_weight"]
            <= upper
        )
    ].copy()

    # 너무 길어지지 않도록
    # 0.02 간격 위주로 표시
    neighborhood = neighborhood[
        (
            (
                neighborhood["global_weight"]
                * 100
            )
            .round()
            .astype(int)
            % 2
        )
        == 0
    ]

    print(
        f"\n[{name} Weight Neighborhood]"
    )

    print(
        neighborhood.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )


# =========================================================
# Load
# =========================================================
print("=" * 80)
print(EXPERIMENT_NAME)
print("=" * 80)

print("\nLoad OOF predictions...")

if not SOURCE_OOF_PATH.exists():
    raise FileNotFoundError(
        f"OOF 파일이 없습니다:\n"
        f"{SOURCE_OOF_PATH}"
    )

oof = pd.read_csv(
    SOURCE_OOF_PATH,
    encoding="utf-8-sig",
)


# =========================================================
# 필요한 컬럼 확인
# =========================================================
REQUIRED_COLS = [
    ID_COL,
    SEASON_COL,
    GROUP_COL,
    TARGET_COL,

    "global_group_pred",
    "split_group_pred",

    "global_R_pred",
    "R_only_pred",

    "global_F_pred",
    "F_only_pred",
]


missing_cols = [
    col
    for col in REQUIRED_COLS
    if col not in oof.columns
]

if missing_cols:
    raise ValueError(
        "OOF에 필요한 컬럼이 없습니다:\n"
        f"{missing_cols}"
    )


print(
    f"OOF rows: {len(oof):,}"
)

print(
    f"Seasons: "
    f"{sorted(oof[SEASON_COL].unique())}"
)

print(
    f"game_type: "
    f"{sorted(oof[GROUP_COL].unique())}"
)


# =========================================================
# R / F 분리
# =========================================================
oof_r = oof[
    oof[GROUP_COL] == R_GROUP
].copy()

oof_f = oof[
    oof[GROUP_COL] == F_GROUP
].copy()


# =========================================================
# prediction NaN 검증
# =========================================================
if oof_r[
    [
        "global_R_pred",
        "R_only_pred",
    ]
].isna().any().any():

    raise ValueError(
        "R prediction에 NaN이 있습니다."
    )


if oof_f[
    [
        "global_F_pred",
        "F_only_pred",
    ]
].isna().any().any():

    raise ValueError(
        "F prediction에 NaN이 있습니다."
    )


# =========================================================
# 기존 성능
# =========================================================
global_group_brier = calc_brier(
    oof[TARGET_COL],
    oof["global_group_pred"],
)

split_group_brier = calc_brier(
    oof[TARGET_COL],
    oof["split_group_pred"],
)


global_r_brier = calc_brier(
    oof_r[TARGET_COL],
    oof_r["global_R_pred"],
)

r_only_brier = calc_brier(
    oof_r[TARGET_COL],
    oof_r["R_only_pred"],
)


global_f_brier = calc_brier(
    oof_f[TARGET_COL],
    oof_f["global_F_pred"],
)

f_only_brier = calc_brier(
    oof_f[TARGET_COL],
    oof_f["F_only_pred"],
)


print("\n[Existing Models]")

print(
    f"Global Group-ES : "
    f"{global_group_brier:.8f}"
)

print(
    f"Split-only      : "
    f"{split_group_brier:.8f}"
)


print("\n[R]")

print(
    f"Global-R : "
    f"{global_r_brier:.8f}"
)

print(
    f"R-only   : "
    f"{r_only_brier:.8f}"
)


print("\n[F]")

print(
    f"Global-F : "
    f"{global_f_brier:.8f}"
)

print(
    f"F-only   : "
    f"{f_only_brier:.8f}"
)


# =========================================================
# R Weight Search
# =========================================================
print("\n")
print("=" * 80)
print("R WEIGHT SEARCH")
print("=" * 80)


(
    best_r_global_weight,
    best_r_brier,
    r_search,
) = search_best_weight(
    df=oof_r,
    global_col="global_R_pred",
    split_col="R_only_pred",
)


best_r_split_weight = (
    1.0 - best_r_global_weight
)


print(
    f"\nBest R Global Weight : "
    f"{best_r_global_weight:.2f}"
)

print(
    f"Best R R-only Weight : "
    f"{best_r_split_weight:.2f}"
)

print(
    f"Best R Brier         : "
    f"{best_r_brier:.8f}"
)


print_neighborhood(
    search_df=r_search,
    best_weight=best_r_global_weight,
    name="R",
)


# =========================================================
# F Weight Search
# =========================================================
print("\n")
print("=" * 80)
print("F WEIGHT SEARCH")
print("=" * 80)


(
    best_f_global_weight,
    best_f_brier,
    f_search,
) = search_best_weight(
    df=oof_f,
    global_col="global_F_pred",
    split_col="F_only_pred",
)


best_f_split_weight = (
    1.0 - best_f_global_weight
)


print(
    f"\nBest F Global Weight : "
    f"{best_f_global_weight:.2f}"
)

print(
    f"Best F F-only Weight : "
    f"{best_f_split_weight:.2f}"
)

print(
    f"Best F Brier         : "
    f"{best_f_brier:.8f}"
)


print_neighborhood(
    search_df=f_search,
    best_weight=best_f_global_weight,
    name="F",
)


# =========================================================
# 최종 Ensemble prediction
# =========================================================
oof["ensemble_pred"] = np.nan


r_mask = (
    oof[GROUP_COL] == R_GROUP
)

f_mask = (
    oof[GROUP_COL] == F_GROUP
)


oof.loc[
    r_mask,
    "ensemble_pred",
] = weighted_prediction(
    global_pred=oof.loc[
        r_mask,
        "global_R_pred",
    ].to_numpy(),

    split_pred=oof.loc[
        r_mask,
        "R_only_pred",
    ].to_numpy(),

    global_weight=(
        best_r_global_weight
    ),
)


oof.loc[
    f_mask,
    "ensemble_pred",
] = weighted_prediction(
    global_pred=oof.loc[
        f_mask,
        "global_F_pred",
    ].to_numpy(),

    split_pred=oof.loc[
        f_mask,
        "F_only_pred",
    ].to_numpy(),

    global_weight=(
        best_f_global_weight
    ),
)


# =========================================================
# 비대칭 Hard Selection도 비교
#
# R = R-only
# F = Global-F
# =========================================================
oof["asymmetric_hard_pred"] = np.nan


oof.loc[
    r_mask,
    "asymmetric_hard_pred",
] = oof.loc[
    r_mask,
    "R_only_pred",
]


oof.loc[
    f_mask,
    "asymmetric_hard_pred",
] = oof.loc[
    f_mask,
    "global_F_pred",
]


# =========================================================
# 전체 성능
# =========================================================
ensemble_brier = calc_brier(
    oof[TARGET_COL],
    oof["ensemble_pred"],
)


asymmetric_hard_brier = calc_brier(
    oof[TARGET_COL],
    oof["asymmetric_hard_pred"],
)


print("\n")
print("#" * 80)
print("FINAL ENSEMBLE RESULT")
print("#" * 80)


print(
    f"\nGlobal Group-ES : "
    f"{global_group_brier:.8f}"
)

print(
    f"Split-only      : "
    f"{split_group_brier:.8f}"
)

print(
    f"Asymmetric Hard : "
    f"{asymmetric_hard_brier:.8f}"
)

print(
    f"Weighted Ensemble: "
    f"{ensemble_brier:.8f}"
)


print(
    f"\nImprovement vs Global Group-ES : "
    f"{global_group_brier - ensemble_brier:+.8f}"
)

print(
    f"Improvement vs Split-only      : "
    f"{split_group_brier - ensemble_brier:+.8f}"
)


print("\n[Final Weights]")

print(
    f"R = Global-R "
    f"{best_r_global_weight:.2f} "
    f"+ R-only "
    f"{best_r_split_weight:.2f}"
)

print(
    f"F = Global-F "
    f"{best_f_global_weight:.2f} "
    f"+ F-only "
    f"{best_f_split_weight:.2f}"
)


# =========================================================
# 시즌별 결과
# =========================================================
season_rows = []


for season in sorted(
    oof[SEASON_COL].unique()
):

    season_df = oof[
        oof[SEASON_COL] == season
    ].copy()


    global_brier = calc_brier(
        season_df[TARGET_COL],
        season_df[
            "global_group_pred"
        ],
    )


    split_brier = calc_brier(
        season_df[TARGET_COL],
        season_df[
            "split_group_pred"
        ],
    )


    hard_brier = calc_brier(
        season_df[TARGET_COL],
        season_df[
            "asymmetric_hard_pred"
        ],
    )


    ens_brier = calc_brier(
        season_df[TARGET_COL],
        season_df[
            "ensemble_pred"
        ],
    )


    # -----------------------------------------------------
    # R season
    # -----------------------------------------------------
    season_r = season_df[
        season_df[GROUP_COL]
        == R_GROUP
    ]


    r_global = calc_brier(
        season_r[TARGET_COL],
        season_r[
            "global_R_pred"
        ],
    )


    r_only = calc_brier(
        season_r[TARGET_COL],
        season_r[
            "R_only_pred"
        ],
    )


    r_ensemble = calc_brier(
        season_r[TARGET_COL],

        weighted_prediction(
            global_pred=season_r[
                "global_R_pred"
            ],

            split_pred=season_r[
                "R_only_pred"
            ],

            global_weight=(
                best_r_global_weight
            ),
        ),
    )


    # -----------------------------------------------------
    # F season
    # -----------------------------------------------------
    season_f = season_df[
        season_df[GROUP_COL]
        == F_GROUP
    ]


    f_global = calc_brier(
        season_f[TARGET_COL],
        season_f[
            "global_F_pred"
        ],
    )


    f_only = calc_brier(
        season_f[TARGET_COL],
        season_f[
            "F_only_pred"
        ],
    )


    f_ensemble = calc_brier(
        season_f[TARGET_COL],

        weighted_prediction(
            global_pred=season_f[
                "global_F_pred"
            ],

            split_pred=season_f[
                "F_only_pred"
            ],

            global_weight=(
                best_f_global_weight
            ),
        ),
    )


    season_rows.append({

        "season":
            int(season),

        "global_group":
            global_brier,

        "split_only":
            split_brier,

        "asymmetric_hard":
            hard_brier,

        "ensemble":
            ens_brier,

        "ensemble_delta_vs_global":
            global_brier - ens_brier,

        "R_global":
            r_global,

        "R_only":
            r_only,

        "R_ensemble":
            r_ensemble,

        "F_global":
            f_global,

        "F_only":
            f_only,

        "F_ensemble":
            f_ensemble,
    })


season_summary = pd.DataFrame(
    season_rows
)


print("\n[Season Summary]")

print(
    season_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =========================================================
# 시즌별 승리 수
# =========================================================
season_wins = int(
    (
        season_summary[
            "ensemble_delta_vs_global"
        ]
        > 0
    ).sum()
)


print(
    f"\nEnsemble wins vs "
    f"Global Group-ES: "
    f"{season_wins}/"
    f"{len(season_summary)} seasons"
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


season_summary.to_csv(
    VALIDATION_DIR
    / "season_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


r_search.to_csv(
    VALIDATION_DIR
    / "R_weight_search.csv",
    index=False,
    encoding="utf-8-sig",
)


f_search.to_csv(
    VALIDATION_DIR
    / "F_weight_search.csv",
    index=False,
    encoding="utf-8-sig",
)


metadata = {

    "experiment_name":
        EXPERIMENT_NAME,

    "source_experiment":
        SOURCE_EXPERIMENT_NAME,

    "weight_definition":
        (
            "prediction = "
            "global_weight * global "
            "+ "
            "(1-global_weight) "
            "* group_only"
        ),

    "weight_step":
        WEIGHT_STEP,

    "R_global_weight":
        best_r_global_weight,

    "R_only_weight":
        best_r_split_weight,

    "F_global_weight":
        best_f_global_weight,

    "F_only_weight":
        best_f_split_weight,

    "global_group_oof_brier":
        global_group_brier,

    "split_only_oof_brier":
        split_group_brier,

    "asymmetric_hard_oof_brier":
        asymmetric_hard_brier,

    "ensemble_oof_brier":
        ensemble_brier,

    "ensemble_improvement_vs_global":
        (
            global_group_brier
            - ensemble_brier
        ),

    "season_wins_vs_global":
        season_wins,
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
    / "season_summary.csv"
)

print(
    VALIDATION_DIR
    / "R_weight_search.csv"
)

print(
    VALIDATION_DIR
    / "F_weight_search.csv"
)

print(
    VALIDATION_DIR
    / "metadata.json"
)


print("\nExperiment complete.")