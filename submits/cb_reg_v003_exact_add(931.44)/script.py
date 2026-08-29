# script.py
#
# cb_reg_v003_exact_add inference
#
# train.py artifact:
#   - CatBoostRegressor
#   - Exact-ASOF 5
#   - Self Interaction 9
#   - frozen HB-EB priors
#   - frozen 2024 OOF conditional residual tables
#
# 규칙:
# - test.csv의 각 행은 독립적으로 예측한다.
# - 다른 test row의 값/통계/빈도/순위/rolling/lag를 사용하지 않는다.
# - Exact-ASOF는 artifact에 저장된 공식 train 통계 + 현재 row만 사용한다.
# - HB-EB는 artifact에 저장된 frozen prior + 현재 row만 사용한다.
# - Residual correction은 artifact에 저장된 frozen table lookup만 사용한다.
# - test groupby / aggregation / fitting은 일절 수행하지 않는다.


from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "cb_reg_v003_exact_add"

DATA_DIR = Path("./data")

MODEL_ROOT = Path("./model")
MODEL_DIR = MODEL_ROOT / EXPERIMENT_NAME
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"

VALIDATION_ROOT = Path("./validation")
VALIDATION_DIR = VALIDATION_ROOT / EXPERIMENT_NAME

OUTPUT_DIR = Path("./output")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


# ============================================================
# 기본 컬럼
# ============================================================

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"


# ============================================================
# Exact-ASOF
# train.py와 동일
# ============================================================

EXACT_FEATURES = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


# ============================================================
# Self Interaction
# train.py와 동일
# ============================================================

SELF_INTERACTION_VERSION = "SELF_INTERACTION_V1"

SELF_INTERACTION_FEATURES = [
    "self_recent_success_blend",
    "self_recent_success_minus_career",
    "self_recent_middle_blend",
    "self_recent_middle_trend",
    "self_count_pressure",
    "self_career_x_pressure",
    "self_recent_x_pressure",
    "self_offspeed_x_same_hand",
    "self_win_uncertainty",
]


# ============================================================
# 공통
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> pd.DataFrame:

    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


# ============================================================
# Artifact 검증
# ============================================================

def validate_artifact(
    artifact: dict,
) -> None:

    required = {
        "experiment_name",
        "model",
        "features",
        "cat_cols",
        "hbeb_priors",
        "pitcher_season_stats",
        "exact_features",
        "self_interaction_version",
        "self_interaction_features",
        "residual_correction",
    }

    missing = sorted(
        required - set(artifact)
    )

    if missing:
        raise KeyError(
            f"Artifact 필수 key 누락: {missing}"
        )

    if (
        artifact["experiment_name"]
        != EXPERIMENT_NAME
    ):
        raise ValueError(
            "Artifact experiment 불일치: "
            f"{artifact['experiment_name']} "
            f"!= {EXPERIMENT_NAME}"
        )

    if (
        artifact.get(
            "uses_test_distribution",
            False,
        )
    ):
        raise RuntimeError(
            "Artifact가 test distribution 사용으로 표시되어 있습니다."
        )

    if not artifact.get(
        "row_independent_inference",
        False,
    ):
        raise RuntimeError(
            "Artifact의 row-independent inference "
            "보장이 확인되지 않습니다."
        )

    if (
        list(
            artifact["exact_features"]
        )
        != EXACT_FEATURES
    ):
        raise RuntimeError(
            "Exact-ASOF feature 정의가 "
            "script.py와 일치하지 않습니다."
        )

    if (
        artifact[
            "self_interaction_version"
        ]
        != SELF_INTERACTION_VERSION
    ):
        raise RuntimeError(
            "Self Interaction version 불일치"
        )

    if (
        list(
            artifact[
                "self_interaction_features"
            ]
        )
        != SELF_INTERACTION_FEATURES
    ):
        raise RuntimeError(
            "Self Interaction feature 목록 불일치"
        )


# ============================================================
# Exact-ASOF
# train.py와 동일한 계산
# ============================================================

def add_pitcher_exact_asof_features(
    df: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    현재 row의 official cumulative ASOF에서
    해당 row season보다 과거인 공식 train 완료 시즌 통계를 빼
    현재 시즌 직전 상태를 복원한다.

    다른 test row의 target/feature/statistic은 사용하지 않는다.

    구현은 season별 batch 처리하지만,
    각 row의 결과는 오직:
        row.season
        row.pitcher_id
        row.asof_pitcher_n
        row.asof_pitcher_success_rate
        frozen train stats
    에 의해서만 결정된다.
    """

    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Exact-ASOF source 누락: {missing}"
        )

    out = df.copy()

    for col in EXACT_FEATURES:
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = (
        pitcher_season_stats
        .copy()
    )

    required_stats = {
        "pitcher_id",
        SEASON_COL,
        "n",
        "success",
    }

    missing_stats = sorted(
        required_stats
        - set(stats.columns)
    )

    if missing_stats:
        raise KeyError(
            "pitcher_season_stats 누락: "
            f"{missing_stats}"
        )

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats["n"] = pd.to_numeric(
        stats["n"],
        errors="raise",
    ).astype(np.float64)

    stats["success"] = pd.to_numeric(
        stats["success"],
        errors="raise",
    ).astype(np.float64)

    # 각 target season에 대한 lookup은
    # frozen train stats만으로 생성된다.
    for target_year in sorted(
        seasons.unique()
    ):

        row_mask = (
            seasons
            == int(target_year)
        )

        rows = out.loc[
            row_mask
        ]

        hist = stats.loc[
            stats[SEASON_COL]
            < int(target_year)
        ]

        if len(hist) > 0:

            hist_agg = (
                hist
                .groupby(
                    "pitcher_id",
                    sort=False,
                    dropna=False,
                )[
                    ["n", "success"]
                ]
                .sum()
            )

            hist_n = (
                rows["pitcher_id"]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                rows["pitcher_id"]
                .map(
                    hist_agg["success"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

        else:

            hist_n = np.zeros(
                len(rows),
                dtype=np.float64,
            )

            hist_success = np.zeros(
                len(rows),
                dtype=np.float64,
            )

        asof_n = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_n"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_rate = (
            pd.to_numeric(
                rows[
                    "asof_pitcher_success_rate"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        asof_success = (
            asof_n
            * asof_rate
        )

        raw_current_n = (
            asof_n
            - hist_n
        )

        current_n = np.maximum(
            raw_current_n,
            0.0,
        )

        raw_current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                raw_current_success
            ),
            np.clip(
                raw_current_success,
                0.0,
                current_n,
            ),
            np.nan,
        )

        current_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_current = (
            (current_n > 0.0)
            & np.isfinite(
                current_success
            )
        )

        current_rate[
            valid_current
        ] = (
            current_success[
                valid_current
            ]
            / current_n[
                valid_current
            ]
        )

        past_rate = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        valid_past = (
            hist_n > 0.0
        )

        past_rate[
            valid_past
        ] = (
            hist_success[
                valid_past
            ]
            / hist_n[
                valid_past
            ]
        )

        idx = rows.index

        out.loc[
            idx,
            "exact_pitcher_season_n",
        ] = current_n

        out.loc[
            idx,
            "exact_pitcher_season_logn",
        ] = np.log1p(
            current_n
        )

        out.loc[
            idx,
            "exact_pitcher_season_success_rate",
        ] = current_rate

        out.loc[
            idx,
            "exact_pitcher_past_success_rate",
        ] = past_rate

        out.loc[
            idx,
            "exact_pitcher_season_minus_past",
        ] = (
            current_rate
            - past_rate
        )

    return out


# ============================================================
# Self Interaction
# train.py와 동일
# ============================================================

def add_self_interaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    현재 row 자체의 공식 ASOF + 현재 경기 상태만 사용한다.
    다른 test row는 사용하지 않는다.
    """

    required = {
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_offspeed_rate",
        "home_win_expectancy",
        "away_win_expectancy",
        "top_bottom",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Self Interaction source 누락: "
            f"{missing}"
        )

    out = df.copy()

    career = pd.to_numeric(
        out[
            "asof_pitcher_success_rate"
        ],
        errors="coerce",
    )

    p1 = pd.to_numeric(
        out[
            "asof_pitcher_prev1_game_success_rate"
        ],
        errors="coerce",
    ).fillna(
        career
    )

    p3 = pd.to_numeric(
        out[
            "asof_pitcher_prev3_game_success_rate"
        ],
        errors="coerce",
    ).fillna(
        p1
    )

    p5 = pd.to_numeric(
        out[
            "asof_pitcher_prev5_game_success_rate"
        ],
        errors="coerce",
    ).fillna(
        p3
    )

    recent = (
        0.5 * p1
        + 0.3 * p3
        + 0.2 * p5
    )

    middle = pd.to_numeric(
        out[
            "asof_pitcher_middle_rate"
        ],
        errors="coerce",
    )

    m1 = pd.to_numeric(
        out[
            "asof_pitcher_prev1_game_middle_rate"
        ],
        errors="coerce",
    ).fillna(
        middle
    )

    m3 = pd.to_numeric(
        out[
            "asof_pitcher_prev3_game_middle_rate"
        ],
        errors="coerce",
    ).fillna(
        m1
    )

    m5 = pd.to_numeric(
        out[
            "asof_pitcher_prev5_game_middle_rate"
        ],
        errors="coerce",
    ).fillna(
        m3
    )

    middle_recent = (
        0.5 * m1
        + 0.3 * m3
        + 0.2 * m5
    )

    balls = pd.to_numeric(
        out["balls_before"],
        errors="coerce",
    ).fillna(
        0.0
    )

    strikes = pd.to_numeric(
        out["strikes_before"],
        errors="coerce",
    ).fillna(
        0.0
    )

    pressure = (
        0.7
        * (
            balls == 3
        ).astype(
            np.float64
        )
        + 1.1
        * (
            (balls == 3)
            & (strikes == 2)
        ).astype(
            np.float64
        )
        + 0.25
        * (
            strikes == 2
        ).astype(
            np.float64
        )
    )

    same_hand = (
        out[
            "pitcher_hand"
        ].astype(str)
        ==
        out[
            "batter_hand"
        ].astype(str)
    ).astype(
        np.float64
    )

    offspeed = pd.to_numeric(
        out[
            "asof_pitcher_offspeed_rate"
        ],
        errors="coerce",
    )

    hwe = (
        pd.to_numeric(
            out[
                "home_win_expectancy"
            ],
            errors="coerce",
        )
        / 100.0
    )

    awe = (
        pd.to_numeric(
            out[
                "away_win_expectancy"
            ],
            errors="coerce",
        )
        / 100.0
    )

    pitcher_we = pd.Series(
        np.where(
            out[
                "top_bottom"
            ].astype(str)
            == "T",
            hwe,
            awe,
        ),
        index=out.index,
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    out[
        "self_recent_success_blend"
    ] = recent

    out[
        "self_recent_success_minus_career"
    ] = (
        recent
        - career
    )

    out[
        "self_recent_middle_blend"
    ] = (
        middle_recent
    )

    out[
        "self_recent_middle_trend"
    ] = (
        m1
        - m5
    )

    out[
        "self_count_pressure"
    ] = pressure

    out[
        "self_career_x_pressure"
    ] = (
        career
        * pressure
    )

    out[
        "self_recent_x_pressure"
    ] = (
        recent
        * pressure
    )

    out[
        "self_offspeed_x_same_hand"
    ] = (
        offspeed
        * same_hand
    )

    out[
        "self_win_uncertainty"
    ] = (
        1.0
        - 2.0
        * (
            pitcher_we
            - 0.5
        ).abs()
    )

    return out


# ============================================================
# Frozen HB-EB
# ============================================================

def apply_hbeb(
    df: pd.DataFrame,
    group_cols: list[str],
    n_col: str,
    rate_col: str,
    prior_df: pd.DataFrame,
    global_prior,
):
    """
    artifact에 저장된 train-only prior lookup과
    현재 row n/rate만 사용한다.
    """

    lookup = (
        df[group_cols]
        .reset_index(
            drop=True
        )
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    alpha = (
        lookup["alpha"]
        .fillna(
            global_prior[0]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    beta = (
        lookup["beta"]
        .fillna(
            global_prior[1]
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    n = (
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    success_count = (
        n * rate
    )

    denominator = (
        n
        + alpha
        + beta
    )

    return np.divide(
        success_count
        + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=(
            denominator > 0.0
        ),
    )


def apply_frozen_hbeb(
    df: pd.DataFrame,
    hbeb_priors: dict,
) -> pd.DataFrame:

    out = df.copy()

    for new_col, info in (
        hbeb_priors.items()
    ):

        required_info = {
            "prior_df",
            "global_prior",
            "group_cols",
            "n_col",
            "rate_col",
        }

        missing = sorted(
            required_info
            - set(info)
        )

        if missing:
            raise KeyError(
                f"{new_col} HB-EB artifact 누락: {missing}"
            )

        out[new_col] = apply_hbeb(
            df=out,
            group_cols=list(
                info[
                    "group_cols"
                ]
            ),
            n_col=info[
                "n_col"
            ],
            rate_col=info[
                "rate_col"
            ],
            prior_df=info[
                "prior_df"
            ],
            global_prior=info[
                "global_prior"
            ],
        )

    return out


# ============================================================
# Conditional residual correction
# ============================================================

def make_pressure_state(
    balls,
    strikes,
):
    """
    train.py와 동일.
    """

    balls = pd.to_numeric(
        balls,
        errors="coerce",
    )

    strikes = pd.to_numeric(
        strikes,
        errors="coerce",
    )

    full_mask = (
        (balls == 3)
        & (strikes == 2)
    )

    high_mask = (
        (
            (balls == 3)
            | (strikes == 2)
        )
        & (~full_mask)
    )

    result = np.full(
        len(balls),
        "normal",
        dtype=object,
    )

    result[
        high_mask.to_numpy()
    ] = "high"

    result[
        full_mask.to_numpy()
    ] = "full"

    return result


def lookup_frozen_effect(
    df: pd.DataFrame,
    table: pd.DataFrame,
    group_cols: list[str],
):
    """
    frozen train-derived table을 현재 row key로 조회한다.

    test에서 groupby/statistics를 만들지 않는다.
    """

    if not isinstance(
        table,
        pd.DataFrame,
    ):
        raise TypeError(
            "Residual effect table이 DataFrame이 아닙니다."
        )

    required = set(
        group_cols
    ) | {
        "effect"
    }

    missing = sorted(
        required
        - set(table.columns)
    )

    if missing:
        raise KeyError(
            "Residual table 필수 컬럼 누락: "
            f"{missing}"
        )

    keys = (
        df[group_cols]
        .copy()
    )

    frozen = (
        table[
            group_cols
            + ["effect"]
        ]
        .copy()
    )

    # build_final_residual_artifact()의 OOF에서는
    # pitcher_id / batter_hand가 문자열로 저장됨.
    for col in group_cols:
        if col in {
            "pitcher_id",
            "batter_hand",
        }:
            keys[col] = (
                keys[col]
                .astype(str)
            )

            frozen[col] = (
                frozen[col]
                .astype(str)
            )

    # frozen table 자체는 train에서 unique group으로 생성.
    if frozen.duplicated(
        subset=group_cols
    ).any():
        raise RuntimeError(
            "Frozen residual table에 "
            "중복 key가 있습니다."
        )

    lookup = (
        keys
        .reset_index(
            drop=True
        )
        .merge(
            frozen,
            on=group_cols,
            how="left",
            sort=False,
            validate="many_to_one",
        )
    )

    effect = (
        pd.to_numeric(
            lookup["effect"],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    found = np.isfinite(
        effect
    )

    effect = np.where(
        found,
        effect,
        0.0,
    )

    return (
        effect,
        found,
    )


def apply_conditional_residual_correction(
    rows: pd.DataFrame,
    base_prediction,
    residual_artifact: dict,
):
    """
    [INFERRED APPLICATION FORMULA]

    train.py는 frozen parent/detail effect table과 다음 상수는 저장하지만
    실제 추론 결합 함수는 제시하지 않았다.

    아래는 상수명의 직접적인 의미에 맞춘 구현:

      detail 있음:
          effect =
              parent_weight * parent_effect
              +
              detail_weight * detail_effect

      parent만 있음:
          effect =
              parent_only_scale * parent_effect

      correction =
          clip(alpha * effect, -cap, +cap)

      final =
          clip(base_prediction + correction, 0, 1)

    test 자체로 effect/table/statistic을 다시 fit하지 않는다.
    """

    base_prediction = np.asarray(
        base_prediction,
        dtype=np.float64,
    )

    if not residual_artifact.get(
        "enabled",
        False,
    ):
        return (
            base_prediction.copy(),
            np.zeros(
                len(rows),
                dtype=np.float64,
            ),
            {
                "n_parent": 0,
                "n_detail": 0,
                "n_parent_only": 0,
                "n_none": int(
                    len(rows)
                ),
            },
        )

    if residual_artifact.get(
        "uses_test_distribution",
        False,
    ):
        raise RuntimeError(
            "Residual artifact가 test distribution "
            "사용으로 표시되어 있습니다."
        )

    if not residual_artifact.get(
        "row_independent_application",
        False,
    ):
        raise RuntimeError(
            "Residual artifact row-independent "
            "application 확인 실패"
        )

    required = {
        "parent_group_cols",
        "detail_group_cols",
        "parent_table",
        "detail_table",
        "parent_only_scale",
        "parent_weight",
        "detail_weight",
        "alpha",
        "cap",
    }

    missing = sorted(
        required
        - set(
            residual_artifact
        )
    )

    if missing:
        raise KeyError(
            f"Residual artifact 누락: {missing}"
        )

    work = rows.copy()

    if (
        "pressure_state"
        in residual_artifact[
            "detail_group_cols"
        ]
    ):
        work[
            "pressure_state"
        ] = make_pressure_state(
            work[
                "balls_before"
            ],
            work[
                "strikes_before"
            ],
        )

    parent_cols = list(
        residual_artifact[
            "parent_group_cols"
        ]
    )

    detail_cols = list(
        residual_artifact[
            "detail_group_cols"
        ]
    )

    (
        parent_effect,
        parent_found,
    ) = lookup_frozen_effect(
        df=work,
        table=residual_artifact[
            "parent_table"
        ],
        group_cols=parent_cols,
    )

    (
        detail_effect,
        detail_found,
    ) = lookup_frozen_effect(
        df=work,
        table=residual_artifact[
            "detail_table"
        ],
        group_cols=detail_cols,
    )

    # detail group가 존재하면 그 parent group의 n은
    # 적어도 detail n 이상이므로, 정상 artifact라면
    # parent도 존재해야 한다.
    detail_without_parent = (
        detail_found
        & (~parent_found)
    )

    if np.any(
        detail_without_parent
    ):
        raise RuntimeError(
            "Residual artifact inconsistency: "
            "detail effect는 있으나 parent effect가 없는 row가 있습니다."
        )

    parent_only_scale = float(
        residual_artifact[
            "parent_only_scale"
        ]
    )

    parent_weight = float(
        residual_artifact[
            "parent_weight"
        ]
    )

    detail_weight = float(
        residual_artifact[
            "detail_weight"
        ]
    )

    alpha = float(
        residual_artifact[
            "alpha"
        ]
    )

    cap = float(
        residual_artifact[
            "cap"
        ]
    )

    combined_effect = np.zeros(
        len(work),
        dtype=np.float64,
    )

    detail_mask = (
        parent_found
        & detail_found
    )

    parent_only_mask = (
        parent_found
        & (~detail_found)
    )

    combined_effect[
        detail_mask
    ] = (
        parent_weight
        * parent_effect[
            detail_mask
        ]
        +
        detail_weight
        * detail_effect[
            detail_mask
        ]
    )

    combined_effect[
        parent_only_mask
    ] = (
        parent_only_scale
        * parent_effect[
            parent_only_mask
        ]
    )

    correction = (
        alpha
        * combined_effect
    )

    correction = np.clip(
        correction,
        -cap,
        cap,
    )

    final_prediction = np.clip(
        base_prediction
        + correction,
        0.0,
        1.0,
    )

    diagnostics = {
        "n_rows": int(
            len(work)
        ),
        "n_parent": int(
            parent_found.sum()
        ),
        "n_detail": int(
            detail_found.sum()
        ),
        "n_parent_only": int(
            parent_only_mask.sum()
        ),
        "n_none": int(
            (
                ~parent_found
            ).sum()
        ),
        "mean_correction": float(
            correction.mean()
        )
        if len(correction)
        else 0.0,
        "mean_abs_correction": float(
            np.abs(
                correction
            ).mean()
        )
        if len(correction)
        else 0.0,
        "min_correction": float(
            correction.min()
        )
        if len(correction)
        else 0.0,
        "max_correction": float(
            correction.max()
        )
        if len(correction)
        else 0.0,
    }

    return (
        final_prediction,
        correction,
        diagnostics,
    )


# ============================================================
# 전체 Feature 변환
# ============================================================

def transform_rows(
    rows: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """
    test row 변환.

    1. Exact-ASOF
    2. Self Interaction
    3. Frozen HB-EB
    4. artifact feature order로 정렬
    5. categorical dtype 복원
    """

    transformed = (
        add_pitcher_exact_asof_features(
            rows,
            artifact[
                "pitcher_season_stats"
            ],
        )
    )

    transformed = (
        add_self_interaction_features(
            transformed
        )
    )

    transformed = (
        apply_frozen_hbeb(
            transformed,
            artifact[
                "hbeb_priors"
            ],
        )
    )

    features = list(
        artifact[
            "features"
        ]
    )

    missing = [
        col
        for col in features
        if col not in transformed.columns
    ]

    if missing:
        raise KeyError(
            "Model feature 생성 실패: "
            f"{missing}"
        )

    X = transformed[
        features
    ].copy()

    X = prepare_features(
        X,
        list(
            artifact[
                "cat_cols"
            ]
        ),
    )

    if (
        list(X.columns)
        != features
    ):
        raise RuntimeError(
            "Feature 순서가 artifact와 다릅니다."
        )

    return X


# ============================================================
# Predict
# ============================================================

def predict_rows(
    rows: pd.DataFrame,
    artifact: dict,
):
    """
    어떤 subset을 넣더라도 각 row 예측값은
    동일해야 한다.
    """

    if len(rows) == 0:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    X = transform_rows(
        rows,
        artifact,
    )

    prediction_raw = np.asarray(
        artifact[
            "model"
        ].predict(
            X
        ),
        dtype=np.float64,
    )

    base_prediction = np.clip(
        prediction_raw,
        0.0,
        1.0,
    )

    (
        final_prediction,
        correction,
        residual_diagnostics,
    ) = (
        apply_conditional_residual_correction(
            rows=rows,
            base_prediction=base_prediction,
            residual_artifact=artifact[
                "residual_correction"
            ],
        )
    )

    if not np.all(
        np.isfinite(
            final_prediction
        )
    ):
        raise RuntimeError(
            "최종 prediction에 NaN/Inf가 있습니다."
        )

    if (
        np.any(
            final_prediction < 0.0
        )
        or np.any(
            final_prediction > 1.0
        )
    ):
        raise RuntimeError(
            "최종 prediction이 [0,1] 범위를 벗어났습니다."
        )

    return {
        "prediction_raw": (
            prediction_raw
        ),
        "base_prediction": (
            base_prediction
        ),
        "residual_correction": (
            correction
        ),
        "prediction": (
            final_prediction
        ),
        "residual_diagnostics": (
            residual_diagnostics
        ),
    }


# ============================================================
# Row independence self-check
# ============================================================

def run_row_independence_selfcheck(
    test: pd.DataFrame,
    artifact: dict,
    full_prediction: np.ndarray,
):
    """
    프로젝트 최우선 규칙의 직접 검증.

    같은 row를:
        1) 전체 test 안에서 예측
        2) 한 행만 넣어 예측

    했을 때 동일해야 한다.
    """

    n = len(test)

    if n == 0:
        raise RuntimeError(
            "test.csv가 비어 있습니다."
        )

    positions = sorted(
        set(
            [
                0,
                n // 4,
                n // 2,
                (3 * n) // 4,
                n - 1,
            ]
        )
    )

    max_abs_diff = 0.0

    for pos in positions:

        single = test.iloc[
            [pos]
        ].copy()

        single_result = (
            predict_rows(
                single,
                artifact,
            )
        )

        single_pred = float(
            single_result[
                "prediction"
            ][0]
        )

        batch_pred = float(
            full_prediction[
                pos
            ]
        )

        diff = abs(
            single_pred
            - batch_pred
        )

        max_abs_diff = max(
            max_abs_diff,
            diff,
        )

        if not np.isclose(
            single_pred,
            batch_pred,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Row independence self-check 실패: "
                f"position={pos}, "
                f"single={single_pred:.17g}, "
                f"batch={batch_pred:.17g}, "
                f"diff={diff:.3e}"
            )

    print(
        "[SELF-CHECK] "
        "single-row vs full-batch PASS | "
        f"max_abs_diff={max_abs_diff:.3e}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help=(
            "한 행 단독 예측과 전체 batch 예측이 "
            "동일한지 검사한다."
        ),
    )

    args = parser.parse_args()

    started = time.time()

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(
            SAMPLE_SUB_PATH
        )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    artifact = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "MODEL_PATH artifact가 dict가 아닙니다."
        )

    validate_artifact(
        artifact
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if TARGET in test.columns:
        raise RuntimeError(
            "test.csv에 target이 존재합니다."
        )

    if ID not in test.columns:
        raise KeyError(
            f"test.csv에 {ID}가 없습니다."
        )

    if not test[
        ID
    ].is_unique:
        raise ValueError(
            "test row_id가 unique하지 않습니다."
        )

    # Artifact model이 요구하는 raw source 확인.
    required_sources = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitcher_hand",
    }

    missing_sources = sorted(
        required_sources
        - set(test.columns)
    )

    if missing_sources:
        raise KeyError(
            "test 필수 source 누락: "
            f"{missing_sources}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("=" * 90)
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )
    print(
        f"Model      : {MODEL_PATH}"
    )
    print(
        f"Test rows  : {len(test):,}"
    )
    print(
        "Exact-ASOF : ON"
    )
    print(
        "Self Interaction: ON (9)"
    )
    print(
        "HB-EB      : frozen artifact"
    )
    print(
        "Residual   : frozen conditional lookup"
    )
    print(
        "Test aggregation: NONE"
    )
    print("=" * 90)

    result = predict_rows(
        test,
        artifact,
    )

    final_prediction = (
        result[
            "prediction"
        ]
    )

    # --------------------------------------------------------
    # Row independence check
    # --------------------------------------------------------

    if args.selfcheck:

        run_row_independence_selfcheck(
            test=test,
            artifact=artifact,
            full_prediction=final_prediction,
        )

    # --------------------------------------------------------
    # Sample submission
    # --------------------------------------------------------

    sample = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    if ID not in sample.columns:
        raise KeyError(
            f"sample_submission에 {ID}가 없습니다."
        )

    if TARGET not in sample.columns:
        raise KeyError(
            f"sample_submission에 {TARGET}이 없습니다."
        )

    if not sample[
        ID
    ].is_unique:
        raise ValueError(
            "sample_submission row_id가 unique하지 않습니다."
        )

    if len(sample) != len(test):
        raise ValueError(
            "test/sample_submission 행 수 불일치: "
            f"{len(test)} != {len(sample)}"
        )

    prediction_by_id = pd.Series(
        final_prediction,
        index=test[ID].to_numpy(),
    )

    # 단순 row order를 가정하지 않고 sample row_id에 맞춰 정렬.
    submission = (
        sample.copy()
    )

    submission[
        TARGET
    ] = (
        submission[
            ID
        ]
        .map(
            prediction_by_id
        )
    )

    if submission[
        TARGET
    ].isna().any():

        missing_ids = (
            submission.loc[
                submission[
                    TARGET
                ].isna(),
                ID,
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "sample_submission row_id에 대응하는 "
            f"prediction 없음: {missing_ids}"
        )

    if not np.all(
        np.isfinite(
            submission[
                TARGET
            ].to_numpy(
                dtype=np.float64
            )
        )
    ):
        raise RuntimeError(
            "submission prediction에 NaN/Inf가 있습니다."
        )

    if (
        (
            submission[
                TARGET
            ] < 0.0
        ).any()
        or (
            submission[
                TARGET
            ] > 1.0
        ).any()
    ):
        raise RuntimeError(
            "submission prediction 범위 오류"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = (
        time.time()
        - started
    )

    residual_diag = (
        result[
            "residual_diagnostics"
        ]
    )

    print()
    print("=" * 90)
    print("Inference summary")
    print("=" * 90)

    print(
        "Raw prediction : "
        f"mean={result['prediction_raw'].mean():.8f} "
        f"std={result['prediction_raw'].std():.8f} "
        f"min={result['prediction_raw'].min():.8f} "
        f"max={result['prediction_raw'].max():.8f}"
    )

    print(
        "Base clipped   : "
        f"mean={result['base_prediction'].mean():.8f} "
        f"std={result['base_prediction'].std():.8f}"
    )

    print(
        "Correction     : "
        f"mean={result['residual_correction'].mean():+.8f} "
        f"mean_abs={np.abs(result['residual_correction']).mean():.8f} "
        f"min={result['residual_correction'].min():+.8f} "
        f"max={result['residual_correction'].max():+.8f}"
    )

    print(
        "Final          : "
        f"mean={final_prediction.mean():.8f} "
        f"std={final_prediction.std():.8f} "
        f"min={final_prediction.min():.8f} "
        f"max={final_prediction.max():.8f}"
    )

    print("-" * 90)

    print(
        "Residual parent hit : "
        f"{residual_diag['n_parent']:,}"
    )

    print(
        "Residual detail hit : "
        f"{residual_diag['n_detail']:,}"
    )

    print(
        "Parent only         : "
        f"{residual_diag['n_parent_only']:,}"
    )

    print(
        "No correction       : "
        f"{residual_diag['n_none']:,}"
    )

    print("-" * 90)

    print(
        f"Output         : {OUTPUT_PATH}"
    )

    print(
        f"Elapsed        : {elapsed:.1f}s"
    )

    print(
        "Row-independent: TRUE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()