# script.py

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

EXPERIMENT_NAME = "cb_reg_v003_exact_pc"

MODEL_PATH = (
    Path("./model")
    / f"{EXPERIMENT_NAME}.pkl"
)

OUTPUT_DIR = Path("./output")
OUTPUT_PATH = (
    OUTPUT_DIR
    / "submission.csv"
)

ID = "row_id"
TARGET = "control_success"
SEASON_COL = "season"
GAME_TYPE_COL = "game_type"

EXACT_COLS = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]

# 기존에 이미 학습한 artifact를 파일명만 바꿔 제출해도
# 재학습 없이 사용할 수 있도록 이전 experiment_name도 허용.
ALLOWED_ARTIFACT_NAMES = {
    "cb_reg_v003_exact_pc",
    "cb_reg_v003_exact_pc_full_seed42",
}


# ============================================================
# Data directory resolver
# ============================================================

def resolve_data_dir() -> Path:
    """
    공식 가이드의 ./data/를 우선 사용한다.

    안내문 마지막에 open/ 표기가 섞여 있으므로,
    실제 환경에 open/만 존재하는 경우에 한해 fallback한다.
    """
    candidates = [
        Path("./data"),
        Path("./open"),
    ]

    for directory in candidates:
        if (
            (directory / "test.csv").exists()
            and (
                directory
                / "sample_submission.csv"
            ).exists()
        ):
            return directory

    raise FileNotFoundError(
        "test.csv와 sample_submission.csv를 찾지 못했습니다. "
        "확인 경로: ./data/, ./open/"
    )


# ============================================================
# Common feature preprocessing
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    cat_cols,
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
# Exact-ASOF
# ============================================================

def add_pitcher_exact_asof_features(
    df: pd.DataFrame,
    pitcher_season_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    검증/기존 cb_reg_v003_exact script와 동일한 Exact-ASOF.

    각 test row의 official cumulative ASOF에서,
    artifact에 저장된 target season 이전 완료 train 시즌 통계만 뺀다.

    같은 test season의 다른 row 값은 사용하지 않는다.
    """

    required = {
        SEASON_COL,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Exact-ASOF 필수 test 컬럼 없음: "
            f"{missing}"
        )

    out = df.copy()

    for col in EXACT_COLS:
        out[col] = np.nan

    seasons = (
        pd.to_numeric(
            out[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    stats = pitcher_season_stats.copy()

    stats[SEASON_COL] = (
        pd.to_numeric(
            stats[SEASON_COL],
            errors="raise",
        )
        .astype(int)
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
            "artifact pitcher_season_stats 컬럼 누락: "
            f"{missing_stats}"
        )

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
                    [
                        "n",
                        "success",
                    ]
                ]
                .sum()
            )

            hist_n = (
                rows[
                    "pitcher_id"
                ]
                .map(
                    hist_agg["n"]
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float64
                )
            )

            hist_success = (
                rows[
                    "pitcher_id"
                ]
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

        current_n = np.maximum(
            asof_n
            - hist_n,
            0.0,
        )

        current_success = (
            asof_success
            - hist_success
        )

        current_success = np.where(
            np.isfinite(
                current_success
            ),
            np.clip(
                current_success,
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
# HB-EB
# ============================================================

def apply_hbeb(
    df: pd.DataFrame,
    group_cols,
    n_col: str,
    rate_col: str,
    prior_df: pd.DataFrame,
    global_prior,
):
    """
    artifact에 저장된 frozen train prior를 row별 lookup한다.
    test row끼리 통계를 만들지 않는다.
    """

    lookup = (
        df[
            group_cols
        ]
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

    denominator = (
        n
        + alpha
        + beta
    )

    return np.divide(
        n * rate
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


def apply_all_hbeb(
    df: pd.DataFrame,
    hbeb_priors,
) -> pd.DataFrame:
    out = df.copy()

    for (
        new_col,
        config,
    ) in hbeb_priors.items():

        out[
            new_col
        ] = apply_hbeb(
            df=out,
            group_cols=list(
                config[
                    "group_cols"
                ]
            ),
            n_col=config[
                "n_col"
            ],
            rate_col=config[
                "rate_col"
            ],
            prior_df=config[
                "prior_df"
            ],
            global_prior=config[
                "global_prior"
            ],
        )

    return out


# ============================================================
# PC contexts
# ============================================================

def make_pc_context_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    검증 v021의 _ctx_frame_v15과 동일한 context 정의.
    """

    x = frame.copy()

    x[
        "v15_count"
    ] = (
        x[
            "balls_before"
        ]
        .astype(str)
        + "_"
        + x[
            "strikes_before"
        ]
        .astype(str)
    )

    x[
        "v15_inning_phase"
    ] = pd.cut(
        pd.to_numeric(
            x["inning"],
            errors="coerce",
        ),
        [
            -np.inf,
            3,
            6,
            9,
            np.inf,
        ],
        labels=[
            "early",
            "middle",
            "late",
            "extra",
        ],
    ).astype(str)

    x[
        "v15_pressure"
    ] = np.where(
        (
            x[
                "balls_before"
            ]
            == 3
        )
        & (
            x[
                "strikes_before"
            ]
            == 2
        ),
        "full",
        np.where(
            x[
                "balls_before"
            ]
            == 3,
            "three_ball",
            np.where(
                x[
                    "strikes_before"
                ]
                == 2,
                "two_strike",
                "normal",
            ),
        ),
    )

    x[
        "v15_runners"
    ] = np.where(
        pd.to_numeric(
            x[
                "num_runners_on"
            ],
            errors="coerce",
        )
        .fillna(0)
        > 0,
        "on",
        "empty",
    )

    x[
        "v15_li_bucket"
    ] = pd.cut(
        pd.to_numeric(
            x["li"],
            errors="coerce",
        ),
        [
            -np.inf,
            0.7,
            1.5,
            np.inf,
        ],
        labels=[
            "low",
            "mid",
            "high",
        ],
    ).astype(str)

    return x


# ============================================================
# Frozen PC lookup
# ============================================================

def pc_lookup(
    table,
    rows: pd.DataFrame,
    keys,
    field: str,
    default: float = 0.0,
):
    """
    검증 v021의 _pc_lookup과 동일.
    unseen key는 0.0.
    """

    if (
        table is None
        or len(table) == 0
    ):
        return np.full(
            len(rows),
            default,
            dtype=np.float32,
        )

    if len(keys) == 1:
        series = (
            table
            .set_index(
                keys[0]
            )[
                field
            ]
        )

        return (
            rows[
                keys[0]
            ]
            .map(series)
            .fillna(default)
            .to_numpy(
                dtype=np.float32
            )
        )

    idx = (
        pd.MultiIndex
        .from_frame(
            rows[
                keys
            ]
        )
    )

    series = (
        table
        .set_index(
            keys
        )[
            field
        ]
    )

    return (
        series
        .reindex(idx)
        .fillna(default)
        .to_numpy(
            dtype=np.float32
        )
    )


def build_pc_features(
    test: pd.DataFrame,
    treatment_artifact,
) -> pd.DataFrame:
    """
    각 target season y row는 artifact에 저장된
    source season y-1의 train lookup table만 사용한다.

    test row끼리 어떠한 통계도 만들지 않는다.
    """

    required = {
        SEASON_COL,
        GAME_TYPE_COL,
        "pitcher_id",
        "batter_hand",
        "base_state",
        "balls_before",
        "strikes_before",
        "inning",
        "num_runners_on",
        "li",
    }

    missing = sorted(
        required
        - set(test.columns)
    )

    if missing:
        raise KeyError(
            "PC 필수 test 컬럼 없음: "
            f"{missing}"
        )

    pc_features = list(
        treatment_artifact[
            "pc_features"
        ]
    )

    pc_fields = list(
        treatment_artifact[
            "pc_fields"
        ]
    )

    pc_contexts = [
        item["context"]
        for item in (
            treatment_artifact[
                "pc_contexts"
            ]
        )
    ]

    tables_by_source_season = (
        treatment_artifact[
            "pc_tables_by_source_season"
        ]
    )

    default = float(
        treatment_artifact.get(
            "pc_unseen_default",
            0.0,
        )
    )

    rows = (
        make_pc_context_frame(
            test.reset_index(
                drop=True
            )
        )
    )

    result = pd.DataFrame(
        0.0,
        index=np.arange(
            len(rows)
        ),
        columns=pc_features,
        dtype=np.float32,
    )

    seasons = (
        pd.to_numeric(
            rows[SEASON_COL],
            errors="raise",
        )
        .astype(int)
    )

    for target_year in sorted(
        seasons.unique()
    ):
        target_mask = (
            seasons
            .eq(
                int(target_year)
            )
            .to_numpy()
        )

        positions = np.flatnonzero(
            target_mask
        )

        rr = (
            rows.loc[
                target_mask
            ]
            .reset_index(
                drop=True
            )
        )

        source_year = (
            int(target_year)
            - 1
        )

        # joblib은 int key를 보존하지만,
        # 혹시 문자열 key로 변환된 경우까지 허용.
        season_tables = (
            tables_by_source_season
            .get(
                source_year
            )
        )

        if season_tables is None:
            season_tables = (
                tables_by_source_season
                .get(
                    str(
                        source_year
                    )
                )
            )

        for context in pc_contexts:
            table = None

            if season_tables is not None:
                table = (
                    season_tables
                    .get(context)
                )

            keys = [
                "pitcher_id",
                context,
            ]

            prefix = (
                f"pc__{context}"
            )

            for field in pc_fields:
                col = (
                    f"{prefix}__{field}"
                )

                if col not in result.columns:
                    raise RuntimeError(
                        "PC artifact schema mismatch: "
                        f"{col}"
                    )

                result.loc[
                    positions,
                    col,
                ] = pc_lookup(
                    table=table,
                    rows=rr,
                    keys=keys,
                    field=field,
                    default=default,
                )

    # 검증과 동일하게 F에서는 PC vector = 0.
    game_type = (
        test[
            GAME_TYPE_COL
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    f_mask = (
        game_type
        == "F"
    )

    result.loc[
        f_mask,
        pc_features,
    ] = 0.0

    values = (
        result[
            pc_features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "PC feature NaN/Inf"
        )

    if f_mask.any():
        if not np.all(
            result.loc[
                f_mask,
                pc_features,
            ]
            .to_numpy(
                dtype=np.float64
            )
            == 0.0
        ):
            raise RuntimeError(
                "F PC masking failed"
            )

    result.index = (
        test.index
    )

    return result


# ============================================================
# Build CONTROL / Treatment frames
# ============================================================

def build_backbone_frame(
    test: pd.DataFrame,
    model_artifact,
) -> pd.DataFrame:
    out = (
        add_pitcher_exact_asof_features(
            df=test,
            pitcher_season_stats=(
                model_artifact[
                    "pitcher_season_stats"
                ]
            ),
        )
    )

    out = apply_all_hbeb(
        df=out,
        hbeb_priors=(
            model_artifact[
                "hbeb_priors"
            ]
        ),
    )

    return out


# ============================================================
# Submission
# ============================================================

def build_submission(
    sample_submission: pd.DataFrame,
    test_ids,
    predictions,
) -> pd.DataFrame:
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            "prediction NaN/Inf"
        )

    if not (
        (
            predictions >= 0.0
        )
        & (
            predictions <= 1.0
        )
    ).all():
        raise ValueError(
            "prediction outside [0,1]"
        )

    pred_df = pd.DataFrame({
        ID: test_ids,
        TARGET: predictions,
    })

    if (
        pred_df[ID]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "prediction row_id 중복"
        )

    if ID not in sample_submission.columns:
        raise KeyError(
            "sample_submission에 row_id가 없습니다."
        )

    submission = (
        sample_submission[
            [ID]
        ]
        .merge(
            pred_df,
            on=ID,
            how="left",
            validate="one_to_one",
        )
    )

    if (
        submission[TARGET]
        .isna()
        .any()
    ):
        raise ValueError(
            "submission prediction NaN"
        )

    if len(submission) != len(
        sample_submission
    ):
        raise ValueError(
            "submission row count mismatch"
        )

    return submission


# ============================================================
# Main
# ============================================================

def main():
    started = time.time()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    data_dir = (
        resolve_data_dir()
    )

    test_path = (
        data_dir
        / "test.csv"
    )

    sample_sub_path = (
        data_dir
        / "sample_submission.csv"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 artifact 없음: "
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load artifact
    # --------------------------------------------------------

    artifact = joblib.load(
        MODEL_PATH
    )

    artifact_name = (
        artifact.get(
            "experiment_name"
        )
    )

    if (
        artifact_name
        not in ALLOWED_ARTIFACT_NAMES
    ):
        raise ValueError(
            "예상하지 못한 artifact experiment_name: "
            f"{artifact_name}"
        )

    if (
        int(
            artifact.get(
                "random_seed",
                -1,
            )
        )
        != 42
    ):
        raise ValueError(
            "이번 checkpoint artifact seed가 42가 아닙니다."
        )

    required_top = {
        "control",
        "treatment",
        "routing",
    }

    missing_top = sorted(
        required_top
        - set(artifact.keys())
    )

    if missing_top:
        raise KeyError(
            "routed artifact key 누락: "
            f"{missing_top}"
        )

    routing = artifact[
        "routing"
    ]

    if (
        routing.get(
            "R_model"
        )
        != "treatment"
        or routing.get(
            "F_model"
        )
        != "control"
    ):
        raise ValueError(
            "artifact routing policy 불일치"
        )

    control_wrapper = (
        artifact[
            "control"
        ]
    )

    control_artifact = (
        control_wrapper[
            "artifact"
        ]
    )

    treatment_artifact = (
        artifact[
            "treatment"
        ]
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    sample_submission = (
        pd.read_csv(
            sample_sub_path,
            encoding="utf-8-sig",
        )
    )

    required_test = {
        ID,
        SEASON_COL,
        GAME_TYPE_COL,
    }

    missing_test = sorted(
        required_test
        - set(test.columns)
    )

    if missing_test:
        raise KeyError(
            "test 필수 컬럼 누락: "
            f"{missing_test}"
        )

    test_ids = (
        test[ID]
        .to_numpy()
    )

    # --------------------------------------------------------
    # CONTROL
    # --------------------------------------------------------

    control_model = (
        control_artifact[
            "model"
        ]
    )

    control_features = list(
        control_artifact[
            "features"
        ]
    )

    control_cat_cols = list(
        control_artifact[
            "cat_cols"
        ]
    )

    if (
        "asof_pitcher_pitchmix_n"
        not in control_features
    ):
        raise RuntimeError(
            "canonical control feature "
            "asof_pitcher_pitchmix_n 누락"
        )

    for col in EXACT_COLS:
        if col not in control_features:
            raise RuntimeError(
                f"CONTROL Exact feature 누락: {col}"
            )

    control_frame = (
        build_backbone_frame(
            test=test,
            model_artifact=(
                control_artifact
            ),
        )
    )

    missing_control = [
        col
        for col in control_features
        if col
        not in control_frame.columns
    ]

    if missing_control:
        raise KeyError(
            "CONTROL test feature 누락: "
            f"{missing_control}"
        )

    X_control = (
        prepare_features(
            control_frame[
                control_features
            ],
            control_cat_cols,
        )
    )

    if hasattr(
        control_model,
        "feature_names_",
    ):
        if (
            list(
                control_model.feature_names_
            )
            != list(
                X_control.columns
            )
        ):
            raise ValueError(
                "CONTROL 학습/추론 feature 순서 불일치"
            )

    control_pred = np.asarray(
        control_model.predict(
            X_control
        ),
        dtype=np.float64,
    )

    if not np.isfinite(
        control_pred
    ).all():
        raise ValueError(
            "CONTROL raw prediction NaN/Inf"
        )

    control_pred = np.clip(
        control_pred,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # TREATMENT
    # --------------------------------------------------------

    treatment_model = (
        treatment_artifact[
            "model"
        ]
    )

    treatment_features = list(
        treatment_artifact[
            "features"
        ]
    )

    treatment_cat_cols = list(
        treatment_artifact[
            "cat_cols"
        ]
    )

    pc_features = list(
        treatment_artifact[
            "pc_features"
        ]
    )

    # Exact A/B backbone guard.
    treatment_backbone = list(
        treatment_artifact[
            "control_backbone_features"
        ]
    )

    if (
        treatment_backbone
        != control_features
    ):
        raise RuntimeError(
            "Treatment CONTROL backbone != existing CONTROL"
        )

    if (
        treatment_features
        != (
            control_features
            + pc_features
        )
    ):
        raise RuntimeError(
            "Treatment feature schema != CONTROL + PC"
        )

    treatment_frame = (
        build_backbone_frame(
            test=test,
            model_artifact=(
                treatment_artifact
            ),
        )
    )

    pc_test = (
        build_pc_features(
            test=test,
            treatment_artifact=(
                treatment_artifact
            ),
        )
    )

    for col in pc_features:
        if col in treatment_frame.columns:
            raise RuntimeError(
                f"PC feature duplicate: {col}"
            )

        treatment_frame[
            col
        ] = pc_test[
            col
        ]

    missing_treatment = [
        col
        for col in treatment_features
        if col
        not in treatment_frame.columns
    ]

    if missing_treatment:
        raise KeyError(
            "TREATMENT test feature 누락: "
            f"{missing_treatment}"
        )

    X_treatment = (
        prepare_features(
            treatment_frame[
                treatment_features
            ],
            treatment_cat_cols,
        )
    )

    if hasattr(
        treatment_model,
        "feature_names_",
    ):
        if (
            list(
                treatment_model.feature_names_
            )
            != list(
                X_treatment.columns
            )
        ):
            raise ValueError(
                "TREATMENT 학습/추론 feature 순서 불일치"
            )

    treatment_pred = np.asarray(
        treatment_model.predict(
            X_treatment
        ),
        dtype=np.float64,
    )

    if not np.isfinite(
        treatment_pred
    ).all():
        raise ValueError(
            "TREATMENT raw prediction NaN/Inf"
        )

    treatment_pred = np.clip(
        treatment_pred,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # R/F routing
    # --------------------------------------------------------

    game_type = (
        test[
            GAME_TYPE_COL
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
        .to_numpy()
    )

    f_mask = (
        game_type
        == "F"
    )

    r_mask = ~f_mask

    pred = (
        control_pred.copy()
    )

    pred[
        r_mask
    ] = (
        treatment_pred[
            r_mask
        ]
    )

    # F는 기존 CONTROL을 정확히 보호.
    if not np.array_equal(
        pred[
            f_mask
        ],
        control_pred[
            f_mask
        ],
    ):
        raise RuntimeError(
            "F routing protection failed"
        )

    if not np.isfinite(
        pred
    ).all():
        raise ValueError(
            "final prediction NaN/Inf"
        )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # submission.csv
    # --------------------------------------------------------

    submission = (
        build_submission(
            sample_submission=(
                sample_submission
            ),
            test_ids=test_ids,
            predictions=pred,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Log only. No extra output files.
    # --------------------------------------------------------

    print("=" * 80)
    print(
        f"Inference : {EXPERIMENT_NAME}"
    )
    print(
        f"artifact  : {artifact_name}"
    )
    print(
        f"data dir  : {data_dir}"
    )
    print(
        f"rows      : {len(test):,}"
    )
    print(
        f"R rows    : {int(r_mask.sum()):,}"
    )
    print(
        f"F rows    : {int(f_mask.sum()):,}"
    )
    print(
        f"CONTROL f : {len(control_features)}"
    )
    print(
        f"TREAT f   : {len(treatment_features)}"
    )
    print(
        f"PC f      : {len(pc_features)}"
    )
    print(
        f"pred min  : {pred.min():.6f}"
    )
    print(
        f"pred mean : {pred.mean():.6f}"
    )
    print(
        f"pred max  : {pred.max():.6f}"
    )
    print(
        f"output    : {OUTPUT_PATH}"
    )
    print(
        f"time      : {time.time() - started:.2f}s"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
