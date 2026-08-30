# ============================================================
# stable_residual_search_v001.py
#
# 목적
# ------------------------------------------------------------
# cb_reg_v003의 OOF prediction을 이용하여
# "시즌이 바뀌어도 같은 방향으로 반복되는 residual domain"이
# 실제로 존재하는지 탐색한다.
#
# 이번 실험에서는 correction을 적용하지 않는다.
#
# Discovery:
#   2021 / 2022 / 2023
#
# Holdout:
#   2024
#
# 중요:
# ------------------------------------------------------------
# 1. 2024는 residual domain을 선택하는 데 사용하지 않는다.
# 2. 2021~2023에서 안정적인 domain을 먼저 고정한다.
# 3. 그 domain이 2024에서도 같은 방향인지 확인한다.
# 4. test.csv는 사용하지 않는다.
# 5. 모델 재학습도 하지 않는다.
#
# residual 정의
# ------------------------------------------------------------
# residual = y_true - prediction
#
# residual > 0
#   -> 모델이 실제보다 낮게 예측
#   -> under-prediction
#
# residual < 0
#   -> 모델이 실제보다 높게 예측
#   -> over-prediction
#
# season_centered_residual
# ------------------------------------------------------------
# 각 시즌 전체 residual 평균을 제거한다.
#
# 즉,
# "2023 전체적으로 모델이 높게 예측했다"
# 같은 global calibration shift가 아니라
#
# "특정 조건에서 다른 데이터보다 유독 높게/낮게 예측했다"
#
# 를 찾는다.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 경로
# ============================================================

DATA_PATH = Path(
    "./data/train.csv"
)

OOF_PATH = Path(
    "./validation/cb_reg_v003/oof_validation.npz"
)

OUTPUT_DIR = Path(
    "./validation/stable_residual_search_v001"
)


# ============================================================
# 기본 설정
# ============================================================

ID_COL = "row_id"

DISCOVERY_SEASONS = [
    2021,
    2022,
    2023,
]

HOLDOUT_SEASON = 2024


# 각 discovery season에서 최소 몇 row가 있어야
# 해당 group의 residual을 신뢰할 것인가
MIN_N_PER_DISCOVERY_SEASON = 100

# 2024에서도 최소 이 정도 데이터가 있어야
# holdout confirmation 대상으로 판단
MIN_N_HOLDOUT = 100


# ============================================================
# 탐색할 Residual Domain
# ============================================================
#
# train.csv에 실제 존재하는 column만 자동으로 사용한다.
#
# 일부러 무작정 모든 feature 조합을 생성하지 않는다.
# 그렇게 하면 multiple testing / validation overfitting이
# 너무 심해질 수 있기 때문이다.
#
# 참고깃허브 아이디어와 현재 프로젝트에서 의미 있는
# categorical/domain 조합 위주로 제한한다.
# ============================================================

CANDIDATE_SCHEMAS = [

    # ---------------------------
    # 단일 Domain
    # ---------------------------

    ("game_type",),

    ("pitcher_hand",),

    ("batter_hand",),

    ("base_state",),

    ("top_bottom",),

    ("outs_before",),


    # ---------------------------
    # Hand matchup
    # ---------------------------

    (
        "pitcher_hand",
        "batter_hand",
    ),

    (
        "game_type",
        "pitcher_hand",
        "batter_hand",
    ),


    # ---------------------------
    # game_type conditional
    # ---------------------------

    (
        "game_type",
        "base_state",
    ),

    (
        "game_type",
        "top_bottom",
    ),

    (
        "game_type",
        "outs_before",
    ),


    # ---------------------------
    # Pitcher conditional
    # ---------------------------

    (
        "pitcher_id",
        "batter_hand",
    ),

    (
        "pitcher_id",
        "game_type",
    ),

    (
        "pitcher_id",
        "game_type",
        "batter_hand",
    ),


    # ---------------------------
    # Batter conditional
    # ---------------------------

    (
        "batter_id",
        "pitcher_hand",
    ),

    (
        "batter_id",
        "game_type",
    ),


    # ---------------------------
    # Team 수준
    # ---------------------------

    (
        "pitcher_team_id",
        "batter_hand",
    ),

    (
        "batter_team_id",
        "pitcher_hand",
    ),
]


# ============================================================
# OOF Load
# ============================================================

def load_oof(path):
    """
    cb_reg_v003의 OOF prediction을 읽는다.
    """

    data = np.load(
        path,
        allow_pickle=True,
    )

    print()
    print("=" * 80)
    print("OOF LOAD")
    print("=" * 80)

    print(
        "Path:",
        path,
    )

    print(
        "Keys:",
        list(data.files),
    )

    required = [
        "row_id",
        "target",
        "validation_season",
    ]

    for key in required:

        if key not in data.files:

            raise KeyError(
                f"OOF에 '{key}'가 없습니다."
            )

    # 최종 evaluation prediction 우선
    if "prediction" in data.files:

        prediction_key = (
            "prediction"
        )

    elif "prediction_clip" in data.files:

        prediction_key = (
            "prediction_clip"
        )

    else:

        raise KeyError(
            "OOF에 prediction 또는 "
            "prediction_clip이 없습니다."
        )

    print(
        "Prediction key:",
        prediction_key,
    )

    oof = pd.DataFrame({

        "row_id":
            data["row_id"],

        "target":
            data["target"].astype(
                np.float64
            ),

        "prediction":
            data[
                prediction_key
            ].astype(
                np.float64
            ),

        "validation_season":
            data[
                "validation_season"
            ].astype(int),
    })

    oof[
        "prediction"
    ] = np.clip(
        oof[
            "prediction"
        ],
        0.0,
        1.0,
    )

    return oof


# ============================================================
# row_id dtype 정리
# ============================================================

def normalize_row_id(
    oof,
    train,
):
    """
    서로 다른 로컬에서 생성된 artifact라도
    row_id dtype 차이 때문에 merge가 실패하지 않도록 정리한다.
    """

    oof = oof.copy()
    train = train.copy()

    oof_numeric = pd.to_numeric(
        oof["row_id"],
        errors="coerce",
    )

    train_numeric = pd.to_numeric(
        train[ID_COL],
        errors="coerce",
    )

    if (
        oof_numeric.notna().all()
        and
        train_numeric.notna().all()
    ):

        oof[
            "row_id"
        ] = (
            oof_numeric
            .astype(np.int64)
        )

        train[
            ID_COL
        ] = (
            train_numeric
            .astype(np.int64)
        )

    else:

        oof[
            "row_id"
        ] = (
            oof[
                "row_id"
            ].astype(str)
        )

        train[
            ID_COL
        ] = (
            train[
                ID_COL
            ].astype(str)
        )

    return (
        oof,
        train,
    )


# ============================================================
# Residual 생성
# ============================================================

def create_residuals(df):
    """
    raw residual과 season-centered residual을 모두 생성한다.
    """

    df = df.copy()

    # --------------------------------------------
    # Raw residual
    # --------------------------------------------

    df[
        "residual"
    ] = (
        df["target"]
        - df["prediction"]
    )

    # --------------------------------------------
    # 시즌별 전체 residual 평균
    # --------------------------------------------

    season_mean = (
        df
        .groupby(
            "validation_season"
        )["residual"]
        .transform("mean")
    )

    # --------------------------------------------
    # Conditional effect 탐색용 residual
    # --------------------------------------------

    df[
        "centered_residual"
    ] = (
        df["residual"]
        - season_mean
    )

    return df


# ============================================================
# Candidate Schema 유효성 확인
# ============================================================

def get_valid_schemas(
    available_columns,
):
    """
    현재 train.csv에 실제 존재하는 column만 사용한다.
    """

    valid = []

    skipped = []

    for schema in (
        CANDIDATE_SCHEMAS
    ):

        missing = [
            col
            for col in schema
            if col
            not in available_columns
        ]

        if missing:

            skipped.append(
                (
                    schema,
                    missing,
                )
            )

        else:

            valid.append(
                schema
            )

    print()
    print("=" * 80)
    print("CANDIDATE SCHEMAS")
    print("=" * 80)

    print(
        f"사용: {len(valid)}개"
    )

    for schema in valid:

        print(
            " +",
            " × ".join(
                schema
            ),
        )

    if skipped:

        print()
        print(
            f"제외: {len(skipped)}개"
        )

        for (
            schema,
            missing,
        ) in skipped:

            print(
                " -",
                " × ".join(
                    schema
                ),
                "| missing:",
                missing,
            )

    return valid


# ============================================================
# 한 Schema의 시즌별 Residual 계산
# ============================================================

def calculate_schema_stats(
    df,
    schema,
):
    """
    schema별 / season별 residual 통계 생성.
    """

    group_cols = list(
        schema
    )

    stats = (

        df
        .groupby(
            group_cols
            + [
                "validation_season"
            ],
            dropna=False,
            sort=False,
        )
        .agg(

            n=(
                "residual",
                "size",
            ),

            mean_residual=(
                "residual",
                "mean",
            ),

            mean_centered_residual=(
                "centered_residual",
                "mean",
            ),

            residual_std=(
                "residual",
                "std",
            ),

        )
        .reset_index()

    )

    return stats


# ============================================================
# Discovery Stable Candidate 생성
# ============================================================

def build_discovery_candidates(
    stats,
    schema,
):
    """
    2021~2023에 모두 존재하면서

    - 각 시즌 n >= threshold
    - centered residual 부호가 3년 모두 동일

    한 group만 stable candidate로 선정한다.

    중요:
    2024는 여기에서 사용하지 않는다.
    """

    group_cols = list(
        schema
    )

    discovery = (
        stats.loc[
            stats[
                "validation_season"
            ].isin(
                DISCOVERY_SEASONS
            )
        ]
        .copy()
    )

    # 최소 sample 조건
    discovery = (
        discovery.loc[
            discovery["n"]
            >= MIN_N_PER_DISCOVERY_SEASON
        ]
        .copy()
    )

    results = []

    grouped = discovery.groupby(
        group_cols,
        dropna=False,
        sort=False,
    )

    for key, g in grouped:

        if not isinstance(
            key,
            tuple,
        ):
            key = (
                key,
            )

        seasons = sorted(
            g[
                "validation_season"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        # 2021/22/23 세 시즌 모두 필요
        if seasons != DISCOVERY_SEASONS:

            continue

        g = (
            g
            .set_index(
                "validation_season"
            )
            .loc[
                DISCOVERY_SEASONS
            ]
        )

        centered_values = (
            g[
                "mean_centered_residual"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        raw_values = (
            g[
                "mean_residual"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        n_values = (
            g["n"]
            .to_numpy(
                dtype=np.float64
            )
        )

        # 0에 정확히 걸린 residual은
        # 안정적 방향이라고 보지 않는다.
        signs = np.sign(
            centered_values
        )

        if np.any(
            signs == 0
        ):

            continue

        # 3년 모두 같은 방향인지 확인
        same_sign = (
            np.all(
                signs
                == signs[0]
            )
        )

        if not same_sign:

            continue

        # weighted mean
        weighted_centered = float(
            np.average(
                centered_values,
                weights=n_values,
            )
        )

        weighted_raw = float(
            np.average(
                raw_values,
                weights=n_values,
            )
        )

        direction = (
            "UNDER_PREDICT"
            if weighted_centered > 0
            else
            "OVER_PREDICT"
        )

        # ----------------------------------------
        # Discovery ranking score
        #
        # residual magnitude와 sample support를
        # 동시에 반영한다.
        #
        # 2024 정보는 절대 사용하지 않는다.
        # ----------------------------------------

        min_n = int(
            np.min(
                n_values
            )
        )

        discovery_score = float(
            abs(
                weighted_centered
            )
            * np.log1p(
                min_n
            )
        )

        row = {
            "schema":
                " × ".join(
                    schema
                ),

            "direction":
                direction,

            "discovery_weighted_centered_residual":
                weighted_centered,

            "discovery_weighted_raw_residual":
                weighted_raw,

            "discovery_mean_abs_centered_residual":
                float(
                    np.mean(
                        np.abs(
                            centered_values
                        )
                    )
                ),

            "discovery_centered_residual_std":
                float(
                    np.std(
                        centered_values
                    )
                ),

            "discovery_min_n":
                min_n,

            "discovery_total_n":
                int(
                    np.sum(
                        n_values
                    )
                ),

            "discovery_score":
                discovery_score,
        }

        # group key 저장
        for col, value in zip(
            group_cols,
            key,
        ):

            row[col] = (
                value
            )

        # 시즌별 값 저장
        for season in (
            DISCOVERY_SEASONS
        ):

            row[
                f"n_{season}"
            ] = int(
                g.loc[
                    season,
                    "n",
                ]
            )

            row[
                f"raw_residual_{season}"
            ] = float(
                g.loc[
                    season,
                    "mean_residual",
                ]
            )

            row[
                f"centered_residual_{season}"
            ] = float(
                g.loc[
                    season,
                    "mean_centered_residual",
                ]
            )

        results.append(
            row
        )

    if not results:

        return pd.DataFrame()

    result = pd.DataFrame(
        results
    )

    result = (
        result
        .sort_values(
            "discovery_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# 2024 Holdout Confirmation
# ============================================================

def attach_holdout(
    candidates,
    stats,
    schema,
):
    """
    Discovery 단계에서 이미 고정된 candidate에만
    2024 residual을 붙인다.

    2024는 후보를 생성하거나 ranking하는 데 사용하지 않는다.
    """

    if candidates.empty:

        return candidates

    group_cols = list(
        schema
    )

    holdout = (

        stats.loc[
            stats[
                "validation_season"
            ]
            == HOLDOUT_SEASON
        ]
        [
            group_cols
            + [
                "n",
                "mean_residual",
                "mean_centered_residual",
            ]
        ]
        .copy()

    )

    holdout = (
        holdout.rename(
            columns={
                "n":
                    f"n_{HOLDOUT_SEASON}",

                "mean_residual":
                    f"raw_residual_{HOLDOUT_SEASON}",

                "mean_centered_residual":
                    f"centered_residual_{HOLDOUT_SEASON}",
            }
        )
    )

    result = (
        candidates
        .merge(
            holdout,
            on=group_cols,
            how="left",
        )
    )

    discovery_sign = np.sign(
        result[
            "discovery_weighted_centered_residual"
        ]
    )

    holdout_sign = np.sign(
        result[
            f"centered_residual_{HOLDOUT_SEASON}"
        ]
    )

    sufficient_holdout = (
        result[
            f"n_{HOLDOUT_SEASON}"
        ]
        .fillna(0)
        >= MIN_N_HOLDOUT
    )

    sign_match = (
        discovery_sign
        == holdout_sign
    )

    result[
        "holdout_available"
    ] = sufficient_holdout

    result[
        "holdout_sign_match"
    ] = (
        sufficient_holdout
        & sign_match
    )

    # 2024 residual magnitude가
    # discovery 평균 대비 어느 정도인지
    result[
        "holdout_to_discovery_ratio"
    ] = (

        np.abs(
            result[
                f"centered_residual_{HOLDOUT_SEASON}"
            ]
        )

        /

        np.maximum(
            np.abs(
                result[
                    "discovery_weighted_centered_residual"
                ]
            ),
            1e-12,
        )

    )

    return result


# ============================================================
# Schema Summary
# ============================================================

def summarize_schema(
    result,
    schema,
):
    """
    해당 schema에서 stable candidate가
    얼마나 많이 2024까지 유지됐는지 계산.
    """

    schema_name = (
        " × ".join(
            schema
        )
    )

    if result.empty:

        return {
            "schema":
                schema_name,

            "n_discovery_stable":
                0,

            "n_holdout_available":
                0,

            "n_holdout_sign_match":
                0,

            "holdout_confirmation_rate":
                np.nan,

            "best_discovery_score":
                np.nan,
        }

    available = (
        result[
            "holdout_available"
        ]
        == True
    )

    confirmed = (
        result[
            "holdout_sign_match"
        ]
        == True
    )

    n_available = int(
        available.sum()
    )

    n_confirmed = int(
        confirmed.sum()
    )

    if n_available > 0:

        confirmation_rate = float(
            n_confirmed
            / n_available
        )

    else:

        confirmation_rate = (
            np.nan
        )

    return {
        "schema":
            schema_name,

        "n_discovery_stable":
            int(
                len(
                    result
                )
            ),

        "n_holdout_available":
            n_available,

        "n_holdout_sign_match":
            n_confirmed,

        "holdout_confirmation_rate":
            confirmation_rate,

        "best_discovery_score":
            float(
                result[
                    "discovery_score"
                ].max()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 파일 확인
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            DATA_PATH
        )

    if not OOF_PATH.exists():

        raise FileNotFoundError(
            OOF_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # OOF
    # --------------------------------------------------------

    oof = load_oof(
        OOF_PATH
    )

    print()
    print(
        "OOF rows:",
        f"{len(oof):,}",
    )

    print(
        "OOF seasons:",
        sorted(
            oof[
                "validation_season"
            ]
            .unique()
            .tolist()
        ),
    )

    # --------------------------------------------------------
    # train.csv header만 먼저 읽어서
    # 사용 가능한 candidate column 확인
    # --------------------------------------------------------

    train_header = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        nrows=0,
    )

    available_columns = (
        set(
            train_header.columns
        )
    )

    valid_schemas = (
        get_valid_schemas(
            available_columns
        )
    )

    if not valid_schemas:

        raise RuntimeError(
            "사용 가능한 candidate schema가 없습니다."
        )

    # 필요한 column만 읽기
    required_feature_cols = set()

    for schema in (
        valid_schemas
    ):

        required_feature_cols.update(
            schema
        )

    usecols = (
        [ID_COL]
        + sorted(
            required_feature_cols
        )
    )

    print()
    print(
        "train.csv에서 읽을 columns:"
    )

    for col in usecols:

        print(
            " -",
            col,
        )

    train = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=usecols,
    )

    # --------------------------------------------------------
    # row_id dtype 맞추기
    # --------------------------------------------------------

    (
        oof,
        train,
    ) = normalize_row_id(
        oof,
        train,
    )

    # --------------------------------------------------------
    # OOF + train metadata merge
    # --------------------------------------------------------

    df = oof.merge(
        train,
        left_on="row_id",
        right_on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(oof):

        raise ValueError(
            "OOF / train merge 이후 row 수가 달라졌습니다."
        )

    # ID column 중복 정리
    if (
        ID_COL != "row_id"
        and ID_COL in df.columns
    ):

        df = df.drop(
            columns=[ID_COL]
        )

    # --------------------------------------------------------
    # categorical NaN 처리
    # --------------------------------------------------------

    for col in (
        required_feature_cols
    ):

        df[col] = (
            df[col]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    df = create_residuals(
        df
    )

    # 시즌 전체 residual도 출력
    season_diag = (

        df
        .groupby(
            "validation_season"
        )
        .agg(

            n=(
                "residual",
                "size",
            ),

            raw_residual_mean=(
                "residual",
                "mean",
            ),

            raw_residual_std=(
                "residual",
                "std",
            ),

        )
        .reset_index()

    )

    season_diag.to_csv(
        OUTPUT_DIR
        / "season_residual_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("SEASON RESIDUAL DIAGNOSTICS")
    print("=" * 80)

    print(
        season_diag.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Stable Residual Search
    # --------------------------------------------------------

    all_results = []

    schema_summaries = []

    for schema in (
        valid_schemas
    ):

        schema_name = (
            " × ".join(
                schema
            )
        )

        print()
        print("=" * 80)
        print(
            "SEARCH:",
            schema_name,
        )
        print("=" * 80)

        stats = (
            calculate_schema_stats(
                df=df,
                schema=schema,
            )
        )

        candidates = (
            build_discovery_candidates(
                stats=stats,
                schema=schema,
            )
        )

        result = (
            attach_holdout(
                candidates=candidates,
                stats=stats,
                schema=schema,
            )
        )

        schema_summary = (
            summarize_schema(
                result=result,
                schema=schema,
            )
        )

        schema_summaries.append(
            schema_summary
        )

        if result.empty:

            print(
                "Stable candidate: 0"
            )

            continue

        print(
            "Discovery stable:",
            len(
                result
            ),
        )

        print(
            "2024 available:",
            int(
                result[
                    "holdout_available"
                ].sum()
            ),
        )

        print(
            "2024 sign confirmed:",
            int(
                result[
                    "holdout_sign_match"
                ].sum()
            ),
        )

        # schema별 결과 저장
        safe_schema_name = (
            "__".join(
                schema
            )
        )

        result.to_csv(
            OUTPUT_DIR
            / (
                f"candidate_"
                f"{safe_schema_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # Schema Summary
    # --------------------------------------------------------

    schema_summary_df = pd.DataFrame(
        schema_summaries
    )

    schema_summary_df = (
        schema_summary_df
        .sort_values(
            [
                "holdout_confirmation_rate",
                "n_holdout_sign_match",
                "n_discovery_stable",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    schema_summary_df.to_csv(
        OUTPUT_DIR
        / "schema_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 전체 Candidate 통합
    # --------------------------------------------------------

    if all_results:

        all_candidates = pd.concat(
            all_results,
            ignore_index=True,
            sort=False,
        )

        # 중요:
        # 2024 performance로 정렬하지 않는다.
        # discovery score만 사용.
        all_candidates = (
            all_candidates
            .sort_values(
                "discovery_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        all_candidates.to_csv(
            OUTPUT_DIR
            / "all_stable_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )

        confirmed = (
            all_candidates.loc[
                all_candidates[
                    "holdout_sign_match"
                ]
                == True
            ]
            .copy()
        )

        confirmed.to_csv(
            OUTPUT_DIR
            / "holdout_2024_confirmed.csv",
            index=False,
            encoding="utf-8-sig",
        )

    else:

        all_candidates = (
            pd.DataFrame()
        )

        confirmed = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Final Output
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print(
        "SCHEMA SUMMARY"
    )
    print("=" * 100)

    print(
        schema_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "TOP DISCOVERY CANDIDATES"
    )
    print(
        "(2024를 사용하지 않고 2021~2023만으로 ranking)"
    )
    print("=" * 100)

    if all_candidates.empty:

        print(
            "Stable candidate 없음."
        )

    else:

        show_cols = [
            "schema",
            "direction",
            "discovery_weighted_centered_residual",
            "discovery_min_n",
            "discovery_score",
            f"n_{HOLDOUT_SEASON}",
            f"centered_residual_{HOLDOUT_SEASON}",
            "holdout_sign_match",
        ]

        print(
            all_candidates[
                show_cols
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 100)
    print(
        "2024 CONFIRMED SUMMARY"
    )
    print("=" * 100)

    n_all = int(
        len(
            all_candidates
        )
    )

    n_available = (
        int(
            all_candidates[
                "holdout_available"
            ].sum()
        )
        if not all_candidates.empty
        else 0
    )

    n_confirmed = int(
        len(
            confirmed
        )
    )

    print(
        f"Discovery stable candidates : "
        f"{n_all}"
    )

    print(
        f"2024 sufficient sample      : "
        f"{n_available}"
    )

    print(
        f"2024 same-sign confirmed    : "
        f"{n_confirmed}"
    )

    if n_available > 0:

        print(
            f"Confirmation rate           : "
            f"{n_confirmed / n_available:.2%}"
        )

    print()
    print(
        "결과 저장:",
        OUTPUT_DIR,
    )

    print()
    print(
        "중요: "
        "이번 결과는 correction을 적용한 성능이 아니라 "
        "'시간적으로 반복되는 residual pattern이 존재하는가' "
        "확인하는 진단 결과입니다."
    )


if __name__ == "__main__":
    main()