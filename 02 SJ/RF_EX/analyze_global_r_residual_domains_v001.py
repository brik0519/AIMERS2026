# analyze_global_r_residual_domains_v001.py
# ============================================================
# Global Base -> R Residual Domain Analysis
#
# 기준 Base
# ---------
# cb_reg_v003_test 계열 Global CatBoostRegressor
#
# 입력 OOF
# --------
# validation/cb_reg_r_group_es_f_global_base_v001/
#     oof_predictions_with_residual.csv
#
# 이 파일에서 사용하는 핵심:
#   global_prediction
#   global_residual = target - global_prediction
#
# 목적
# ----
# R 전체를 별도 모델로 교체하지 않고,
# "Global Base가 R 안에서 시간적으로 반복해서 같은 방향으로 틀리는
#  상황(domain)"이 있는지 찾는다.
#
# 검증 원칙
# ---------
# Discovery:
#   2021, 2022, 2023
#
# Temporal transfer 내부 검증:
#   2021 residual -> 2022 correction
#   2021~2022 residual -> 2023 correction
#
# Domain selection:
#   오직 2021~2023 정보만 사용
#
# Final holdout:
#   Discovery에서 LOCK된 Top-K domain만
#   2024에서 마지막으로 평가
#
# 2024 결과는 domain 선택/순위 결정에 절대 사용하지 않는다.
#
# Correction 정의
# ---------------
# residual = y - pred
#
# 과거 domain 평균 residual = c 라면
#
#   corrected_pred = base_pred + GAMMA * c
#
# 이 분석에서는 GAMMA=0.5를 미리 고정한다.
# 즉, 과거 오차를 100% 따라가지 않고 절반만 보정한다.
#
# 대회 규칙
# ---------
# 사용 domain은 현재 row 자체의 공식 입력 feature만으로 정의한다.
# test의 다른 row, test 분포, rolling, lag, test 집계는 사용하지 않는다.
#
# 고카디널리티 ID:
# pitcher_id / batter_id / team_id는 domain discovery에서 제외한다.
# 특정 선수 암기보다 일반화 가능한 상황 패턴을 먼저 찾는다.
# ============================================================

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

SOURCE_EXPERIMENT = "cb_reg_r_group_es_f_global_base_v001"

OOF_PATH = (
    Path("./validation")
    / SOURCE_EXPERIMENT
    / "oof_predictions_with_residual.csv"
)

TRAIN_PATH = Path("./data/train.csv")

EXPERIMENT_NAME = "global_r_residual_domain_analysis_v001"

OUTPUT_DIR = (
    Path("./validation")
    / EXPERIMENT_NAME
)


# ============================================================
# Columns
# ============================================================

ID = "row_id"
TARGET = "target"
SEASON = "validation_season"
GROUP = "game_type"

BASE_PRED = "global_prediction"
RESIDUAL = "global_residual"

R_GROUP = "R"


# ============================================================
# Discovery / Holdout
# ============================================================

DISCOVERY_SEASONS = [
    2021,
    2022,
    2023,
]

HOLDOUT_SEASON = 2024


# ============================================================
# Conservative correction policy
# ============================================================

GAMMA = 0.50

# 후보 domain이 너무 작은 표본에 의존하지 않도록 제한
MIN_N_PER_SEASON = 1000

# 각 시즌 R row의 최소 0.5%
MIN_SHARE_PER_SEASON = 0.005

# R 전체와 거의 같은 domain은 제외
MAX_SHARE_PER_SEASON = 0.80

# discovery에서 LOCK할 domain 개수
TOP_K_LOCKED = 10


# ============================================================
# Domain features
# ============================================================

# 그대로 사용하는 저카디널리티 / 상황 피처
RAW_DOMAIN_FEATURES = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "pitcher_hand",
    "batter_hand",
]

# discovery R 데이터에서만 quantile edge를 만든 뒤
# 2021~2024 전체에 동일 edge를 적용
QUANTILE_BIN_FEATURES = [
    "li",
    "score_diff_pitcher_team",
    "run_total_before",
    "home_win_expectancy",
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_success_rate",
    "asof_batter_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

# 2-way interaction은 전부 돌리지 않고,
# 해석 가능하고 test row 단독으로 만들 수 있는 상황 조합 위주로 제한
PAIR_FEATURES = [
    "count_state",
    "base_state",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "hand_matchup",
    "inning_bucket",
    "score_bucket",
    "li_qbin",
    "pitcher_success_qbin",
    "pitcher_middle_qbin",
    "pitchmix_qbin",
]


# ============================================================
# Helpers
# ============================================================

def brier(y, pred):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    return float(
        np.mean(
            (pred - y) ** 2
        )
    )


def safe_string(series):
    return (
        series
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def correction_gain(
    frame,
    mask,
    correction,
):
    """
    frame 전체 R Brier 기준 gain.
    해당 domain row에만 correction을 적용한다.

    gain > 0 = correction 개선
    """
    y = (
        frame[TARGET]
        .to_numpy(
            dtype=np.float64
        )
    )

    base = (
        frame[BASE_PRED]
        .to_numpy(
            dtype=np.float64
        )
    )

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    corrected = base.copy()

    corrected[mask] = np.clip(
        corrected[mask]
        + correction,
        0.0,
        1.0,
    )

    return (
        brier(y, base)
        - brier(y, corrected)
    )


def domain_brier_gain(
    frame,
    mask,
    correction,
):
    """
    domain 내부에서만 계산한 Brier gain.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.sum() == 0:
        return np.nan

    y = (
        frame.loc[
            mask,
            TARGET,
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    base = (
        frame.loc[
            mask,
            BASE_PRED,
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    corrected = np.clip(
        base + correction,
        0.0,
        1.0,
    )

    return (
        brier(y, base)
        - brier(y, corrected)
    )


# ============================================================
# Load / QA
# ============================================================

def load_oof():
    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF 파일이 없습니다:\n{OOF_PATH}\n"
            "먼저 cb_reg_r_group_es_f_global_base_v001을 실행하세요."
        )

    df = pd.read_csv(
        OOF_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        ID,
        TARGET,
        SEASON,
        GROUP,
        BASE_PRED,
        RESIDUAL,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"OOF 필수 컬럼 누락: {sorted(missing)}"
        )

    df[SEASON] = (
        pd.to_numeric(
            df[SEASON],
            errors="raise",
        )
        .astype(int)
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise",
    )

    df[BASE_PRED] = pd.to_numeric(
        df[BASE_PRED],
        errors="raise",
    )

    # 혹시 저장된 residual과 계산값이 다른지 검증
    recalculated = (
        df[TARGET]
        - df[BASE_PRED]
    )

    saved = pd.to_numeric(
        df[RESIDUAL],
        errors="raise",
    )

    if not np.allclose(
        recalculated.to_numpy(),
        saved.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "global_residual != target - global_prediction"
        )

    expected_seasons = {
        2021,
        2022,
        2023,
        2024,
    }

    if not expected_seasons.issubset(
        set(df[SEASON].unique())
    ):
        raise ValueError(
            "2021~2024 OOF가 모두 필요합니다."
        )

    return df


def merge_missing_raw_features(df):
    """
    이전 OOF 저장 파일에 필요한 raw feature가 빠져 있을 경우
    train.csv에서 row_id 기준으로만 가져온다.

    train.csv는 공식 학습 데이터이며,
    OOF row와 one-to-one join만 수행한다.
    """
    wanted = set(
        RAW_DOMAIN_FEATURES
        + QUANTILE_BIN_FEATURES
        + [
            "score_diff_pitcher_team",
            "run_total_before",
            "balls_before",
            "strikes_before",
            "inning",
            "pitcher_hand",
            "batter_hand",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    )

    missing = [
        col
        for col in wanted
        if col not in df.columns
    ]

    if not missing:
        return df

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"필요 feature가 OOF에 없고 train.csv도 없습니다: {missing}"
        )

    train_cols = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    available = [
        col
        for col in missing
        if col in train_cols
    ]

    if not available:
        return df

    extra = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=(
            [ID]
            + available
        ),
        low_memory=False,
    )

    if not extra[ID].is_unique:
        raise AssertionError(
            "train.csv row_id가 unique하지 않습니다."
        )

    df = df.merge(
        extra,
        on=ID,
        how="left",
        validate="one_to_one",
    )

    return df


# ============================================================
# Derived row-local domains
# ============================================================

def fit_quantile_edges(
    discovery,
    feature,
    n_bins=4,
):
    x = numeric(
        discovery[feature]
    ).dropna()

    if x.nunique() < 3:
        return None

    qs = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        x.to_numpy(
            dtype=np.float64
        ),
        qs,
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 3:
        return None

    # pd.cut에서 최솟값/최댓값을 안정적으로 포함
    edges = edges.astype(
        np.float64
    )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def apply_qbin(
    df,
    feature,
    edges,
    new_col,
):
    x = numeric(
        df[feature]
    )

    df[new_col] = (
        pd.cut(
            x,
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )


def add_domains(df):
    out = df.copy()

    # --------------------------------------------------------
    # 기본 derived categories
    # --------------------------------------------------------

    if {
        "balls_before",
        "strikes_before",
    }.issubset(out.columns):
        b = (
            numeric(
                out["balls_before"]
            )
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        s = (
            numeric(
                out["strikes_before"]
            )
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        out["count_state"] = (
            b + "_" + s
        )

    if {
        "pitcher_hand",
        "batter_hand",
    }.issubset(out.columns):
        out["hand_matchup"] = (
            safe_string(
                out["pitcher_hand"]
            )
            + "_"
            + safe_string(
                out["batter_hand"]
            )
        )

    if "inning" in out.columns:
        inn = numeric(
            out["inning"]
        )

        out["inning_bucket"] = pd.cut(
            inn,
            bins=[
                -np.inf,
                3,
                6,
                9,
                np.inf,
            ],
            labels=[
                "1-3",
                "4-6",
                "7-9",
                "10+",
            ],
        ).astype("string").fillna(
            "__MISSING__"
        ).astype(str)

    if "score_diff_pitcher_team" in out.columns:
        score = numeric(
            out[
                "score_diff_pitcher_team"
            ]
        )

        out["score_bucket"] = pd.cut(
            score,
            bins=[
                -np.inf,
                -4,
                -2,
                -1,
                0,
                1,
                3,
                np.inf,
            ],
            labels=[
                "<=-4",
                "-3~-2",
                "-1",
                "0",
                "+1",
                "+2~+3",
                ">=+4",
            ],
        ).astype("string").fillna(
            "__MISSING__"
        ).astype(str)

    # --------------------------------------------------------
    # Discovery R only로 quantile edges fit
    # --------------------------------------------------------

    discovery_mask = (
        out[GROUP]
        .astype(str)
        .eq(R_GROUP)
        & out[SEASON]
        .isin(
            DISCOVERY_SEASONS
        )
    )

    discovery = (
        out.loc[
            discovery_mask
        ]
        .copy()
    )

    bin_meta = {}

    # raw feature -> readable qbin name
    qbin_names = {
        "li": "li_qbin",
        "score_diff_pitcher_team": "score_diff_qbin",
        "run_total_before": "run_total_qbin",
        "home_win_expectancy": "home_winexp_qbin",
        "asof_pitcher_n": "pitcher_n_qbin",
        "asof_batter_n": "batter_n_qbin",
        "asof_pitcher_success_rate": "pitcher_success_qbin",
        "asof_batter_success_rate": "batter_success_qbin",
        "asof_pitcher_reverse_rate": "pitcher_reverse_qbin",
        "asof_pitcher_middle_rate": "pitcher_middle_qbin",
        "asof_pitcher_ball_rate": "pitcher_ball_qbin",
        "asof_pitcher_strike_rate": "pitcher_strike_qbin",
        "asof_pitcher_prev1_game_success_rate": "prev1_success_qbin",
        "asof_pitcher_prev3_game_success_rate": "prev3_success_qbin",
        "asof_pitcher_prev5_game_success_rate": "prev5_success_qbin",
        "asof_pitcher_fastball_rate": "fastball_qbin",
        "asof_pitcher_breaking_rate": "breaking_qbin",
        "asof_pitcher_offspeed_rate": "offspeed_qbin",
    }

    for feature in (
        QUANTILE_BIN_FEATURES
    ):
        if feature not in out.columns:
            continue

        edges = fit_quantile_edges(
            discovery,
            feature,
            n_bins=4,
        )

        if edges is None:
            continue

        new_col = qbin_names.get(
            feature,
            f"{feature}_qbin",
        )

        apply_qbin(
            out,
            feature,
            edges,
            new_col,
        )

        bin_meta[
            new_col
        ] = {
            "source_feature": (
                feature
            ),
            "edges": [
                (
                    "-inf"
                    if np.isneginf(v)
                    else (
                        "inf"
                        if np.isposinf(v)
                        else float(v)
                    )
                )
                for v in edges
            ],
        }

    # pitch mix를 하나의 coarse state로도 표현
    pitchmix_sources = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]

    if all(
        c in out.columns
        for c in pitchmix_sources
    ):
        rates = pd.DataFrame(
            {
                "fastball": numeric(
                    out[
                        "asof_pitcher_fastball_rate"
                    ]
                ),
                "breaking": numeric(
                    out[
                        "asof_pitcher_breaking_rate"
                    ]
                ),
                "offspeed": numeric(
                    out[
                        "asof_pitcher_offspeed_rate"
                    ]
                ),
            },
            index=out.index,
        )

        all_missing = rates.isna().all(
            axis=1
        )

        dominant = (
            rates.fillna(-np.inf)
            .idxmax(axis=1)
            .astype("string")
        )

        dominant.loc[
            all_missing
        ] = "__MISSING__"

        out["pitchmix_qbin"] = (
            dominant.astype(str)
        )

    return (
        out,
        bin_meta,
    )


# ============================================================
# Candidate generation
# ============================================================

def domain_features(df):
    features = []

    for col in RAW_DOMAIN_FEATURES:
        if col in df.columns:
            # raw continuous-ish inning은 derived bucket도 있으므로
            # 너무 많은 distinct value가 아니면 사용
            nunique = (
                df[col]
                .nunique(
                    dropna=False
                )
            )

            if nunique <= 31:
                features.append(
                    col
                )

    derived = [
        "count_state",
        "hand_matchup",
        "inning_bucket",
        "score_bucket",
        "pitchmix_qbin",
    ]

    derived += [
        col
        for col in df.columns
        if col.endswith(
            "_qbin"
        )
    ]

    for col in derived:
        if (
            col in df.columns
            and col not in features
        ):
            features.append(
                col
            )

    return features


def build_domain_key(
    frame,
    features,
):
    if len(features) == 1:
        col = features[0]
        return safe_string(
            frame[col]
        )

    left = safe_string(
        frame[
            features[0]
        ]
    )

    right = safe_string(
        frame[
            features[1]
        ]
    )

    return (
        left
        + " || "
        + right
    )


def candidate_specs(
    available_features,
):
    specs = []

    # 1-way
    for feature in (
        available_features
    ):
        specs.append(
            (feature,)
        )

    # 2-way: curated list 안에 있는 것만
    pair_available = [
        col
        for col in PAIR_FEATURES
        if col in available_features
    ]

    for left, right in combinations(
        pair_available,
        2,
    ):
        specs.append(
            (
                left,
                right,
            )
        )

    return specs


# ============================================================
# Candidate evaluation
# ============================================================

def evaluate_candidate_value(
    r_df,
    features,
    value,
):
    key = build_domain_key(
        r_df,
        features,
    )

    mask = (
        key == value
    )

    # --------------------------------------------------------
    # 시즌별 residual / support
    # --------------------------------------------------------

    season_stats = {}

    for season in (
        DISCOVERY_SEASONS
        + [HOLDOUT_SEASON]
    ):
        season_mask = (
            r_df[SEASON]
            .eq(season)
            .to_numpy()
        )

        domain_mask = (
            mask.to_numpy()
            & season_mask
        )

        n_total = int(
            season_mask.sum()
        )

        n = int(
            domain_mask.sum()
        )

        share = (
            n / n_total
            if n_total > 0
            else 0.0
        )

        if n > 0:
            mean_residual = float(
                r_df.loc[
                    domain_mask,
                    RESIDUAL,
                ].mean()
            )
        else:
            mean_residual = np.nan

        season_stats[
            season
        ] = {
            "n": n,
            "share": share,
            "mean_residual": (
                mean_residual
            ),
        }

    # --------------------------------------------------------
    # Discovery support filter
    # --------------------------------------------------------

    for season in (
        DISCOVERY_SEASONS
    ):
        stat = season_stats[
            season
        ]

        if (
            stat["n"]
            < MIN_N_PER_SEASON
        ):
            return None

        if (
            stat["share"]
            < MIN_SHARE_PER_SEASON
        ):
            return None

        if (
            stat["share"]
            > MAX_SHARE_PER_SEASON
        ):
            return None

    discovery_means = [
        season_stats[s][
            "mean_residual"
        ]
        for s in DISCOVERY_SEASONS
    ]

    if not np.isfinite(
        discovery_means
    ).all():
        return None

    signs = np.sign(
        discovery_means
    )

    # 2021~2023 모두 같은 방향인 domain만
    if (
        np.any(signs == 0)
        or not np.all(
            signs == signs[0]
        )
    ):
        return None

    # --------------------------------------------------------
    # Temporal transfer 1:
    # 2021 residual -> 2022 correction
    # --------------------------------------------------------

    mask_2021 = (
        mask.to_numpy()
        & r_df[SEASON]
        .eq(2021)
        .to_numpy()
    )

    mask_2022 = (
        mask.to_numpy()
        & r_df[SEASON]
        .eq(2022)
        .to_numpy()
    )

    source_2021_mean = float(
        r_df.loc[
            mask_2021,
            RESIDUAL,
        ].mean()
    )

    correction_2022 = (
        GAMMA
        * source_2021_mean
    )

    frame_2022 = (
        r_df.loc[
            r_df[SEASON]
            .eq(2022)
        ]
        .copy()
    )

    key_2022 = build_domain_key(
        frame_2022,
        features,
    )

    local_mask_2022 = (
        key_2022
        .eq(value)
        .to_numpy()
    )

    gain_2022_all_r = (
        correction_gain(
            frame_2022,
            local_mask_2022,
            correction_2022,
        )
    )

    gain_2022_domain = (
        domain_brier_gain(
            frame_2022,
            local_mask_2022,
            correction_2022,
        )
    )

    # --------------------------------------------------------
    # Temporal transfer 2:
    # 2021~2022 residual -> 2023 correction
    # --------------------------------------------------------

    source_2122_mask = (
        mask.to_numpy()
        & r_df[SEASON]
        .isin(
            [
                2021,
                2022,
            ]
        )
        .to_numpy()
    )

    source_2122_mean = float(
        r_df.loc[
            source_2122_mask,
            RESIDUAL,
        ].mean()
    )

    correction_2023 = (
        GAMMA
        * source_2122_mean
    )

    frame_2023 = (
        r_df.loc[
            r_df[SEASON]
            .eq(2023)
        ]
        .copy()
    )

    key_2023 = build_domain_key(
        frame_2023,
        features,
    )

    local_mask_2023 = (
        key_2023
        .eq(value)
        .to_numpy()
    )

    gain_2023_all_r = (
        correction_gain(
            frame_2023,
            local_mask_2023,
            correction_2023,
        )
    )

    gain_2023_domain = (
        domain_brier_gain(
            frame_2023,
            local_mask_2023,
            correction_2023,
        )
    )

    # 두 번 모두 미래 시즌으로 넘겼을 때 개선된 후보만
    if not (
        gain_2022_all_r > 0.0
        and gain_2023_all_r > 0.0
    ):
        return None

    # --------------------------------------------------------
    # Discovery pooled residual
    # --------------------------------------------------------

    discovery_mask = (
        mask.to_numpy()
        & r_df[SEASON]
        .isin(
            DISCOVERY_SEASONS
        )
        .to_numpy()
    )

    discovery_mean = float(
        r_df.loc[
            discovery_mask,
            RESIDUAL,
        ].mean()
    )

    discovery_std_across_seasons = float(
        np.std(
            discovery_means,
            ddof=0,
        )
    )

    abs_means = np.abs(
        discovery_means
    )

    stability_ratio = float(
        np.min(abs_means)
        / (
            np.max(abs_means)
            + 1e-12
        )
    )

    # rank는 오직 discovery temporal transfer로 계산
    transfer_sum = float(
        gain_2022_all_r
        + gain_2023_all_r
    )

    transfer_min = float(
        min(
            gain_2022_all_r,
            gain_2023_all_r,
        )
    )

    return {
        "features": (
            " + ".join(
                features
            )
        ),
        "value": str(
            value
        ),
        "n_2021": (
            season_stats[2021][
                "n"
            ]
        ),
        "n_2022": (
            season_stats[2022][
                "n"
            ]
        ),
        "n_2023": (
            season_stats[2023][
                "n"
            ]
        ),
        "share_2021": (
            season_stats[2021][
                "share"
            ]
        ),
        "share_2022": (
            season_stats[2022][
                "share"
            ]
        ),
        "share_2023": (
            season_stats[2023][
                "share"
            ]
        ),
        "resid_2021": (
            discovery_means[0]
        ),
        "resid_2022": (
            discovery_means[1]
        ),
        "resid_2023": (
            discovery_means[2]
        ),
        "discovery_mean_residual": (
            discovery_mean
        ),
        "discovery_season_std": (
            discovery_std_across_seasons
        ),
        "stability_ratio": (
            stability_ratio
        ),
        "transfer_2022_gain_all_R": (
            gain_2022_all_r
        ),
        "transfer_2023_gain_all_R": (
            gain_2023_all_r
        ),
        "transfer_2022_gain_domain": (
            gain_2022_domain
        ),
        "transfer_2023_gain_domain": (
            gain_2023_domain
        ),
        "transfer_sum_gain_all_R": (
            transfer_sum
        ),
        "transfer_min_gain_all_R": (
            transfer_min
        ),
        # 이 아래 2024 관련 값은 아직 selection에 쓰지 않음
        "_n_2024": (
            season_stats[2024][
                "n"
            ]
        ),
        "_share_2024": (
            season_stats[2024][
                "share"
            ]
        ),
        "_resid_2024": (
            season_stats[2024][
                "mean_residual"
            ]
        ),
    }


def discover_candidates(r_df):
    available = domain_features(
        r_df
    )

    specs = candidate_specs(
        available
    )

    print(
        "Domain features:",
        len(available),
    )

    print(
        "Candidate specs:",
        len(specs),
    )

    rows = []

    for idx, features in enumerate(
        specs,
        start=1,
    ):
        key = build_domain_key(
            r_df,
            features,
        )

        # discovery에서 실제 존재하는 value만 후보
        discovery_mask = (
            r_df[SEASON]
            .isin(
                DISCOVERY_SEASONS
            )
        )

        values = (
            key.loc[
                discovery_mask
            ]
            .drop_duplicates()
            .tolist()
        )

        for value in values:
            row = (
                evaluate_candidate_value(
                    r_df=r_df,
                    features=features,
                    value=value,
                )
            )

            if row is not None:
                rows.append(
                    row
                )

        if (
            idx % 10 == 0
            or idx == len(specs)
        ):
            print(
                f"Processed specs: "
                f"{idx}/{len(specs)} | "
                f"stable candidates={len(rows)}"
            )

    if not rows:
        return (
            pd.DataFrame(),
            available,
        )

    candidates = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # 중요:
    # 2024 정보 없이 순위 결정
    #
    # 1) 두 transfer 중 더 약한 gain이 큰 것
    # 2) transfer 합이 큰 것
    # 3) 시즌 residual 크기 안정성
    # --------------------------------------------------------

    candidates = (
        candidates.sort_values(
            [
                "transfer_min_gain_all_R",
                "transfer_sum_gain_all_R",
                "stability_ratio",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    candidates[
        "discovery_rank"
    ] = (
        np.arange(
            len(candidates)
        )
        + 1
    )

    return (
        candidates,
        available,
    )


# ============================================================
# Locked 2024 evaluation
# ============================================================

def evaluate_locked_2024(
    r_df,
    locked,
):
    if locked.empty:
        return pd.DataFrame()

    frame_2024 = (
        r_df.loc[
            r_df[SEASON]
            .eq(HOLDOUT_SEASON)
        ]
        .copy()
    )

    rows = []

    for _, candidate in (
        locked.iterrows()
    ):
        features = tuple(
            candidate[
                "features"
            ].split(
                " + "
            )
        )

        value = str(
            candidate["value"]
        )

        key_2024 = (
            build_domain_key(
                frame_2024,
                features,
            )
        )

        mask_2024 = (
            key_2024
            .eq(value)
            .to_numpy()
        )

        # LOCKED correction:
        # 2021~2023 pooled mean residual만 사용
        correction = float(
            GAMMA
            * candidate[
                "discovery_mean_residual"
            ]
        )

        gain_all_r = correction_gain(
            frame_2024,
            mask_2024,
            correction,
        )

        gain_domain = (
            domain_brier_gain(
                frame_2024,
                mask_2024,
                correction,
            )
        )

        rows.append({
            "discovery_rank": int(
                candidate[
                    "discovery_rank"
                ]
            ),
            "features": (
                candidate[
                    "features"
                ]
            ),
            "value": value,
            "locked_correction": (
                correction
            ),
            "n_2024": int(
                mask_2024.sum()
            ),
            "share_2024": float(
                mask_2024.mean()
            ),
            "actual_mean_residual_2024": float(
                frame_2024.loc[
                    mask_2024,
                    RESIDUAL,
                ].mean()
            )
            if mask_2024.any()
            else np.nan,
            "holdout_2024_gain_all_R": (
                gain_all_r
            ),
            "holdout_2024_gain_domain": (
                gain_domain
            ),
            "holdout_win": bool(
                gain_all_r > 0.0
            ),
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "discovery_rank"
    )


# ============================================================
# Main
# ============================================================

def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Change EXPERIMENT_NAME."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    df = load_oof()

    df = merge_missing_raw_features(
        df
    )

    (
        df,
        bin_meta,
    ) = add_domains(
        df
    )

    # R만 분석
    r_df = (
        df.loc[
            df[GROUP]
            .astype(str)
            .eq(R_GROUP)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("=" * 100)
    print(
        "Global Base -> R Residual Domain Analysis"
    )
    print("=" * 100)
    print(
        "Discovery seasons :",
        DISCOVERY_SEASONS,
    )
    print(
        "Holdout season    :",
        HOLDOUT_SEASON,
    )
    print(
        "Residual          : target - global_prediction"
    )
    print(
        "Gamma             :",
        GAMMA,
    )
    print(
        "R rows            :",
        f"{len(r_df):,}",
    )

    for season in (
        DISCOVERY_SEASONS
        + [HOLDOUT_SEASON]
    ):
        temp = (
            r_df.loc[
                r_df[SEASON]
                .eq(season)
            ]
        )

        print(
            f"{season} | "
            f"n={len(temp):,} | "
            f"mean residual="
            f"{temp[RESIDUAL].mean():+.8f} | "
            f"base Brier="
            f"{brier(temp[TARGET], temp[BASE_PRED]):.10f}"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # Discovery
    # --------------------------------------------------------

    (
        candidates,
        available_features,
    ) = discover_candidates(
        r_df
    )

    candidates.to_csv(
        OUTPUT_DIR
        / "discovery_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2024 관련 임시 컬럼은 discovery 공개 파일에서 제거한 버전도 저장
    if not candidates.empty:
        discovery_only_cols = [
            col
            for col in candidates.columns
            if not col.startswith("_")
        ]

        candidates[
            discovery_only_cols
        ].to_csv(
            OUTPUT_DIR
            / "discovery_rank_locked.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Top K LOCK
    # --------------------------------------------------------

    locked = (
        candidates.head(
            TOP_K_LOCKED
        )
        .copy()
    )

    if not locked.empty:
        locked.to_csv(
            OUTPUT_DIR
            / "locked_top_domains_before_2024.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # 2024 holdout
    # --------------------------------------------------------

    holdout = (
        evaluate_locked_2024(
            r_df,
            locked,
        )
    )

    holdout.to_csv(
        OUTPUT_DIR
        / "locked_domains_2024_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "source_oof": str(
            OOF_PATH
        ),
        "base_prediction": (
            BASE_PRED
        ),
        "residual": (
            "target - global_prediction"
        ),
        "analysis_group": (
            R_GROUP
        ),
        "discovery_seasons": (
            DISCOVERY_SEASONS
        ),
        "holdout_season": (
            HOLDOUT_SEASON
        ),
        "gamma": (
            GAMMA
        ),
        "min_n_per_season": (
            MIN_N_PER_SEASON
        ),
        "min_share_per_season": (
            MIN_SHARE_PER_SEASON
        ),
        "max_share_per_season": (
            MAX_SHARE_PER_SEASON
        ),
        "top_k_locked": (
            TOP_K_LOCKED
        ),
        "available_domain_features": (
            available_features
        ),
        "quantile_bins": (
            bin_meta
        ),
        "selection_rule": (
            "2021~2023 same residual sign, sufficient support, "
            "2021->2022 correction gain > 0, "
            "2021~2022->2023 correction gain > 0; "
            "ranked without using 2024."
        ),
        "competition_rule_note": (
            "All domain definitions use only current-row official features. "
            "No test-row aggregation or test distribution is used."
        ),
    }

    with open(
        OUTPUT_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "DISCOVERY RESULT: 2021~2023 only"
    )
    print("=" * 100)

    if candidates.empty:
        print(
            "조건을 만족한 안정적 residual domain이 없습니다."
        )
        print(
            "=> 이 경우 억지로 residual correction을 진행하지 않는 것이 맞습니다."
        )
        return

    show_cols = [
        "discovery_rank",
        "features",
        "value",
        "resid_2021",
        "resid_2022",
        "resid_2023",
        "transfer_2022_gain_all_R",
        "transfer_2023_gain_all_R",
    ]

    print(
        candidates.head(
            TOP_K_LOCKED
        )[show_cols]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "LOCKED TOP DOMAINS -> 2024 HOLDOUT"
    )
    print("=" * 100)

    if holdout.empty:
        print(
            "Holdout 평가 대상이 없습니다."
        )
        return

    print(
        holdout[
            [
                "discovery_rank",
                "features",
                "value",
                "locked_correction",
                "n_2024",
                "actual_mean_residual_2024",
                "holdout_2024_gain_all_R",
                "holdout_win",
            ]
        ]
        .to_string(
            index=False
        )
    )

    wins = int(
        holdout[
            "holdout_win"
        ].sum()
    )

    print()
    print(
        f"2024 holdout wins: "
        f"{wins}/{len(holdout)}"
    )

    print()
    print(
        "해석:"
    )
    print(
        "- 여러 상위 domain이 2024에서도 +gain이면 residual correction 진행 가치가 큼"
    )
    print(
        "- discovery에서는 좋아도 2024에서 대부분 깨지면 correction은 보류"
    )
    print(
        "- 2024 결과를 보고 새 domain을 다시 고르면 holdout 의미가 사라지므로 금지"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
