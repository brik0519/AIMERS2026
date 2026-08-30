# analyze_middle5_2024_cause.py
"""
LG Aimers Phase2 - Middle Temporal 5의 2024 개선 원인 진단

목적
----
Multi-Seed 실험에서 Middle Temporal 5가 전체 OOF에서는 불안정했지만,
2024 Validation에서는 5/5 seed 모두 Basic보다 Brier가 개선되었다.

이 스크립트는 새 모델을 학습하지 않는다.
이미 저장된 Basic / Middle5 Multi-Seed OOF prediction을 이용해
"2024에서 어디서, 어떤 조건에서 개선이 발생했는가?"를 분석한다.

핵심 질문
---------
1) 특정 Middle Temporal 값 구간에서 개선이 집중되는가?
2) 2024에서 Middle Temporal의 분포가 2021~2023과 달라졌는가?
3) Middle Level + Trend가 함께 특정 상태일 때 개선이 커지는가?
4) count / outs / base_state / inning 등 특정 경기 상황에서 개선이 집중되는가?
5) Basic의 예측 확률 구간 중 어디에서 Middle5가 보정 효과를 내는가?
6) 위 패턴이 5개 seed에서 같은 방향으로 반복되는가?

주의
----
- 이 분석은 원인을 "증명"하는 causal analysis가 아니다.
- 2024의 안정적인 개선을 설명할 수 있는 재현 가능한 패턴을 찾는 진단 단계다.
- 여기서 발견한 패턴은 후속 Feature Engineering 가설일 뿐이며,
  반드시 동일 Expanding Window에서 다시 검증해야 한다.
- Temporal Feature 계산은 동일 row의 공식 asof_* 값끼리만 수행한다.
- 평가 데이터(test)의 다른 row를 사용하는 연산은 없다.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "middle5_2024_cause_analysis_v001"

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
# 기본 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

SOURCE_EXPERIMENT_NAME = "basic_vs_middle5_multiseed_v001"
SEEDS = [42, 7, 2026, 123, 999]

HISTORICAL_VALID_SEASONS = [2021, 2022, 2023]
FOCUS_SEASON = 2024

# 분위수 기반 진단 구간 수
N_BINS = 5

# 콘솔에서 너무 작은 그룹이 상단에 노출되는 것을 막기 위한 표시 기준.
# CSV에는 작은 그룹도 모두 저장한다.
CONSOLE_MIN_N = 1000


# =========================================================
# Middle Temporal Feature 정의
# =========================================================

MIDDLE_LEVEL_FEATURES = [
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",
]

MIDDLE_TREND_FEATURES = [
    "middle_trend_1_3",
    "middle_trend_3_5",
]

MIDDLE_TEMPORAL_FEATURES = (
    MIDDLE_LEVEL_FEATURES
    + MIDDLE_TREND_FEATURES
)


# =========================================================
# Feature 생성
# =========================================================

def add_middle_temporal_features(df):
    """
    기존 실험과 동일한 Middle Temporal 5개를 생성한다.

    *_vs_career는 기존 변수명을 유지하지만 실제 의미는
    Recent vs 해당 시점까지 Cumulative(as-of)이다.
    """
    df = df.copy()

    required_cols = [
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Middle Temporal 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    # Level: 최근 Middle 비율 - 누적 Middle 비율
    df["recent1_middle_vs_career"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["recent3_middle_vs_career"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    df["recent5_middle_vs_career"] = (
        df["asof_pitcher_prev5_game_middle_rate"]
        - df["asof_pitcher_middle_rate"]
    )

    # Trend: 더 짧은 최근 구간 - 더 긴 최근 구간
    df["middle_trend_1_3"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev3_game_middle_rate"]
    )

    df["middle_trend_3_5"] = (
        df["asof_pitcher_prev3_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    return df


# =========================================================
# 공통 유틸
# =========================================================

def calculate_brier(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )


def make_quantile_edges(series, n_bins=N_BINS):
    """
    Historical 데이터로 고정된 quantile bin edge를 만든다.

    시즌별로 각자 qcut하면 항상 20%씩 나뉘어 분포 변화가 숨겨질 수 있으므로,
    2021~2023에서 만든 동일 edge를 2024에도 적용한다.
    """
    values = (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    if len(values) == 0:
        return None

    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        values,
        quantiles,
    )

    edges = np.unique(edges)

    # 값의 종류가 너무 적어 bin 생성이 불가능한 경우
    if len(edges) < 2:
        return None

    # min/max 바깥 값도 동일 bin에 포함
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def apply_bins(series, edges, prefix="Q"):
    """
    고정된 edge를 이용해 bin label을 생성한다.
    NaN은 'Missing'으로 별도 표시한다.
    """
    if edges is None:
        return pd.Series(
            ["All"] * len(series),
            index=series.index,
            dtype="object",
        )

    n_actual_bins = len(edges) - 1

    labels = [
        f"{prefix}{i + 1}"
        for i in range(n_actual_bins)
    ]

    binned = pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    result = binned.astype("object")
    result[pd.isna(result)] = "Missing"

    return result.astype(str)


def safe_std(series):
    value = float(
        pd.to_numeric(
            series,
            errors="coerce",
        ).std(ddof=0)
    )

    if not np.isfinite(value):
        return np.nan

    return value


# =========================================================
# Multi-Seed OOF 로드
# =========================================================

def load_multiseed_oof():
    """
    각 seed의 Basic / Middle5 OOF를 로드해 seed-row 단위 long table로 만든다.

    반환 컬럼 예
    ---------
    row_id, season, target, seed,
    pred_basic, pred_middle5,
    se_basic, se_middle5, delta_se, pred_shift

    delta_se = Basic SE - Middle5 SE
    delta_se > 0 이면 Middle5 개선
    """
    source_dir = (
        VALIDATION_ROOT
        / SOURCE_EXPERIMENT_NAME
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Multi-Seed 결과 폴더가 없습니다: {source_dir}\n"
            "train_basic_vs_middle5_multiseed.py를 먼저 실행해야 합니다."
        )

    seed_frames = []

    for seed in SEEDS:
        basic_path = (
            source_dir
            / f"seed_{seed}"
            / "basic"
            / "oof_pred.csv"
        )

        middle_path = (
            source_dir
            / f"seed_{seed}"
            / "middle5"
            / "oof_pred.csv"
        )

        if not basic_path.exists():
            raise FileNotFoundError(
                f"Basic OOF 없음: {basic_path}"
            )

        if not middle_path.exists():
            raise FileNotFoundError(
                f"Middle5 OOF 없음: {middle_path}"
            )

        basic = pd.read_csv(
            basic_path,
            encoding="utf-8-sig",
        )

        middle = pd.read_csv(
            middle_path,
            encoding="utf-8-sig",
        )

        required = {
            ID_COL,
            "season",
            "target",
            "prediction",
        }

        if not required.issubset(
            basic.columns
        ):
            raise ValueError(
                f"{basic_path} 컬럼 부족"
            )

        if not required.issubset(
            middle.columns
        ):
            raise ValueError(
                f"{middle_path} 컬럼 부족"
            )

        if not basic[ID_COL].is_unique:
            raise AssertionError(
                f"seed={seed} Basic OOF row_id 중복"
            )

        if not middle[ID_COL].is_unique:
            raise AssertionError(
                f"seed={seed} Middle5 OOF row_id 중복"
            )

        merged = basic[
            [
                ID_COL,
                "season",
                "target",
                "prediction",
            ]
        ].merge(
            middle[
                [
                    ID_COL,
                    "season",
                    "target",
                    "prediction",
                ]
            ],
            on=ID_COL,
            how="inner",
            suffixes=("_basic", "_middle5"),
            validate="one_to_one",
        )

        if len(merged) != len(basic):
            raise AssertionError(
                f"seed={seed}: Basic/Middle5 OOF row 수 불일치"
            )

        if not np.array_equal(
            merged["season_basic"].to_numpy(),
            merged["season_middle5"].to_numpy(),
        ):
            raise AssertionError(
                f"seed={seed}: season 불일치"
            )

        if not np.array_equal(
            merged["target_basic"].to_numpy(),
            merged["target_middle5"].to_numpy(),
        ):
            raise AssertionError(
                f"seed={seed}: target 불일치"
            )

        merged = merged.rename(
            columns={
                "season_basic": "season",
                "target_basic": "target",
                "prediction_basic": "pred_basic",
                "prediction_middle5": "pred_middle5",
            }
        )

        merged = merged.drop(
            columns=[
                "season_middle5",
                "target_middle5",
            ]
        )

        merged["seed"] = int(seed)

        merged["se_basic"] = (
            merged["target"]
            - merged["pred_basic"]
        ) ** 2

        merged["se_middle5"] = (
            merged["target"]
            - merged["pred_middle5"]
        ) ** 2

        merged["delta_se"] = (
            merged["se_basic"]
            - merged["se_middle5"]
        )

        merged["pred_shift"] = (
            merged["pred_middle5"]
            - merged["pred_basic"]
        )

        merged["middle5_row_better"] = (
            merged["delta_se"] > 0
        )

        seed_frames.append(
            merged
        )

    stacked = pd.concat(
        seed_frames,
        axis=0,
        ignore_index=True,
    )

    return stacked


# =========================================================
# Row-level Multi-Seed 평균
# =========================================================

def aggregate_rows_across_seeds(stacked):
    """
    같은 row에 대한 5개 seed 결과를 평균한다.

    이 테이블은 "한 seed에서만 우연히 좋아진 row"보다
    여러 seed에서 반복되는 row-level 개선을 보기 위한 것이다.
    """
    row_level = (
        stacked
        .groupby(
            [
                ID_COL,
                "season",
                "target",
            ],
            as_index=False,
        )
        .agg(
            mean_pred_basic=(
                "pred_basic",
                "mean",
            ),
            mean_pred_middle5=(
                "pred_middle5",
                "mean",
            ),
            mean_se_basic=(
                "se_basic",
                "mean",
            ),
            mean_se_middle5=(
                "se_middle5",
                "mean",
            ),
            mean_delta_se=(
                "delta_se",
                "mean",
            ),
            median_delta_se=(
                "delta_se",
                "median",
            ),
            mean_pred_shift=(
                "pred_shift",
                "mean",
            ),
            middle5_row_win_count=(
                "middle5_row_better",
                "sum",
            ),
        )
    )

    row_level[
        "middle5_row_win_rate"
    ] = (
        row_level[
            "middle5_row_win_count"
        ]
        / len(SEEDS)
    )

    return row_level


# =========================================================
# Seed별 2024 전체 Brier
# =========================================================

def summarize_2024_by_seed(stacked):
    focus = stacked.loc[
        stacked["season"] == FOCUS_SEASON
    ].copy()

    rows = []

    for seed in SEEDS:
        part = focus.loc[
            focus["seed"] == seed
        ]

        basic_brier = float(
            part["se_basic"].mean()
        )
        middle_brier = float(
            part["se_middle5"].mean()
        )

        rows.append(
            {
                "seed": int(seed),
                "n": int(len(part)),
                "basic_brier": basic_brier,
                "middle5_brier": middle_brier,
                "delta_brier_basic_minus_middle5": (
                    basic_brier
                    - middle_brier
                ),
                "middle5_better": bool(
                    basic_brier
                    - middle_brier
                    > 0
                ),
                "mean_abs_prediction_shift": float(
                    np.abs(
                        part["pred_shift"]
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Group 단위 robust summary
# =========================================================

def summarize_group_seed_robustness(
    stacked_with_features,
    group_cols,
    period_name,
):
    """
    특정 그룹에서 Basic vs Middle5의 Brier 차이가
    몇 개 seed에서 같은 방향인지 계산한다.

    그룹별 seed Brier를 먼저 계산한 뒤,
    seed 간 평균 delta / std / 승리 횟수를 요약한다.
    """
    temp = stacked_with_features.copy()

    # 그룹 내 seed별 Brier
    seed_group = (
        temp
        .groupby(
            ["seed"] + list(group_cols),
            dropna=False,
            as_index=False,
        )
        .agg(
            n=(
                ID_COL,
                "size",
            ),
            target_rate=(
                "target",
                "mean",
            ),
            basic_brier=(
                "se_basic",
                "mean",
            ),
            middle5_brier=(
                "se_middle5",
                "mean",
            ),
            mean_pred_shift=(
                "pred_shift",
                "mean",
            ),
        )
    )

    seed_group["delta_brier"] = (
        seed_group["basic_brier"]
        - seed_group["middle5_brier"]
    )

    seed_group["middle5_better"] = (
        seed_group["delta_brier"] > 0
    )

    group_summary = (
        seed_group
        .groupby(
            list(group_cols),
            dropna=False,
            as_index=False,
        )
        .agg(
            n_per_seed=(
                "n",
                "mean",
            ),
            target_rate=(
                "target_rate",
                "mean",
            ),
            basic_brier_mean=(
                "basic_brier",
                "mean",
            ),
            middle5_brier_mean=(
                "middle5_brier",
                "mean",
            ),
            mean_delta_across_seeds=(
                "delta_brier",
                "mean",
            ),
            std_delta_across_seeds=(
                "delta_brier",
                "std",
            ),
            middle5_seed_win_count=(
                "middle5_better",
                "sum",
            ),
            mean_pred_shift=(
                "mean_pred_shift",
                "mean",
            ),
        )
    )

    group_summary["period"] = period_name

    # 해당 partition 안에서의 weighted contribution.
    # 동일 grouping 내 contribution 합은 overall delta와 대응한다.
    total_n = float(
        group_summary[
            "n_per_seed"
        ].sum()
    )

    if total_n > 0:
        group_summary[
            "weighted_delta_contribution"
        ] = (
            group_summary[
                "n_per_seed"
            ]
            / total_n
            * group_summary[
                "mean_delta_across_seeds"
            ]
        )
    else:
        group_summary[
            "weighted_delta_contribution"
        ] = np.nan

    return (
        group_summary,
        seed_group,
    )


# =========================================================
# Middle Feature 구간 분석
# =========================================================

def analyze_middle_feature_bins(
    stacked,
    row_features,
):
    """
    2021~2023에서 만든 고정 quantile bin을
    2021~2023과 2024 모두에 적용한다.

    이를 통해
    - Feature-target 관계 변화
    - Feature 구간별 Middle5 효과 변화
    - 5개 seed에서 방향이 반복되는지
    를 확인한다.
    """
    historical_rows = row_features.loc[
        row_features["season"].isin(
            HISTORICAL_VALID_SEASONS
        )
    ]

    all_summaries = []
    bin_edges_meta = {}

    for feature in MIDDLE_TEMPORAL_FEATURES:
        edges = make_quantile_edges(
            historical_rows[feature],
            n_bins=N_BINS,
        )

        if edges is None:
            continue

        bin_edges_meta[feature] = [
            None
            if not np.isfinite(x)
            else float(x)
            for x in edges
        ]

        feature_bins = row_features[
            [
                ID_COL,
                "season",
                feature,
            ]
        ].copy()

        feature_bins["feature_bin"] = (
            apply_bins(
                feature_bins[feature],
                edges,
                prefix="Q",
            )
        )

        temp = stacked.merge(
            feature_bins[
                [
                    ID_COL,
                    "feature_bin",
                ]
            ],
            on=ID_COL,
            how="left",
            validate="many_to_one",
        )

        # Historical 2021~2023
        hist = temp.loc[
            temp["season"].isin(
                HISTORICAL_VALID_SEASONS
            )
        ].copy()

        hist_summary, _ = (
            summarize_group_seed_robustness(
                hist,
                group_cols=["feature_bin"],
                period_name="2021_2023",
            )
        )

        hist_summary.insert(
            0,
            "feature",
            feature,
        )

        # 2024
        focus = temp.loc[
            temp["season"] == FOCUS_SEASON
        ].copy()

        focus_summary, _ = (
            summarize_group_seed_robustness(
                focus,
                group_cols=["feature_bin"],
                period_name="2024",
            )
        )

        focus_summary.insert(
            0,
            "feature",
            feature,
        )

        all_summaries.extend(
            [
                hist_summary,
                focus_summary,
            ]
        )

    if not all_summaries:
        return (
            pd.DataFrame(),
            bin_edges_meta,
        )

    result = pd.concat(
        all_summaries,
        axis=0,
        ignore_index=True,
    )

    return (
        result,
        bin_edges_meta,
    )


# =========================================================
# Middle Feature 분포 변화
# =========================================================

def analyze_distribution_shift(
    row_features,
    bin_edges_meta,
):
    """
    2021~2023 vs 2024의 Middle Feature 분포 차이를 확인한다.

    SMD:
      |값|이 클수록 평균 위치가 많이 이동했다는 단순 진단 지표.

    PSI:
      같은 고정 bin에서 두 기간의 비율 차이를 요약한 분포 변화 지표.
      이 값만으로 Feature 품질을 판단하지 않는다.
    """
    hist = row_features.loc[
        row_features["season"].isin(
            HISTORICAL_VALID_SEASONS
        )
    ]

    focus = row_features.loc[
        row_features["season"] == FOCUS_SEASON
    ]

    rows = []

    for feature in MIDDLE_TEMPORAL_FEATURES:
        h = pd.to_numeric(
            hist[feature],
            errors="coerce",
        )

        f = pd.to_numeric(
            focus[feature],
            errors="coerce",
        )

        h_non = h.dropna()
        f_non = f.dropna()

        h_mean = float(
            h_non.mean()
        ) if len(h_non) else np.nan

        f_mean = float(
            f_non.mean()
        ) if len(f_non) else np.nan

        h_std = safe_std(
            h_non
        )

        f_std = safe_std(
            f_non
        )

        pooled_std = np.sqrt(
            np.nanmean(
                [
                    h_std ** 2
                    if np.isfinite(h_std)
                    else np.nan,
                    f_std ** 2
                    if np.isfinite(f_std)
                    else np.nan,
                ]
            )
        )

        if (
            np.isfinite(pooled_std)
            and pooled_std > 0
        ):
            smd = float(
                (f_mean - h_mean)
                / pooled_std
            )
        else:
            smd = np.nan

        # PSI
        edges_raw = (
            bin_edges_meta
            .get(feature)
        )

        psi = np.nan

        if edges_raw is not None:
            edges = np.array(
                [
                    -np.inf
                    if x is None and i == 0
                    else (
                        np.inf
                        if x is None
                        else x
                    )
                    for i, x
                    in enumerate(edges_raw)
                ],
                dtype=np.float64,
            )

            # 마지막 None은 +inf, 첫 None은 -inf가 되도록 수정
            if len(edges) >= 2:
                edges[0] = -np.inf
                edges[-1] = np.inf

            h_bin = apply_bins(
                h,
                edges,
            )
            f_bin = apply_bins(
                f,
                edges,
            )

            categories = sorted(
                set(h_bin.unique())
                | set(f_bin.unique())
            )

            epsilon = 1e-6
            psi_sum = 0.0

            for category in categories:
                hp = float(
                    (h_bin == category).mean()
                )
                fp = float(
                    (f_bin == category).mean()
                )

                hp = max(
                    hp,
                    epsilon,
                )
                fp = max(
                    fp,
                    epsilon,
                )

                psi_sum += (
                    (fp - hp)
                    * np.log(fp / hp)
                )

            psi = float(psi_sum)

        rows.append(
            {
                "feature": feature,
                "hist_n": int(
                    h.notna().sum()
                ),
                "focus_2024_n": int(
                    f.notna().sum()
                ),
                "hist_missing_rate": float(
                    h.isna().mean()
                ),
                "focus_2024_missing_rate": float(
                    f.isna().mean()
                ),
                "hist_mean": h_mean,
                "focus_2024_mean": f_mean,
                "hist_std": h_std,
                "focus_2024_std": f_std,
                "hist_median": float(
                    h_non.median()
                )
                if len(h_non)
                else np.nan,
                "focus_2024_median": float(
                    f_non.median()
                )
                if len(f_non)
                else np.nan,
                "hist_q10": float(
                    h_non.quantile(0.10)
                )
                if len(h_non)
                else np.nan,
                "focus_2024_q10": float(
                    f_non.quantile(0.10)
                )
                if len(f_non)
                else np.nan,
                "hist_q90": float(
                    h_non.quantile(0.90)
                )
                if len(h_non)
                else np.nan,
                "focus_2024_q90": float(
                    f_non.quantile(0.90)
                )
                if len(f_non)
                else np.nan,
                "standardized_mean_difference": smd,
                "psi_fixed_bins": psi,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Middle Level × Trend 결합 상태 분석
# =========================================================

def add_middle_level_trend_indices(
    row_features,
):
    """
    분석용 Middle Level index / Trend index를 만든다.

    주의:
    이 index는 모델에 넣기 위한 새로운 Feature가 아니다.
    3개의 Level과 2개의 Trend를 한눈에 보기 위한 진단용 요약값이다.

    각 feature를 2021~2023 평균/표준편차로 z-score한 뒤
    Level 3개 평균, Trend 2개 평균을 사용한다.
    """
    result = row_features.copy()

    hist_mask = (
        result["season"]
        .isin(
            HISTORICAL_VALID_SEASONS
        )
    )

    standardization = {}

    level_z_cols = []
    trend_z_cols = []

    for feature in MIDDLE_TEMPORAL_FEATURES:
        hist_values = pd.to_numeric(
            result.loc[
                hist_mask,
                feature,
            ],
            errors="coerce",
        )

        mean = float(
            hist_values.mean()
        )
        std = float(
            hist_values.std(ddof=0)
        )

        standardization[feature] = {
            "mean": mean,
            "std": std,
        }

        z_col = f"__z_{feature}"

        if (
            np.isfinite(std)
            and std > 0
        ):
            result[z_col] = (
                pd.to_numeric(
                    result[feature],
                    errors="coerce",
                )
                - mean
            ) / std
        else:
            result[z_col] = np.nan

        if feature in MIDDLE_LEVEL_FEATURES:
            level_z_cols.append(
                z_col
            )
        else:
            trend_z_cols.append(
                z_col
            )

    result[
        "middle_level_index"
    ] = result[
        level_z_cols
    ].mean(
        axis=1,
        skipna=True,
    )

    result[
        "middle_trend_index"
    ] = result[
        trend_z_cols
    ].mean(
        axis=1,
        skipna=True,
    )

    result = result.drop(
        columns=(
            level_z_cols
            + trend_z_cols
        )
    )

    return (
        result,
        standardization,
    )


def analyze_level_trend_joint(
    stacked,
    row_features,
):
    """
    분석용 Level index와 Trend index를 3분위로 나눠
    3 x 3 결합 상태에서 2024 개선이 어디에 집중되는지 확인한다.
    """
    indexed, standardization = (
        add_middle_level_trend_indices(
            row_features
        )
    )

    hist = indexed.loc[
        indexed["season"].isin(
            HISTORICAL_VALID_SEASONS
        )
    ]

    level_edges = make_quantile_edges(
        hist["middle_level_index"],
        n_bins=3,
    )

    trend_edges = make_quantile_edges(
        hist["middle_trend_index"],
        n_bins=3,
    )

    indexed["level_bin"] = (
        apply_bins(
            indexed[
                "middle_level_index"
            ],
            level_edges,
            prefix="L",
        )
    )

    indexed["trend_bin"] = (
        apply_bins(
            indexed[
                "middle_trend_index"
            ],
            trend_edges,
            prefix="T",
        )
    )

    temp = stacked.merge(
        indexed[
            [
                ID_COL,
                "level_bin",
                "trend_bin",
                "middle_level_index",
                "middle_trend_index",
            ]
        ],
        on=ID_COL,
        how="left",
        validate="many_to_one",
    )

    summaries = []

    for period_name, mask in [
        (
            "2021_2023",
            temp["season"].isin(
                HISTORICAL_VALID_SEASONS
            ),
        ),
        (
            "2024",
            temp["season"]
            == FOCUS_SEASON,
        ),
    ]:
        summary, _ = (
            summarize_group_seed_robustness(
                temp.loc[mask].copy(),
                group_cols=[
                    "level_bin",
                    "trend_bin",
                ],
                period_name=period_name,
            )
        )

        summaries.append(
            summary
        )

    joint = pd.concat(
        summaries,
        axis=0,
        ignore_index=True,
    )

    meta = {
        "standardization": standardization,
        "level_edges": (
            None
            if level_edges is None
            else [
                None
                if not np.isfinite(x)
                else float(x)
                for x in level_edges
            ]
        ),
        "trend_edges": (
            None
            if trend_edges is None
            else [
                None
                if not np.isfinite(x)
                else float(x)
                for x in trend_edges
            ]
        ),
    }

    return (
        joint,
        indexed[
            [
                ID_COL,
                "season",
                "middle_level_index",
                "middle_trend_index",
                "level_bin",
                "trend_bin",
            ]
        ],
        meta,
    )


# =========================================================
# 경기 상황 분석
# =========================================================

def analyze_situations(
    stacked,
    row_features,
):
    """
    Middle5 개선이 특정 경기 상황에 집중되는지 확인한다.

    각각 별도 partition으로 분석하기 때문에
    서로 다른 situation table의 contribution을 직접 합치면 안 된다.
    """
    situation_specs = {
        "count": [
            "count_state",
        ],
        "outs": [
            "outs_before",
        ],
        "base_state": [
            "base_state",
        ],
        "num_runners": [
            "num_runners_on",
        ],
        "inning": [
            "inning",
        ],
        "top_bottom": [
            "top_bottom",
        ],
        "game_type": [
            "game_type",
        ],
    }

    join_cols = [
        ID_COL,
        "balls_before",
        "strikes_before",
        "outs_before",
        "base_state",
        "num_runners_on",
        "inning",
        "top_bottom",
        "game_type",
    ]

    info = row_features[
        join_cols
    ].copy()

    info["count_state"] = (
        info["balls_before"]
        .astype("Int64")
        .astype(str)
        + "-"
        + info["strikes_before"]
        .astype("Int64")
        .astype(str)
    )

    temp = stacked.merge(
        info.drop(
            columns=[
                "balls_before",
                "strikes_before",
            ]
        ),
        on=ID_COL,
        how="left",
        validate="many_to_one",
    )

    all_summary = []

    for analysis_name, group_cols in (
        situation_specs.items()
    ):
        for period_name, mask in [
            (
                "2021_2023",
                temp["season"].isin(
                    HISTORICAL_VALID_SEASONS
                ),
            ),
            (
                "2024",
                temp["season"]
                == FOCUS_SEASON,
            ),
        ]:
            summary, _ = (
                summarize_group_seed_robustness(
                    temp.loc[mask].copy(),
                    group_cols=group_cols,
                    period_name=period_name,
                )
            )

            summary.insert(
                0,
                "analysis",
                analysis_name,
            )

            # 어떤 상황 컬럼이든 한 CSV에서 보기 쉽게 문자열 label 생성
            summary["group_label"] = (
                summary[
                    group_cols
                ]
                .astype(str)
                .agg(
                    "|".join,
                    axis=1,
                )
            )

            all_summary.append(
                summary
            )

    return pd.concat(
        all_summary,
        axis=0,
        ignore_index=True,
    )


# =========================================================
# Basic prediction 확률 구간 분석
# =========================================================

def analyze_prediction_bins(
    stacked,
    row_level,
):
    """
    Middle5가 Basic의 어떤 확률 구간에서 주로 개선되는지 확인한다.

    2021~2023의 Basic 평균 prediction으로 고정 bin을 만들고,
    2024에도 같은 edge를 적용한다.
    """
    hist_rows = row_level.loc[
        row_level["season"].isin(
            HISTORICAL_VALID_SEASONS
        )
    ]

    edges = make_quantile_edges(
        hist_rows[
            "mean_pred_basic"
        ],
        n_bins=10,
    )

    bins = row_level[
        [
            ID_COL,
            "season",
            "mean_pred_basic",
        ]
    ].copy()

    bins["basic_pred_bin"] = (
        apply_bins(
            bins["mean_pred_basic"],
            edges,
            prefix="P",
        )
    )

    temp = stacked.merge(
        bins[
            [
                ID_COL,
                "basic_pred_bin",
            ]
        ],
        on=ID_COL,
        how="left",
        validate="many_to_one",
    )

    summaries = []

    for period_name, mask in [
        (
            "2021_2023",
            temp["season"].isin(
                HISTORICAL_VALID_SEASONS
            ),
        ),
        (
            "2024",
            temp["season"]
            == FOCUS_SEASON,
        ),
    ]:
        summary, _ = (
            summarize_group_seed_robustness(
                temp.loc[mask].copy(),
                group_cols=[
                    "basic_pred_bin",
                ],
                period_name=period_name,
            )
        )

        summaries.append(
            summary
        )

    result = pd.concat(
        summaries,
        axis=0,
        ignore_index=True,
    )

    meta = {
        "basic_prediction_bin_edges": (
            None
            if edges is None
            else [
                None
                if not np.isfinite(x)
                else float(x)
                for x in edges
            ]
        )
    }

    return (
        result,
        meta,
    )


# =========================================================
# 콘솔용 핵심 결과 출력
# =========================================================

def print_top_segments(
    df,
    label_cols,
    title,
    n=8,
):
    """
    2024에서 seed 평균 delta가 큰/작은 그룹을 콘솔에 표시한다.
    작은 그룹은 CSV에는 남기되 콘솔 ranking에서는 제외한다.
    """
    focus = df.loc[
        (df["period"] == "2024")
        & (
            df["n_per_seed"]
            >= CONSOLE_MIN_N
        )
    ].copy()

    if len(focus) == 0:
        return

    focus["segment"] = (
        focus[label_cols]
        .astype(str)
        .agg(
            " | ".join,
            axis=1,
        )
    )

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)

    print("[2024 개선 상위]")
    improved = (
        focus
        .sort_values(
            "mean_delta_across_seeds",
            ascending=False,
        )
        .head(n)
    )

    print(
        improved[
            [
                "segment",
                "n_per_seed",
                "mean_delta_across_seeds",
                "middle5_seed_win_count",
                "weighted_delta_contribution",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("[2024 악화 상위]")
    worsened = (
        focus
        .sort_values(
            "mean_delta_across_seeds",
            ascending=True,
        )
        .head(n)
    )

    print(
        worsened[
            [
                "segment",
                "n_per_seed",
                "mean_delta_across_seeds",
                "middle5_seed_win_count",
                "weighted_delta_contribution",
            ]
        ].to_string(
            index=False
        )
    )


# =========================================================
# Main
# =========================================================

def main():
    if VALIDATION_DIR.exists():
        raise FileExistsError(
            f"기존 분석 폴더가 있습니다: {VALIDATION_DIR}\n"
            "기존 결과 보호를 위해 자동 덮어쓰지 않습니다."
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        Path(__file__).resolve(),
        VALIDATION_DIR
        / "analyze_middle5_2024_cause.py",
    )

    # -----------------------------------------------------
    # 1. Multi-Seed OOF 로드
    # -----------------------------------------------------

    stacked = load_multiseed_oof()

    row_level = (
        aggregate_rows_across_seeds(
            stacked
        )
    )

    # -----------------------------------------------------
    # 2. Train에서 분석용 Feature / 상황 정보 로드
    # -----------------------------------------------------

    needed_cols = [
        ID_COL,
        "season",
        TARGET_COL,
        "balls_before",
        "strikes_before",
        "outs_before",
        "base_state",
        "num_runners_on",
        "inning",
        "top_bottom",
        "game_type",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=needed_cols,
    )

    if not train[ID_COL].is_unique:
        raise AssertionError(
            "train.csv row_id 중복"
        )

    train = (
        add_middle_temporal_features(
            train
        )
    )

    # OOF에 실제 포함된 2021~2024 row만 사용
    oof_ids = row_level[
        [
            ID_COL,
            "season",
            "target",
        ]
    ].copy()

    row_features = oof_ids.merge(
        train,
        on=ID_COL,
        how="left",
        suffixes=("_oof", "_train"),
        validate="one_to_one",
    )

    if len(row_features) != len(
        row_level
    ):
        raise AssertionError(
            "OOF / train join row 수 불일치"
        )

    if row_features[
        "season_train"
    ].isna().any():
        raise AssertionError(
            "train.csv에서 찾지 못한 OOF row가 있습니다."
        )

    if not np.array_equal(
        row_features[
            "season_oof"
        ].to_numpy(),
        row_features[
            "season_train"
        ].to_numpy(),
    ):
        raise AssertionError(
            "OOF / train season 불일치"
        )

    if not np.array_equal(
        row_features[
            "target"
        ].to_numpy(),
        row_features[
            TARGET_COL
        ].to_numpy(),
    ):
        raise AssertionError(
            "OOF / train target 불일치"
        )

    row_features = (
        row_features
        .drop(
            columns=[
                "season_train",
                TARGET_COL,
            ]
        )
        .rename(
            columns={
                "season_oof": "season",
            }
        )
    )

    # row-level prediction 평균도 분석 테이블에 결합
    row_features = row_features.merge(
        row_level.drop(
            columns=[
                "season",
                "target",
            ]
        ),
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # 3. 2024 seed별 전체 결과 재확인
    # -----------------------------------------------------

    seed_2024 = summarize_2024_by_seed(
        stacked
    )

    seed_2024.to_csv(
        VALIDATION_DIR
        / "2024_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 4. Middle Feature 고정 bin 분석
    # -----------------------------------------------------

    (
        middle_bin_summary,
        bin_edges_meta,
    ) = analyze_middle_feature_bins(
        stacked=stacked,
        row_features=row_features,
    )

    middle_bin_summary.to_csv(
        VALIDATION_DIR
        / "middle_feature_bin_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 5. Feature 분포 변화 분석
    # -----------------------------------------------------

    distribution_shift = (
        analyze_distribution_shift(
            row_features=row_features,
            bin_edges_meta=bin_edges_meta,
        )
    )

    distribution_shift.to_csv(
        VALIDATION_DIR
        / "middle_feature_distribution_shift.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 6. Middle Level × Trend 결합 분석
    # -----------------------------------------------------

    (
        level_trend_joint,
        level_trend_row,
        level_trend_meta,
    ) = analyze_level_trend_joint(
        stacked=stacked,
        row_features=row_features,
    )

    level_trend_joint.to_csv(
        VALIDATION_DIR
        / "middle_level_trend_joint.csv",
        index=False,
        encoding="utf-8-sig",
    )

    level_trend_row.to_csv(
        VALIDATION_DIR
        / "middle_level_trend_row_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 7. 경기 상황 분석
    # -----------------------------------------------------

    situation_summary = (
        analyze_situations(
            stacked=stacked,
            row_features=row_features,
        )
    )

    situation_summary.to_csv(
        VALIDATION_DIR
        / "situation_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 8. Basic Prediction 확률 구간 분석
    # -----------------------------------------------------

    (
        prediction_bin_summary,
        prediction_bin_meta,
    ) = analyze_prediction_bins(
        stacked=stacked,
        row_level=row_level,
    )

    prediction_bin_summary.to_csv(
        VALIDATION_DIR
        / "basic_prediction_bin_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 9. 2024 row-level 상세 저장
    # -----------------------------------------------------

    row_2024 = row_features.loc[
        row_features["season"]
        == FOCUS_SEASON
    ].copy()

    row_2024.to_csv(
        VALIDATION_DIR
        / "2024_row_level_multiseed_mean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 10. Metadata
    # -----------------------------------------------------

    metadata = {
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "source_experiment_name": (
            SOURCE_EXPERIMENT_NAME
        ),
        "seeds": SEEDS,
        "historical_validation_seasons": (
            HISTORICAL_VALID_SEASONS
        ),
        "focus_season": FOCUS_SEASON,
        "middle_level_features": (
            MIDDLE_LEVEL_FEATURES
        ),
        "middle_trend_features": (
            MIDDLE_TREND_FEATURES
        ),
        "delta_definition": (
            "Basic squared error - Middle5 squared error; "
            "positive means Middle5 is better"
        ),
        "fixed_bin_note": (
            "Feature/prediction bins are derived from 2021~2023 "
            "and applied unchanged to 2024."
        ),
        "middle_feature_bin_edges": (
            bin_edges_meta
        ),
        "middle_level_trend_meta": (
            level_trend_meta
        ),
        "prediction_bin_meta": (
            prediction_bin_meta
        ),
        "interpretation_warning": (
            "This is diagnostic analysis, not causal proof. "
            "Any pattern found here must be converted into a hypothesis "
            "and re-tested with the fixed expanding-window validation."
        ),
    }

    with (
        VALIDATION_DIR
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # 콘솔 요약
    # -----------------------------------------------------

    print()
    print("=" * 96)
    print("2024 Basic vs Middle5 - Seed별 재확인")
    print("=" * 96)
    print(
        seed_2024.to_string(
            index=False
        )
    )

    print()
    print("=" * 96)
    print("2021~2023 vs 2024 - Middle Feature 분포 변화")
    print("=" * 96)

    distribution_display = (
        distribution_shift[
            [
                "feature",
                "hist_mean",
                "focus_2024_mean",
                "standardized_mean_difference",
                "psi_fixed_bins",
                "hist_missing_rate",
                "focus_2024_missing_rate",
            ]
        ]
        .sort_values(
            "psi_fixed_bins",
            ascending=False,
        )
    )

    print(
        distribution_display
        .to_string(
            index=False
        )
    )

    print_top_segments(
        middle_bin_summary,
        label_cols=[
            "feature",
            "feature_bin",
        ],
        title=(
            "Middle Temporal 값 구간별 "
            "2024 개선/악화"
        ),
        n=10,
    )

    print_top_segments(
        level_trend_joint,
        label_cols=[
            "level_bin",
            "trend_bin",
        ],
        title=(
            "Middle Level × Trend 결합 상태별 "
            "2024 개선/악화"
        ),
        n=9,
    )

    # 경기상황은 2024만 추려 개선량 상위 표시
    situation_2024 = (
        situation_summary.loc[
            situation_summary["period"]
            == "2024"
        ]
        .copy()
    )

    situation_2024 = (
        situation_2024.loc[
            situation_2024[
                "n_per_seed"
            ]
            >= CONSOLE_MIN_N
        ]
    )

    if len(situation_2024):
        print()
        print("=" * 96)
        print("경기 상황별 2024 개선 상위")
        print("=" * 96)

        print(
            situation_2024[
                [
                    "analysis",
                    "group_label",
                    "n_per_seed",
                    "mean_delta_across_seeds",
                    "middle5_seed_win_count",
                    "weighted_delta_contribution",
                ]
            ]
            .sort_values(
                "mean_delta_across_seeds",
                ascending=False,
            )
            .head(15)
            .to_string(
                index=False
            )
        )

        print()
        print("=" * 96)
        print("경기 상황별 2024 악화 상위")
        print("=" * 96)

        print(
            situation_2024[
                [
                    "analysis",
                    "group_label",
                    "n_per_seed",
                    "mean_delta_across_seeds",
                    "middle5_seed_win_count",
                    "weighted_delta_contribution",
                ]
            ]
            .sort_values(
                "mean_delta_across_seeds",
                ascending=True,
            )
            .head(15)
            .to_string(
                index=False
            )
        )

    print_top_segments(
        prediction_bin_summary,
        label_cols=[
            "basic_pred_bin",
        ],
        title=(
            "Basic 예측 확률 구간별 "
            "2024 Middle5 보정 효과"
        ),
        n=10,
    )

    print()
    print("=" * 96)
    print("분석 완료")
    print("=" * 96)
    print(
        "저장 위치:",
        VALIDATION_DIR,
    )
    print()
    print(
        "가장 먼저 볼 파일:"
    )
    print(
        "1) 2024_seed_summary.csv"
    )
    print(
        "2) middle_feature_bin_comparison.csv"
    )
    print(
        "3) middle_feature_distribution_shift.csv"
    )
    print(
        "4) middle_level_trend_joint.csv"
    )
    print(
        "5) situation_comparison.csv"
    )
    print(
        "6) basic_prediction_bin_comparison.csv"
    )
    print()
    print(
        "해석: mean_delta_across_seeds > 0 이면 "
        "해당 구간에서 Middle5가 평균적으로 개선."
    )
    print(
        "middle5_seed_win_count=5 이면 "
        "그 구간에서도 5개 seed 모두 같은 방향으로 개선."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
