# ============================================================
# v18_on_exact_asof_v001.py
#
# 목적
# ------------------------------------------------------------
# 이미 생성된 v003_recent_method_ablation_v001의 OOF를 이용해
#
#   1) canonical v003 + V18
#   2) v003 + Exact-ASOF + V18
#
# 을 strict next-season 방식으로 비교한다.
#
# 모델을 다시 학습하지 않는다.
#
# ------------------------------------------------------------
# 입력
# ------------------------------------------------------------
# validation/v003_recent_method_ablation_v001/oof_predictions.csv
# data/train.csv
#
# OOF 파일에는 최소:
#   row_id
#   target
#   prediction
#   game_type
#   validation_season
#   variant
#
# 가 있어야 한다.
#
# ------------------------------------------------------------
# V18 설정
# ------------------------------------------------------------
# 기존 cb_reg_v018_gated에서 성공한 설정을 그대로 고정한다.
#
# Parent:
#   pitcher_id × batter_hand
#   min_n = 200
#   shrink_k = 300
#
# Detail:
#   pitcher_id × pressure_state × batter_hand
#   min_n = 200
#   shrink_k = 100
#
# Gate:
#   Parent + Detail same sign
#       -> 50:50
#
#   Parent only
#       -> Parent effect × 0.5
#
#   Conflict
#       -> correction 0
#
#   Detail only
#       -> correction 0
#
# alpha = 0.25
# cap   = 0.02
#
# ------------------------------------------------------------
# Strict Temporal Transfer
# ------------------------------------------------------------
# 2021 residual -> 2022
# 2022 residual -> 2023
# 2023 residual -> 2024
#
# 2024를 이용한 parameter tuning 없음.
#
# ------------------------------------------------------------
# residual
# ------------------------------------------------------------
# residual = y - prediction
#
# 이후 source season의 game_type별 평균 residual을 빼서
# R/F 전체 shift가 conditional effect에 섞이지 않도록 한다.
#
# ------------------------------------------------------------
# 가장 중요한 질문
# ------------------------------------------------------------
# Exact-ASOF가 이미 Base를 크게 개선했는데,
# 그 위에서도 V18 residual pattern이 추가로 남아 있는가?
#
# YES:
#   Exact-ASOF + V18 최종 모델 제작
#
# NO:
#   Exact-ASOF와 V18이 같은 오차를 해결했을 가능성이 큼
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 경로
# ============================================================

EXPERIMENT_NAME = "v18_on_exact_asof_v001"

DATA_DIR = Path("./data")

INPUT_OOF_PATH = (
    Path("./validation")
    / "v003_recent_method_ablation_v001"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    Path("./validation")
    / EXPERIMENT_NAME
)

TRAIN_PATH = DATA_DIR / "train.csv"


# ============================================================
# 기본 컬럼
# ============================================================

ID = "row_id"
TARGET = "target"
PRED = "prediction"

SEASON_COL = "validation_season"
VARIANT_COL = "variant"

GAME_TYPE_COL = "game_type"

PITCHER_COL = "pitcher_id"
BATTER_HAND_COL = "batter_hand"

BALLS_COL = "balls_before"
STRIKES_COL = "strikes_before"


# ============================================================
# 비교할 Base
# ============================================================

BASE_VARIANTS = [
    "base",
    "exact_asof",
]


# ============================================================
# Strict Next-Season Transfer
# ============================================================

TRANSITIONS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]


# ============================================================
# 기존 V18 frozen configuration
# ============================================================

MIN_N_PARENT = 200
MIN_N_DETAIL = 200

SHRINK_PARENT = 300.0
SHRINK_DETAIL = 100.0

PARENT_ONLY_SCALE = 0.5

DETAIL_WEIGHT = 0.5
PARENT_WEIGHT = 0.5

ALPHA = 0.25
CAP = 0.02


# ============================================================
# 기존 v003 + V18 결과
#
# 현재 코드가 과거 V18 구현과 같은지 sanity check용.
# ============================================================

EXPECTED_BASE_V18_GAIN = {
    2022: 0.00003800,
    2023: 0.00014050,
    2024: 0.00003061,
}

VERIFY_TOL = 2e-5


# ============================================================
# Metric
# ============================================================

def brier_score(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def domain_brier(
    df,
    pred_col,
    domain,
):

    mask = (
        df[GAME_TYPE_COL]
        .astype(str)
        == str(domain)
    )

    if mask.sum() == 0:
        return np.nan

    return brier_score(
        df.loc[
            mask,
            TARGET,
        ],
        df.loc[
            mask,
            pred_col,
        ],
    )


# ============================================================
# pressure_state
# ============================================================

def make_pressure_state(
    balls,
    strikes,
):
    """
    기존 V18 실험의 operational definition.

    full:
        3 balls AND 2 strikes

    high:
        3 balls OR 2 strikes
        단 full 제외

    normal:
        나머지
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


# ============================================================
# residual centering
# ============================================================

def add_centered_residual(
    source_df,
):
    """
    residual = target - prediction

    source season 안에서 game_type별 평균 residual을 제거한다.

    목적:
    R/F 전체적인 base-rate shift를 conditional correction으로
    잘못 학습하는 것을 방지.
    """

    out = source_df.copy()

    out["residual"] = (
        out[TARGET]
        - out[PRED]
    )

    domain_mean = (
        out
        .groupby(
            GAME_TYPE_COL,
            dropna=False,
        )["residual"]
        .transform("mean")
    )

    out["centered_residual"] = (
        out["residual"]
        - domain_mean
    )

    return out


# ============================================================
# correction table 생성
# ============================================================

def build_effect_table(
    source_df,
    group_cols,
    min_n,
    shrink_k,
):
    """
    source season OOF residual에서
    conditional residual effect를 계산한다.

    shrinkage:
        effect
        = mean residual
          × n / (n + shrink_k)
    """

    grouped = (
        source_df
        .groupby(
            group_cols,
            dropna=False,
            sort=False,
        )["centered_residual"]
        .agg(
            n="size",
            mean_residual="mean",
        )
        .reset_index()
    )

    # 충분한 표본이 있는 group만 사용
    grouped = (
        grouped.loc[
            grouped["n"]
            >= int(min_n)
        ]
        .copy()
    )

    grouped["reliability"] = (
        grouped["n"]
        / (
            grouped["n"]
            + float(shrink_k)
        )
    )

    grouped["effect"] = (
        grouped["mean_residual"]
        * grouped["reliability"]
    )

    return grouped


# ============================================================
# target row에 effect lookup
# ============================================================

def lookup_effect(
    target_df,
    table,
    group_cols,
    effect_name,
):
    """
    Frozen correction table lookup.

    target 내부 다른 row를 사용하지 않는다.
    """

    lookup_table = (
        table[
            group_cols
            + ["effect"]
        ]
        .rename(
            columns={
                "effect": effect_name
            }
        )
    )

    result = (
        target_df[
            group_cols
        ]
        .reset_index(
            drop=True
        )
        .merge(
            lookup_table,
            on=group_cols,
            how="left",
            sort=False,
        )
    )

    return result[
        effect_name
    ].to_numpy(
        dtype=np.float64
    )


# ============================================================
# V18 Gate
# ============================================================

def apply_v18_gate(
    target_df,
    parent_effect,
    detail_effect,
):
    """
    기존 Gated V18 방식.

    1. Parent + Detail 존재 & 같은 방향
       -> 50:50

    2. Parent만 존재
       -> Parent × 0.5

    3. 둘 다 존재하지만 방향 충돌
       -> correction 0

    4. Detail만 존재
       -> correction 0

    5. 둘 다 없음
       -> correction 0
    """

    parent_effect = np.asarray(
        parent_effect,
        dtype=np.float64,
    )

    detail_effect = np.asarray(
        detail_effect,
        dtype=np.float64,
    )

    parent_exists = np.isfinite(
        parent_effect
    )

    detail_exists = np.isfinite(
        detail_effect
    )

    both = (
        parent_exists
        & detail_exists
    )

    # 0 effect는 사실상 보정 의미가 없으므로
    # same sign으로 취급하지 않는다.
    same_sign = (
        both
        & (
            parent_effect
            * detail_effect
            > 0.0
        )
    )

    conflict = (
        both
        & (~same_sign)
    )

    parent_only = (
        parent_exists
        & (~detail_exists)
    )

    detail_only = (
        (~parent_exists)
        & detail_exists
    )

    neither = (
        (~parent_exists)
        & (~detail_exists)
    )

    gated_effect = np.zeros(
        len(target_df),
        dtype=np.float64,
    )

    # Parent + Detail 동의
    gated_effect[
        same_sign
    ] = (
        PARENT_WEIGHT
        * parent_effect[
            same_sign
        ]
        + DETAIL_WEIGHT
        * detail_effect[
            same_sign
        ]
    )

    # Parent only
    gated_effect[
        parent_only
    ] = (
        PARENT_ONLY_SCALE
        * parent_effect[
            parent_only
        ]
    )

    # alpha 적용
    correction = (
        ALPHA
        * gated_effect
    )

    # cap
    correction = np.clip(
        correction,
        -CAP,
        CAP,
    )

    raw_pred = (
        target_df[PRED]
        .to_numpy(
            dtype=np.float64
        )
    )

    corrected_pred = np.clip(
        raw_pred
        + correction,
        0.0,
        1.0,
    )

    diagnostics = {
        "n_rows": int(
            len(target_df)
        ),

        "same_sign_rows": int(
            same_sign.sum()
        ),

        "parent_only_rows": int(
            parent_only.sum()
        ),

        "conflict_rows": int(
            conflict.sum()
        ),

        "detail_only_rows": int(
            detail_only.sum()
        ),

        "neither_rows": int(
            neither.sum()
        ),

        "corrected_rows": int(
            (
                same_sign
                | parent_only
            ).sum()
        ),

        "coverage": float(
            (
                same_sign
                | parent_only
            ).mean()
        ),

        "mean_abs_correction": float(
            np.mean(
                np.abs(correction)
            )
        ),

        "max_abs_correction": float(
            np.max(
                np.abs(correction)
            )
        ),

        "positive_correction_rows": int(
            (correction > 0).sum()
        ),

        "negative_correction_rows": int(
            (correction < 0).sum()
        ),
    }

    return (
        corrected_pred,
        correction,
        diagnostics,
    )


# ============================================================
# 단일 transition
# ============================================================

def run_transition(
    variant_df,
    variant_name,
    source_season,
    target_season,
):

    source = (
        variant_df.loc[
            variant_df[
                SEASON_COL
            ]
            == source_season
        ]
        .copy()
    )

    target = (
        variant_df.loc[
            variant_df[
                SEASON_COL
            ]
            == target_season
        ]
        .copy()
    )

    if len(source) == 0:
        raise ValueError(
            f"{variant_name}: "
            f"source {source_season} 없음"
        )

    if len(target) == 0:
        raise ValueError(
            f"{variant_name}: "
            f"target {target_season} 없음"
        )

    # --------------------------------------------------------
    # Source residual
    # --------------------------------------------------------

    source = (
        add_centered_residual(
            source
        )
    )

    # --------------------------------------------------------
    # Parent table
    # pitcher_id × batter_hand
    # --------------------------------------------------------

    parent_cols = [
        PITCHER_COL,
        BATTER_HAND_COL,
    ]

    parent_table = (
        build_effect_table(
            source_df=source,
            group_cols=parent_cols,
            min_n=MIN_N_PARENT,
            shrink_k=SHRINK_PARENT,
        )
    )

    # --------------------------------------------------------
    # Detail table
    # pitcher_id × pressure_state × batter_hand
    # --------------------------------------------------------

    detail_cols = [
        PITCHER_COL,
        "pressure_state",
        BATTER_HAND_COL,
    ]

    detail_table = (
        build_effect_table(
            source_df=source,
            group_cols=detail_cols,
            min_n=MIN_N_DETAIL,
            shrink_k=SHRINK_DETAIL,
        )
    )

    # --------------------------------------------------------
    # Frozen lookup
    # --------------------------------------------------------

    parent_effect = (
        lookup_effect(
            target_df=target,
            table=parent_table,
            group_cols=parent_cols,
            effect_name="parent_effect",
        )
    )

    detail_effect = (
        lookup_effect(
            target_df=target,
            table=detail_table,
            group_cols=detail_cols,
            effect_name="detail_effect",
        )
    )

    # --------------------------------------------------------
    # Gate
    # --------------------------------------------------------

    (
        corrected_pred,
        correction,
        diagnostics,
    ) = apply_v18_gate(
        target_df=target,
        parent_effect=parent_effect,
        detail_effect=detail_effect,
    )

    target[
        "v18_prediction"
    ] = corrected_pred

    target[
        "v18_correction"
    ] = correction

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    raw_brier = (
        brier_score(
            target[TARGET],
            target[PRED],
        )
    )

    corrected_brier = (
        brier_score(
            target[TARGET],
            target[
                "v18_prediction"
            ],
        )
    )

    gain = (
        raw_brier
        - corrected_brier
    )

    raw_r = (
        domain_brier(
            target,
            PRED,
            "R",
        )
    )

    corrected_r = (
        domain_brier(
            target,
            "v18_prediction",
            "R",
        )
    )

    raw_f = (
        domain_brier(
            target,
            PRED,
            "F",
        )
    )

    corrected_f = (
        domain_brier(
            target,
            "v18_prediction",
            "F",
        )
    )

    r_gain = (
        raw_r
        - corrected_r
    )

    f_gain = (
        raw_f
        - corrected_f
    )

    result = {
        "variant": (
            variant_name
        ),

        "source_season": int(
            source_season
        ),

        "target_season": int(
            target_season
        ),

        "raw_brier": float(
            raw_brier
        ),

        "v18_brier": float(
            corrected_brier
        ),

        "gain": float(
            gain
        ),

        "raw_r_brier": float(
            raw_r
        ),

        "v18_r_brier": float(
            corrected_r
        ),

        "r_gain": float(
            r_gain
        ),

        "raw_f_brier": float(
            raw_f
        ),

        "v18_f_brier": float(
            corrected_f
        ),

        "f_gain": float(
            f_gain
        ),

        "parent_groups": int(
            len(parent_table)
        ),

        "detail_groups": int(
            len(detail_table)
        ),

        **diagnostics,
    }

    return (
        result,
        target,
    )


# ============================================================
# Variant 실행
# ============================================================

def run_variant(
    data,
    variant_name,
):

    df = (
        data.loc[
            data[
                VARIANT_COL
            ]
            == variant_name
        ]
        .copy()
    )

    if len(df) == 0:

        raise ValueError(
            f"OOF에 variant "
            f"'{variant_name}' 없음"
        )

    results = []
    corrected_frames = []

    print()
    print()
    print("#" * 110)
    print(
        f"VARIANT = {variant_name}"
    )
    print("#" * 110)

    for (
        source_season,
        target_season,
    ) in TRANSITIONS:

        (
            result,
            target,
        ) = run_transition(
            variant_df=df,
            variant_name=(
                variant_name
            ),
            source_season=(
                source_season
            ),
            target_season=(
                target_season
            ),
        )

        results.append(
            result
        )

        corrected_frames.append(
            target
        )

        print()
        print(
            f"{source_season}"
            f" -> "
            f"{target_season}"
        )

        print(
            f"Raw Brier  : "
            f"{result['raw_brier']:.10f}"
        )

        print(
            f"V18 Brier  : "
            f"{result['v18_brier']:.10f}"
        )

        print(
            f"Gain       : "
            f"{result['gain']:+.10f}"
        )

        print(
            f"R Gain     : "
            f"{result['r_gain']:+.10f}"
        )

        print(
            f"F Gain     : "
            f"{result['f_gain']:+.10f}"
        )

        print(
            f"Coverage   : "
            f"{result['coverage']:.6f}"
        )

        print(
            f"Same sign  : "
            f"{result['same_sign_rows']:,}"
        )

        print(
            f"Parent only: "
            f"{result['parent_only_rows']:,}"
        )

        print(
            f"Conflict   : "
            f"{result['conflict_rows']:,}"
        )

    return (
        pd.DataFrame(
            results
        ),
        pd.concat(
            corrected_frames,
            ignore_index=True,
        ),
    )


# ============================================================
# 기존 V18 sanity check
# ============================================================

def verify_base_v18(
    base_results,
):
    print()
    print("=" * 100)
    print("BASE V18 IMPLEMENTATION CHECK")
    print("=" * 100)

    for _, row in (
        base_results
        .sort_values(
            "target_season"
        )
        .iterrows()
    ):

        target_season = int(
            row[
                "target_season"
            ]
        )

        actual = float(
            row["gain"]
        )

        expected = (
            EXPECTED_BASE_V18_GAIN[
                target_season
            ]
        )

        diff = (
            actual
            - expected
        )

        status = (
            "PASS"
            if abs(diff)
            <= VERIFY_TOL
            else "CHECK"
        )

        print(
            f"{target_season}: "
            f"actual={actual:+.8f} | "
            f"expected≈{expected:+.8f} | "
            f"diff={diff:+.8f} | "
            f"{status}"
        )

    print("=" * 100)


# ============================================================
# Pooled summary
# ============================================================

def make_pooled_summary(
    corrected_df,
):

    rows = []

    for variant in (
        BASE_VARIANTS
    ):

        df = (
            corrected_df.loc[
                corrected_df[
                    VARIANT_COL
                ]
                == variant
            ]
            .copy()
        )

        raw_brier = (
            brier_score(
                df[TARGET],
                df[PRED],
            )
        )

        v18_brier = (
            brier_score(
                df[TARGET],
                df[
                    "v18_prediction"
                ],
            )
        )

        rows.append(
            {
                "variant": variant,

                "raw_brier_2022_2024": (
                    raw_brier
                ),

                "v18_brier_2022_2024": (
                    v18_brier
                ),

                "v18_gain": (
                    raw_brier
                    - v18_brier
                ),

                "n_rows": int(
                    len(df)
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Exact-ASOF + V18이 기존 Base + V18보다
    # 얼마나 좋은지
    # --------------------------------------------------------

    base_row = (
        summary.loc[
            summary[
                "variant"
            ]
            == "base"
        ]
        .iloc[0]
    )

    exact_row = (
        summary.loc[
            summary[
                "variant"
            ]
            == "exact_asof"
        ]
        .iloc[0]
    )

    stacking_gain = (
        base_row[
            "v18_brier_2022_2024"
        ]
        - exact_row[
            "v18_brier_2022_2024"
        ]
    )

    return (
        summary,
        stacking_gain,
    )


# ============================================================
# Main
# ============================================================

def main():

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            f"{OUTPUT_DIR} already exists.\n"
            "기존 결과를 보존하려면 "
            "EXPERIMENT_NAME을 변경하세요."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # OOF load
    # ========================================================

    if not INPUT_OOF_PATH.exists():

        raise FileNotFoundError(
            f"OOF 파일이 없습니다:\n"
            f"{INPUT_OOF_PATH}"
        )

    oof = pd.read_csv(
        INPUT_OOF_PATH,
        encoding="utf-8-sig",
    )

    required_oof_cols = {
        ID,
        TARGET,
        PRED,
        GAME_TYPE_COL,
        SEASON_COL,
        VARIANT_COL,
    }

    missing = (
        required_oof_cols
        - set(
            oof.columns
        )
    )

    if missing:

        raise KeyError(
            "OOF 필수 컬럼 없음: "
            f"{sorted(missing)}"
        )

    # 필요한 variant만 사용
    oof = (
        oof.loc[
            oof[
                VARIANT_COL
            ].isin(
                BASE_VARIANTS
            )
        ]
        .copy()
    )

    # ========================================================
    # train metadata load
    # ========================================================

    train_meta = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID,
            PITCHER_COL,
            BATTER_HAND_COL,
            BALLS_COL,
            STRIKES_COL,
        ],
    )

    if not train_meta[
        ID
    ].is_unique:

        raise ValueError(
            "train row_id가 unique하지 않습니다."
        )

    # ========================================================
    # merge
    # ========================================================

    data = (
        oof.merge(
            train_meta,
            on=ID,
            how="left",
            validate="many_to_one",
        )
    )

    if data[
        PITCHER_COL
    ].isna().all():

        raise ValueError(
            "train metadata merge 실패"
        )

    # categorical key 문자열 통일
    data[PITCHER_COL] = (
        data[PITCHER_COL]
        .fillna("__MISSING__")
        .astype(str)
    )

    data[BATTER_HAND_COL] = (
        data[BATTER_HAND_COL]
        .fillna("__MISSING__")
        .astype(str)
    )

    # pressure state
    data["pressure_state"] = (
        make_pressure_state(
            data[BALLS_COL],
            data[STRIKES_COL],
        )
    )

    print()
    print("=" * 110)

    print(
        f"Experiment : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Input OOF  : "
        f"{INPUT_OOF_PATH}"
    )

    print(
        "Variants   : "
        "base / exact_asof"
    )

    print(
        "V18 params : "
        "FROZEN"
    )

    print(
        f"Parent     : "
        f"min_n={MIN_N_PARENT}, "
        f"shrink={SHRINK_PARENT}"
    )

    print(
        f"Detail     : "
        f"min_n={MIN_N_DETAIL}, "
        f"shrink={SHRINK_DETAIL}"
    )

    print(
        f"alpha={ALPHA}, "
        f"cap={CAP}"
    )

    print("=" * 110)

    # ========================================================
    # 각 variant
    # ========================================================

    all_results = []
    all_corrected = []

    for variant in (
        BASE_VARIANTS
    ):

        (
            results,
            corrected,
        ) = run_variant(
            data=data,
            variant_name=variant,
        )

        all_results.append(
            results
        )

        all_corrected.append(
            corrected
        )

    all_results = (
        pd.concat(
            all_results,
            ignore_index=True,
        )
    )

    all_corrected = (
        pd.concat(
            all_corrected,
            ignore_index=True,
        )
    )

    # ========================================================
    # 기존 V18 재현 확인
    # ========================================================

    base_results = (
        all_results.loc[
            all_results[
                "variant"
            ]
            == "base"
        ]
    )

    verify_base_v18(
        base_results
    )

    # ========================================================
    # Pooled
    # ========================================================

    (
        pooled_summary,
        stacking_gain,
    ) = make_pooled_summary(
        all_corrected
    )

    # ========================================================
    # 저장
    # ========================================================

    all_results.to_csv(
        OUTPUT_DIR
        / "transition_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_corrected.to_csv(
        OUTPUT_DIR
        / "corrected_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_summary.to_csv(
        OUTPUT_DIR
        / "pooled_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 최종 출력
    # ========================================================

    print()
    print()
    print("=" * 120)
    print("TRANSITION RESULTS")
    print("gain > 0 = V18 추가 개선")
    print("=" * 120)

    print(
        all_results[
            [
                "variant",
                "source_season",
                "target_season",
                "raw_brier",
                "v18_brier",
                "gain",
                "r_gain",
                "f_gain",
                "coverage",
                "parent_groups",
                "detail_groups",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print()
    print("=" * 120)
    print("POOLED 2022~2024")
    print("=" * 120)

    print(
        pooled_summary
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)

    print(
        "Exact-ASOF + V18 "
        "vs Base + V18 gain:"
    )

    print(
        f"{stacking_gain:+.10f}"
    )

    if stacking_gain > 0:

        print(
            "\nRESULT: "
            "Exact-ASOF + V18이 "
            "기존 Base + V18보다 좋음."
        )

    else:

        print(
            "\nRESULT: "
            "Exact-ASOF가 Base를 개선했지만 "
            "V18과 결합했을 때는 "
            "기존 Base + V18을 넘지 못함."
        )

    print()
    print(
        f"Saved to: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()