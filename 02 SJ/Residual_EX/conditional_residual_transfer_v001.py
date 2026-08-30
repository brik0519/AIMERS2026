# ============================================================
# conditional_residual_transfer_v001.py
#
# 목적
# ------------------------------------------------------------
# cb_reg_v003 / cb_reg_v009의 기존 OOF prediction을 이용하여
#
#   F + pitcher_id × batter_hand
#
# 조건에서 과거 시즌에 반복적으로 발생한 residual을 계산하고,
# 그 보정값을 "다음 시즌"에 그대로 적용했을 때
# Brier Score가 개선되는지 확인한다.
#
# 중요
# ------------------------------------------------------------
# - 모델 재학습 없음
# - Feature 추가 없음
# - Target season의 label로 correction을 만들지 않음
# - 2023 residual -> 2024 correction
# - test 데이터 사용 없음
#
# residual = y_true - prediction
#
# residual > 0 : 모델이 너무 낮게 예측
# residual < 0 : 모델이 너무 높게 예측
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 경로
# ============================================================

DATA_PATH = Path("./data/train.csv")

OOF_PATHS = {
    "cb_reg_v003": Path(
        "./validation/cb_reg_v003/oof_validation.npz"
    ),
    "cb_reg_v009": Path(
        "./validation/cb_reg_v009/oof_validation.npz"
    ),
}

OUTPUT_DIR = Path(
    "./validation/conditional_residual_transfer_v001"
)


# ============================================================
# 실험 설정
# ============================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

# 이번 실험에서는 F만 보정
DOMAIN_COL = "game_type"
DOMAIN_VALUE = "F"

# 참고깃허브에서 효과가 있었던 conditional 단위 중
# 가장 먼저 확인할 조합
GROUP_COLS = [
    "pitcher_id",
    "batter_hand",
]

# 표본이 적은 group의 보정량을 0 쪽으로 줄이기 위한 값
#
# correction =
#     n / (n + SHRINK_K)
#     * mean_residual
#
# 이번 실험에서는 100으로 고정하고
# 2024 결과를 보고 값을 튜닝하지 않는다.
SHRINK_K = 100.0


# next-season transfer
TRANSFER_PAIRS = [
    (2022, 2023),
    (2023, 2024),
]


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


# ============================================================
# OOF 불러오기
# ============================================================

def load_oof(npz_path):
    """
    기존 validation artifact의 OOF prediction을 불러온다.

    prediction 필드를 최종 평가 prediction으로 사용한다.
    """

    data = np.load(
        npz_path,
        allow_pickle=True,
    )

    print()
    print(f"[LOAD] {npz_path}")
    print("keys:", list(data.files))

    required = [
        "row_id",
        "target",
        "prediction",
        "validation_season",
    ]

    for key in required:
        if key not in data.files:
            raise KeyError(
                f"{npz_path}에 '{key}'가 없습니다."
            )

    oof = pd.DataFrame({
        "row_id": data["row_id"],
        "target": data["target"].astype(
            np.float64
        ),
        "prediction": data[
            "prediction"
        ].astype(
            np.float64
        ),
        "validation_season": data[
            "validation_season"
        ].astype(int),
    })

    return oof


# ============================================================
# Train 정보와 merge
# ============================================================

def attach_features(oof, train):
    """
    OOF의 row_id를 이용해 train.csv에서
    game_type / pitcher_id / batter_hand를 가져온다.
    """

    meta_cols = [
        ID_COL,
        DOMAIN_COL,
        *GROUP_COLS,
    ]

    meta = (
        train[meta_cols]
        .drop_duplicates(
            subset=[ID_COL]
        )
        .copy()
    )

    merged = oof.merge(
        meta,
        left_on="row_id",
        right_on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if merged[DOMAIN_COL].isna().any():
        n_missing = int(
            merged[DOMAIN_COL]
            .isna()
            .sum()
        )

        raise ValueError(
            f"train.csv와 merge되지 않은 row가 "
            f"{n_missing}개 있습니다."
        )

    return merged


# ============================================================
# Correction Table 생성
# ============================================================

def make_correction_table(
    source_df,
):
    """
    Source season의 F row만 사용하여

        pitcher_id × batter_hand

    별 평균 residual을 계산한다.

    Target season의 label은 절대 사용하지 않는다.
    """

    source_f = (
        source_df.loc[
            source_df[DOMAIN_COL]
            .astype(str)
            == DOMAIN_VALUE
        ]
        .copy()
    )

    source_f["residual"] = (
        source_f["target"]
        - source_f["prediction"]
    )

    table = (
        source_f
        .groupby(
            GROUP_COLS,
            dropna=False,
        )["residual"]
        .agg(
            n="size",
            mean_residual="mean",
        )
        .reset_index()
    )

    # --------------------------------------------
    # Shrinkage
    #
    # 데이터 많음:
    #   weight -> 1
    #
    # 데이터 적음:
    #   weight -> 0
    # --------------------------------------------

    table["shrink_weight"] = (
        table["n"]
        / (
            table["n"]
            + SHRINK_K
        )
    )

    table["correction"] = (
        table["mean_residual"]
        * table["shrink_weight"]
    )

    return table


# ============================================================
# 다음 시즌에 Correction 적용
# ============================================================

def apply_correction(
    target_df,
    correction_table,
):
    """
    다음 시즌의 F row에만
    과거 시즌에서 계산한 correction을 적용한다.

    R은 prediction 그대로 유지한다.
    """

    result = target_df.copy()

    result = result.merge(
        correction_table[
            GROUP_COLS
            + [
                "n",
                "mean_residual",
                "shrink_weight",
                "correction",
            ]
        ],
        on=GROUP_COLS,
        how="left",
    )

    result[
        "correction"
    ] = (
        result[
            "correction"
        ]
        .fillna(0.0)
    )

    result[
        "corrected_prediction"
    ] = result[
        "prediction"
    ]

    f_mask = (
        result[DOMAIN_COL]
        .astype(str)
        == DOMAIN_VALUE
    )

    result.loc[
        f_mask,
        "corrected_prediction",
    ] = (
        result.loc[
            f_mask,
            "prediction",
        ]
        + result.loc[
            f_mask,
            "correction",
        ]
    )

    result[
        "corrected_prediction"
    ] = np.clip(
        result[
            "corrected_prediction"
        ],
        0.0,
        1.0,
    )

    return result


# ============================================================
# 한 Transfer 평가
# ============================================================

def evaluate_transfer(
    model_name,
    df,
    source_season,
    target_season,
):
    source = (
        df.loc[
            df["validation_season"]
            == source_season
        ]
        .copy()
    )

    target = (
        df.loc[
            df["validation_season"]
            == target_season
        ]
        .copy()
    )

    if source.empty:
        raise ValueError(
            f"{model_name}: "
            f"{source_season} OOF 없음"
        )

    if target.empty:
        raise ValueError(
            f"{model_name}: "
            f"{target_season} OOF 없음"
        )

    correction_table = (
        make_correction_table(
            source
        )
    )

    corrected = apply_correction(
        target,
        correction_table,
    )

    # --------------------------------------------
    # 전체 Brier
    # --------------------------------------------

    raw_brier = brier_score(
        corrected["target"],
        corrected["prediction"],
    )

    corrected_brier = brier_score(
        corrected["target"],
        corrected[
            "corrected_prediction"
        ],
    )

    gain = (
        raw_brier
        - corrected_brier
    )

    # --------------------------------------------
    # F Brier
    # --------------------------------------------

    f_mask = (
        corrected[DOMAIN_COL]
        .astype(str)
        == DOMAIN_VALUE
    )

    f_raw_brier = brier_score(
        corrected.loc[
            f_mask,
            "target",
        ],
        corrected.loc[
            f_mask,
            "prediction",
        ],
    )

    f_corrected_brier = brier_score(
        corrected.loc[
            f_mask,
            "target",
        ],
        corrected.loc[
            f_mask,
            "corrected_prediction",
        ],
    )

    f_gain = (
        f_raw_brier
        - f_corrected_brier
    )

    # --------------------------------------------
    # correction table에 존재했던 F 비율
    # --------------------------------------------

    f_rows = corrected.loc[
        f_mask
    ]

    matched = (
        f_rows["n"]
        .notna()
    )

    coverage = float(
        matched.mean()
    )

    summary = {
        "model": model_name,
        "source_season": (
            source_season
        ),
        "target_season": (
            target_season
        ),
        "raw_brier": (
            raw_brier
        ),
        "corrected_brier": (
            corrected_brier
        ),
        "gain": gain,
        "f_raw_brier": (
            f_raw_brier
        ),
        "f_corrected_brier": (
            f_corrected_brier
        ),
        "f_gain": (
            f_gain
        ),
        "f_group_coverage": (
            coverage
        ),
        "n_correction_groups": int(
            len(correction_table)
        ),
    }

    return (
        summary,
        corrected,
        correction_table,
    )


# ============================================================
# Main
# ============================================================

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            DATA_PATH
        )

    for path in OOF_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # train.csv
    # --------------------------------------------------------

    train = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID_COL,
            DOMAIN_COL,
            "pitcher_id",
            "batter_hand",
        ],
    )

    summaries = []

    # --------------------------------------------------------
    # v003 / v009 동일 실험
    # --------------------------------------------------------

    for model_name, path in (
        OOF_PATHS.items()
    ):

        print()
        print("=" * 80)
        print(model_name)
        print("=" * 80)

        oof = load_oof(
            path
        )

        df = attach_features(
            oof,
            train,
        )

        for (
            source_season,
            target_season,
        ) in TRANSFER_PAIRS:

            (
                summary,
                corrected,
                correction_table,
            ) = evaluate_transfer(
                model_name=model_name,
                df=df,
                source_season=(
                    source_season
                ),
                target_season=(
                    target_season
                ),
            )

            summaries.append(
                summary
            )

            prefix = (
                f"{model_name}_"
                f"{source_season}_to_"
                f"{target_season}"
            )

            corrected.to_csv(
                OUTPUT_DIR
                / f"{prefix}_pred.csv",
                index=False,
                encoding="utf-8-sig",
            )

            correction_table.to_csv(
                OUTPUT_DIR
                / f"{prefix}_correction.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print()
            print(
                f"[{source_season} -> "
                f"{target_season}]"
            )

            print(
                f"Raw Brier       : "
                f"{summary['raw_brier']:.8f}"
            )

            print(
                f"Corrected Brier : "
                f"{summary['corrected_brier']:.8f}"
            )

            print(
                f"Overall Gain    : "
                f"{summary['gain']:+.8f}"
            )

            print(
                f"F Raw Brier     : "
                f"{summary['f_raw_brier']:.8f}"
            )

            print(
                f"F Corr Brier    : "
                f"{summary['f_corrected_brier']:.8f}"
            )

            print(
                f"F Gain          : "
                f"{summary['f_gain']:+.8f}"
            )

            print(
                f"F coverage      : "
                f"{summary['f_group_coverage']:.2%}"
            )

    # --------------------------------------------------------
    # 최종 summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        summary_df[
            [
                "model",
                "source_season",
                "target_season",
                "raw_brier",
                "corrected_brier",
                "gain",
                "f_gain",
                "f_group_coverage",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "결과 저장:",
        OUTPUT_DIR
        / "summary.csv"
    )


if __name__ == "__main__":
    main()