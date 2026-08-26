# ============================================================
# stable_hand_residual_correction_v001.py
#
# 목적
# ------------------------------------------------------------
# cb_reg_v003 / cb_reg_v009 OOF prediction을 대상으로
#
#   pitcher_hand × batter_hand
#
# 조건에서 여러 과거 시즌에 걸쳐 같은 방향으로 반복되는
# conditional residual만 다음 시즌 prediction에 보정한다.
#
# 모델 재학습 없음.
# Feature 추가 없음.
# Test 사용 없음.
#
# residual = target - prediction
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
    "./validation/stable_hand_residual_correction_v001"
)


# ============================================================
# 설정
# ============================================================

ID_COL = "row_id"

GROUP_COLS = [
    "pitcher_hand",
    "batter_hand",
]

# 최소 2개 과거 시즌에서 같은 방향이어야 보정
MIN_SOURCE_SEASONS = 2

# 보정을 너무 강하게 적용하지 않기 위한 shrinkage
# 이번 실험에서는 고정하고 튜닝하지 않는다.
SHRINK_K = 1000.0

# 2023:
#   2021~2022 residual로 보정
#
# 2024:
#   2021~2023 residual로 보정
TARGET_SEASONS = [
    2023,
    2024,
]


# ============================================================
# Metric
# ============================================================

def brier_score(
    y_true,
    y_pred,
):
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
# OOF Load
# ============================================================

def load_oof(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    if "prediction" in data.files:
        pred_key = "prediction"

    elif "prediction_clip" in data.files:
        pred_key = "prediction_clip"

    else:
        raise KeyError(
            f"{path}: prediction이 없습니다."
        )

    required = [
        "row_id",
        "target",
        "validation_season",
    ]

    for key in required:
        if key not in data.files:
            raise KeyError(
                f"{path}: {key} 없음"
            )

    return pd.DataFrame({
        "row_id": (
            data["row_id"]
        ),
        "target": (
            data["target"]
            .astype(np.float64)
        ),
        "prediction": (
            np.clip(
                data[pred_key]
                .astype(np.float64),
                0.0,
                1.0,
            )
        ),
        "validation_season": (
            data["validation_season"]
            .astype(int)
        ),
    })


# ============================================================
# row_id dtype 통일
# ============================================================

def normalize_ids(
    oof,
    train,
):
    oof = oof.copy()
    train = train.copy()

    oof_num = pd.to_numeric(
        oof["row_id"],
        errors="coerce",
    )

    train_num = pd.to_numeric(
        train[ID_COL],
        errors="coerce",
    )

    if (
        oof_num.notna().all()
        and train_num.notna().all()
    ):
        oof["row_id"] = (
            oof_num.astype(np.int64)
        )

        train[ID_COL] = (
            train_num.astype(np.int64)
        )

    else:
        oof["row_id"] = (
            oof["row_id"].astype(str)
        )

        train[ID_COL] = (
            train[ID_COL].astype(str)
        )

    return oof, train


# ============================================================
# Stable Correction Table 생성
# ============================================================

def build_stable_correction(
    source_df,
):
    """
    여러 과거 시즌의 residual을 이용한다.

    1. 시즌별 전체 residual 평균 계산
    2. hand matchup별 residual에서 전체 평균 제거
    3. 여러 시즌에서 부호가 같은 group만 선택
    4. 시즌별 effect를 동일 가중 평균
    5. shrinkage 적용
    """

    source_df = (
        source_df.copy()
    )

    source_df["residual"] = (
        source_df["target"]
        - source_df["prediction"]
    )

    # --------------------------------------------
    # 시즌 전체 residual 평균 제거
    #
    # 모델 전체 calibration 문제와
    # hand matchup 특유의 오차를 분리
    # --------------------------------------------

    season_global = (
        source_df
        .groupby(
            "validation_season"
        )["residual"]
        .transform("mean")
    )

    source_df[
        "centered_residual"
    ] = (
        source_df["residual"]
        - season_global
    )

    # --------------------------------------------
    # season × hand matchup 통계
    # --------------------------------------------

    stats = (
        source_df
        .groupby(
            [
                "validation_season",
                *GROUP_COLS,
            ],
            dropna=False,
        )
        .agg(
            n=(
                "centered_residual",
                "size",
            ),
            effect=(
                "centered_residual",
                "mean",
            ),
        )
        .reset_index()
    )

    rows = []

    for key, group in (
        stats.groupby(
            GROUP_COLS,
            dropna=False,
        )
    ):

        if not isinstance(
            key,
            tuple,
        ):
            key = (key,)

        group = (
            group
            .sort_values(
                "validation_season"
            )
            .copy()
        )

        # 최소 source season 수
        if (
            group[
                "validation_season"
            ]
            .nunique()
            < MIN_SOURCE_SEASONS
        ):
            continue

        effects = (
            group["effect"]
            .to_numpy(
                dtype=np.float64
            )
        )

        signs = np.sign(
            effects
        )

        # 0이거나 방향이 바뀌면
        # 안정적인 패턴이 아니므로 사용하지 않음
        if np.any(
            signs == 0
        ):
            continue

        if not np.all(
            signs == signs[0]
        ):
            continue

        # ----------------------------------------
        # 시즌 equal-weight
        #
        # 데이터가 많은 시즌 하나가
        # correction을 지배하지 않도록 함
        # ----------------------------------------

        mean_effect = float(
            np.mean(
                effects
            )
        )

        total_n = int(
            group["n"].sum()
        )

        shrink_weight = float(
            total_n
            / (
                total_n
                + SHRINK_K
            )
        )

        correction = float(
            mean_effect
            * shrink_weight
        )

        row = {
            "pitcher_hand": (
                key[0]
            ),
            "batter_hand": (
                key[1]
            ),
            "n_source_seasons": int(
                group[
                    "validation_season"
                ].nunique()
            ),
            "total_n": (
                total_n
            ),
            "mean_effect": (
                mean_effect
            ),
            "shrink_weight": (
                shrink_weight
            ),
            "correction": (
                correction
            ),
            "direction": (
                "UNDER_PREDICT"
                if correction > 0
                else "OVER_PREDICT"
            ),
        }

        for _, r in (
            group.iterrows()
        ):
            season = int(
                r[
                    "validation_season"
                ]
            )

            row[
                f"effect_{season}"
            ] = float(
                r["effect"]
            )

            row[
                f"n_{season}"
            ] = int(
                r["n"]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Correction 적용
# ============================================================

def apply_correction(
    target_df,
    correction_table,
):

    result = (
        target_df.copy()
    )

    result = result.merge(
        correction_table[
            GROUP_COLS
            + [
                "correction",
                "direction",
                "total_n",
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
    ] = np.clip(
        result["prediction"]
        + result["correction"],
        0.0,
        1.0,
    )

    return result


# ============================================================
# 한 모델 평가
# ============================================================

def evaluate_model(
    model_name,
    df,
):

    summaries = []

    for target_season in (
        TARGET_SEASONS
    ):

        # ----------------------------------------
        # target보다 과거 OOF만 사용
        # ----------------------------------------

        source = (
            df.loc[
                df[
                    "validation_season"
                ]
                < target_season
            ]
            .copy()
        )

        target = (
            df.loc[
                df[
                    "validation_season"
                ]
                == target_season
            ]
            .copy()
        )

        if source.empty:
            raise ValueError(
                f"{target_season} 이전 OOF 없음"
            )

        if target.empty:
            raise ValueError(
                f"{target_season} OOF 없음"
            )

        source_seasons = sorted(
            source[
                "validation_season"
            ]
            .unique()
            .tolist()
        )

        # ----------------------------------------
        # stable correction 생성
        # ----------------------------------------

        correction_table = (
            build_stable_correction(
                source
            )
        )

        corrected = (
            apply_correction(
                target,
                correction_table,
            )
        )

        # ----------------------------------------
        # Metric
        # ----------------------------------------

        raw_brier = (
            brier_score(
                corrected["target"],
                corrected[
                    "prediction"
                ],
            )
        )

        corrected_brier = (
            brier_score(
                corrected["target"],
                corrected[
                    "corrected_prediction"
                ],
            )
        )

        gain = float(
            raw_brier
            - corrected_brier
        )

        corrected_mask = (
            corrected[
                "correction"
            ]
            != 0.0
        )

        coverage = float(
            corrected_mask.mean()
        )

        summaries.append({
            "model": (
                model_name
            ),
            "source_seasons": (
                ",".join(
                    map(
                        str,
                        source_seasons,
                    )
                )
            ),
            "target_season": (
                target_season
            ),
            "n_stable_groups": int(
                len(
                    correction_table
                )
            ),
            "coverage": (
                coverage
            ),
            "raw_brier": (
                raw_brier
            ),
            "corrected_brier": (
                corrected_brier
            ),
            "gain": (
                gain
            ),
        })

        # ----------------------------------------
        # 저장
        # ----------------------------------------

        prefix = (
            f"{model_name}_"
            f"to_{target_season}"
        )

        correction_table.to_csv(
            OUTPUT_DIR
            / (
                f"{prefix}_"
                "correction_table.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        corrected.to_csv(
            OUTPUT_DIR
            / (
                f"{prefix}_"
                "predictions.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        # ----------------------------------------
        # 출력
        # ----------------------------------------

        print()
        print("=" * 80)

        print(
            f"{model_name} | "
            f"{source_seasons} "
            f"-> {target_season}"
        )

        print("=" * 80)

        print(
            f"Stable groups    : "
            f"{len(correction_table)}"
        )

        print(
            f"Coverage         : "
            f"{coverage:.2%}"
        )

        print(
            f"Raw Brier        : "
            f"{raw_brier:.8f}"
        )

        print(
            f"Corrected Brier  : "
            f"{corrected_brier:.8f}"
        )

        print(
            f"Gain             : "
            f"{gain:+.8f}"
        )

        print()
        print(
            "[Correction table]"
        )

        if correction_table.empty:
            print(
                "Stable group 없음"
            )
        else:
            print(
                correction_table
                .to_string(
                    index=False
                )
            )

    return summaries


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------
    # train
    # --------------------------------------------

    train = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=[
            ID_COL,
            "pitcher_hand",
            "batter_hand",
        ],
    )

    for col in GROUP_COLS:

        train[col] = (
            train[col]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )

    all_summaries = []

    # --------------------------------------------
    # v003 / v009
    # --------------------------------------------

    for (
        model_name,
        oof_path,
    ) in OOF_PATHS.items():

        if not oof_path.exists():
            raise FileNotFoundError(
                oof_path
            )

        oof = load_oof(
            oof_path
        )

        (
            oof,
            train_normalized,
        ) = normalize_ids(
            oof,
            train,
        )

        df = oof.merge(
            train_normalized,
            left_on="row_id",
            right_on=ID_COL,
            how="left",
            validate="one_to_one",
        )

        if len(df) != len(oof):
            raise ValueError(
                "OOF / train merge 실패"
            )

        summaries = (
            evaluate_model(
                model_name,
                df,
            )
        )

        all_summaries.extend(
            summaries
        )

    # --------------------------------------------
    # 최종 Summary
    # --------------------------------------------

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print()
    print("=" * 100)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 100)

    print(
        summary_df[
            [
                "model",
                "source_seasons",
                "target_season",
                "n_stable_groups",
                "coverage",
                "raw_brier",
                "corrected_brier",
                "gain",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "gain > 0 : 개선"
    )

    print(
        "gain < 0 : 악화"
    )

    print()
    print(
        "결과 저장:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()