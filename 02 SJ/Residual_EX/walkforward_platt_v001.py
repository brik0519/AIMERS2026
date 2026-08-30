# ============================================================
# walkforward_platt_v001.py
#
# 목적
# ------------------------------------------------------------
# cb_reg_v003 / cb_reg_v009의 기존 OOF prediction을 이용하여
# Walk-Forward Platt Scaling을 적용한다.
#
# 예:
#   2021 OOF       -> calibrator 학습 -> 2022 평가
#   2021~2022 OOF  -> calibrator 학습 -> 2023 평가
#   2021~2023 OOF  -> calibrator 학습 -> 2024 평가
#
# 중요
# ------------------------------------------------------------
# - CatBoost 재학습 없음
# - test.csv 사용 없음
# - target season label로 calibrator 학습하지 않음
# - v003 / v009에 완전히 동일한 calibration 적용
#
# Platt Scaling
# ------------------------------------------------------------
# 기존 확률 p를 logit(p)로 변환한 뒤,
#
#   calibrated_p = sigmoid(A * logit(p) + B)
#
# 형태의 Logistic Regression을 과거 OOF에서 학습한다.
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ============================================================
# 경로
# ============================================================

OOF_PATHS = {
    "cb_reg_v003": Path(
        "./validation/cb_reg_v003/oof_validation.npz"
    ),
    "cb_reg_v009": Path(
        "./validation/cb_reg_v009/oof_validation.npz"
    ),
}

OUTPUT_DIR = Path(
    "./validation/walkforward_platt_v001"
)


# ============================================================
# 설정
# ============================================================

EPS = 1e-6

# 2021은 calibration 학습용 첫 시즌으로 사용
# 따라서 실제 strict transfer 평가는 2022부터 가능
TARGET_SEASONS = [
    2022,
    2023,
    2024,
]


# ============================================================
# Metric
# ============================================================

def calculate_brier(
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


def calculate_bss(
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

    brier = calculate_brier(
        y_true,
        y_pred,
    )

    positive_rate = float(
        np.mean(y_true)
    )

    baseline = (
        positive_rate
        * (1.0 - positive_rate)
    )

    if baseline <= 0:
        return 0.0

    return float(
        100000.0
        * (
            1.0
            - brier / baseline
        )
    )


# ============================================================
# Prediction -> logit
# ============================================================

def probability_to_logit(
    prediction,
):
    """
    Platt Scaling의 입력값을 만든다.

    p -> log(p / (1-p))
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    prediction = np.clip(
        prediction,
        EPS,
        1.0 - EPS,
    )

    return np.log(
        prediction
        / (1.0 - prediction)
    )


# ============================================================
# OOF 로드
# ============================================================

def load_oof(
    path,
):
    data = np.load(
        path,
        allow_pickle=True,
    )

    print()
    print(
        f"[LOAD] {path}"
    )
    print(
        "keys:",
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
                f"{path}에 "
                f"'{key}'가 없습니다."
            )

    # 최종 평가에 사용했던 prediction을 우선 사용
    if "prediction" in data.files:
        pred_key = "prediction"

    elif "prediction_clip" in data.files:
        pred_key = "prediction_clip"

    else:
        raise KeyError(
            f"{path}에 prediction 또는 "
            "prediction_clip이 없습니다."
        )

    print(
        f"사용 prediction key: "
        f"{pred_key}"
    )

    df = pd.DataFrame({
        "row_id": (
            data["row_id"]
        ),
        "target": (
            data["target"]
            .astype(np.float64)
        ),
        "prediction": (
            data[pred_key]
            .astype(np.float64)
        ),
        "validation_season": (
            data[
                "validation_season"
            ]
            .astype(int)
        ),
    })

    df[
        "prediction"
    ] = np.clip(
        df["prediction"],
        0.0,
        1.0,
    )

    return df


# ============================================================
# Platt Scaling 학습
# ============================================================

def fit_platt(
    train_prediction,
    train_target,
):
    """
    과거 OOF만 사용하여 Platt calibrator를 학습한다.

    입력:
        logit(original_probability)

    출력:
        LogisticRegression calibrator
    """

    x = probability_to_logit(
        train_prediction
    ).reshape(-1, 1)

    y = np.asarray(
        train_target,
        dtype=np.int64,
    )

    # calibration 자체의 규제를 너무 강하게 두지 않도록
    # 큰 C 사용.
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    calibrator.fit(
        x,
        y,
    )

    return calibrator


def apply_platt(
    calibrator,
    prediction,
):
    x = probability_to_logit(
        prediction
    ).reshape(-1, 1)

    calibrated = (
        calibrator
        .predict_proba(x)[:, 1]
    )

    return np.clip(
        calibrated,
        0.0,
        1.0,
    )


# ============================================================
# 한 모델 Walk-Forward 평가
# ============================================================

def evaluate_model(
    model_name,
    oof,
):
    summaries = []
    prediction_frames = []

    for target_season in (
        TARGET_SEASONS
    ):

        # ----------------------------------------------------
        # 반드시 target season보다 과거 OOF만 사용
        # ----------------------------------------------------

        calibration_train = (
            oof.loc[
                oof[
                    "validation_season"
                ]
                < target_season
            ]
            .copy()
        )

        validation = (
            oof.loc[
                oof[
                    "validation_season"
                ]
                == target_season
            ]
            .copy()
        )

        if calibration_train.empty:
            raise ValueError(
                f"{model_name}: "
                f"{target_season} 이전 "
                "OOF 데이터가 없습니다."
            )

        if validation.empty:
            raise ValueError(
                f"{model_name}: "
                f"{target_season} "
                "OOF 데이터가 없습니다."
            )

        source_seasons = sorted(
            calibration_train[
                "validation_season"
            ]
            .unique()
            .tolist()
        )

        # ----------------------------------------------------
        # Platt 학습
        # ----------------------------------------------------

        calibrator = fit_platt(
            train_prediction=(
                calibration_train[
                    "prediction"
                ]
            ),
            train_target=(
                calibration_train[
                    "target"
                ]
            ),
        )

        # ----------------------------------------------------
        # 다음 시즌에 적용
        # ----------------------------------------------------

        calibrated_prediction = (
            apply_platt(
                calibrator,
                validation[
                    "prediction"
                ],
            )
        )

        validation[
            "calibrated_prediction"
        ] = calibrated_prediction

        # ----------------------------------------------------
        # Metric
        # ----------------------------------------------------

        raw_brier = (
            calculate_brier(
                validation[
                    "target"
                ],
                validation[
                    "prediction"
                ],
            )
        )

        calibrated_brier = (
            calculate_brier(
                validation[
                    "target"
                ],
                validation[
                    "calibrated_prediction"
                ],
            )
        )

        gain = (
            raw_brier
            - calibrated_brier
        )

        raw_bss = (
            calculate_bss(
                validation[
                    "target"
                ],
                validation[
                    "prediction"
                ],
            )
        )

        calibrated_bss = (
            calculate_bss(
                validation[
                    "target"
                ],
                validation[
                    "calibrated_prediction"
                ],
            )
        )

        bss_gain = (
            calibrated_bss
            - raw_bss
        )

        # Platt 계수
        coefficient = float(
            calibrator.coef_[0][0]
        )

        intercept = float(
            calibrator.intercept_[0]
        )

        result = {
            "model": model_name,
            "calibration_seasons": (
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
            "n_calibration": int(
                len(
                    calibration_train
                )
            ),
            "n_validation": int(
                len(validation)
            ),
            "raw_brier": (
                raw_brier
            ),
            "calibrated_brier": (
                calibrated_brier
            ),
            "gain": gain,
            "raw_bss": (
                raw_bss
            ),
            "calibrated_bss": (
                calibrated_bss
            ),
            "bss_gain": (
                bss_gain
            ),
            "platt_coef": (
                coefficient
            ),
            "platt_intercept": (
                intercept
            ),
            "raw_pred_mean": float(
                validation[
                    "prediction"
                ].mean()
            ),
            "calibrated_pred_mean": float(
                validation[
                    "calibrated_prediction"
                ].mean()
            ),
            "actual_mean": float(
                validation[
                    "target"
                ].mean()
            ),
        }

        summaries.append(
            result
        )

        validation[
            "model"
        ] = model_name

        prediction_frames.append(
            validation
        )

        print()
        print(
            "=" * 72
        )
        print(
            f"{model_name} | "
            f"{source_seasons} "
            f"-> {target_season}"
        )
        print(
            "=" * 72
        )

        print(
            f"Raw Brier        : "
            f"{raw_brier:.8f}"
        )

        print(
            f"Calibrated Brier : "
            f"{calibrated_brier:.8f}"
        )

        print(
            f"Gain             : "
            f"{gain:+.8f}"
        )

        print(
            f"BSS gain         : "
            f"{bss_gain:+.2f}"
        )

        print(
            f"Platt coef       : "
            f"{coefficient:.6f}"
        )

        print(
            f"Platt intercept  : "
            f"{intercept:+.6f}"
        )

        print(
            f"Pred mean        : "
            f"{result['raw_pred_mean']:.6f}"
            f" -> "
            f"{result['calibrated_pred_mean']:.6f}"
        )

        print(
            f"Actual mean      : "
            f"{result['actual_mean']:.6f}"
        )

    return (
        summaries,
        prediction_frames,
    )


# ============================================================
# Main
# ============================================================

def main():

    for path in (
        OOF_PATHS.values()
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []
    all_predictions = []

    # --------------------------------------------------------
    # v003 / v009 동일한 방식으로 평가
    # --------------------------------------------------------

    for model_name, path in (
        OOF_PATHS.items()
    ):

        oof = load_oof(
            path
        )

        (
            summaries,
            prediction_frames,
        ) = evaluate_model(
            model_name=model_name,
            oof=oof,
        )

        all_summaries.extend(
            summaries
        )

        all_predictions.extend(
            prediction_frames
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    prediction_df.to_csv(
        OUTPUT_DIR
        / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Pooled strict-transfer 결과
    # --------------------------------------------------------

    pooled_rows = []

    for model_name in (
        OOF_PATHS.keys()
    ):

        model_pred = (
            prediction_df.loc[
                prediction_df[
                    "model"
                ]
                == model_name
            ]
        )

        raw_brier = (
            calculate_brier(
                model_pred["target"],
                model_pred[
                    "prediction"
                ],
            )
        )

        calibrated_brier = (
            calculate_brier(
                model_pred["target"],
                model_pred[
                    "calibrated_prediction"
                ],
            )
        )

        pooled_rows.append({
            "model": model_name,
            "raw_brier": (
                raw_brier
            ),
            "calibrated_brier": (
                calibrated_brier
            ),
            "gain": (
                raw_brier
                - calibrated_brier
            ),
        })

    pooled_df = pd.DataFrame(
        pooled_rows
    )

    pooled_df.to_csv(
        OUTPUT_DIR
        / "pooled_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print()
    print("=" * 90)
    print(
        "WALK-FORWARD PLATT SUMMARY"
    )
    print("=" * 90)

    print(
        summary_df[
            [
                "model",
                "calibration_seasons",
                "target_season",
                "raw_brier",
                "calibrated_brier",
                "gain",
                "bss_gain",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print(
        "POOLED STRICT-TRANSFER SUMMARY"
    )
    print("=" * 90)

    print(
        pooled_df.to_string(
            index=False
        )
    )

    print()
    print(
        "결과 저장:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()