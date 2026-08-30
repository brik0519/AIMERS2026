# script.py

# ============================================================
# LG Aimers KBO - Submission Inference
#
# 서버에서 자동 실행되는 제출용 코드.
#
#
# 처리 과정
# ------------------------------------------------------------
#
# 1. ./model/ensemble.pkl 로드
#
# 2. 저장된 mode 확인
#
#       exp1
#       exp3
#       exp4
#
# 3. ./data/test.csv 로드
#
# 4. 학습 때와 완전히 동일한 Feature Engineering 적용
#
# 5. predict_proba()
#
# 6. sample_submission의 row_id 순서에 맞춤
#
# 7. 반드시
#
#       ./output/submission.csv
#
#    로 저장
#
#
# 중요
# ------------------------------------------------------------
#
# test의 다른 행을 이용해서
# rolling / aggregation / frequency 등을 만들지 않는다.
#
# 모든 추가 Feature는 현재 행에 이미 제공된
# official asof 정보만 이용한다.
# ============================================================


import os

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. 기본 설정
# ============================================================

DATA_DIR = "./data"

MODEL_PATH = "./model/ensemble.pkl"

OUTPUT_DIR = "./output"

ID_COL = "row_id"
TARGET_COL = "control_success"


# ============================================================
# 2. Feature Engineering 함수
# ============================================================

def build_features(df, mode):
    """
    train_final.py와 반드시 동일한 Feature를 만든다.

    mode:

        exp1
            Temporal 10 + Count 1

        exp3
            Temporal 10

        exp4
            Count 1
    """

    df = df.copy()


    # ========================================================
    # Temporal Features
    #
    # EXP-1 / EXP-3
    # ========================================================

    if mode in ["exp1", "exp3"]:

        # ----------------------------------------------------
        # Recent vs Career - Success
        # ----------------------------------------------------

        df["recent1_success_vs_career"] = (
            df["asof_pitcher_prev1_game_success_rate"]
            - df["asof_pitcher_success_rate"]
        )

        df["recent3_success_vs_career"] = (
            df["asof_pitcher_prev3_game_success_rate"]
            - df["asof_pitcher_success_rate"]
        )

        df["recent5_success_vs_career"] = (
            df["asof_pitcher_prev5_game_success_rate"]
            - df["asof_pitcher_success_rate"]
        )


        # ----------------------------------------------------
        # Recent vs Career - Middle
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Success Trend
        # ----------------------------------------------------

        df["success_trend_1_3"] = (
            df["asof_pitcher_prev1_game_success_rate"]
            - df["asof_pitcher_prev3_game_success_rate"]
        )

        df["success_trend_3_5"] = (
            df["asof_pitcher_prev3_game_success_rate"]
            - df["asof_pitcher_prev5_game_success_rate"]
        )


        # ----------------------------------------------------
        # Middle Trend
        # ----------------------------------------------------

        df["middle_trend_1_3"] = (
            df["asof_pitcher_prev1_game_middle_rate"]
            - df["asof_pitcher_prev3_game_middle_rate"]
        )

        df["middle_trend_3_5"] = (
            df["asof_pitcher_prev3_game_middle_rate"]
            - df["asof_pitcher_prev5_game_middle_rate"]
        )


    # ========================================================
    # Count State
    #
    # EXP-1 / EXP-4
    # ========================================================

    if mode in ["exp1", "exp4"]:

        balls = (
            df["balls_before"]
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        strikes = (
            df["strikes_before"]
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        df["count_state"] = (
            balls
            + "_"
            + strikes
        )


    return df


# ============================================================
# 3. Main
# ============================================================

def main():

    print("=" * 60)
    print("KBO Submission Inference")
    print("=" * 60)


    # ========================================================
    # 모델 불러오기
    # ========================================================

    print("\nLoad model...")


    bundle = joblib.load(
        MODEL_PATH
    )


    mode = bundle[
        "mode"
    ]

    model = bundle[
        "model"
    ]

    features = bundle[
        "features"
    ]

    raw_features = bundle[
        "raw_features"
    ]


    print(
        f"Mode: "
        f"{mode}"
    )

    print(
        f"Features: "
        f"{len(features)}"
    )


    # ========================================================
    # Test 데이터 불러오기
    # ========================================================

    print("\nLoad test data...")


    test = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "test.csv"
        ),
        encoding="utf-8-sig",
    )


    # row_id 보존
    test_ids = (
        test[ID_COL]
        .copy()
    )


    print(
        f"Test rows: "
        f"{len(test):,}"
    )


    # ========================================================
    # Raw Feature 확인
    # ========================================================

    missing_raw = [
        col
        for col in raw_features
        if col not in test.columns
    ]


    if missing_raw:

        raise ValueError(
            "test.csv에 필요한 Raw Feature가 없습니다: "
            f"{missing_raw}"
        )


    # ========================================================
    # Feature Engineering
    # ========================================================

    print("\nBuild features...")


    test = build_features(
        test,
        mode,
    )


    # 최종 Feature가 모두 생성됐는지 확인
    missing_features = [
        col
        for col in features
        if col not in test.columns
    ]


    if missing_features:

        raise ValueError(
            "Inference Feature 생성 실패: "
            f"{missing_features}"
        )


    X_test = test[
        features
    ].copy()


    # ========================================================
    # Prediction
    # ========================================================

    print("\nPredict...")


    pred = model.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # Prediction 검증
    # ========================================================

    # NaN / inf가 있으면 제출하면 안 됨
    if not np.isfinite(pred).all():

        raise ValueError(
            "Prediction에 NaN 또는 inf가 있습니다."
        )


    # 확률 범위를 벗어나면 문제
    if (
        pred.min() < 0
        or pred.max() > 1
    ):

        raise ValueError(
            "Prediction이 [0, 1] 범위를 벗어났습니다."
        )


    print("\nPrediction statistics:")


    print(
        pd.Series(
            pred
        ).describe()
    )


    # ========================================================
    # Prediction DataFrame
    # ========================================================

    prediction = pd.DataFrame({
        ID_COL:
            test_ids,

        TARGET_COL:
            pred,
    })


    # row_id 중복 검사
    if prediction[ID_COL].duplicated().any():

        raise ValueError(
            "test row_id가 중복되어 있습니다."
        )


    # ========================================================
    # Sample Submission 불러오기
    # ========================================================

    sample = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "sample_submission.csv"
        ),
        encoding="utf-8-sig",
    )


    # ========================================================
    # sample_submission row_id 순서에 맞추기
    # ========================================================
    #
    # test.csv 순서를 그대로 믿기보다
    # 공식 sample_submission의 row_id 순서에 맞춘다.
    # ========================================================

    submission = (
        sample[[ID_COL]]
        .merge(
            prediction,
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
    )


    # 예측이 빠진 row가 있는지 확인
    if submission[TARGET_COL].isna().any():

        raise ValueError(
            "submission에 예측값이 없는 row가 있습니다."
        )


    # ========================================================
    # 최종 컬럼 확인
    # ========================================================

    submission = submission[
        [
            ID_COL,
            TARGET_COL,
        ]
    ]


    if len(submission) != len(sample):

        raise ValueError(
            "submission row 수가 sample_submission과 다릅니다."
        )


    # ========================================================
    # 저장
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        "submission.csv"
    )


    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


    print("\n")
    print("=" * 60)
    print("Inference Complete")
    print("=" * 60)

    print(
        f"Mode: "
        f"{mode}"
    )

    print(
        f"Saved: "
        f"{output_path}"
    )

    print(
        f"Rows: "
        f"{len(submission):,}"
    )


if __name__ == "__main__":
    main()