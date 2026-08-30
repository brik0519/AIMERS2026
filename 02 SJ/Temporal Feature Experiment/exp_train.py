# train_final.py

# ============================================================
# LG Aimers KBO - Feature Experiment Final Training
#
# 목적:
#   Validation 실험이 끝난 EXP-1 / EXP-3 / EXP-4 모델을
#   2019~2024 전체 train 데이터로 재학습해서 저장한다.
#
#
# 실행 방법
# ------------------------------------------------------------
#
# EXP-1:
#   python train_final.py --mode exp1
#
# EXP-3:
#   python train_final.py --mode exp3
#
# EXP-4:
#   python train_final.py --mode exp4
#
#
# 각 Mode의 Feature
# ------------------------------------------------------------
#
# exp1
#   Raw 47
#   + Temporal 10
#   + count_state 1
#
# exp3
#   Raw 47
#   + Temporal 10
#
# exp4
#   Raw 47
#   + count_state 1
#
#
# 모델
# ------------------------------------------------------------
#
# CatBoost : 56.55685%
# XGBoost  : 43.44315%
#
# 기존 실험과 동일한 모델/하이퍼파라미터를 사용한다.
#
#
# 저장 위치
# ------------------------------------------------------------
#
# ./model/exp1_final/ensemble.pkl
# ./model/exp3_final/ensemble.pkl
# ./model/exp4_final/ensemble.pkl
#
#
# 중요
# ------------------------------------------------------------
#
# - Validation 없이 2019~2024 전체 학습
# - 기존 ./model/rf.pkl은 건드리지 않음
# - 저장 파일에는 model뿐 아니라 mode 정보도 함께 저장
# - script.py가 mode를 읽고 동일한 Feature를 생성함
# ============================================================


import argparse
import os
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. 기본 설정
# ============================================================

DATA_DIR = "./data"

ID_COL = "row_id"
TARGET_COL = "control_success"


# 기존 최고 앙상블 weight 그대로 유지
CAT_WEIGHT = 0.5655685
XGB_WEIGHT = 0.4344315


# ============================================================
# 2. Temporal Feature 목록
# ============================================================

TEMPORAL_FEATURES = [

    # Recent vs Career - Success
    "recent1_success_vs_career",
    "recent3_success_vs_career",
    "recent5_success_vs_career",

    # Recent vs Career - Middle
    "recent1_middle_vs_career",
    "recent3_middle_vs_career",
    "recent5_middle_vs_career",

    # Trend - Success
    "success_trend_1_3",
    "success_trend_3_5",

    # Trend - Middle
    "middle_trend_1_3",
    "middle_trend_3_5",
]


COUNT_FEATURES = [
    "count_state",
]


# ============================================================
# 3. Feature Engineering 함수
# ============================================================

def build_features(df, mode):
    """
    mode에 따라 필요한 Feature를 생성한다.

    exp1:
        Temporal 10 + Count 1

    exp3:
        Temporal 10

    exp4:
        Count 1
    """

    df = df.copy()


    # ========================================================
    # Temporal Feature
    #
    # exp1과 exp3에서만 생성
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
    # Count Feature
    #
    # exp1과 exp4에서만 생성
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
# 4. Mode별 Feature 구성
# ============================================================

def get_feature_config(
    mode,
    raw_features,
):
    """
    mode에 맞는 전체 Feature와
    categorical Feature 목록을 반환한다.
    """

    # 기존 categorical Feature
    cat_cols = [
        "top_bottom",
        "game_type",
        "base_state",
    ]


    # --------------------------------------------------------
    # EXP-1
    # Raw + Temporal + Count
    # --------------------------------------------------------

    if mode == "exp1":

        features = (
            raw_features
            + TEMPORAL_FEATURES
            + COUNT_FEATURES
        )

        cat_cols = (
            cat_cols
            + ["count_state"]
        )


    # --------------------------------------------------------
    # EXP-3
    # Raw + Temporal
    # --------------------------------------------------------

    elif mode == "exp3":

        features = (
            raw_features
            + TEMPORAL_FEATURES
        )


    # --------------------------------------------------------
    # EXP-4
    # Raw + Count
    # --------------------------------------------------------

    elif mode == "exp4":

        features = (
            raw_features
            + COUNT_FEATURES
        )

        cat_cols = (
            cat_cols
            + ["count_state"]
        )


    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )


    return (
        features,
        cat_cols,
    )


# ============================================================
# 5. 모델 생성
# ============================================================

def build_model(
    features,
    cat_cols,
):
    """
    기존 실험과 동일한

        preprocessing
        + CatBoost
        + XGBoost
        + Soft Voting

    모델을 생성한다.
    """

    num_cols = [
        col
        for col in features
        if col not in cat_cols
    ]


    # ========================================================
    # 범주형 전처리
    # ========================================================

    cat_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        ),
    ])


    # ========================================================
    # 수치형 전처리
    # ========================================================

    num_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
    ])


    # ========================================================
    # 전체 전처리
    # ========================================================

    preprocessor = ColumnTransformer([
        (
            "cat",
            cat_pipeline,
            cat_cols,
        ),
        (
            "num",
            num_pipeline,
            num_cols,
        ),
    ])


    # ========================================================
    # CatBoost
    # ========================================================

    catboost = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,

        loss_function="Logloss",
        l2_leaf_reg=3.0,

        random_seed=42,

        verbose=False,
        thread_count=-1,

        # catboost_info 폴더 생성 방지
        allow_writing_files=False,
    )


    # ========================================================
    # XGBoost
    # ========================================================

    xgboost = XGBClassifier(
        objective="binary:logistic",

        n_estimators=500,
        learning_rate=0.05,

        max_depth=6,
        min_child_weight=10,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.0,
        reg_lambda=1.0,

        tree_method="hist",
        eval_metric="logloss",

        n_jobs=-1,
        random_state=42,
    )


    # ========================================================
    # Soft Voting
    # ========================================================

    ensemble = VotingClassifier(
        estimators=[
            (
                "catboost",
                catboost,
            ),
            (
                "xgboost",
                xgboost,
            ),
        ],

        voting="soft",

        weights=[
            CAT_WEIGHT,
            XGB_WEIGHT,
        ],
    )


    # ========================================================
    # 최종 Pipeline
    # ========================================================

    model = Pipeline([
        (
            "pre",
            preprocessor,
        ),
        (
            "clf",
            ensemble,
        ),
    ])


    return model


# ============================================================
# 6. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 실행할 실험 선택
    #
    # 예:
    #
    # python train_final.py --mode exp1
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "exp1",
            "exp3",
            "exp4",
        ],
    )

    args = parser.parse_args()

    mode = args.mode


    print("=" * 60)
    print(f"FINAL TRAINING: {mode.upper()}")
    print("=" * 60)


    # ========================================================
    # Raw Feature 목록
    # ========================================================

    test_cols = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "test.csv"
        ),
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()


    raw_features = [
        col
        for col in test_cols
        if col != ID_COL
    ]


    # ========================================================
    # Mode별 Feature 구성
    # ========================================================

    (
        features,
        cat_cols,
    ) = get_feature_config(
        mode,
        raw_features,
    )


    print(
        f"\nRaw features   : "
        f"{len(raw_features)}"
    )

    print(
        f"Total features : "
        f"{len(features)}"
    )

    print(
        f"Categorical    : "
        f"{len(cat_cols)}"
    )


    # ========================================================
    # Train 데이터 불러오기
    # ========================================================

    print("\nLoad full train data...")


    train = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "train.csv"
        ),
        encoding="utf-8-sig",
        usecols=(
            raw_features
            + [TARGET_COL]
        ),
    )


    print(
        f"Train shape: "
        f"{train.shape}"
    )


    # ========================================================
    # Feature Engineering
    # ========================================================

    print("\nBuild features...")


    train = build_features(
        train,
        mode,
    )


    # 생성 여부 검사
    missing_features = [
        col
        for col in features
        if col not in train.columns
    ]


    if missing_features:

        raise ValueError(
            "Feature 생성 실패: "
            f"{missing_features}"
        )


    # ========================================================
    # 전체 Train 사용
    #
    # 2019~2024 전체 데이터
    # ========================================================

    X = train[
        features
    ].copy()


    y = train[
        TARGET_COL
    ].copy()


    print(
        f"\nTraining rows: "
        f"{len(X):,}"
    )

    print(
        f"Season: "
        f"{train['season'].min()} "
        f"~ "
        f"{train['season'].max()}"
    )


    # ========================================================
    # 모델 생성
    # ========================================================

    print("\nBuild model...")


    model = build_model(
        features,
        cat_cols,
    )


    # ========================================================
    # 전체 데이터 재학습
    # ========================================================

    print("\nTrain full model...")


    start = time.time()


    model.fit(
        X,
        y,
    )


    elapsed = (
        time.time()
        - start
    )


    print(
        f"Training complete :: "
        f"{elapsed:.1f}s"
    )


    # ========================================================
    # 모델 저장
    # ========================================================

    save_dir = (
        f"./model/{mode}_final"
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    save_path = os.path.join(
        save_dir,
        "ensemble.pkl"
    )


    # --------------------------------------------------------
    # model만 저장하는 것이 아니라
    # inference에 필요한 설정도 함께 저장한다.
    #
    # 이렇게 하면 script.py가 어떤 Feature Set인지
    # 자동으로 알 수 있다.
    # --------------------------------------------------------

    bundle = {

        "mode":
            mode,

        "model":
            model,

        "raw_features":
            raw_features,

        "features":
            features,

        "cat_cols":
            cat_cols,

        "catboost_weight":
            CAT_WEIGHT,

        "xgboost_weight":
            XGB_WEIGHT,
    }


    joblib.dump(
        bundle,
        save_path,
        compress=3,
    )


    # ========================================================
    # 종료
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Mode: {mode}"
    )

    print(
        f"Features: "
        f"{len(features)}"
    )

    print(
        f"Saved: "
        f"{save_path}"
    )

    print(
        f"CatBoost weight: "
        f"{CAT_WEIGHT}"
    )

    print(
        f"XGBoost weight: "
        f"{XGB_WEIGHT}"
    )


if __name__ == "__main__":
    main()