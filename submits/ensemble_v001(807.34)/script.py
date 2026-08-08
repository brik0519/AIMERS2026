# ensemble_submit.py
#
# LG Aimers Phase2
# CatBoostClassifier + XGBClassifier exact-weight ensemble 제출 생성
#
# Validation exact optimum:
#   CatBoostClassifier = 0.5655685195400963
#   XGBClassifier      = 0.43443148045990365
#
# XGBoost는 xgb_v001/train.py에서 저장한 artifact 구조를 그대로 사용:
# {
#   "model": final_model,
#   "preprocessor": final_preprocessor,
#   "features": features,
#   "cat_cols": cat_cols,
#   "id_like_cols": id_like_cols,
#   "num_cols": num_cols,
#   ...
# }

import os

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 설정
# =========================================================

ID_COL = "row_id"
TARGET_COL = "control_success"

CB_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
]

DATA_DIR = "./data"
MODEL_ROOT = "./model"
OUTPUT_DIR = "./output"

TEST_PATH = os.path.join(
    DATA_DIR,
    "test.csv",
)

CB_EXPERIMENT = "cb_v001"
XGB_EXPERIMENT = "xgb_v001"

CB_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    CB_EXPERIMENT,
    f"{CB_EXPERIMENT}.pkl",
)

XGB_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    XGB_EXPERIMENT,
    f"{XGB_EXPERIMENT}.pkl",
)

OUT_PATH = os.path.join(
    OUTPUT_DIR,
    "submission.csv",
)

CB_WEIGHT = 0.5655685195400963
XGB_WEIGHT = 0.43443148045990365


# =========================================================
# 데이터 로드
# =========================================================

def load_test(path):
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    if ID_COL not in df.columns:
        raise ValueError(
            f"test 데이터에 {ID_COL} 컬럼이 없습니다."
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"test 데이터의 {ID_COL}에 중복이 있습니다."
        )

    return df


# =========================================================
# CatBoost 전처리
# =========================================================

def prepare_catboost_features(
    df,
    features,
    cat_cols,
):
    """
    cb_v001 train.py와 동일한 CatBoost 전처리.

    - 학습 Feature 순서 고정
    - 범주형 결측값은 "__MISSING__"
    - 범주형 컬럼은 문자열 dtype으로 통일
    - 수치형 결측값은 CatBoost가 직접 처리
    """

    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "CatBoost 입력 Feature가 없습니다: "
            + ", ".join(missing_features)
        )

    X = df[
        features
    ].copy()

    for col in cat_cols:
        if col in X.columns:
            X[col] = (
                X[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return X


# =========================================================
# XGBoost raw feature 전처리
# =========================================================

def prepare_xgb_raw_features(
    df,
    features,
    id_like_cols,
):
    """
    xgb_v001 train.py의 prepare_raw_features()와 동일.

    1. 학습 Feature 순서 고정
    2. ID 계열 numeric 강제 변환
    3. 나머지 numeric NaN은 그대로 유지
    """

    missing_features = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "XGBoost 입력 Feature가 없습니다: "
            + ", ".join(missing_features)
        )

    X = df[
        features
    ].copy()

    for col in id_like_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    return X


# =========================================================
# Prediction diagnostics
# =========================================================

def print_prediction_summary(
    name,
    pred,
):
    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    print(f"[{name}]")

    print(
        f"mean : {pred.mean():.6f}"
    )

    print(
        f"std  : {pred.std():.6f}"
    )

    print(
        f"min  : {pred.min():.6f}"
    )

    print(
        f"max  : {pred.max():.6f}"
    )

    print()


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 68)
    print("CatBoost + XGBoost Exact Ensemble Submission")
    print("=" * 68)

    print(
        f"CB weight  : {CB_WEIGHT:.10f}"
    )

    print(
        f"XGB weight : {XGB_WEIGHT:.10f}"
    )

    print()

    # -----------------------------------------------------
    # Weight sanity check
    # -----------------------------------------------------

    if not np.isclose(
        CB_WEIGHT + XGB_WEIGHT,
        1.0,
    ):
        raise ValueError(
            "앙상블 weight 합이 1이 아닙니다."
        )

    # -----------------------------------------------------
    # 파일 존재 확인
    # -----------------------------------------------------

    for path in [
        TEST_PATH,
        CB_MODEL_PATH,
        XGB_MODEL_PATH,
    ]:
        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {path}"
            )

    # -----------------------------------------------------
    # 모델 로드
    # -----------------------------------------------------

    print("[1] 모델 로드")

    cb_model = joblib.load(
        CB_MODEL_PATH
    )

    xgb_artifact = joblib.load(
        XGB_MODEL_PATH
    )

    if not isinstance(
        xgb_artifact,
        dict,
    ):
        raise TypeError(
            "xgb_v001.pkl이 예상한 artifact dict 구조가 아닙니다."
        )

    required_xgb_keys = {
        "model",
        "preprocessor",
        "features",
        "cat_cols",
        "id_like_cols",
        "num_cols",
    }

    missing_keys = (
        required_xgb_keys
        - set(
            xgb_artifact.keys()
        )
    )

    if missing_keys:
        raise KeyError(
            "XGBoost artifact 필수 key가 없습니다: "
            + ", ".join(
                sorted(
                    missing_keys
                )
            )
        )

    xgb_model = (
        xgb_artifact["model"]
    )

    xgb_preprocessor = (
        xgb_artifact[
            "preprocessor"
        ]
    )

    xgb_features = list(
        xgb_artifact[
            "features"
        ]
    )

    xgb_id_like_cols = list(
        xgb_artifact[
            "id_like_cols"
        ]
    )

    print(
        f"CatBoost model   : {type(cb_model).__name__}"
    )

    print(
        f"XGBoost model    : {type(xgb_model).__name__}"
    )

    print(
        f"XGB preprocessor : {type(xgb_preprocessor).__name__}"
    )

    # -----------------------------------------------------
    # Test 데이터 로드
    # -----------------------------------------------------

    print()
    print("[2] Test 데이터 로드")

    test = load_test(
        TEST_PATH
    )

    print(
        f"test rows : {len(test)}"
    )

    # -----------------------------------------------------
    # 공통 Feature 정책
    # -----------------------------------------------------

    test_features = [
        col
        for col in test.columns
        if col != ID_COL
    ]

    cb_features = (
        test_features
    )

    cb_cat_cols = [
        col
        for col in CB_CAT_COLS
        if col in cb_features
    ]

    if xgb_features != test_features:
        raise ValueError(
            "저장된 XGBoost 학습 Feature 순서와 "
            "현재 test.csv Feature 순서가 다릅니다.\n"
            f"xgb_features={xgb_features}\n"
            f"test_features={test_features}"
        )

    print()
    print("[3] Feature 구성")

    print(
        f"Feature 수          : {len(test_features)}"
    )

    print(
        f"CatBoost cat 수     : {len(cb_cat_cols)}"
    )

    print(
        f"XGBoost cat 수      : {len(xgb_artifact['cat_cols'])}"
    )

    print(
        f"XGBoost ID-like 수  : {len(xgb_id_like_cols)}"
    )

    # -----------------------------------------------------
    # CatBoost Feature
    # -----------------------------------------------------

    X_cb = (
        prepare_catboost_features(
            df=test,
            features=cb_features,
            cat_cols=cb_cat_cols,
        )
    )

    cb_model_feature_names = getattr(
        cb_model,
        "feature_names_",
        None,
    )

    if cb_model_feature_names:
        cb_model_feature_names = list(
            cb_model_feature_names
        )

        if (
            cb_model_feature_names
            != cb_features
        ):
            raise ValueError(
                "CatBoost 저장 모델 Feature 순서와 "
                "test Feature 순서가 다릅니다."
            )

    # -----------------------------------------------------
    # XGBoost Feature
    # -----------------------------------------------------

    X_xgb_raw = (
        prepare_xgb_raw_features(
            df=test,
            features=xgb_features,
            id_like_cols=
                xgb_id_like_cols,
        )
    )

    X_xgb = (
        xgb_preprocessor
        .transform(
            X_xgb_raw
        )
    )

    if (
        X_xgb.shape[0]
        != len(test)
    ):
        raise RuntimeError(
            "XGBoost 전처리 후 row 수가 test와 다릅니다."
        )

    print(
        f"XGB transformed shape: {X_xgb.shape}"
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    print()
    print("[4] Inference")

    cb_pred = (
        cb_model
        .predict_proba(
            X_cb
        )[:, 1]
    )

    xgb_pred = (
        xgb_model
        .predict_proba(
            X_xgb
        )[:, 1]
    )

    cb_pred = np.asarray(
        cb_pred,
        dtype=np.float64,
    )

    xgb_pred = np.asarray(
        xgb_pred,
        dtype=np.float64,
    )

    for name, pred in [
        (
            "CatBoost",
            cb_pred,
        ),
        (
            "XGBoost",
            xgb_pred,
        ),
    ]:
        if len(pred) != len(test):
            raise RuntimeError(
                f"{name} prediction 길이가 test와 다릅니다."
            )

        if not np.isfinite(
            pred
        ).all():
            raise RuntimeError(
                f"{name} prediction에 NaN 또는 Inf가 있습니다."
            )

        if not (
            (
                pred >= 0.0
            )
            & (
                pred <= 1.0
            )
        ).all():
            raise RuntimeError(
                f"{name} prediction이 [0,1] 범위를 벗어났습니다."
            )

    # -----------------------------------------------------
    # Exact ensemble
    # -----------------------------------------------------

    ensemble_pred = (
        CB_WEIGHT
        * cb_pred
        + XGB_WEIGHT
        * xgb_pred
    )

    ensemble_pred = np.clip(
        ensemble_pred,
        0.0,
        1.0,
    )

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    print()
    print("[5] Prediction diagnostics")
    print()

    print_prediction_summary(
        "CatBoost",
        cb_pred,
    )

    print_prediction_summary(
        "XGBoost",
        xgb_pred,
    )

    print_prediction_summary(
        "Ensemble",
        ensemble_pred,
    )

    if len(test) > 1:
        test_corr = float(
            np.corrcoef(
                cb_pred,
                xgb_pred,
            )[0, 1]
        )

        print(
            "CB-XGB test prediction corr: "
            f"{test_corr:.6f}"
        )

    # -----------------------------------------------------
    # Submission 생성
    # -----------------------------------------------------

    print()
    print("[6] Submission 생성")

    submission = pd.DataFrame({
        ID_COL:
            test[ID_COL].to_numpy(),
        TARGET_COL:
            ensemble_pred,
    })

    # -----------------------------------------------------
    # Submission 검증
    # -----------------------------------------------------

    if len(submission) != len(test):
        raise RuntimeError(
            "submission row 수와 test row 수가 다릅니다."
        )

    if submission[
        ID_COL
    ].duplicated().any():
        raise ValueError(
            "submission row_id에 중복이 있습니다."
        )

    if submission[
        TARGET_COL
    ].isna().any():
        raise ValueError(
            "submission prediction에 NaN이 있습니다."
        )

    if not np.isfinite(
        submission[
            TARGET_COL
        ].to_numpy()
    ).all():
        raise ValueError(
            "submission prediction에 NaN 또는 Inf가 있습니다."
        )

    if not (
        submission[
            TARGET_COL
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):
        raise ValueError(
            "submission prediction이 [0,1] 범위를 벗어났습니다."
        )

    # -----------------------------------------------------
    # Submission 저장
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    submission.to_csv(
        OUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 68)
    print("제출 파일 저장 완료")

    print(
        f"- {OUT_PATH}"
    )

    print(
        f"- rows: {len(submission)}"
    )

    print(
        f"- prediction mean: "
        f"{submission[TARGET_COL].mean():.6f}"
    )

    print(
        f"- prediction std : "
        f"{submission[TARGET_COL].std(ddof=0):.6f}"
    )

    print(
        f"- prediction min : "
        f"{submission[TARGET_COL].min():.6f}"
    )

    print(
        f"- prediction max : "
        f"{submission[TARGET_COL].max():.6f}"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()