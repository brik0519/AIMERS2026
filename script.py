# script.py
#
# CatBoost + XGBoost + MLP v002 3-model ensemble inference
#
# Validation ensemble_v003 exact optimum:
#   CatBoost = 0.471775504844901
#   XGBoost  = 0.36992172427041964
#   MLP v002 = 0.1583027708846793
#
# 각 모델은 학습 당시의 전처리 방식을 그대로 사용한다.
# - CatBoost: cb_v001 feature schema
# - XGBoost : xgb_v001 bundle preprocessor
# - MLP v002: final preprocessor + processed variant
#
# 출력:
#   ./output/submission.csv
#
# Ensemble weight는 validation/ensemble_v003의
# triple exact convex optimum을 하드코딩한다.


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = "ensemble_v003"

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

ID = "row_id"
TARGET = "control_success"

CAT_COLS = [
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


# ---------------------------------------------------------
# Ensemble weight
# ---------------------------------------------------------
# validation/ensemble_v003 triple exact convex optimum.
# 합은 반드시 1.0이어야 한다.
# ---------------------------------------------------------

CAT_WEIGHT = 0.471775504844901
XGB_WEIGHT = 0.36992172427041964
MLP_WEIGHT = 0.1583027708846793

WEIGHT_SUM = (
    CAT_WEIGHT
    + XGB_WEIGHT
    + MLP_WEIGHT
)

if not np.isclose(
    WEIGHT_SUM,
    1.0,
    atol=1e-12,
):
    raise ValueError(
        f"Ensemble weight 합이 1이 아닙니다: {WEIGHT_SUM}"
    )


# =========================================================
# Device
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# CatBoost 전처리
# =========================================================

def prepare_catboost_features(
    df,
    features,
    cat_cols,
):
    """
    cb_v001 train.py와 동일한 categorical 처리.

    - categorical NaN -> "__MISSING__"
    - categorical dtype -> str
    - numerical은 그대로 유지
    """

    out = df[
        features
    ].copy()

    for col in cat_cols:

        if col not in out.columns:
            continue

        out[col] = (
            out[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return out


# =========================================================
# MLP model
# =========================================================

class TabularMLP(nn.Module):

    def __init__(
        self,
        cardinalities,
        embedding_dims,
        n_num_features,
        hidden_dims,
        dropout,
    ):

        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=int(cardinality),
                embedding_dim=int(dim),
                padding_idx=0,
            )
            for cardinality, dim
            in zip(
                cardinalities,
                embedding_dims,
            )
        ])

        embedding_total = int(
            sum(embedding_dims)
        )

        input_dim = (
            embedding_total
            + int(n_num_features)
        )

        if input_dim <= 0:
            raise ValueError(
                "MLP input feature가 없습니다."
            )

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.extend([
                nn.Linear(
                    prev_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(
                    hidden_dim
                ),
                nn.SiLU(),
                nn.Dropout(
                    dropout
                ),
            ])

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(
            *layers
        )

        self.head = nn.Linear(
            prev_dim,
            1,
        )

    def forward(
        self,
        x_cat,
        x_num,
    ):

        parts = []

        if len(self.embeddings):

            embedded = [
                emb(
                    x_cat[:, i]
                )
                for i, emb
                in enumerate(
                    self.embeddings
                )
            ]

            parts.append(
                torch.cat(
                    embedded,
                    dim=1,
                )
            )

        if x_num.shape[1] > 0:
            parts.append(
                x_num
            )

        x = torch.cat(
            parts,
            dim=1,
        )

        x = self.backbone(
            x
        )

        return (
            self.head(x)
            .squeeze(1)
        )


# =========================================================
# MLP v002 preprocessing
# =========================================================

MLP_ALWAYS_DROP_COLUMNS = [
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "score_diff_home",
]

MLP_COUNT_COLUMNS = [
    "balls_before",
    "strikes_before",
]

MLP_MISSING_TOKEN = "__MISSING__"


def add_count_state(
    frame,
):
    """
    mlp_preprocess.py와 동일한 count_state 생성.
    """

    frame = frame.copy()

    required = {
        "balls_before",
        "strikes_before",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "MLP count_state 생성에 필요한 컬럼 누락: "
            f"{missing}"
        )

    balls = pd.to_numeric(
        frame["balls_before"],
        errors="coerce",
    )

    strikes = pd.to_numeric(
        frame["strikes_before"],
        errors="coerce",
    )

    invalid = (
        balls.isna()
        | strikes.isna()
        | ~balls.isin([0, 1, 2, 3])
        | ~strikes.isin([0, 1, 2])
    )

    if invalid.any():

        examples = frame.loc[
            invalid,
            [
                "balls_before",
                "strikes_before",
            ],
        ].head(5)

        raise ValueError(
            "Invalid ball/strike count detected:\n"
            + examples.to_string(index=False)
        )

    frame["count_state"] = (
        balls.astype("int8").astype(str)
        + "-"
        + strikes.astype("int8").astype(str)
    )

    return frame


def prepare_mlp_frame(
    frame,
    variant,
):
    """
    mlp_preprocess.py의 prepare_frame()과 동일한 처리.

    processed:
      - count_state 생성
      - raw balls/strikes 제거

    hybrid:
      - count_state 생성
      - raw balls/strikes 유지
    """

    if variant not in {
        "processed",
        "hybrid",
    }:
        raise ValueError(
            f"지원하지 않는 MLP variant: {variant}"
        )

    frame = add_count_state(
        frame
    )

    drop_columns = list(
        MLP_ALWAYS_DROP_COLUMNS
    )

    if variant == "processed":
        drop_columns.extend(
            MLP_COUNT_COLUMNS
        )

    missing = sorted(
        set(drop_columns)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "MLP preprocessing 필수 컬럼 누락: "
            f"{missing}"
        )

    return frame.drop(
        columns=drop_columns
    )


def categorical_strings(
    series,
):
    return (
        series
        .astype("string")
        .fillna(MLP_MISSING_TOKEN)
    )


def transform_mlp_categorical(
    df,
    preprocessor,
):
    """
    Final preprocessor의 category mapping을 그대로 적용한다.
    """

    cat_cols = list(
        preprocessor[
            "categorical_columns"
        ]
    )

    if not cat_cols:
        return np.empty(
            (len(df), 0),
            dtype=np.int64,
        )

    columns = []

    unknown_index = int(
        preprocessor[
            "unknown_index"
        ]
    )

    for col in cat_cols:

        if col not in df.columns:
            raise ValueError(
                f"MLP categorical feature 누락: {col}"
            )

        mapping = (
            preprocessor[
                "category_maps"
            ][col]
        )

        encoded = (
            categorical_strings(
                df[col]
            )
            .map(mapping)
            .fillna(
                unknown_index
            )
            .astype("int64")
            .to_numpy()
        )

        columns.append(
            encoded
        )

    return np.column_stack(
        columns
    ).astype(
        np.int64,
        copy=False,
    )


def transform_mlp_numerical(
    df,
    preprocessor,
):
    """
    mlp_preprocess.py와 동일하게
    median 대체 후 mean/std standardization을 적용한다.

    mlp_v002에는 numeric clipping을 적용하지 않는다.
    """

    num_cols = list(
        preprocessor[
            "numeric_columns"
        ]
    )

    if not num_cols:
        return np.empty(
            (len(df), 0),
            dtype=np.float32,
        )

    columns = []

    for col in num_cols:

        if col not in df.columns:
            raise ValueError(
                f"MLP numeric feature 누락: {col}"
            )

        stats = (
            preprocessor[
                "numeric_stats"
            ][col]
        )

        values = (
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
            .fillna(
                float(
                    stats["median"]
                )
            )
            .astype("float64")
        )

        standardized = (
            values
            - float(
                stats["mean"]
            )
        ) / float(
            stats["std"]
        )

        columns.append(
            standardized.to_numpy(
                dtype=np.float32
            )
        )

    result = np.column_stack(
        columns
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        result
    ).all():
        raise ValueError(
            "MLP numerical transform 결과에 NaN/Inf가 있습니다."
        )

    return result


# =========================================================
# MLP inference
# =========================================================

@torch.no_grad()
def predict_mlp(
    bundle,
    test,
    device,
    batch_size=8192,
):
    """
    mlp_v002 final bundle을 이용한 runtime inference.

    Hidden test.csv를 Final preprocessor로 직접 변환하므로
    로컬 final/test/*.npy 파일에 의존하지 않는다.
    """

    required_bundle_keys = {
        "model_state_dict",
        "model_config",
        "preprocessor",
        "metadata",
    }

    missing_bundle_keys = sorted(
        required_bundle_keys
        - set(bundle.keys())
    )

    if missing_bundle_keys:
        raise ValueError(
            "MLP bundle 필수 key 누락: "
            f"{missing_bundle_keys}"
        )

    metadata = bundle[
        "metadata"
    ]

    preprocessor = bundle[
        "preprocessor"
    ]

    variant = (
        metadata.get(
            "variant"
        )
        or metadata.get(
            "preprocessing",
            {},
        ).get(
            "variant"
        )
        or preprocessor.get(
            "variant"
        )
    )

    if variant != "processed":
        raise ValueError(
            "현재 ensemble_v003은 processed MLP를 기대합니다. "
            f"bundle variant={variant}"
        )

    prepared = prepare_mlp_frame(
        test,
        variant,
    )

    x_cat = transform_mlp_categorical(
        prepared,
        preprocessor,
    )

    x_num = transform_mlp_numerical(
        prepared,
        preprocessor,
    )

    config = bundle[
        "model_config"
    ]

    expected_n_cat = len(
        config[
            "cardinalities"
        ]
    )

    expected_n_num = int(
        config[
            "n_num_features"
        ]
    )

    if x_cat.shape[1] != expected_n_cat:
        raise ValueError(
            "MLP categorical feature 수 불일치: "
            f"{x_cat.shape[1]} != {expected_n_cat}"
        )

    if x_num.shape[1] != expected_n_num:
        raise ValueError(
            "MLP numeric feature 수 불일치: "
            f"{x_num.shape[1]} != {expected_n_num}"
        )

    cardinalities = list(
        config[
            "cardinalities"
        ]
    )

    for i, cardinality in enumerate(
        cardinalities
    ):
        if x_cat.shape[0] == 0:
            continue

        col_min = int(
            x_cat[:, i].min()
        )

        col_max = int(
            x_cat[:, i].max()
        )

        if col_min < 0 or col_max >= int(
            cardinality
        ):
            raise ValueError(
                "MLP categorical index 범위 오류: "
                f"column={i}, min={col_min}, max={col_max}, "
                f"cardinality={cardinality}"
            )

    model = TabularMLP(
        cardinalities=(
            config[
                "cardinalities"
            ]
        ),
        embedding_dims=(
            config[
                "embedding_dims"
            ]
        ),
        n_num_features=(
            config[
                "n_num_features"
            ]
        ),
        hidden_dims=(
            config[
                "hidden_dims"
            ]
        ),
        dropout=(
            config[
                "dropout"
            ]
        ),
    )

    model.load_state_dict(
        bundle[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    predictions = []

    n = len(
        test
    )

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        batch_cat = torch.from_numpy(
            np.ascontiguousarray(
                x_cat[
                    start:end
                ],
                dtype=np.int64,
            )
        ).to(
            device,
            non_blocking=True,
        )

        batch_num = torch.from_numpy(
            np.ascontiguousarray(
                x_num[
                    start:end
                ],
                dtype=np.float32,
            )
        ).to(
            device,
            non_blocking=True,
        )

        logits = model(
            batch_cat,
            batch_num,
        )

        pred = torch.sigmoid(
            logits
        )

        predictions.append(
            pred
            .cpu()
            .numpy()
        )

    prediction = np.concatenate(
        predictions
    ).astype(
        np.float64,
        copy=False,
    )

    return prediction


# =========================================================
# Generic model helper
# =========================================================

def unwrap_model(
    loaded,
):
    """
    plain estimator와 bundle 형태를 모두 어느 정도 지원한다.

    가능한 bundle key:
    - model
    - estimator
    """

    if not isinstance(
        loaded,
        dict,
    ):
        return loaded

    for key in [
        "model",
        "estimator",
    ]:

        if key in loaded:
            return loaded[
                key
            ]

    return loaded


def predict_binary_proba(
    estimator,
    X,
    model_name,
):

    if hasattr(
        estimator,
        "predict_proba",
    ):

        pred = (
            estimator
            .predict_proba(
                X
            )
        )

        pred = np.asarray(
            pred
        )

        if (
            pred.ndim == 2
            and pred.shape[1] >= 2
        ):

            return (
                pred[:, 1]
                .astype(
                    np.float64,
                    copy=False,
                )
            )

        if pred.ndim == 1:

            return (
                pred.astype(
                    np.float64,
                    copy=False,
                )
            )

    if hasattr(
        estimator,
        "predict",
    ):

        pred = np.asarray(
            estimator.predict(
                X
            )
        ).reshape(-1)

        return pred.astype(
            np.float64,
            copy=False,
        )

    raise TypeError(
        f"{model_name}에서 "
        "predict_proba 또는 predict를 찾지 못했습니다."
    )


# =========================================================
# XGBoost inference
# =========================================================

def predict_xgboost(
    loaded,
    test,
):
    """
    xgb_v001 저장 구조를 최대한 보수적으로 지원한다.

    지원 우선순위:

    1. sklearn Pipeline / estimator 자체가 raw DataFrame을
       직접 처리하는 경우

    2. bundle 안에 callable preprocessor가 있는 경우:
       loaded["preprocessor"].transform(...)

    3. bundle 안에 명시적 features가 있는 경우

    중요:
    xgb_v001에서 별도 encoder를 train.py 바깥에서 사용했는데
    해당 encoder가 model bundle에 저장되어 있지 않다면,
    정확한 inference 재현은 불가능하다.
    """

    # -----------------------------------------------------
    # Bundle
    # -----------------------------------------------------

    if isinstance(
        loaded,
        dict,
    ):

        estimator = unwrap_model(
            loaded
        )

        features = loaded.get(
            "features",
            [
                col
                for col in test.columns
                if col != ID
            ],
        )

        X = test[
            features
        ].copy()

        preprocessor = loaded.get(
            "preprocessor"
        )

        if (
            preprocessor is not None
            and hasattr(
                preprocessor,
                "transform",
            )
        ):

            X = preprocessor.transform(
                X
            )

        return predict_binary_proba(
            estimator,
            X,
            "XGBoost",
        )

    # -----------------------------------------------------
    # Plain estimator / sklearn Pipeline
    # -----------------------------------------------------

    estimator = loaded

    feature_names = getattr(
        estimator,
        "feature_names_in_",
        None,
    )

    if feature_names is not None:

        features = list(
            feature_names
        )

    else:

        features = [
            col
            for col in test.columns
            if col != ID
        ]

    X = test[
        features
    ].copy()

    return predict_binary_proba(
        estimator,
        X,
        "XGBoost",
    )


# =========================================================
# Probability audit
# =========================================================

def validate_probability(
    prediction,
    model_name,
    expected_length,
):
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(-1)

    if len(
        prediction
    ) != expected_length:
        raise ValueError(
            f"{model_name} prediction 길이 불일치: "
            f"{len(prediction)} != {expected_length}"
        )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            f"{model_name} prediction에 NaN/Inf가 있습니다."
        )

    if len(
        prediction
    ) > 0 and (
        prediction.min() < 0.0
        or prediction.max() > 1.0
    ):
        raise ValueError(
            f"{model_name} prediction이 [0, 1] 범위를 벗어났습니다: "
            f"min={prediction.min()}, max={prediction.max()}"
        )

    return prediction


# =========================================================
# Input / output audit
# =========================================================

def validate_test_and_submission(
    test,
    sample_sub,
):

    if ID not in test.columns:

        raise ValueError(
            f"test.csv에 {ID}가 없습니다."
        )

    required_sub = {
        ID,
        TARGET,
    }

    missing = (
        required_sub
        - set(
            sample_sub.columns
        )
    )

    if missing:

        raise ValueError(
            "sample_submission 필수 컬럼 누락: "
            f"{missing}"
        )

    if test[
        ID
    ].duplicated().any():

        raise ValueError(
            "test.csv row_id 중복이 있습니다."
        )

    if sample_sub[
        ID
    ].duplicated().any():

        raise ValueError(
            "sample_submission row_id 중복이 있습니다."
        )

    if set(
        test[
            ID
        ]
    ) != set(
        sample_sub[
            ID
        ]
    ):

        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "row_id 집합이 다릅니다."
        )


def build_submission(
    sample_sub,
    row_ids,
    prediction,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if len(
        prediction
    ) != len(
        row_ids
    ):

        raise ValueError(
            "row_id와 prediction 길이가 다릅니다."
        )

    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            "최종 prediction에 NaN/Inf가 있습니다."
        )

    if (
        np.min(
            prediction
        ) < 0.0
        or np.max(
            prediction
        ) > 1.0
    ):

        raise ValueError(
            "최종 prediction이 [0, 1] 범위를 벗어났습니다."
        )

    pred_df = pd.DataFrame({
        ID:
            np.asarray(
                row_ids
            ),
        TARGET:
            prediction,
    })

    out = (
        sample_sub[
            [ID]
        ]
        .merge(
            pred_df,
            on=ID,
            how="left",
            validate="one_to_one",
        )
    )

    if out[
        TARGET
    ].isna().any():

        n_missing = int(
            out[
                TARGET
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            f"Submission prediction 누락: {n_missing}"
        )

    return out


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 72)
    print(
        "CatBoost + XGBoost + MLP v002 Ensemble Inference"
    )
    print("=" * 72)

    # -----------------------------------------------------
    # 실제 현재 모델 위치
    # -----------------------------------------------------

    cat_model_path = (
        MODEL_DIR
        / "cb_v001"
        / "cb_v001.pkl"
    )

    xgb_model_path = (
        MODEL_DIR
        / "xgb_v001"
        / "xgb_v001.pkl"
    )

    # mlp_v002 train.py의 표준 저장 경로는
    # ./model/mlp_v002.pkl 이지만,
    # 기존 실험별 하위 디렉터리 구조도 함께 지원한다.
    mlp_model_candidates = [
        MODEL_DIR / "mlp_v002.pkl",
        MODEL_DIR / "mlp_v002" / "mlp_v002.pkl",
    ]

    mlp_model_path = next(
        (
            path
            for path in mlp_model_candidates
            if path.exists()
        ),
        mlp_model_candidates[0],
    )

    for path in [
        cat_model_path,
        xgb_model_path,
        mlp_model_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"모델 파일이 없습니다: {path}"
            )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"test.csv가 없습니다: {TEST_PATH}"
        )

    if not SAMPLE_SUB_PATH.exists():

        raise FileNotFoundError(
            "sample_submission.csv가 없습니다: "
            f"{SAMPLE_SUB_PATH}"
        )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    print()
    print("[1] 데이터 로드")

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    sample_sub = pd.read_csv(
        SAMPLE_SUB_PATH,
        encoding="utf-8-sig",
    )

    validate_test_and_submission(
        test,
        sample_sub,
    )

    print(
        f"test rows       : {len(test):,}"
    )

    print(
        f"submission rows : {len(sample_sub):,}"
    )

    # -----------------------------------------------------
    # Model load
    # -----------------------------------------------------

    print()
    print("[2] 모델 로드")

    cat_loaded = joblib.load(
        cat_model_path
    )

    xgb_loaded = joblib.load(
        xgb_model_path
    )

    mlp_bundle = joblib.load(
        mlp_model_path
    )

    print(
        f"CatBoost : {cat_model_path}"
    )

    print(
        f"XGBoost  : {xgb_model_path}"
    )

    print(
        f"MLP      : {mlp_model_path}"
    )

    # -----------------------------------------------------
    # CatBoost
    # -----------------------------------------------------

    print()
    print("[3] CatBoost inference")

    cat_model = unwrap_model(
        cat_loaded
    )

    cat_features = getattr(
        cat_model,
        "feature_names_",
        None,
    )

    if (
        cat_features is None
        or len(
            cat_features
        ) == 0
    ):

        cat_features = [
            col
            for col in test.columns
            if col != ID
        ]

    else:

        cat_features = list(
            cat_features
        )

    missing = [
        col
        for col in cat_features
        if col not in test.columns
    ]

    if missing:

        raise ValueError(
            "CatBoost feature 누락: "
            f"{missing}"
        )

    cat_cols = [
        col
        for col in CAT_COLS
        if col in cat_features
    ]

    X_cat = (
        prepare_catboost_features(
            test,
            cat_features,
            cat_cols,
        )
    )

    p_cat = predict_binary_proba(
        cat_model,
        X_cat,
        "CatBoost",
    )

    print(
        f"mean={p_cat.mean():.6f} "
        f"std={p_cat.std():.6f} "
        f"min={p_cat.min():.6f} "
        f"max={p_cat.max():.6f}"
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    print()
    print("[4] XGBoost inference")

    p_xgb = predict_xgboost(
        xgb_loaded,
        test,
    )

    print(
        f"mean={p_xgb.mean():.6f} "
        f"std={p_xgb.std():.6f} "
        f"min={p_xgb.min():.6f} "
        f"max={p_xgb.max():.6f}"
    )

    # -----------------------------------------------------
    # MLP
    # -----------------------------------------------------

    print()
    print("[5] MLP v002 inference")

    device = get_device()

    print(
        f"device={device}"
    )

    p_mlp = predict_mlp(
        mlp_bundle,
        test,
        device,
    )

    print(
        f"mean={p_mlp.mean():.6f} "
        f"std={p_mlp.std():.6f} "
        f"min={p_mlp.min():.6f} "
        f"max={p_mlp.max():.6f}"
    )

    # -----------------------------------------------------
    # Alignment audit
    # -----------------------------------------------------

    n_test = len(
        test
    )

    p_cat = validate_probability(
        p_cat,
        "CatBoost",
        n_test,
    )

    p_xgb = validate_probability(
        p_xgb,
        "XGBoost",
        n_test,
    )

    p_mlp = validate_probability(
        p_mlp,
        "MLP v002",
        n_test,
    )

    # -----------------------------------------------------
    # Ensemble
    # -----------------------------------------------------

    print()
    print("[6] 3-model ensemble")

    print(
        f"CatBoost weight : {CAT_WEIGHT:.6f}"
    )

    print(
        f"XGBoost weight  : {XGB_WEIGHT:.6f}"
    )

    print(
        f"MLP weight      : {MLP_WEIGHT:.6f}"
    )

    prediction = (
        CAT_WEIGHT
        * p_cat
        + XGB_WEIGHT
        * p_xgb
        + MLP_WEIGHT
        * p_mlp
    )

    # Exact optimum이 양수이며 합이 1인 convex blend이므로
    # 각 모델 확률이 [0, 1]이면 ensemble도 자동으로 [0, 1]이다.
    # clipping으로 upstream 오류를 숨기지 않고 엄격하게 검증한다.
    prediction = validate_probability(
        prediction,
        "Ensemble",
        n_test,
    )

    print(
        f"mean={prediction.mean():.6f} "
        f"std={prediction.std():.6f} "
        f"min={prediction.min():.6f} "
        f"max={prediction.max():.6f}"
    )

    # -----------------------------------------------------
    # Submission
    # -----------------------------------------------------

    print()
    print("[7] submission.csv 생성")

    submission = build_submission(
        sample_sub,
        test[
            ID
        ].to_numpy(),
        prediction,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Inference 완료")
    print("=" * 72)

    print(
        f"Output : {OUTPUT_PATH}"
    )

    print(
        f"Rows   : {len(submission):,}"
    )

    print(
        f"Mean   : "
        f"{submission[TARGET].mean():.6f}"
    )

    print(
        f"Std    : "
        f"{submission[TARGET].std(ddof=0):.6f}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
