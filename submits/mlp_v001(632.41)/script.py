# script.py: mlp_v001
#
# 목적:
# Kaggle에서 학습한 mlp_v001.pkl bundle을 로드하여
# 대회 평가 환경에서 test.csv를 추론하고 submission.csv를 생성한다.
#
# 학습 코드와의 일관성:
# - 동일한 raw feature set 사용
# - 동일 categorical vocabulary 사용
# - unseen category -> 0
# - 동일 numerical median / mean / std 사용
# - 동일 numeric clipping 사용
# - 동일 TabularMLP architecture 복원
#
# 대회 제출 구조:
#
# .
# ├── data/
# │   ├── test.csv
# │   └── sample_submission.csv
# ├── model/
# │   └── mlp_v001.pkl
# └── script.py
#
# 출력:
# /output/submission.csv
#
# 중요:
# 제출 경로는 /output/submission.csv로 고정한다.


import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# 경로
# =========================================================

DATA_DIR = Path("./data")

MODEL_PATH = Path("./model/mlp_v001.pkl")

TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = Path("./output")
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


# =========================================================
# 기본 설정
# =========================================================

ID = "row_id"
TARGET = "control_success"

BATCH_SIZE = 8192
NUM_WORKERS = 0


# =========================================================
# Device
# =========================================================

def get_device():

    if torch.cuda.is_available():

        try:
            # 실제 CUDA kernel을 간단히 실행하여
            # 설치 binary와 GPU가 호환되는지 확인한다.
            probe = torch.zeros(
                1,
                device="cuda",
            )

            _ = probe + 1

            return torch.device(
                "cuda"
            )

        except Exception as e:

            print(
                "[WARNING] CUDA 사용 불가. "
                "CPU로 fallback합니다."
            )

            print(
                f"[CUDA Error] {e}"
            )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):

        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# =========================================================
# Model
# train.py와 동일
# =========================================================

class TabularMLP(
    nn.Module
):

    def __init__(
        self,
        cardinalities,
        embedding_dims,
        n_num_features,
        hidden_dims,
        dropout,
    ):

        super().__init__()

        self.embeddings = (
            nn.ModuleList([
                nn.Embedding(
                    num_embeddings=int(
                        cardinality
                    ),
                    embedding_dim=int(
                        dim
                    ),
                    padding_idx=0,
                )
                for cardinality, dim
                in zip(
                    cardinalities,
                    embedding_dims,
                )
            ])
        )

        embedding_total = int(
            sum(
                embedding_dims
            )
        )

        input_dim = (
            embedding_total
            + int(
                n_num_features
            )
        )

        if input_dim <= 0:

            raise ValueError(
                "MLP input feature가 없습니다."
            )

        layers = []

        prev_dim = (
            input_dim
        )

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

            prev_dim = (
                hidden_dim
            )

        self.backbone = (
            nn.Sequential(
                *layers
            )
        )

        self.head = (
            nn.Linear(
                prev_dim,
                1,
            )
        )

    def forward(
        self,
        x_cat,
        x_num,
    ):

        parts = []

        if len(
            self.embeddings
        ):

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

        if (
            x_num.shape[1]
            > 0
        ):

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
            self.head(
                x
            )
            .squeeze(1)
        )


# =========================================================
# Bundle 검증
# =========================================================

def validate_bundle(
    bundle,
):

    required = {
        "model_state_dict",
        "model_config",
        "features",
        "cat_cols",
        "num_cols",
        "preprocessor",
        "metadata",
    }

    missing = (
        required
        - set(
            bundle.keys()
        )
    )

    if missing:

        raise KeyError(
            "Model bundle에 필요한 key가 없습니다: "
            f"{sorted(missing)}"
        )

    experiment_name = (
        bundle[
            "metadata"
        ]
        .get(
            "experiment_name"
        )
    )

    if (
        experiment_name
        != "mlp_v001"
    ):

        raise ValueError(
            "잘못된 model bundle입니다.\n"
            "expected: mlp_v001\n"
            f"actual  : {experiment_name}"
        )


# =========================================================
# Categorical transform
# =========================================================

def transform_categorical(
    df,
    cat_cols,
    category_maps,
):

    if not cat_cols:

        return np.empty(
            (
                len(df),
                0,
            ),
            dtype=np.int64,
        )

    columns = []

    for col in cat_cols:

        if col not in df.columns:

            raise KeyError(
                f"test.csv에 categorical feature가 없습니다: {col}"
            )

        if col not in category_maps:

            raise KeyError(
                f"category map이 없습니다: {col}"
            )

        s = (
            df[col]
            .fillna(
                "__MISSING__"
            )
            .astype(
                str
            )
        )

        mapped = (
            s.map(
                category_maps[
                    col
                ]
            )
            .fillna(0)
            .astype(
                np.int64
            )
            .to_numpy()
        )

        columns.append(
            mapped
        )

    return (
        np.column_stack(
            columns
        )
        .astype(
            np.int64,
            copy=False,
        )
    )


# =========================================================
# Numerical transform
# =========================================================

def transform_numerical(
    df,
    num_cols,
    preprocessor,
):

    if not num_cols:

        return np.empty(
            (
                len(df),
                0,
            ),
            dtype=np.float32,
        )

    numeric_median = (
        preprocessor[
            "numeric_median"
        ]
    )

    numeric_mean = (
        preprocessor[
            "numeric_mean"
        ]
    )

    numeric_std = (
        preprocessor[
            "numeric_std"
        ]
    )

    numeric_clip = float(
        preprocessor.get(
            "numeric_clip",
            10.0,
        )
    )

    columns = []

    for col in num_cols:

        if col not in df.columns:

            raise KeyError(
                f"test.csv에 numerical feature가 없습니다: {col}"
            )

        s = pd.to_numeric(
            df[col],
            errors="coerce",
        ).astype(
            np.float64
        )

        median = float(
            numeric_median[
                col
            ]
        )

        mean = float(
            numeric_mean[
                col
            ]
        )

        std = float(
            numeric_std[
                col
            ]
        )

        if (
            not np.isfinite(
                std
            )
            or std < 1e-12
        ):

            std = 1.0

        z = (
            s.fillna(
                median
            )
            .sub(
                mean
            )
            .div(
                std
            )
            .clip(
                lower=-numeric_clip,
                upper=numeric_clip,
            )
        )

        columns.append(
            z.to_numpy(
                dtype=np.float32,
            )
        )

    return (
        np.column_stack(
            columns
        )
        .astype(
            np.float32,
            copy=False,
        )
    )


# =========================================================
# Dataset
# =========================================================

def make_dataset(
    x_cat,
    x_num,
):

    x_cat_tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                x_cat,
                dtype=np.int64,
            )
        )
    )

    x_num_tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                x_num,
                dtype=np.float32,
            )
        )
    )

    return TensorDataset(
        x_cat_tensor,
        x_num_tensor,
    )


# =========================================================
# Inference
# =========================================================

@torch.inference_mode()
def predict(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []

    for (
        x_cat,
        x_num,
    ) in loader:

        x_cat = x_cat.to(
            device,
            non_blocking=True,
        )

        x_num = x_num.to(
            device,
            non_blocking=True,
        )

        logits = model(
            x_cat,
            x_num,
        )

        probs = torch.sigmoid(
            logits
        )

        predictions.append(
            probs
            .detach()
            .cpu()
            .numpy()
        )

    pred = np.concatenate(
        predictions
    ).astype(
        np.float64,
        copy=False,
    )

    return pred


# =========================================================
# Submission
# =========================================================

def build_submission(
    test,
    prediction,
):

    sample = pd.read_csv(
        SAMPLE_SUBMISSION_PATH,
        encoding="utf-8-sig",
    )

    if (
        len(sample)
        != len(test)
    ):

        raise ValueError(
            "test.csv와 sample_submission.csv의 "
            "행 수가 다릅니다.\n"
            f"test={len(test):,}, "
            f"sample={len(sample):,}"
        )

    if (
        TARGET
        not in sample.columns
    ):

        raise KeyError(
            "sample_submission.csv에 "
            f"{TARGET} 컬럼이 없습니다."
        )

    # row_id가 양쪽에 있으면 순서 검증
    if (
        ID in sample.columns
        and ID in test.columns
    ):

        sample_ids = (
            sample[ID]
            .astype(str)
            .to_numpy()
        )

        test_ids = (
            test[ID]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            sample_ids,
            test_ids,
        ):

            raise ValueError(
                "test.csv와 sample_submission.csv의 "
                "row_id 순서가 다릅니다."
            )

    sample[
        TARGET
    ] = (
        prediction
    )

    return sample


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 70)
    print("mlp_v001 Inference")
    print("=" * 70)

    # -----------------------------------------------------
    # 파일 확인
    # -----------------------------------------------------

    for path in [
        MODEL_PATH,
        TEST_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"필수 파일을 찾을 수 없습니다: {path}"
            )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = get_device()

    print(
        f"Device: {device}"
    )

    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "PyTorch:",
        torch.__version__,
    )

    # -----------------------------------------------------
    # Bundle load
    # -----------------------------------------------------

    bundle = joblib.load(
        MODEL_PATH
    )

    validate_bundle(
        bundle
    )

    model_config = (
        bundle[
            "model_config"
        ]
    )

    features = list(
        bundle[
            "features"
        ]
    )

    cat_cols = list(
        bundle[
            "cat_cols"
        ]
    )

    num_cols = list(
        bundle[
            "num_cols"
        ]
    )

    preprocessor = (
        bundle[
            "preprocessor"
        ]
    )

    metadata = (
        bundle[
            "metadata"
        ]
    )

    print()
    print(
        "Experiment:",
        metadata.get(
            "experiment_name"
        ),
    )

    print(
        "Best epoch:",
        metadata.get(
            "best_epoch"
        ),
    )

    print(
        "Feature 수:",
        len(
            features
        ),
    )

    print(
        "Categorical:",
        len(
            cat_cols
        ),
    )

    print(
        "Numerical:",
        len(
            num_cols
        ),
    )

    # -----------------------------------------------------
    # Test load
    # -----------------------------------------------------

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
    )

    print()
    print(
        "test shape:",
        test.shape,
    )

    missing_features = [
        col
        for col in features
        if col not in test.columns
    ]

    if missing_features:

        raise KeyError(
            "test.csv에 학습 feature가 없습니다: "
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    x_cat = (
        transform_categorical(
            test,
            cat_cols,
            preprocessor[
                "category_maps"
            ],
        )
    )

    x_num = (
        transform_numerical(
            test,
            num_cols,
            preprocessor,
        )
    )

    print()
    print("=" * 70)
    print("Preprocessing")
    print("=" * 70)

    print(
        "Categorical shape:",
        x_cat.shape,
    )

    print(
        "Numerical shape  :",
        x_num.shape,
    )

    if cat_cols:

        print()
        print(
            "[Unknown category ratio]"
        )

        for i, col in enumerate(
            cat_cols
        ):

            ratio = float(
                np.mean(
                    x_cat[:, i]
                    == 0
                )
            )

            print(
                f"{col:<22}: "
                f"{ratio:.6%}"
            )

    # -----------------------------------------------------
    # Model restore
    # -----------------------------------------------------

    model = TabularMLP(
        cardinalities=(
            model_config[
                "cardinalities"
            ]
        ),
        embedding_dims=(
            model_config[
                "embedding_dims"
            ]
        ),
        n_num_features=(
            model_config[
                "n_num_features"
            ]
        ),
        hidden_dims=(
            model_config[
                "hidden_dims"
            ]
        ),
        dropout=(
            model_config[
                "dropout"
            ]
        ),
    )

    model.load_state_dict(
        bundle[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    # -----------------------------------------------------
    # Loader
    # -----------------------------------------------------

    dataset = (
        make_dataset(
            x_cat,
            x_num,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            == "cuda"
        ),
        drop_last=False,
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("Prediction 시작")
    print("=" * 70)

    prediction = predict(
        model,
        loader,
        device,
    )

    if (
        len(prediction)
        != len(test)
    ):

        raise ValueError(
            "Prediction 행 수가 test와 다릅니다."
        )

    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            "Prediction에 NaN 또는 inf가 존재합니다."
        )

    prediction = np.clip(
        prediction,
        0.0,
        1.0,
    )

    print(
        f"Prediction mean : "
        f"{prediction.mean():.6f}"
    )

    print(
        f"Prediction std  : "
        f"{prediction.std():.6f}"
    )

    print(
        f"Prediction min  : "
        f"{prediction.min():.6f}"
    )

    print(
        f"Prediction max  : "
        f"{prediction.max():.6f}"
    )

    # -----------------------------------------------------
    # Submission
    # -----------------------------------------------------

    submission = (
        build_submission(
            test,
            prediction,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Submission 생성 완료")
    print("=" * 70)

    print(
        f"Output : {SUBMISSION_PATH}"
    )

    print(
        f"Rows   : {len(submission):,}"
    )

    print(
        f"Target : {TARGET}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
