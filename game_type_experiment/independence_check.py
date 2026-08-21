from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPERIMENT_NAME = "cb_v002_game_type_final_v001"

ID_COL = "row_id"

DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_NAME}.pkl"
TEST_PATH = DATA_DIR / "test.csv"


def prepare_categorical(df: pd.DataFrame, cat_cols):
    df = df.copy()

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return df


def predict_dataframe(df: pd.DataFrame, bundle):
    features = bundle["features"]
    cat_cols = bundle["cat_cols"]

    group_col = bundle["group_col"]
    r_group = bundle["R_group"]
    f_group = bundle["F_group"]

    missing_features = [
        col for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = prepare_categorical(
        df[features].copy(),
        cat_cols,
    )

    r_mask = X[group_col].astype(str).eq(r_group)
    f_mask = X[group_col].astype(str).eq(f_group)

    if (~(r_mask | f_mask)).any():
        bad_values = (
            X.loc[
                ~(r_mask | f_mask),
                group_col,
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unexpected game_type: {bad_values}"
        )

    pred = np.empty(len(X), dtype=float)

    if r_mask.any():
        X_r = X.loc[r_mask, features]

        global_r_pred = (
            bundle["global_R_model"]
            .predict_proba(X_r)[:, 1]
        )

        r_only_pred = (
            bundle["R_only_model"]
            .predict_proba(X_r)[:, 1]
        )

        pred[r_mask.to_numpy()] = (
            bundle["R_global_weight"] * global_r_pred
            + bundle["R_only_weight"] * r_only_pred
        )

    if f_mask.any():
        X_f = X.loc[f_mask, features]

        global_f_pred = (
            bundle["global_F_model"]
            .predict_proba(X_f)[:, 1]
        )

        f_only_pred = (
            bundle["F_only_model"]
            .predict_proba(X_f)[:, 1]
        )

        pred[f_mask.to_numpy()] = (
            bundle["F_global_weight"] * global_f_pred
            + bundle["F_only_weight"] * f_only_pred
        )

    return pred


print("=" * 80)
print("INDEPENDENT-ROW INFERENCE CHECK")
print("=" * 80)

print("\nLoad model...")
bundle = joblib.load(MODEL_PATH)

print("Load test...")
test = pd.read_csv(
    TEST_PATH,
    encoding="utf-8-sig",
)

if ID_COL not in test.columns:
    raise ValueError(
        f"{ID_COL} not found in test.csv"
    )


# 1) 전체 test 예측
print("\n[1] Predict full test...")

full_pred = predict_dataframe(
    test,
    bundle,
)

full_result = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    "full_pred": full_pred,
})


# 2) 각 행을 하나씩 예측
print("[2] Predict each row separately...")

single_preds = []

for i in range(len(test)):
    one_row = test.iloc[[i]].copy()

    one_pred = predict_dataframe(
        one_row,
        bundle,
    )[0]

    single_preds.append(one_pred)

single_preds = np.asarray(
    single_preds,
    dtype=float,
)


# 3) 전체 예측과 단일행 예측 비교
abs_diff = np.abs(
    full_pred - single_preds
)

max_diff = float(abs_diff.max())

same = np.allclose(
    full_pred,
    single_preds,
    rtol=0.0,
    atol=1e-12,
)

check_df = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    "full_pred": full_pred,
    "single_row_pred": single_preds,
    "abs_diff": abs_diff,
})

print("\n[Full vs Single-row]")
print(
    check_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.15f}",
    )
)

print(
    f"\nMax absolute difference: "
    f"{max_diff:.15e}"
)


# 4) 행 순서를 섞어도 동일한지 확인
print("\n[3] Shuffle-order check...")

shuffled = (
    test.sample(
        frac=1.0,
        random_state=42,
    )
    .reset_index(drop=True)
)

shuffled_pred = predict_dataframe(
    shuffled,
    bundle,
)

shuffled_result = pd.DataFrame({
    ID_COL: shuffled[ID_COL].values,
    "shuffled_pred": shuffled_pred,
})

order_check = (
    full_result
    .merge(
        shuffled_result,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
)

order_check["abs_diff"] = np.abs(
    order_check["full_pred"]
    - order_check["shuffled_pred"]
)

order_max_diff = float(
    order_check["abs_diff"].max()
)

order_same = np.allclose(
    order_check["full_pred"],
    order_check["shuffled_pred"],
    rtol=0.0,
    atol=1e-12,
)

print("\n[Full vs Shuffled]")
print(
    order_check.to_string(
        index=False,
        float_format=lambda x: f"{x:.15f}",
    )
)

print(
    f"\nShuffle max absolute difference: "
    f"{order_max_diff:.15e}"
)


# 최종 판정
print("\n")
print("=" * 80)
print("FINAL CHECK")
print("=" * 80)

if same:
    print(
        "PASS: 각 행을 단독으로 예측해도 "
        "전체 test에서의 예측값과 동일합니다."
    )
else:
    print(
        "FAIL: 단일행 예측과 전체 test 예측이 다릅니다."
    )

if order_same:
    print(
        "PASS: test 행 순서를 바꿔도 "
        "각 row_id의 예측값이 동일합니다."
    )
else:
    print(
        "FAIL: test 행 순서에 따라 예측값이 달라집니다."
    )

if not (same and order_same):
    raise RuntimeError(
        "독립행 예측 규칙 검증 실패"
    )

print("\n독립행 예측 규칙 검증 완료.")
