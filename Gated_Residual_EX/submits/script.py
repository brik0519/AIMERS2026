from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# Paths / constants
# =========================================================

DATA_DIR = Path("./data")
MODEL_PATH = Path("./model/cb_reg_v018_gated.pkl")
OUTPUT_DIR = Path("./output")
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

ID_COL = "row_id"
TARGET_COL = "control_success"

HBEB_CONFIGS = [
    {
        "group_cols": ["pitcher_hand"],
        "n_col": "asof_pitcher_n",
        "rate_col": "asof_pitcher_success_rate",
        "new_col": "hbeb_pitcher_success_rate",
    },
    {
        "group_cols": ["batter_hand"],
        "n_col": "asof_batter_n",
        "rate_col": "asof_batter_success_rate",
        "new_col": "hbeb_batter_success_rate",
    },
]


# =========================================================
# HB-EB using frozen train-only priors
# =========================================================

def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    original_index = df.index

    lookup_df = df[group_cols].merge(
        prior_df,
        on=group_cols,
        how="left",
        sort=False,
    )

    alpha = (
        lookup_df["alpha"]
        .fillna(global_prior[0])
        .to_numpy(dtype=np.float64)
    )

    beta = (
        lookup_df["beta"]
        .fillna(global_prior[1])
        .to_numpy(dtype=np.float64)
    )

    n = (
        pd.to_numeric(df[n_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        pd.to_numeric(df[rate_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    success_count = n * rate
    denominator = n + alpha + beta
    eb_rate = (success_count + alpha) / denominator

    return pd.Series(
        eb_rate,
        index=original_index,
        dtype=np.float64,
    )


def apply_saved_hbeb_priors(
    df_apply,
    base_features,
    priors,
):
    transformed = df_apply.copy()
    current_features = list(base_features)

    for config in HBEB_CONFIGS:
        new_col = config["new_col"]

        if new_col not in priors:
            raise KeyError(
                f"Missing saved HB-EB prior: {new_col}"
            )

        prior_info = priors[new_col]

        transformed[new_col] = apply_hbeb(
            df=transformed,
            group_cols=prior_info["group_cols"],
            n_col=prior_info["n_col"],
            rate_col=prior_info["rate_col"],
            prior_df=prior_info["prior_df"],
            global_prior=prior_info["global_prior"],
        )

        if config["rate_col"] in current_features:
            current_features.remove(config["rate_col"])

        if new_col not in current_features:
            current_features.append(new_col)

    return transformed, current_features


def prepare_features(df, cat_cols):
    df = df.copy()

    for col in cat_cols:
        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return df


# =========================================================
# V18 gated residual correction
# =========================================================

def add_pressure_state(df):
    df = df.copy()

    required = [
        "balls_before",
        "strikes_before",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing pressure columns: "
            + ", ".join(missing)
        )

    balls = (
        pd.to_numeric(
            df["balls_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    strikes = (
        pd.to_numeric(
            df["strikes_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2),
        "full",
        np.where(
            (balls == 3) | (strikes == 2),
            "high",
            "normal",
        ),
    )

    return df


def lookup_effect(
    df,
    table,
    key_cols,
):
    if table is None or len(table) == 0:
        return np.zeros(
            len(df),
            dtype=np.float64,
        )

    left = df[key_cols].copy()
    right = table.copy()

    for col in key_cols:
        left[col] = (
            left[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        right[col] = (
            right[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return (
        left.merge(
            right[
                key_cols
                + ["shrunk_effect"]
            ],
            on=key_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )["shrunk_effect"]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def build_gated_correction(
    parent_effect,
    detail_effect,
    config,
):
    parent_effect = np.asarray(
        parent_effect,
        dtype=np.float64,
    )

    detail_effect = np.asarray(
        detail_effect,
        dtype=np.float64,
    )

    if len(parent_effect) != len(detail_effect):
        raise ValueError(
            "Residual effect lengths differ."
        )

    parent_available = (
        parent_effect != 0.0
    )

    detail_available = (
        detail_effect != 0.0
    )

    same_sign = (
        parent_available
        & detail_available
        & (
            np.sign(parent_effect)
            == np.sign(detail_effect)
        )
    )

    parent_only = (
        parent_available
        & ~detail_available
    )

    correction = np.zeros(
        len(parent_effect),
        dtype=np.float64,
    )

    # parent만 있으면 약하게 보정
    correction[parent_only] = (
        parent_effect[parent_only]
        * float(config["parent_only_scale"])
    )

    # parent와 detail이 같은 방향일 때만 detail 반영
    detail_weight = float(
        config["detail_weight"]
    )

    correction[same_sign] = (
        (1.0 - detail_weight)
        * parent_effect[same_sign]
        + detail_weight
        * detail_effect[same_sign]
    )

    correction *= float(
        config["alpha"]
    )

    cap = float(
        config["cap"]
    )

    return np.clip(
        correction,
        -cap,
        cap,
    )


# =========================================================
# Main inference
# =========================================================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    test_path = DATA_DIR / "test.csv"
    sample_path = DATA_DIR / "sample_submission.csv"

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    artifact = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "hbeb_priors",
        "features",
        "cat_cols",
        "v18_gated_correction",
    ]

    missing = [
        k for k in required_keys
        if k not in artifact
    ]

    if missing:
        raise KeyError(
            "Submission artifact is missing keys: "
            + ", ".join(missing)
        )

    model = artifact["model"]
    priors = artifact["hbeb_priors"]
    final_features = list(artifact["features"])
    cat_cols = list(artifact["cat_cols"])
    correction_info = artifact[
        "v18_gated_correction"
    ]

    if not correction_info.get("enabled", False):
        raise ValueError(
            "V18 gated correction is not enabled."
        )

    # -----------------------------------------------------
    # Load test. No cross-row aggregation/statistics.
    # -----------------------------------------------------
    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_test_cols = [
        ID_COL,
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ]

    missing_test = [
        c for c in required_test_cols
        if c not in test.columns
    ]

    if missing_test:
        raise KeyError(
            "test.csv is missing columns: "
            + ", ".join(missing_test)
        )

    base_features = [
        c for c in test.columns
        if c != ID_COL
    ]

    transformed, _ = apply_saved_hbeb_priors(
        df_apply=test,
        base_features=base_features,
        priors=priors,
    )

    missing_features = [
        c for c in final_features
        if c not in transformed.columns
    ]

    if missing_features:
        raise KeyError(
            "Transformed test is missing model features: "
            + ", ".join(missing_features)
        )

    X_test = prepare_features(
        transformed[final_features],
        cat_cols,
    )

    # -----------------------------------------------------
    # Step 1: exact canonical v003 base prediction
    # -----------------------------------------------------
    pred_raw = model.predict(X_test)

    pred_base = np.clip(
        np.asarray(
            pred_raw,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    # -----------------------------------------------------
    # Step 2: row-local pressure state
    # -----------------------------------------------------
    gate_frame = add_pressure_state(
        test[
            [
                "pitcher_id",
                "batter_hand",
                "balls_before",
                "strikes_before",
            ]
        ]
    )

    # -----------------------------------------------------
    # Step 3: frozen train-only V18 lookup
    # -----------------------------------------------------
    parent_effect = lookup_effect(
        df=gate_frame,
        table=correction_info["parent_table"],
        key_cols=correction_info["parent_keys"],
    )

    detail_effect = lookup_effect(
        df=gate_frame,
        table=correction_info["detail_table"],
        key_cols=correction_info["detail_keys"],
    )

    correction = build_gated_correction(
        parent_effect=parent_effect,
        detail_effect=detail_effect,
        config=correction_info["config"],
    )

    pred_final = np.clip(
        pred_base + correction,
        0.0,
        1.0,
    )

    # -----------------------------------------------------
    # Submission
    # -----------------------------------------------------
    pred_frame = pd.DataFrame(
        {
            ID_COL: test[ID_COL].to_numpy(),
            TARGET_COL: pred_final,
        }
    )

    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
    )

    if ID_COL not in sample.columns:
        raise KeyError(
            f"sample_submission.csv is missing {ID_COL}"
        )

    submission = sample[[ID_COL]].merge(
        pred_frame,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if submission[TARGET_COL].isna().any():
        raise ValueError(
            "Some submission rows are missing predictions."
        )

    if (
        (submission[TARGET_COL] < 0.0).any()
        or (submission[TARGET_COL] > 1.0).any()
    ):
        raise ValueError(
            "Predictions are outside [0, 1]."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Diagnostics only.
    corrected_mask = correction != 0.0

    print("=" * 72)
    print("Inference complete")
    print("=" * 72)
    print(f"Rows             : {len(submission):,}")
    print(
        f"Correction source: "
        f"{correction_info['source_season']}"
    )
    print(
        f"Parent groups    : "
        f"{len(correction_info['parent_table']):,}"
    )
    print(
        f"Detail groups    : "
        f"{len(correction_info['detail_table']):,}"
    )
    print(
        f"Corrected rows   : "
        f"{int(corrected_mask.sum()):,} "
        f"({corrected_mask.mean():.2%})"
    )
    print(f"Pred min         : {pred_final.min():.8f}")
    print(f"Pred mean        : {pred_final.mean():.8f}")
    print(f"Pred max         : {pred_final.max():.8f}")
    print(f"Saved            : {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
