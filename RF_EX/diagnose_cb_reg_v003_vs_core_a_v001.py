# diagnose_cb_reg_v003_vs_core_a_v001.py
# ============================================================
# 목적
# ============================================================
# 기존 LB 840.8056의 cb_reg_v003와
# 새 LB 783.3495의 Global + CORE_A 제출을 직접 비교한다.
#
# A = 기존 cb_reg_v003 prediction
# B = 새 model.joblib 내부 Global prediction
# C = 새 routed prediction
#
# 확인 항목
# 1) 기존/새 Global artifact 구조
# 2) feature 목록/순서
# 3) categorical feature 목록
# 4) HB-EB prior 동일성
# 5) CatBoost tree 수 / 주요 파라미터
# 6) test prediction A vs B
# 7) B vs C 변화가 CORE_A row에만 존재하는지
# 8) CORE_A / non-CORE_A별 prediction 변화량
#
# 판정
# ----
# - A == B:
#     새 Global은 기존 cb_reg_v003와 동일.
#     LB 하락 원인은 CORE_A specialist routing으로 판단 가능.
#
# - A != B:
#     CORE_A 효과를 논하기 전에 Global 재현 문제부터 해결해야 함.
#
# 대회 규칙
# ---------
# test.csv 행별 feature와 train에서 저장된 prior/model artifact만 사용.
# test row 간 집계 feature를 만들지 않는다.
# 이 코드는 제출용이 아니라 로컬 진단용이다.
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 경로
# ============================================================

DATA_DIR = Path("./data")
TEST_PATH = DATA_DIR / "test.csv"

# 기존 840.8056 모델
OLD_MODEL_PATH = (
    Path("./model")
    / "cb_reg_v003_test"
    / "cb_reg_v003_test.pkl"
)

# 방금 제출한 Global + CORE_A 모델
NEW_MODEL_PATH = (
    Path("./model")
    / "model.joblib"
)

OUTPUT_DIR = (
    Path("./diagnostics")
    / "cb_reg_v003_vs_core_a_v001"
)

ROW_OUTPUT_PATH = (
    OUTPUT_DIR
    / "test_prediction_comparison.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "summary.json"
)


# ============================================================
# CORE_A
# ============================================================

GROUP_COL = "game_type"
PITCHER_HAND_COL = "pitcher_hand"
BATTER_HAND_COL = "batter_hand"
ID = "row_id"


def core_a_mask(df):
    return (
        df[GROUP_COL]
        .fillna("__MISSING__")
        .astype(str)
        .eq("R")
        &
        df[PITCHER_HAND_COL]
        .fillna("__MISSING__")
        .astype(str)
        .eq("1")
        &
        df[BATTER_HAND_COL]
        .fillna("__MISSING__")
        .astype(str)
        .eq("1")
    ).to_numpy()


# ============================================================
# 공통 helpers
# ============================================================

def prepare_features(df, cat_cols):
    out = df.copy()

    for col in cat_cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    return out


def apply_hbeb(
    df,
    group_cols,
    n_col,
    rate_col,
    prior_df,
    global_prior,
):
    """
    기존/새 artifact에 저장된 train-only HB-EB prior를
    현재 row에 lookup하여 적용한다.
    """

    original_index = df.index

    lookup_df = (
        df[group_cols]
        .reset_index(drop=True)
        .merge(
            prior_df,
            on=group_cols,
            how="left",
            sort=False,
        )
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
        pd.to_numeric(
            df[n_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    rate = (
        pd.to_numeric(
            df[rate_col],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    denominator = (
        n + alpha + beta
    )

    eb = np.divide(
        (n * rate) + alpha,
        denominator,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=(denominator > 0.0),
    )

    return pd.Series(
        eb,
        index=original_index,
        dtype=np.float64,
    )


def predict_with_artifact(
    test,
    model,
    priors,
    features,
    cat_cols,
):
    transformed = test.copy()

    # artifact 내부 prior 정의를 그대로 사용
    for new_col, info in priors.items():
        transformed[new_col] = apply_hbeb(
            df=transformed,
            group_cols=list(
                info["group_cols"]
            ),
            n_col=info["n_col"],
            rate_col=info["rate_col"],
            prior_df=info["prior_df"],
            global_prior=tuple(
                info["global_prior"]
            ),
        )

    X = prepare_features(
        transformed[features],
        cat_cols,
    )

    raw = np.asarray(
        model.predict(X),
        dtype=np.float64,
    )

    return (
        np.clip(
            raw,
            0.0,
            1.0,
        ),
        X,
    )


def numeric_compare(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )
    b = np.asarray(
        b,
        dtype=np.float64,
    )

    d = b - a
    ad = np.abs(d)

    return {
        "n": int(len(a)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a)),
        "std_b": float(np.std(b)),
        "mean_signed_diff_b_minus_a": float(
            np.mean(d)
        ),
        "mean_abs_diff": float(
            np.mean(ad)
        ),
        "rmse_diff": float(
            np.sqrt(
                np.mean(d ** 2)
            )
        ),
        "max_abs_diff": float(
            np.max(ad)
        ),
        "n_abs_diff_gt_1e_12": int(
            np.sum(ad > 1e-12)
        ),
        "n_abs_diff_gt_1e_10": int(
            np.sum(ad > 1e-10)
        ),
        "n_abs_diff_gt_1e_8": int(
            np.sum(ad > 1e-8)
        ),
        "n_abs_diff_gt_1e_6": int(
            np.sum(ad > 1e-6)
        ),
        "corr": float(
            np.corrcoef(a, b)[0, 1]
        ),
    }


def model_info(model):
    params = model.get_params()

    wanted = [
        "loss_function",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "random_seed",
        "max_ctr_complexity",
    ]

    return {
        "type": type(model).__name__,
        "tree_count": int(
            model.tree_count_
        ),
        "params": {
            key: params.get(key)
            for key in wanted
        },
        "feature_names": list(
            model.feature_names_
        ),
    }


def normalize_key_df(df, group_cols):
    x = df[
        list(group_cols)
        + ["alpha", "beta"]
    ].copy()

    for col in group_cols:
        x[col] = (
            x[col]
            .fillna("__NA__")
            .astype(str)
        )

    return (
        x.sort_values(
            list(group_cols)
        )
        .reset_index(drop=True)
    )


def compare_priors(
    old_priors,
    new_priors,
):
    result = {}

    all_keys = sorted(
        set(old_priors)
        | set(new_priors)
    )

    for key in all_keys:
        if (
            key not in old_priors
            or key not in new_priors
        ):
            result[key] = {
                "exists_old": (
                    key in old_priors
                ),
                "exists_new": (
                    key in new_priors
                ),
                "identical": False,
            }
            continue

        old = old_priors[key]
        new = new_priors[key]

        old_gp = tuple(
            old["global_prior"]
        )
        new_gp = tuple(
            new["global_prior"]
        )

        global_prior_abs_diff = [
            abs(
                float(a) - float(b)
            )
            for a, b in zip(
                old_gp,
                new_gp,
            )
        ]

        old_groups = list(
            old["group_cols"]
        )
        new_groups = list(
            new["group_cols"]
        )

        same_group_definition = (
            old_groups == new_groups
            and old["n_col"]
            == new["n_col"]
            and old["rate_col"]
            == new["rate_col"]
        )

        prior_df_equal = False
        prior_df_max_abs_diff = None

        if same_group_definition:
            old_df = normalize_key_df(
                old["prior_df"],
                old_groups,
            )
            new_df = normalize_key_df(
                new["prior_df"],
                new_groups,
            )

            key_cols = old_groups

            merged = old_df.merge(
                new_df,
                on=key_cols,
                how="outer",
                suffixes=(
                    "_old",
                    "_new",
                ),
                indicator=True,
            )

            if (
                merged["_merge"]
                .eq("both")
                .all()
            ):
                diffs = []

                for col in [
                    "alpha",
                    "beta",
                ]:
                    diff = np.abs(
                        pd.to_numeric(
                            merged[
                                f"{col}_old"
                            ],
                            errors="coerce",
                        ).to_numpy(
                            dtype=np.float64
                        )
                        -
                        pd.to_numeric(
                            merged[
                                f"{col}_new"
                            ],
                            errors="coerce",
                        ).to_numpy(
                            dtype=np.float64
                        )
                    )

                    diffs.append(diff)

                if diffs:
                    all_diffs = (
                        np.concatenate(
                            diffs
                        )
                    )
                    prior_df_max_abs_diff = float(
                        np.nanmax(
                            all_diffs
                        )
                    )
                    prior_df_equal = bool(
                        np.nanmax(
                            all_diffs
                        )
                        <= 1e-12
                    )

        result[key] = {
            "exists_old": True,
            "exists_new": True,
            "same_group_definition": (
                same_group_definition
            ),
            "global_prior_old": [
                float(x)
                for x in old_gp
            ],
            "global_prior_new": [
                float(x)
                for x in new_gp
            ],
            "global_prior_max_abs_diff": float(
                max(
                    global_prior_abs_diff
                )
            ),
            "prior_df_equal_1e_12": (
                prior_df_equal
            ),
            "prior_df_max_abs_diff": (
                prior_df_max_abs_diff
            ),
        }

    return result


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            to_jsonable(x)
            for x in obj
        ]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


# ============================================================
# Main
# ============================================================

def main():
    if not OLD_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\n기존 cb_reg_v003 artifact를 찾을 수 없습니다.\n"
            f"기대 경로: {OLD_MODEL_PATH}\n\n"
            "실제 경로가 다르면 코드 상단 OLD_MODEL_PATH만 수정하세요."
        )

    if not NEW_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\n새 CORE_A artifact를 찾을 수 없습니다.\n"
            f"기대 경로: {NEW_MODEL_PATH}\n\n"
            "실제 경로가 다르면 코드 상단 NEW_MODEL_PATH만 수정하세요."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_PATH} not found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("=" * 100)
    print("LOAD ARTIFACTS")
    print("=" * 100)

    old = joblib.load(
        OLD_MODEL_PATH
    )
    new = joblib.load(
        NEW_MODEL_PATH
    )

    print(
        f"OLD: {OLD_MODEL_PATH}"
    )
    print(
        f"NEW: {NEW_MODEL_PATH}"
    )

    print(
        f"OLD keys: {sorted(old.keys())}"
    )
    print(
        f"NEW keys: {sorted(new.keys())}"
    )

    # 기존 cb_reg_v003 artifact
    required_old = {
        "model",
        "hbeb_priors",
        "features",
        "cat_cols",
    }

    missing_old = (
        required_old
        - set(old.keys())
    )

    if missing_old:
        raise KeyError(
            f"OLD artifact missing keys: "
            f"{sorted(missing_old)}"
        )

    # 새 artifact
    required_new = {
        "global_model",
        "core_a_model",
        "priors",
        "metadata",
    }

    missing_new = (
        required_new
        - set(new.keys())
    )

    if missing_new:
        raise KeyError(
            f"NEW artifact missing keys: "
            f"{sorted(missing_new)}"
        )

    old_model = old["model"]
    old_priors = old[
        "hbeb_priors"
    ]
    old_features = list(
        old["features"]
    )
    old_cat_cols = list(
        old["cat_cols"]
    )

    new_global = new[
        "global_model"
    ]
    new_core = new[
        "core_a_model"
    ]
    new_priors = new[
        "priors"
    ]

    new_metadata = new[
        "metadata"
    ]

    new_features = list(
        new_metadata[
            "features"
        ]
    )
    new_cat_cols = list(
        new_metadata[
            "categorical_features"
        ]
    )

    # --------------------------------------------------------
    # Artifact comparison
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("1. ARTIFACT / FEATURE COMPARISON")
    print("=" * 100)

    features_exact = (
        old_features
        == new_features
    )

    feature_sets_equal = (
        set(old_features)
        == set(new_features)
    )

    cats_exact = (
        old_cat_cols
        == new_cat_cols
    )

    print(
        f"Feature count OLD/NEW : "
        f"{len(old_features)} / "
        f"{len(new_features)}"
    )
    print(
        f"Feature set equal     : "
        f"{feature_sets_equal}"
    )
    print(
        f"Feature order equal   : "
        f"{features_exact}"
    )
    print(
        f"Cat cols equal        : "
        f"{cats_exact}"
    )

    if not features_exact:
        only_old = [
            x
            for x in old_features
            if x not in new_features
        ]
        only_new = [
            x
            for x in new_features
            if x not in old_features
        ]

        order_mismatches = [
            (
                i,
                a,
                b,
            )
            for i, (a, b)
            in enumerate(
                zip(
                    old_features,
                    new_features,
                )
            )
            if a != b
        ]

        print(
            f"Only OLD: {only_old}"
        )
        print(
            f"Only NEW: {only_new}"
        )
        print(
            "First feature-order mismatches:"
        )
        for row in (
            order_mismatches[:10]
        ):
            print(
                f"  index={row[0]} "
                f"OLD={row[1]} "
                f"NEW={row[2]}"
            )

    old_model_info = (
        model_info(
            old_model
        )
    )
    new_global_info = (
        model_info(
            new_global
        )
    )
    new_core_info = (
        model_info(
            new_core
        )
    )

    print()
    print(
        f"OLD tree_count        : "
        f"{old_model_info['tree_count']}"
    )
    print(
        f"NEW Global tree_count : "
        f"{new_global_info['tree_count']}"
    )
    print(
        f"NEW CORE tree_count   : "
        f"{new_core_info['tree_count']}"
    )

    print()
    print(
        "OLD params       :",
        old_model_info["params"],
    )
    print(
        "NEW Global params:",
        new_global_info["params"],
    )
    print(
        "NEW CORE params  :",
        new_core_info["params"],
    )

    # HB-EB prior 직접 비교
    prior_comparison = (
        compare_priors(
            old_priors,
            new_priors,
        )
    )

    print()
    print("HB-EB prior comparison")

    for key, value in (
        prior_comparison.items()
    ):
        print(
            f"  {key}"
        )
        for k, v in (
            value.items()
        ):
            print(
                f"    {k}: {v}"
            )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("2. TEST PREDICTION COMPARISON")
    print("=" * 100)

    test = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    core_mask = (
        core_a_mask(test)
    )

    n_core = int(
        core_mask.sum()
    )

    n_non_core = int(
        (~core_mask).sum()
    )

    print(
        f"Test rows       : {len(test):,}"
    )
    print(
        f"CORE_A rows     : {n_core:,} "
        f"({core_mask.mean():.4%})"
    )
    print(
        f"Non-CORE_A rows : {n_non_core:,}"
    )

    # A = old cb_reg_v003
    pred_old, X_old = (
        predict_with_artifact(
            test=test,
            model=old_model,
            priors=old_priors,
            features=old_features,
            cat_cols=old_cat_cols,
        )
    )

    # B = new Global
    pred_new_global, X_new = (
        predict_with_artifact(
            test=test,
            model=new_global,
            priors=new_priors,
            features=new_features,
            cat_cols=new_cat_cols,
        )
    )

    # CORE_A specialist prediction
    # 같은 새 HB-EB transformed X를 사용
    pred_core = np.asarray(
        new_core.predict(
            X_new.loc[
                core_mask
            ]
        ),
        dtype=np.float64,
    )

    pred_core = np.clip(
        pred_core,
        0.0,
        1.0,
    )

    # C = routed
    pred_routed = (
        pred_new_global.copy()
    )

    pred_routed[
        core_mask
    ] = pred_core

    # --------------------------------------------------------
    # Compare A vs B
    # --------------------------------------------------------

    old_vs_new_global = (
        numeric_compare(
            pred_old,
            pred_new_global,
        )
    )

    # A vs C
    old_vs_routed_all = (
        numeric_compare(
            pred_old,
            pred_routed,
        )
    )

    # CORE_A
    old_vs_global_core = (
        numeric_compare(
            pred_old[
                core_mask
            ],
            pred_new_global[
                core_mask
            ],
        )
    )

    old_vs_routed_core = (
        numeric_compare(
            pred_old[
                core_mask
            ],
            pred_routed[
                core_mask
            ],
        )
    )

    global_vs_core_specialist = (
        numeric_compare(
            pred_new_global[
                core_mask
            ],
            pred_routed[
                core_mask
            ],
        )
    )

    # non CORE
    old_vs_routed_non_core = (
        numeric_compare(
            pred_old[
                ~core_mask
            ],
            pred_routed[
                ~core_mask
            ],
        )
    )

    print()
    print("[A = OLD cb_reg_v003 vs B = NEW Global]")
    for k, v in (
        old_vs_new_global.items()
    ):
        print(
            f"{k:30s}: {v}"
        )

    print()
    print("[A = OLD vs C = ROUTED | ALL]")
    for k, v in (
        old_vs_routed_all.items()
    ):
        print(
            f"{k:30s}: {v}"
        )

    print()
    print("[A = OLD vs C = ROUTED | NON-CORE_A]")
    for k, v in (
        old_vs_routed_non_core.items()
    ):
        print(
            f"{k:30s}: {v}"
        )

    print()
    print("[B = NEW Global vs C = CORE specialist | CORE_A]")
    for k, v in (
        global_vs_core_specialist.items()
    ):
        print(
            f"{k:30s}: {v}"
        )

    # --------------------------------------------------------
    # Routing integrity
    # --------------------------------------------------------

    routing_diff = np.abs(
        pred_routed
        - pred_new_global
    )

    changed_mask = (
        routing_diff > 1e-12
    )

    changed_outside_core = int(
        np.sum(
            changed_mask
            & (~core_mask)
        )
    )

    unchanged_inside_core = int(
        np.sum(
            (~changed_mask)
            & core_mask
        )
    )

    print()
    print("=" * 100)
    print("3. ROUTING INTEGRITY")
    print("=" * 100)
    print(
        f"Rows changed by routing       : "
        f"{changed_mask.sum():,}"
    )
    print(
        f"Changed outside CORE_A        : "
        f"{changed_outside_core:,}"
    )
    print(
        f"Unchanged inside CORE_A       : "
        f"{unchanged_inside_core:,}"
    )
    print(
        f"Expected CORE_A rows          : "
        f"{n_core:,}"
    )

    # --------------------------------------------------------
    # Save per-row
    # --------------------------------------------------------

    row_df = pd.DataFrame({
        ID: (
            test[ID].to_numpy()
            if ID in test.columns
            else np.arange(
                len(test)
            )
        ),
        "is_core_a": core_mask,
        "pred_old_cb_reg_v003": (
            pred_old
        ),
        "pred_new_global": (
            pred_new_global
        ),
        "pred_new_routed": (
            pred_routed
        ),
        "diff_new_global_minus_old": (
            pred_new_global
            - pred_old
        ),
        "diff_routed_minus_old": (
            pred_routed
            - pred_old
        ),
        "diff_routed_minus_new_global": (
            pred_routed
            - pred_new_global
        ),
    })

    row_df.to_csv(
        ROW_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Automatic verdict
    # --------------------------------------------------------

    global_same = (
        old_vs_new_global[
            "max_abs_diff"
        ] <= 1e-10
    )

    non_core_same = (
        old_vs_routed_non_core[
            "max_abs_diff"
        ] <= 1e-10
    )

    routing_clean = (
        changed_outside_core == 0
    )

    if global_same:
        verdict = (
            "GLOBAL_REPRODUCTION_PASS: "
            "기존 cb_reg_v003와 새 Global 예측이 사실상 동일합니다. "
            "따라서 LB 하락은 CORE_A specialist routing의 "
            "test generalization 실패로 해석할 수 있습니다."
        )
    else:
        verdict = (
            "GLOBAL_REPRODUCTION_FAIL: "
            "기존 cb_reg_v003와 새 Global 예측이 다릅니다. "
            "CORE_A 효과를 해석하기 전에 Global artifact/feature/HB-EB "
            "재현 차이부터 해결해야 합니다."
        )

    if not routing_clean:
        verdict += (
            " 추가로 CORE_A 밖의 row도 routing에 의해 변경되어 "
            "routing 구현 문제도 존재합니다."
        )

    print()
    print("=" * 100)
    print("4. FINAL VERDICT")
    print("=" * 100)
    print(verdict)
    print()
    print(
        f"Global same <=1e-10     : {global_same}"
    )
    print(
        f"Non-core same <=1e-10   : {non_core_same}"
    )
    print(
        f"Routing outside CORE_A=0: {routing_clean}"
    )

    summary = {
        "paths": {
            "old_model": str(
                OLD_MODEL_PATH
            ),
            "new_model": str(
                NEW_MODEL_PATH
            ),
            "test": str(
                TEST_PATH
            ),
        },
        "artifact": {
            "features_exact_order_equal": (
                features_exact
            ),
            "feature_sets_equal": (
                feature_sets_equal
            ),
            "cat_cols_exact_equal": (
                cats_exact
            ),
            "old_model": (
                old_model_info
            ),
            "new_global": (
                new_global_info
            ),
            "new_core": (
                new_core_info
            ),
            "hbeb_prior_comparison": (
                prior_comparison
            ),
        },
        "rows": {
            "test": int(
                len(test)
            ),
            "core_a": n_core,
            "non_core_a": n_non_core,
        },
        "comparisons": {
            "old_vs_new_global_all": (
                old_vs_new_global
            ),
            "old_vs_routed_all": (
                old_vs_routed_all
            ),
            "old_vs_global_core_a": (
                old_vs_global_core
            ),
            "old_vs_routed_core_a": (
                old_vs_routed_core
            ),
            "new_global_vs_routed_core_a": (
                global_vs_core_specialist
            ),
            "old_vs_routed_non_core_a": (
                old_vs_routed_non_core
            ),
        },
        "routing": {
            "rows_changed": int(
                changed_mask.sum()
            ),
            "changed_outside_core_a": (
                changed_outside_core
            ),
            "unchanged_inside_core_a": (
                unchanged_inside_core
            ),
        },
        "verdict": {
            "global_same_1e_10": (
                global_same
            ),
            "non_core_same_1e_10": (
                non_core_same
            ),
            "routing_clean": (
                routing_clean
            ),
            "text": verdict,
        },
    }

    with open(
        SUMMARY_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            to_jsonable(summary),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Per-row CSV : {ROW_OUTPUT_PATH}"
    )
    print(
        f"Summary JSON: {SUMMARY_OUTPUT_PATH}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
