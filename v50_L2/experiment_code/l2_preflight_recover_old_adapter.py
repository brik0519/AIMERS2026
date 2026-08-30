#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
l2_preflight_recover_old_adapter.py
===================================

목적
----
새 Final L1(v42) 기준 L2 ExtraTrees residual adapter를 재학습하기 전에
옛 L2의 정확한 adapter feature schema / estimator params / source 후보를
로컬 프로젝트에서 자동 복원한다.

이 스크립트는 학습하지 않는다.
파일/모델 metadata만 읽는다.

실행
----
python l2_preflight_recover_old_adapter.py

출력 블록 전체를 ChatGPT에 보내면 다음 단계:
    v42 OOF residual
    -> exact old adapter core + Exact5
    -> ExtraTrees 3 seeds
    -> shrink sweep
코드를 바로 생성한다.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(".")

# -------------------------------------------------------------------------
# Known project paths
# -------------------------------------------------------------------------

V42_OOF_CANDIDATES = [
    ROOT
    / "validation"
    / "cb_reg_v042_success_full_r_only_submit"
    / "oof_predictions.csv",

    ROOT
    / "validation"
    / "cb_reg_v042_success_full_r_only_submit"
    / "oof_predictions.csv.gz",
]

TRAIN_CANDIDATES = [
    ROOT / "data" / "train.csv",
    ROOT / "open" / "train.csv",
]

# Exact later L2 family first.
MODEL_GLOBS = [
    "model/cb_reg_v028_l1pc_l2_extratrees_no_pc/**/*.pkl",
    "model/cb_reg_v028_l1pc_l2_extratrees_no_pc/**/*.joblib",

    # Earlier residual adapter family, useful for recovering adapter-core schema.
    "model/cb_reg_domain_residual_adapter_v001/adapter_folds/adapter_R_to_2024.pkl",
    "model/cb_reg_domain_residual_adapter_v001/adapter_folds/*.pkl",

    # General fallback.
    "model/**/*l2*.pkl",
    "model/**/*adapter*.pkl",
]

SOURCE_PRIORITY_TOKENS = [
    "cb_reg_v028_l1pc_l2_extratrees_no_pc",
    "l1pc_l2",
    "extratrees",
    "residual_adapter",
    "domain_residual_adapter",
]

EXPECTED_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 200,
    "max_features": 0.7,
}

EXACT5 = [
    "exact_pitcher_season_n",
    "exact_pitcher_season_logn",
    "exact_pitcher_season_success_rate",
    "exact_pitcher_past_success_rate",
    "exact_pitcher_season_minus_past",
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def section(title):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def safe_repr(x, limit=500):
    try:
        s = repr(x)
    except Exception as exc:
        return f"<repr failed: {exc}>"

    if len(s) > limit:
        return s[:limit] + " ..."

    return s


def load_any(path):
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, exc


def iter_children(obj):
    """
    Conservative recursive traversal of likely sklearn/joblib artifacts.
    Avoid traversing huge model internals.
    """

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"[{k!r}]", v
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:100]):
            yield f"[{i}]", v
        return

    # sklearn Pipeline
    if hasattr(obj, "named_steps"):
        try:
            for k, v in obj.named_steps.items():
                yield f".named_steps[{k!r}]", v
        except Exception:
            pass

    # common wrapper attrs
    for attr in [
        "model",
        "estimator",
        "regressor",
        "adapter",
        "pipeline",
        "preprocessor",
    ]:
        if hasattr(obj, attr):
            try:
                yield f".{attr}", getattr(obj, attr)
            except Exception:
                pass


def recursive_find_estimators(root_obj, max_depth=4):
    found = []
    seen = set()

    def rec(obj, path, depth):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        cls = obj.__class__.__name__

        if (
            "ExtraTrees" in cls
            or "RandomForest" in cls
            or "Pipeline" in cls
        ):
            found.append(
                {
                    "path": path,
                    "class": cls,
                    "object": obj,
                }
            )

        if depth >= max_depth:
            return

        for suffix, child in iter_children(obj):
            rec(
                child,
                path + suffix,
                depth + 1,
            )

    rec(
        root_obj,
        "root",
        0,
    )

    return found


def extract_feature_candidates(root_obj, max_depth=4):
    results = []
    seen = set()

    interesting_keys = {
        "features",
        "feature_names",
        "feature_cols",
        "feature_columns",
        "adapter_features",
        "adapter_cols",
        "l2_features",
        "numeric_features",
        "num_features",
    }

    def normalize_feature_seq(value):
        if isinstance(
            value,
            np.ndarray,
        ):
            value = value.tolist()

        if isinstance(
            value,
            pd.Index,
        ):
            value = value.tolist()

        if isinstance(
            value,
            (list, tuple),
        ):
            if (
                len(value) > 0
                and all(
                    isinstance(x, str)
                    for x in value
                )
            ):
                return list(value)

        return None

    def rec(obj, path, depth):
        oid = id(obj)

        if oid in seen:
            return

        seen.add(oid)

        # sklearn feature_names_in_
        if hasattr(
            obj,
            "feature_names_in_",
        ):
            try:
                value = (
                    getattr(
                        obj,
                        "feature_names_in_",
                    )
                )

                seq = normalize_feature_seq(
                    value
                )

                if seq:
                    results.append(
                        {
                            "source": (
                                path
                                + ".feature_names_in_"
                            ),
                            "features": seq,
                        }
                    )
            except Exception:
                pass

        if isinstance(
            obj,
            dict,
        ):
            for key, value in (
                obj.items()
            ):
                key_str = str(
                    key
                )

                if (
                    key_str
                    in interesting_keys
                    or
                    "feature" in key_str.lower()
                    and "importance" not in key_str.lower()
                ):
                    seq = normalize_feature_seq(
                        value
                    )

                    if seq:
                        results.append(
                            {
                                "source": (
                                    path
                                    + f"[{key!r}]"
                                ),
                                "features": seq,
                            }
                        )

        if depth >= max_depth:
            return

        for suffix, child in (
            iter_children(
                obj
            )
        ):
            rec(
                child,
                path + suffix,
                depth + 1,
            )

    rec(
        root_obj,
        "root",
        0,
    )

    # de-duplicate exact sequences
    dedup = []

    fingerprints = set()

    for item in results:
        fp = tuple(
            item[
                "features"
            ]
        )

        if fp in fingerprints:
            continue

        fingerprints.add(
            fp
        )

        dedup.append(
            item
        )

    return dedup


def get_model_params(obj):
    if not hasattr(
        obj,
        "get_params",
    ):
        return None

    try:
        params = (
            obj.get_params(
                deep=False
            )
        )
    except Exception:
        return None

    keys = [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "random_state",
        "n_jobs",
        "bootstrap",
        "criterion",
    ]

    return {
        key: params.get(
            key,
            "<missing>",
        )
        for key in keys
    }


def source_score(path, text):
    name = str(
        path
    ).lower()

    score = 0

    for rank, token in enumerate(
        SOURCE_PRIORITY_TOKENS
    ):
        if token in name:
            score += (
                100
                - rank * 10
            )

        if token in text.lower():
            score += (
                30
                - rank * 2
            )

    if "extratreesregressor" in text.lower():
        score += 60

    if "min_samples_leaf" in text.lower():
        score += 20

    if "residual" in text.lower():
        score += 20

    if "exact_pitcher_season_n" in text:
        score += 20

    if "pc_features" in text.lower():
        score += 5

    return score


def discover_source_candidates():
    rows = []

    for p in ROOT.rglob("*.py"):
        if not p.is_file():
            continue

        if any(
            part in {
                ".git",
                "__pycache__",
                ".venv",
                "venv",
            }
            for part in p.parts
        ):
            continue

        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        s = source_score(
            p,
            text,
        )

        if s <= 0:
            continue

        rows.append(
            (
                s,
                p,
                text,
            )
        )

    rows.sort(
        key=lambda x: (
            -x[0],
            str(x[1]),
        )
    )

    return rows


def ast_string_lists(path):
    """
    Extract top-level Python assignments whose value is a literal list/tuple of strings.
    This often recovers ADAPTER_FEATURES exactly.
    """

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        tree = ast.parse(
            text
        )
    except Exception as exc:
        return [
            (
                "<AST ERROR>",
                [str(exc)],
            )
        ]

    out = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        names = []

        if isinstance(
            node,
            ast.Assign,
        ):
            for t in node.targets:
                if isinstance(
                    t,
                    ast.Name,
                ):
                    names.append(
                        t.id
                    )

            value_node = (
                node.value
            )

        else:
            if isinstance(
                node.target,
                ast.Name,
            ):
                names.append(
                    node.target.id
                )

            value_node = (
                node.value
            )

        if (
            not names
            or value_node is None
        ):
            continue

        try:
            value = ast.literal_eval(
                value_node
            )
        except Exception:
            continue

        if (
            isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            )
            and len(value) > 0
            and all(
                isinstance(x, str)
                for x in value
            )
        ):
            for name in names:
                lname = name.lower()

                if (
                    "feature" in lname
                    or "adapter" in lname
                    or "numeric" in lname
                    or "column" in lname
                    or lname.endswith(
                        "_cols"
                    )
                ):
                    out.append(
                        (
                            name,
                            list(value),
                        )
                    )

    return out


def discover_model_candidates():
    paths = []

    seen = set()

    for pattern in MODEL_GLOBS:
        for p in ROOT.glob(
            pattern
        ):
            if not p.is_file():
                continue

            key = str(
                p.resolve()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            paths.append(
                p
            )

    def priority(p):
        s = str(
            p
        ).lower()

        score = 0

        if (
            "cb_reg_v028_l1pc_l2_extratrees_no_pc"
            in s
        ):
            score += 1000

        if "adapter_r_to_2024" in s:
            score += 500

        if "domain_residual_adapter" in s:
            score += 200

        if "2024" in s:
            score += 50

        return (
            -score,
            str(p),
        )

    return sorted(
        paths,
        key=priority,
    )


def main():
    section(
        "L2 PREFLIGHT — REQUIRED FILES"
    )

    train_path = first_existing(
        TRAIN_CANDIDATES
    )

    v42_oof_path = (
        first_existing(
            V42_OOF_CANDIDATES
        )
    )

    print(
        "train.csv :",
        train_path
        if train_path
        else "NOT FOUND",
    )

    print(
        "v42 OOF   :",
        v42_oof_path
        if v42_oof_path
        else "NOT FOUND",
    )

    if v42_oof_path is not None:
        try:
            oof = pd.read_csv(
                v42_oof_path,
                encoding="utf-8-sig",
                low_memory=False,
            )

            print(
                f"v42 OOF rows: {len(oof):,}"
            )

            print(
                "v42 OOF columns:"
            )

            print(
                list(
                    oof.columns
                )
            )

            required_any = [
                "routed_pred",
                "y_true",
                "row_id",
                "season",
            ]

            print(
                "required columns present:",
                {
                    c: (
                        c in oof.columns
                    )
                    for c in required_any
                },
            )
        except Exception as exc:
            print(
                "v42 OOF read ERROR:",
                repr(exc),
            )

    # ---------------------------------------------------------------------
    # Old source
    # ---------------------------------------------------------------------
    section(
        "OLD L2 SOURCE CANDIDATES"
    )

    source_candidates = (
        discover_source_candidates()
    )

    if not source_candidates:
        print(
            "No relevant .py source found."
        )

    for rank, (
        score,
        path,
        text,
    ) in enumerate(
        source_candidates[:12],
        start=1,
    ):
        print(
            f"[{rank}] score={score:4d}  {path}"
        )

        indicators = {
            "ExtraTreesRegressor": (
                "ExtraTreesRegressor"
                in text
            ),
            "min_samples_leaf": (
                "min_samples_leaf"
                in text
            ),
            "max_features": (
                "max_features"
                in text
            ),
            "residual": (
                "residual"
                in text.lower()
            ),
            "Exact5": (
                "exact_pitcher_season_n"
                in text
            ),
        }

        print(
            "    indicators:",
            indicators,
        )

        lists = (
            ast_string_lists(
                path
            )
        )

        for name, values in lists[:12]:
            print(
                f"    {name} ({len(values)}):"
            )

            print(
                "       ",
                values,
            )

    # ---------------------------------------------------------------------
    # Old model
    # ---------------------------------------------------------------------
    section(
        "OLD L2 MODEL CANDIDATES"
    )

    model_candidates = (
        discover_model_candidates()
    )

    if not model_candidates:
        print(
            "No old adapter/L2 model artifacts found."
        )

    for rank, path in enumerate(
        model_candidates[:20],
        start=1,
    ):
        print(
            f"[{rank}] {path}"
        )

    # Inspect the best few loadable candidates.
    inspected = 0

    for path in (
        model_candidates
    ):
        if inspected >= 5:
            break

        obj, err = load_any(
            path
        )

        if err is not None:
            print()
            print(
                f"[LOAD FAIL] {path}"
            )
            print(
                repr(err)
            )
            continue

        inspected += 1

        section(
            f"MODEL INSPECT #{inspected}: {path}"
        )

        print(
            "root class:",
            obj.__class__.__module__
            + "."
            + obj.__class__.__name__,
        )

        if isinstance(
            obj,
            dict,
        ):
            print(
                "artifact keys:"
            )

            print(
                sorted(
                    map(
                        str,
                        obj.keys(),
                    )
                )
            )

        estimators = (
            recursive_find_estimators(
                obj
            )
        )

        if not estimators:
            print(
                "No ExtraTrees/RandomForest/Pipeline estimator "
                "found in shallow artifact traversal."
            )

        for est_item in estimators:
            est = (
                est_item[
                    "object"
                ]
            )

            print()
            print(
                "Estimator path :",
                est_item[
                    "path"
                ],
            )

            print(
                "Estimator class:",
                est_item[
                    "class"
                ],
            )

            params = get_model_params(
                est
            )

            if params is not None:
                print(
                    "params:",
                    params,
                )

                comparison = {}

                for key, expected in (
                    EXPECTED_PARAMS.items()
                ):
                    actual = params.get(
                        key,
                        "<missing>",
                    )

                    comparison[
                        key
                    ] = {
                        "actual": actual,
                        "expected": expected,
                        "match": (
                            actual == expected
                        ),
                    }

                print(
                    "known-param comparison:",
                    comparison,
                )

        feature_sets = (
            extract_feature_candidates(
                obj
            )
        )

        if not feature_sets:
            print()
            print(
                "No explicit feature name sequence found."
            )
        else:
            print()
            print(
                "Feature-name candidates:"
            )

            for i, item in enumerate(
                feature_sets,
                start=1,
            ):
                features = (
                    item[
                        "features"
                    ]
                )

                print(
                    f"  [{i}] source={item['source']} "
                    f"n={len(features)}"
                )

                print(
                    "      ",
                    features,
                )

                exact_present = [
                    c
                    for c in EXACT5
                    if c in features
                ]

                print(
                    "      Exact5 present:",
                    exact_present,
                )

    # ---------------------------------------------------------------------
    # Final instructions
    # ---------------------------------------------------------------------
    section(
        "WHAT TO SEND BACK"
    )

    print(
        "위 출력에서 다음 블록을 전부 복붙:"
    )

    print(
        "1) L2 PREFLIGHT — REQUIRED FILES"
    )

    print(
        "2) OLD L2 SOURCE CANDIDATES"
    )

    print(
        "3) MODEL INSPECT #... 블록"
    )

    print()
    print(
        "이 결과로 adapter core를 추측하지 않고 "
        "정확히 복원한 뒤 v42 residual L2 코드를 생성합니다."
    )


if __name__ == "__main__":
    main()
