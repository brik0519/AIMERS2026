#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
l2_extract_exact_adapter_builder_to_file.py

old L2 train.py에서 필요한 정확한 L2 adapter 정의를 추출하여
터미널 출력이 아니라 파일 하나로 저장한다.

실행:
    python l2_extract_exact_adapter_builder_to_file.py

생성:
    l2_adapter_exact_extract.txt

그 파일만 ChatGPT에 업로드하면 됨.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(".")

SOURCE_CANDIDATES = [
    ROOT / "model" / "cb_reg_v003_exact_pc_l2" / "train.py",
    ROOT / "validation" / "~0824" / "cb_reg_domain_residual_adapter_v001"
    / "train_cb_reg_domain_residual_adapter_v001.py",
]

OUT_PATH = ROOT / "l2_adapter_exact_extract.txt"

WANTED_ASSIGNMENTS = {
    "REFERENCE_RAWNUM",
    "REFERENCE_CAT_SMALL",
    "ENG_COLS",
    "CAT_COLS",
    "EXACT_FEATURES",
    "ADAPTER_SEEDS",
    "V15_CF_CFG",
    "FINAL_ADAPTER_CFG",
    "ADAPTER_SHRINK",
    "SOURCE_YEAR",
    "EXPECTED_TARGET_YEAR",
    "N_JOBS",
}

WANTED_FUNCTIONS = {
    "numeric_series",
    "add_eng",
    "fit_adapter_prep",
    "transform_adapter_core",
    "exact5_matrix",
    "fit_final_l2_models",
}


def find_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p

    matches = sorted(
        p
        for p in ROOT.rglob("train.py")
        if p.is_file()
        and "cb_reg_v003_exact_pc_l2" in str(p)
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "model/cb_reg_v003_exact_pc_l2/train.py 를 찾지 못했습니다."
    )


def node_source(lines, node) -> str:
    start = int(node.lineno) - 1
    end = int(getattr(node, "end_lineno", node.lineno))
    return "\n".join(lines[start:end])


def assignment_names(node):
    names = []

    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)

    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            names.append(node.target.id)

    return names


def main():
    source = find_source()

    text = source.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()
    tree = ast.parse(text)

    chunks = []

    chunks.append("=" * 120)
    chunks.append("SOURCE")
    chunks.append("=" * 120)
    chunks.append(str(source))
    chunks.append("")

    chunks.append("=" * 120)
    chunks.append("EXACT CONSTANTS")
    chunks.append("=" * 120)

    found_assignments = set()

    for node in tree.body:
        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):
            continue

        names = assignment_names(node)

        if not any(
            name in WANTED_ASSIGNMENTS
            for name in names
        ):
            continue

        found_assignments.update(
            name
            for name in names
            if name in WANTED_ASSIGNMENTS
        )

        chunks.append("")
        chunks.append(
            f"# lines {node.lineno}-"
            f"{getattr(node, 'end_lineno', node.lineno)}"
        )
        chunks.append(
            node_source(lines, node)
        )

    missing_assignments = sorted(
        WANTED_ASSIGNMENTS
        - found_assignments
    )

    chunks.append("")
    chunks.append(
        "MISSING CONSTANTS: "
        + repr(missing_assignments)
    )

    chunks.append("")
    chunks.append("=" * 120)
    chunks.append("EXACT FUNCTIONS")
    chunks.append("=" * 120)

    found_functions = set()

    for node in tree.body:
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        if node.name not in WANTED_FUNCTIONS:
            continue

        found_functions.add(node.name)

        chunks.append("")
        chunks.append("-" * 120)
        chunks.append(
            f"FUNCTION {node.name} "
            f"(lines {node.lineno}-"
            f"{getattr(node, 'end_lineno', node.lineno)})"
        )
        chunks.append("-" * 120)
        chunks.append(
            node_source(lines, node)
        )

    missing_functions = sorted(
        WANTED_FUNCTIONS
        - found_functions
    )

    chunks.append("")
    chunks.append(
        "MISSING FUNCTIONS: "
        + repr(missing_functions)
    )

    # Also include the exact final L2 construction block.
    chunks.append("")
    chunks.append("=" * 120)
    chunks.append("FINAL L2 CONSTRUCTION CONTEXT")
    chunks.append("=" * 120)

    needles = [
        "adapter_prep = fit_adapter_prep",
        "core24 = transform_adapter_core",
        "X24 = pd.concat",
        "fit_final_l2_models(",
        '"adapter_core_feature_cols"',
        '"adapter_feature_cols"',
        '"reference_rawnum"',
        '"reference_cat_small"',
        '"eng_cols"',
    ]

    emitted = set()

    for needle in needles:
        hits = [
            i
            for i, line in enumerate(lines)
            if needle in line
        ]

        for i in hits:
            lo = max(0, i - 10)
            hi = min(len(lines), i + 20)

            key = (lo, hi)

            if key in emitted:
                continue

            emitted.add(key)

            chunks.append("")
            chunks.append(
                f"# context lines {lo + 1}-{hi}"
            )

            for j in range(lo, hi):
                chunks.append(
                    f"{j + 1:5d}: {lines[j]}"
                )

    chunks.append("")
    chunks.append("=" * 120)
    chunks.append("END")
    chunks.append("=" * 120)

    OUT_PATH.write_text(
        "\n".join(chunks) + "\n",
        encoding="utf-8",
    )

    print(f"Source : {source}")
    print(f"Saved  : {OUT_PATH}")
    print(
        f"Size   : {OUT_PATH.stat().st_size / 1024:.1f} KB"
    )
    print()
    print(
        "이제 터미널 내용을 복붙하지 말고 "
        "l2_adapter_exact_extract.txt 파일 자체를 업로드하세요."
    )


if __name__ == "__main__":
    main()
