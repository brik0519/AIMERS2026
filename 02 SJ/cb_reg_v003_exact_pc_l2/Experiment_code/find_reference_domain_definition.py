#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_reference_domain_definition.py

프로젝트 폴더 안의 .py / .ipynb / .md / .txt 파일을 훑어서
reference의 R_CORE / R_ANCHOR / F domain 정의가 들어간 코드/문맥을 찾는다.

임의로 domain을 추측하지 않고, 실제 reference 코드가 로컬에 남아 있는지
먼저 복구하기 위한 audit script.

실행:
    python find_reference_domain_definition.py

출력:
    validation/reference_domain_audit/domain_hits.txt
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(".")
OUT_DIR = ROOT / "validation" / "reference_domain_audit"
OUT_PATH = OUT_DIR / "domain_hits.txt"

TARGET_EXTS = {".py", ".ipynb", ".md", ".txt"}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "validation/reference_domain_audit",
}

KEYWORDS = [
    "R_CORE",
    "R_ANCHOR",
    "three-domain",
    "three_domain",
    "domain",
]

STRONG_PATTERNS = [
    r"R_CORE",
    r"R_ANCHOR",
    r"np\.where",
    r"\.isin\(",
    r"game_type",
    r"pitcher_team_id",
    r"batter_team_id",
    r"DOMAIN",
    r"domain_mask",
    r"assign_domain",
    r"get_domain",
    r"make_domain",
]


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if any(x in parts for x in {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}):
        return True
    if OUT_DIR in path.parents:
        return True
    return False


def extract_ipynb(path: Path):
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    chunks = []
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if not source:
            continue
        chunks.append((f"cell {i} [{cell.get('cell_type', 'unknown')}]", source))
    return chunks


def extract_text(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except Exception:
            return []
    except Exception:
        return []

    return [("text", text)]


def score_block(text: str) -> int:
    score = 0
    for pat in STRONG_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            score += 1
    if "R_CORE" in text and "R_ANCHOR" in text:
        score += 5
    return score


def context_from_lines(text: str, radius: int = 18):
    lines = text.splitlines()
    hits = []

    target_idxs = [
        i for i, line in enumerate(lines)
        if ("R_CORE" in line or "R_ANCHOR" in line)
    ]

    merged = []
    for idx in target_idxs:
        lo = max(0, idx - radius)
        hi = min(len(lines), idx + radius + 1)
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    for lo, hi in merged:
        snippet = "\n".join(
            f"{i+1:5d}: {lines[i]}"
            for i in range(lo, hi)
        )
        hits.append(snippet)

    return hits


def main():
    candidates = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TARGET_EXTS:
            continue
        if should_skip(path):
            continue

        if path.suffix.lower() == ".ipynb":
            blocks = extract_ipynb(path)
        else:
            blocks = extract_text(path)

        for label, text in blocks:
            if "R_CORE" not in text and "R_ANCHOR" not in text:
                continue

            score = score_block(text)
            contexts = context_from_lines(text)

            candidates.append({
                "path": str(path),
                "label": label,
                "score": score,
                "contexts": contexts,
            })

    candidates.sort(
        key=lambda x: (-x["score"], x["path"], x["label"])
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = []
    out.append("=" * 120)
    out.append("REFERENCE DOMAIN AUDIT")
    out.append("=" * 120)
    out.append(f"files/cells with R_CORE or R_ANCHOR: {len(candidates)}")
    out.append("")

    if not candidates:
        out.append(
            "NO HITS.\n"
            "현재 프로젝트 폴더에는 exact domain definition을 복구할 수 있는 "
            "R_CORE/R_ANCHOR 코드가 없습니다."
        )
    else:
        for rank, item in enumerate(candidates, start=1):
            out.append("#" * 120)
            out.append(
                f"[{rank}] score={item['score']} | "
                f"{item['path']} | {item['label']}"
            )
            out.append("#" * 120)

            for j, snippet in enumerate(item["contexts"], start=1):
                out.append(f"\n--- context {j} ---")
                out.append(snippet)
                out.append("")

    OUT_PATH.write_text("\n".join(out), encoding="utf-8")

    print("=" * 100)
    print("REFERENCE DOMAIN AUDIT")
    print("=" * 100)
    print(f"hits: {len(candidates)}")
    print(f"saved: {OUT_PATH}")

    if candidates:
        print("\nTop candidates:")
        for item in candidates[:10]:
            print(
                f"score={item['score']:2d} | "
                f"{item['path']} | {item['label']}"
            )

        print(
            "\n다음 파일의 상위 후보 문맥을 보내주면 "
            "R_CORE/R_ANCHOR 정의를 그대로 사용해 L2 domain A/B를 만들 수 있습니다:"
        )
        print(OUT_PATH)
    else:
        print(
            "\n로컬에도 정의가 없습니다. "
            "이 경우 reference V13/V14 원본 notebook/code를 추가로 확보해야 합니다."
        )


if __name__ == "__main__":
    main()
