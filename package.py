# package.py

import shutil
from pathlib import Path

import train


# =========================================================
# 경로
# =========================================================

EXPERIMENT_NAME = train.EXPERIMENT_NAME

MODEL_PATH = Path(train.MODEL_PATH)

SCRIPT_PATH = Path("./script.py")
REQUIREMENTS_PATH = Path("./requirements.txt")

SUBMIT_ROOT = Path("./submits")
SUBMIT_DIR = SUBMIT_ROOT / EXPERIMENT_NAME


# =========================================================
# requirements.txt 확인
# =========================================================

def inspect_requirements(path):
    """
    requirements.txt 내용을 출력하고
    중복 package가 있으면 경고한다.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"requirements.txt를 찾을 수 없습니다: {path}"
        )

    packages = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            packages.append(line)

    print()
    print("=" * 60)
    print("requirements.txt")
    print("=" * 60)

    for pkg in packages:
        print(f"- {pkg}")

    lower = [
        pkg.lower()
        for pkg in packages
    ]

    duplicates = sorted({
        pkg
        for pkg in lower
        if lower.count(pkg) > 1
    })

    if duplicates:

        print()
        print("[경고] 중복 package 발견")

        for pkg in duplicates:
            print(f"- {pkg}")

    print("=" * 60)


# =========================================================
# 제출 디렉터리 확인
# =========================================================

def check_submit_directory():

    if SUBMIT_DIR.exists():
        raise FileExistsError(
            f"Submit directory already exists: {SUBMIT_DIR}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )


# =========================================================
# 파일 존재 확인
# =========================================================

def check_files():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다: {MODEL_PATH}"
        )

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"script.py를 찾을 수 없습니다."
        )

    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(
            f"requirements.txt를 찾을 수 없습니다."
        )


# =========================================================
# 제출 패키지 생성
# =========================================================

def build_package():

    SUBMIT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_dir = SUBMIT_DIR / "model"

    model_dir.mkdir()

    shutil.copy2(
        MODEL_PATH,
        model_dir / MODEL_PATH.name,
    )

    shutil.copy2(
        SCRIPT_PATH,
        SUBMIT_DIR / "script.py",
    )

    shutil.copy2(
        REQUIREMENTS_PATH,
        SUBMIT_DIR / "requirements.txt",
    )

    archive_path = shutil.make_archive(
        str(SUBMIT_DIR),
        "zip",
        root_dir=SUBMIT_DIR,
    )

    print()
    print("=" * 60)
    print("Package 생성 완료")
    print("=" * 60)
    print(f"Directory : {SUBMIT_DIR}")
    print(f"Archive   : {archive_path}")
    print("=" * 60)


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 60)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Model      : {MODEL_PATH}")
    print("=" * 60)

    inspect_requirements(
        REQUIREMENTS_PATH,
    )

    check_submit_directory()

    check_files()

    build_package()


if __name__ == "__main__":
    main()