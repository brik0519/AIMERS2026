# package.py

import argparse
import importlib
import shutil
from pathlib import Path


# =========================================================
# 경로
# =========================================================

TRAIN_PATH = Path("./train.py")
SCRIPT_PATH = Path("./script.py")
REQUIREMENTS_PATH = Path("./requirements.txt")

SUBMIT_ROOT = Path("./submits")

LOCAL_DATA_PATH = './data'
KAGGLE_DATA_PATH = '/kaggle/input/datasets/bemorehyeon/aimersdata'

LOCAL_OUTPUT_PATH = './output'
KAGGLE_OUTPUT_PATH = '/kaggle/working'

PATH_TOGGLE_MAP = {
    LOCAL_DATA_PATH: KAGGLE_DATA_PATH,
    LOCAL_OUTPUT_PATH: KAGGLE_OUTPUT_PATH,
}


# =========================================================
# Argument parser
# =========================================================

def parse_args():
    """
    Toggle 대상 파일과 실행 환경을 입력받는다.

    옵션을 지정하지 않으면 기존과 동일하게 package만 생성한다.
    Toggle을 사용할 경우 대상 파일과 실행 환경을 모두 지정해야 한다.
    """

    parser = argparse.ArgumentParser()

    target_group = parser.add_mutually_exclusive_group()

    target_group.add_argument(
        "--train",
        dest="target",
        action="store_const",
        const=TRAIN_PATH,
        help="train.py의 실행 경로를 변경한다.",
    )

    target_group.add_argument(
        "--script",
        dest="target",
        action="store_const",
        const=SCRIPT_PATH,
        help="script.py의 실행 경로를 변경한다.",
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "-k",
        "--kaggle",
        dest="mode",
        action="store_const",
        const="kaggle",
        help="선택한 파일의 경로를 Kaggle 환경으로 변경한다.",
    )

    mode_group.add_argument(
        "-l",
        "--local",
        dest="mode",
        action="store_const",
        const="local",
        help="선택한 파일의 경로를 Local 환경으로 변경한다.",
    )

    args = parser.parse_args()

    if (args.target is None) != (args.mode is None):
        parser.error(
            "Toggle을 사용하려면 대상(--train/--script)과 "
            "환경(--kaggle/--local)을 함께 지정해야 합니다."
        )

    return args


# =========================================================
# Local / Kaggle 경로 Toggle
# =========================================================

def toggle_paths(path, mode):
    """
    선택한 Python 파일의 Local/Kaggle 경로를 전환한다.

    Local:
        ./data
        ./output

    Kaggle:
        /kaggle/input/datasets/bemorehyeon/aimersdata
        /kaggle/working
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Toggle 대상 파일을 찾을 수 없습니다: {path}"
        )

    content = path.read_text(encoding="utf-8")
    changed = []
    already = []

    for local_path, kaggle_path in PATH_TOGGLE_MAP.items():

        if mode == "kaggle":
            source = local_path
            target = kaggle_path
        else:
            source = kaggle_path
            target = local_path

        if source in content:
            content = content.replace(
                source,
                target,
            )
            changed.append(
                (source, target)
            )

        elif target in content:
            already.append(target)

    if not changed and not already:
        raise ValueError(
            f"{path}에서 Toggle 가능한 경로를 찾을 수 없습니다."
        )

    if changed:
        path.write_text(
            content,
            encoding="utf-8",
        )

    print()
    print("=" * 60)
    print(f"Path Toggle : {path}")
    print(f"Mode        : {mode}")
    print("=" * 60)

    for source, target in changed:
        print(f"[변경] {source}")
        print(f"    -> {target}")

    for target in already:
        print(f"[유지] {target}")

    print("=" * 60)


# =========================================================
# train.py 설정 확인
# =========================================================

def load_train_config():
    """
    Toggle이 끝난 뒤 train.py를 import하여
    package 생성에 필요한 설정을 가져온다.
    """

    train = importlib.import_module("train")

    experiment_name = train.EXPERIMENT_NAME
    model_path = Path(train.MODEL_PATH)
    submit_dir = SUBMIT_ROOT / experiment_name

    return (
        experiment_name,
        model_path,
        submit_dir,
    )


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

def check_submit_directory(submit_dir):

    if submit_dir.exists():
        raise FileExistsError(
            f"Submit directory already exists: {submit_dir}\n"
            "새 EXPERIMENT_NAME을 사용하세요."
        )


# =========================================================
# 파일 존재 확인
# =========================================================

def check_files(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다: {model_path}"
        )

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "script.py를 찾을 수 없습니다."
        )

    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(
            "requirements.txt를 찾을 수 없습니다."
        )


# =========================================================
# 제출 패키지 생성
# =========================================================

def build_package(
    model_path,
    submit_dir,
):

    submit_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_dir = submit_dir / "model"

    model_dir.mkdir()

    shutil.copy2(
        model_path,
        model_dir / model_path.name,
    )

    shutil.copy2(
        SCRIPT_PATH,
        submit_dir / "script.py",
    )

    shutil.copy2(
        REQUIREMENTS_PATH,
        submit_dir / "requirements.txt",
    )

    archive_path = shutil.make_archive(
        str(submit_dir),
        "zip",
        root_dir=submit_dir,
    )

    print()
    print("=" * 60)
    print("Package 생성 완료")
    print("=" * 60)
    print(f"Directory : {submit_dir}")
    print(f"Archive   : {archive_path}")
    print("=" * 60)


# =========================================================
# Main
# =========================================================
def main():

    args = parse_args()

    if args.target is not None:
        toggle_paths(
            args.target,
            args.mode,
        )
        return

    (
        experiment_name,
        model_path,
        submit_dir,
    ) = load_train_config()

    print()
    print("=" * 60)
    print(f"Experiment : {experiment_name}")
    print(f"Model      : {model_path}")
    print("=" * 60)

    inspect_requirements(
        REQUIREMENTS_PATH,
    )

    check_submit_directory(
        submit_dir,
    )

    check_files(
        model_path,
    )

    build_package(
        model_path,
        submit_dir,
    )


if __name__ == "__main__":
    main()
