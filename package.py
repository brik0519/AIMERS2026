# package.py
#
# Single model / Ensemble 공통 packaging workflow.
#
# Package layout
# --------------
# submits/<experiment_name>/
# ├── model/
# │   ├── <model_a>/
# │   │   └── <model_a>.pkl
# │   └── ...
# ├── script.py
# ├── requirements.txt
# └── <experiment_name>.zip
#
# ZIP 내부도 동일하게:
# model/<model_name>/<model_name>.pkl
# script.py
# requirements.txt

import argparse
import ast
import importlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


# =========================================================
# 경로
# =========================================================

TRAIN_PATH = Path("./train.py")
SCRIPT_PATH = Path("./script.py")
REQUIREMENTS_PATH = Path("./requirements.txt")

MODEL_ROOT = Path("./model")

ENSEMBLE_ROOT = Path("./ensemble")
VALIDATION_ROOT = Path("./validation")

SUBMIT_ROOT = Path("./submits")

LOCAL_DATA_PATH = "./data"
KAGGLE_DATA_PATH = (
    "/kaggle/input/datasets/bemorehyeon/aimersdata"
)

LOCAL_OUTPUT_PATH = "./output"
KAGGLE_OUTPUT_PATH = "/kaggle/working"

PATH_TOGGLE_MAP = {
    LOCAL_DATA_PATH: KAGGLE_DATA_PATH,
    LOCAL_OUTPUT_PATH: KAGGLE_OUTPUT_PATH,
}


# =========================================================
# 이름 규칙
# =========================================================

ENSEMBLE_PATTERN = re.compile(
    r"^ensemble_v(\d+)$"
)


# =========================================================
# Package Spec
# =========================================================

@dataclass(frozen=True)
class PackageSpec:
    experiment_name: str
    package_type: str
    model_paths: dict[str, Path]
    script_path: Path
    requirements_path: Path
    submit_dir: Path
    metadata_path: Path | None = None

    @property
    def archive_path(self) -> Path:
        return (
            self.submit_dir
            / f"{self.experiment_name}.zip"
        )


# =========================================================
# Argument parser
# =========================================================

def parse_args():
    """
    옵션이 없으면 현재 train.py 기준 단일 모델 package를 생성한다.

    -e / --ensemble:
        최신 ensemble_vNNN package 생성

    --train/--script + --kaggle/--local:
        기존 Local/Kaggle 경로 Toggle
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--ensemble",
        action="store_true",
        help=(
            "가장 최신 ensemble experiment를 "
            "package로 생성한다."
        ),
    )

    target_group = (
        parser.add_mutually_exclusive_group()
    )

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

    mode_group = (
        parser.add_mutually_exclusive_group()
    )

    mode_group.add_argument(
        "-k",
        "--kaggle",
        dest="mode",
        action="store_const",
        const="kaggle",
        help=(
            "선택한 파일의 경로를 "
            "Kaggle 환경으로 변경한다."
        ),
    )

    mode_group.add_argument(
        "-l",
        "--local",
        dest="mode",
        action="store_const",
        const="local",
        help=(
            "선택한 파일의 경로를 "
            "Local 환경으로 변경한다."
        ),
    )

    args = parser.parse_args()

    if (
        (args.target is None)
        != (args.mode is None)
    ):
        parser.error(
            "Toggle을 사용하려면 "
            "대상(--train/--script)과 "
            "환경(--kaggle/--local)을 "
            "함께 지정해야 합니다."
        )

    if (
        args.ensemble
        and args.target is not None
    ):
        parser.error(
            "--ensemble은 경로 Toggle 옵션과 "
            "함께 사용할 수 없습니다."
        )

    return args


# =========================================================
# Local / Kaggle 경로 Toggle
# =========================================================

def toggle_paths(
    path,
    mode,
):
    if not path.exists():
        raise FileNotFoundError(
            "Toggle 대상 파일을 찾을 수 없습니다: "
            f"{path}"
        )

    content = path.read_text(
        encoding="utf-8"
    )

    changed = []
    already = []

    for (
        local_path,
        kaggle_path,
    ) in PATH_TOGGLE_MAP.items():

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
                (
                    source,
                    target,
                )
            )
        elif target in content:
            already.append(
                target
            )

    if not changed and not already:
        raise ValueError(
            f"{path}에서 Toggle 가능한 "
            "경로를 찾을 수 없습니다."
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
        print(
            f"[변경] {source}"
        )
        print(
            f"    -> {target}"
        )

    for target in already:
        print(
            f"[유지] {target}"
        )

    print("=" * 60)


# =========================================================
# Python 상수 정적 읽기
# =========================================================

def read_string_constant(
    path: Path,
    constant_name: str,
) -> str | None:
    """
    Python 파일을 실행하지 않고
    NAME = "literal" 형태의 문자열 상수를 읽는다.

    script.py의 EXPERIMENT_NAME 검증에 사용한다.
    """

    if not path.is_file():
        return None

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if (
            isinstance(
                target,
                ast.Name,
            )
            and target.id
            == constant_name
            and isinstance(
                node.value,
                ast.Constant,
            )
            and isinstance(
                node.value.value,
                str,
            )
        ):
            return node.value.value

    return None


# =========================================================
# Single model resolver
# =========================================================

def load_train_config():
    """
    기존 workflow와 동일하게 train.py를 import하여
    EXPERIMENT_NAME / MODEL_PATH를 가져온다.
    """

    train = importlib.import_module(
        "train"
    )

    experiment_name = str(
        train.EXPERIMENT_NAME
    )

    model_path = Path(
        train.MODEL_PATH
    )

    return (
        experiment_name,
        model_path,
    )


def resolve_single_package():
    (
        experiment_name,
        model_path,
    ) = load_train_config()

    model_name = (
        experiment_name
    )

    return PackageSpec(
        experiment_name=(
            experiment_name
        ),
        package_type="single",
        model_paths={
            model_name:
                model_path
        },
        script_path=SCRIPT_PATH,
        requirements_path=(
            REQUIREMENTS_PATH
        ),
        submit_dir=(
            SUBMIT_ROOT
            / experiment_name
        ),
        metadata_path=None,
    )


# =========================================================
# Ensemble resolver
# =========================================================

def find_latest_ensemble():
    """
    ensemble root와 validation root 양쪽에서
    ensemble_vNNN을 탐색하고 가장 큰 version을 반환한다.

    ensemble/<name>이 존재하면 이를 우선 사용한다.
    """

    candidates: dict[
        int,
        Path,
    ] = {}

    if ENSEMBLE_ROOT.is_dir():
        for path in (
            ENSEMBLE_ROOT.iterdir()
        ):
            if not path.is_dir():
                continue

            match = (
                ENSEMBLE_PATTERN.fullmatch(
                    path.name
                )
            )

            if match is None:
                continue

            version = int(
                match.group(1)
            )

            candidates[
                version
            ] = path

    if VALIDATION_ROOT.is_dir():
        for path in (
            VALIDATION_ROOT.iterdir()
        ):
            if not path.is_dir():
                continue

            match = (
                ENSEMBLE_PATTERN.fullmatch(
                    path.name
                )
            )

            if match is None:
                continue

            version = int(
                match.group(1)
            )

            # 동일 version의 canonical ensemble dir가
            # 이미 있으면 validation path로 덮어쓰지 않는다.
            candidates.setdefault(
                version,
                path,
            )

    if not candidates:
        return None

    latest_version = max(
        candidates
    )

    return candidates[
        latest_version
    ]


def confirm_ensemble():
    while True:
        answer = input(
            "이 모델로 진행할까요? [y/n]: "
        ).strip().lower()

        if answer == "y":
            return True

        if answer == "n":
            return False

        print(
            "Please enter 'y' or 'n'."
        )


def resolve_ensemble_metadata_path(
    experiment_name: str,
    discovered_dir: Path,
) -> Path:
    """
    우선순위:
    1) ensemble/<name>/metadata.json
    2) 발견된 directory/metadata.json
    3) validation/<name>/metadata.json
    """

    candidates = [
        (
            ENSEMBLE_ROOT
            / experiment_name
            / "metadata.json"
        ),
        (
            discovered_dir
            / "metadata.json"
        ),
        (
            VALIDATION_ROOT
            / experiment_name
            / "metadata.json"
        ),
    ]

    seen = set()

    for path in candidates:
        key = str(
            path.resolve()
            if path.exists()
            else path
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        if path.is_file():
            return path

    raise FileNotFoundError(
        "Ensemble metadata.json을 찾을 수 없습니다.\n"
        + "\n".join(
            f"- {path}"
            for path in candidates
        )
    )


def read_ensemble_model_names(
    metadata_path: Path,
) -> list[str]:
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    model_names = metadata.get(
        "models"
    )

    if (
        not isinstance(
            model_names,
            list,
        )
        or not model_names
    ):
        # 일부 metadata는 global_best.weights만
        # source of truth로 가질 수 있으므로 fallback.
        global_best = metadata.get(
            "global_best",
            {}
        )

        weights = (
            global_best.get(
                "weights"
            )
            if isinstance(
                global_best,
                dict,
            )
            else None
        )

        if (
            isinstance(
                weights,
                dict,
            )
            and weights
        ):
            model_names = list(
                weights.keys()
            )
        else:
            raise ValueError(
                "Ensemble metadata에 "
                "models 또는 global_best.weights가 없습니다: "
                f"{metadata_path}"
            )

    cleaned = []

    for model_name in model_names:
        if not isinstance(
            model_name,
            str,
        ):
            raise ValueError(
                "Ensemble model name은 문자열이어야 합니다."
            )

        if (
            Path(
                model_name
            ).name
            != model_name
        ):
            raise ValueError(
                "Invalid ensemble dependency model name: "
                f"{model_name!r}"
            )

        if model_name in cleaned:
            raise ValueError(
                "Duplicate ensemble dependency model: "
                f"{model_name}"
            )

        cleaned.append(
            model_name
        )

    return cleaned


def resolve_ensemble_package(
    ensemble_dir: Path,
):
    experiment_name = (
        ensemble_dir.name
    )

    metadata_path = (
        resolve_ensemble_metadata_path(
            experiment_name,
            ensemble_dir,
        )
    )

    model_names = (
        read_ensemble_model_names(
            metadata_path
        )
    )

    model_paths = {
        model_name: (
            MODEL_ROOT
            / model_name
            / f"{model_name}.pkl"
        )
        for model_name in model_names
    }

    return PackageSpec(
        experiment_name=(
            experiment_name
        ),
        package_type=(
            "ensemble"
        ),
        model_paths=(
            model_paths
        ),
        script_path=(
            SCRIPT_PATH
        ),
        requirements_path=(
            REQUIREMENTS_PATH
        ),
        submit_dir=(
            SUBMIT_ROOT
            / experiment_name
        ),
        metadata_path=(
            metadata_path
        ),
    )


# =========================================================
# requirements.txt 확인
# =========================================================

def inspect_requirements(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            "requirements.txt를 찾을 수 없습니다: "
            f"{path}"
        )

    packages = []

    with path.open(
        encoding="utf-8"
    ) as file:
        for line in file:
            line = (
                line.strip()
            )

            if (
                not line
                or line.startswith(
                    "#"
                )
            ):
                continue

            packages.append(
                line
            )

    print()
    print("=" * 60)
    print("requirements.txt")
    print("=" * 60)

    for package in packages:
        print(
            f"- {package}"
        )

    lower = [
        package.lower()
        for package in packages
    ]

    duplicates = sorted({
        package
        for package in lower
        if lower.count(
            package
        ) > 1
    })

    if duplicates:
        print()
        print(
            "[경고] 중복 package 발견"
        )

        for package in duplicates:
            print(
                f"- {package}"
            )

    print("=" * 60)


# =========================================================
# 공통 검증
# =========================================================

def validate_package_spec(
    spec: PackageSpec,
):
    if not spec.experiment_name:
        raise ValueError(
            "experiment_name이 비어 있습니다."
        )

    if (
        spec.package_type
        not in {
            "single",
            "ensemble",
        }
    ):
        raise ValueError(
            "지원하지 않는 package_type: "
            f"{spec.package_type}"
        )

    if not spec.model_paths:
        raise ValueError(
            "Package에 포함할 모델이 없습니다."
        )

    if not spec.script_path.is_file():
        raise FileNotFoundError(
            "script.py를 찾을 수 없습니다: "
            f"{spec.script_path}"
        )

    if not (
        spec.requirements_path
        .is_file()
    ):
        raise FileNotFoundError(
            "requirements.txt를 찾을 수 없습니다: "
            f"{spec.requirements_path}"
        )

    for (
        model_name,
        model_path,
    ) in spec.model_paths.items():

        if (
            Path(
                model_name
            ).name
            != model_name
        ):
            raise ValueError(
                "Invalid model name: "
                f"{model_name!r}"
            )

        expected_dir = (
            MODEL_ROOT
            / model_name
        )

        expected_path = (
            expected_dir
            / f"{model_name}.pkl"
        )

        if not expected_dir.is_dir():
            raise FileNotFoundError(
                "Required model directory not found:\n"
                f"{expected_dir}"
            )

        if not model_path.is_file():
            raise FileNotFoundError(
                "Required model not found:\n"
                f"{model_path}"
            )

        # 표준 경로를 강제한다.
        try:
            same_path = (
                model_path.resolve()
                == expected_path.resolve()
            )
        except OSError:
            same_path = (
                model_path
                == expected_path
            )

        if not same_path:
            raise ValueError(
                "MODEL_PATH가 프로젝트 표준과 다릅니다.\n"
                f"model_name: {model_name}\n"
                f"actual    : {model_path}\n"
                f"expected  : {expected_path}"
            )

    # root script.py가 현재 package 대상인지 정적 검증.
    script_experiment = (
        read_string_constant(
            spec.script_path,
            "EXPERIMENT_NAME",
        )
    )

    if script_experiment is None:
        raise ValueError(
            "script.py에서 literal "
            "EXPERIMENT_NAME을 찾지 못했습니다."
        )

    if (
        script_experiment
        != spec.experiment_name
    ):
        raise ValueError(
            "script.py의 EXPERIMENT_NAME이 "
            "package 대상과 다릅니다.\n"
            f"package: {spec.experiment_name}\n"
            f"script : {script_experiment}"
        )

    if spec.submit_dir.exists():
        raise FileExistsError(
            "Submit directory already exists: "
            f"{spec.submit_dir}\n"
            "기존 제출 폴더를 정리하거나 "
            "새 EXPERIMENT_NAME을 사용하세요."
        )


# =========================================================
# Preview
# =========================================================

def print_package_preview(
    spec: PackageSpec,
):
    print()
    print("=" * 60)
    print("Package Preview")
    print("=" * 60)

    print(
        f"Experiment : "
        f"{spec.experiment_name}"
    )

    print(
        f"Type       : "
        f"{spec.package_type}"
    )

    print("Models")

    for (
        model_name,
        model_path,
    ) in spec.model_paths.items():
        print(
            f"- {model_name}"
        )
        print(
            f"  {model_path}"
        )

    if spec.metadata_path is not None:
        print(
            "Metadata"
        )
        print(
            f"  {spec.metadata_path}"
        )

    print("Script")
    print(
        f"  {spec.script_path}"
    )

    print("Requirements")
    print(
        f"  {spec.requirements_path}"
    )

    print("Output")
    print(
        f"  {spec.submit_dir}/"
    )
    print(
        f"  {spec.archive_path}"
    )

    print("=" * 60)


# =========================================================
# ZIP 검증
# =========================================================

def validate_staged_package(
    package_dir: Path,
    spec: PackageSpec,
):
    expected_files = {
        "script.py",
        "requirements.txt",
    }

    actual_root_files = {
        path.name
        for path in (
            package_dir.iterdir()
        )
        if path.is_file()
    }

    if (
        actual_root_files
        != expected_files
    ):
        raise RuntimeError(
            "Staged package root file 구성이 "
            "예상과 다릅니다.\n"
            f"expected={sorted(expected_files)}\n"
            f"actual={sorted(actual_root_files)}"
        )

    model_root = (
        package_dir
        / "model"
    )

    if not model_root.is_dir():
        raise RuntimeError(
            "Staged package에 model/이 없습니다."
        )

    actual_model_names = sorted(
        [
            path.name
            for path in (
                model_root.iterdir()
            )
            if path.is_dir()
        ]
    )

    expected_model_names = sorted(
        spec.model_paths.keys()
    )

    if (
        actual_model_names
        != expected_model_names
    ):
        raise RuntimeError(
            "Staged model dependency가 "
            "예상과 다릅니다.\n"
            f"expected={expected_model_names}\n"
            f"actual={actual_model_names}"
        )

    for model_name in (
        expected_model_names
    ):
        expected_model_file = (
            model_root
            / model_name
            / f"{model_name}.pkl"
        )

        if not (
            expected_model_file
            .is_file()
        ):
            raise RuntimeError(
                "Staged model file 누락:\n"
                f"{expected_model_file}"
            )


# =========================================================
# 공통 Package Builder
# =========================================================

def build_package(
    spec: PackageSpec,
):
    """
    Single / Ensemble 공통 builder.

    임시 위치에서:
        model/<name>/<name>.pkl
        script.py
        requirements.txt
        zip
    을 모두 완성한 뒤 submits/<experiment>/로 이동한다.
    """

    submit_root_existed = (
        SUBMIT_ROOT.exists()
    )

    directory_created = False

    try:
        with tempfile.TemporaryDirectory(
            prefix=(
                f".{spec.experiment_name}"
                "_package_"
            ),
            dir=SUBMIT_ROOT.parent,
        ) as temp_dir:

            temp_root = Path(
                temp_dir
            )

            temp_package_dir = (
                temp_root
                / spec.experiment_name
            )

            temp_model_root = (
                temp_package_dir
                / "model"
            )

            temp_model_root.mkdir(
                parents=True,
                exist_ok=False,
            )

            # -----------------------------------------
            # models
            # -----------------------------------------

            for (
                model_name,
                model_path,
            ) in spec.model_paths.items():

                destination_dir = (
                    temp_model_root
                    / model_name
                )

                destination_dir.mkdir(
                    exist_ok=False
                )

                shutil.copy2(
                    model_path,
                    destination_dir
                    / f"{model_name}.pkl",
                )

            # -----------------------------------------
            # script / requirements
            # -----------------------------------------

            shutil.copy2(
                spec.script_path,
                temp_package_dir
                / "script.py",
            )

            shutil.copy2(
                spec.requirements_path,
                temp_package_dir
                / "requirements.txt",
            )

            validate_staged_package(
                temp_package_dir,
                spec,
            )

            # -----------------------------------------
            # ZIP 생성
            #
            # archive는 temp package 밖에 생성하여
            # 자기 자신이 ZIP 안에 들어가는 것을 막는다.
            # -----------------------------------------

            temp_archive_path = Path(
                shutil.make_archive(
                    str(
                        temp_root
                        / spec.experiment_name
                    ),
                    "zip",
                    root_dir=(
                        temp_package_dir
                    ),
                )
            )

            if not (
                temp_archive_path
                .is_file()
            ):
                raise RuntimeError(
                    "ZIP 생성에 실패했습니다: "
                    f"{temp_archive_path}"
                )

            # -----------------------------------------
            # 최종 submit 이동
            # -----------------------------------------

            SUBMIT_ROOT.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(
                    temp_package_dir
                ),
                str(
                    spec.submit_dir
                ),
            )

            directory_created = (
                True
            )

            final_archive_path = (
                spec.archive_path
            )

            shutil.move(
                str(
                    temp_archive_path
                ),
                str(
                    final_archive_path
                ),
            )

    except Exception:
        if (
            directory_created
            and spec.submit_dir.exists()
        ):
            shutil.rmtree(
                spec.submit_dir
            )

        if (
            not submit_root_existed
            and SUBMIT_ROOT.exists()
            and not any(
                SUBMIT_ROOT.iterdir()
            )
        ):
            SUBMIT_ROOT.rmdir()

        raise

    print()
    print("=" * 60)
    print("Package 생성 완료")
    print("=" * 60)

    print(
        f"Directory : "
        f"{spec.submit_dir}"
    )

    print(
        f"Archive   : "
        f"{spec.archive_path}"
    )

    print("=" * 60)


# =========================================================
# Packaging entry points
# =========================================================

def package_single():
    spec = (
        resolve_single_package()
    )

    validate_package_spec(
        spec
    )

    inspect_requirements(
        spec.requirements_path
    )

    print_package_preview(
        spec
    )

    build_package(
        spec
    )


def package_latest_ensemble():
    print()
    print("=" * 60)
    print("Ensemble Packaging")
    print("=" * 60)

    ensemble_dir = (
        find_latest_ensemble()
    )

    if ensemble_dir is None:
        raise FileNotFoundError(
            "No ensemble experiment found."
        )

    experiment_name = (
        ensemble_dir.name
    )

    print(
        "Latest ensemble model: "
        f"{experiment_name}"
    )

    if not confirm_ensemble():
        print(
            "Packaging cancelled."
        )
        return

    spec = (
        resolve_ensemble_package(
            ensemble_dir
        )
    )

    validate_package_spec(
        spec
    )

    inspect_requirements(
        spec.requirements_path
    )

    print_package_preview(
        spec
    )

    build_package(
        spec
    )


def print_packaging_error(
    error,
):
    print("[ERROR]")
    print(error)
    print(
        "Packaging aborted."
    )


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

    try:
        if args.ensemble:
            package_latest_ensemble()
        else:
            package_single()

    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        shutil.Error,
        SyntaxError,
        ValueError,
    ) as error:
        print_packaging_error(
            error
        )


if __name__ == "__main__":
    main()
