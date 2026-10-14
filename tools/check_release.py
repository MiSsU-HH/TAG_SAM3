"""Check the baseline-only source tree and optionally make a public ZIP archive."""

import argparse
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    ".gitignore", "LICENSE", "README.md", "requirements.txt", "evaluate.py",
    "assets/fig_intro.pdf", "assets/fig_intro.svg", "configs/ade20k_instance_mapping.json",
    "configs/sam3_ade20k.json", "configs/sam3_cityscapes.json", "configs/sam3_isaid.json",
    "tag/__init__.py", "tag/sam3_baseline.py", "tag/datasets.py", "tag/metrics.py",
    "tag/readouts.py", "tag/runtime.py", "tools/prepare_annotations.py",
    "tools/prepare_isaid.py", "tools/check_release.py",
    "tests/test_metrics.py", "tests/test_baseline.py", "tests/test_evaluate.py",
}
RUNTIME_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "data",
    "annotations", "outputs", "weights", "checkpoints", "build", "dist",
}
PRIVATE_IDENTIFIERS = (
    "TAG" + "Model", "Text" + "Adapter", "Adapter" + "Attention",
    "adapter_" + "condition", "presence_token_" + "override",
    "presence_" + "only", "forward_" + "presence", "torch." + "optim",
    "back" + "ward(",
)


def public_files():
    errors = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if set(relative.parts) & RUNTIME_DIRS or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            errors.append(f"Symlink is not a release source: {relative}")
        elif path.is_file() and relative.as_posix() not in FILES:
            errors.append(f"Unexpected file in the baseline release: {relative}")
    for name in sorted(FILES):
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            errors.append(f"Missing or invalid public source: {name}")
            continue
        if path.suffix == ".py" and not name.startswith("tests/"):
            source = path.read_text()
            for identifier in PRIVATE_IDENTIFIERS:
                if identifier in source:
                    errors.append(f"Withheld implementation identifier {identifier!r}: {name}")
            if re.search(r"(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})", source):
                errors.append(f"Credential-like token in {name}")
    if errors:
        raise ValueError("\n".join(errors))
    return [ROOT / name for name in sorted(FILES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Write an allowlisted public ZIP outside this repository.")
    args = parser.parse_args()
    try:
        files = public_files()
    except ValueError as error:
        parser.exit(1, f"Release check failed:\n{error}\n")
    print(f"Release check passed: {len(files)} public files; runtime data and weights excluded.")
    if args.archive is not None:
        archive = args.archive.resolve()
        if archive == ROOT or ROOT in archive.parents:
            parser.error("Write the archive outside the public source directory.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in files:
                output.write(path, Path("TAG") / path.relative_to(ROOT))
        print(f"Public archive: {archive}")


if __name__ == "__main__":
    main()
