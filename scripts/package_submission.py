"""Cross-platform packager for submission.zip.

Creates `submission.zip` with POSIX paths so Linux evaluators can
resolve entries like `src/agent.py` reliably.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_FILES = (
    "src/agent.py",
    "src/agent_base.py",
    "src/requirements.txt",
)


def build_zip(submission_dir: Path, output_zip: Path) -> None:
    if not submission_dir.exists():
        raise FileNotFoundError(f"submission dir not found: {submission_dir}")

    for required in REQUIRED_FILES:
        required_path = submission_dir / Path(required)
        if not required_path.exists():
            raise FileNotFoundError(f"required file missing: {required_path}")

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            arcname = path.relative_to(submission_dir).as_posix()
            archive.write(path, arcname)


def verify_zip(output_zip: Path) -> list[str]:
    with zipfile.ZipFile(output_zip, "r") as archive:
        names = sorted(archive.namelist())

    for required in REQUIRED_FILES:
        if required not in names:
            raise FileNotFoundError(f"required entry missing in zip: {required}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Package competition submission.")
    parser.add_argument(
        "--submission-dir",
        default="submission",
        help="Path to submission directory (default: submission)",
    )
    parser.add_argument(
        "--output",
        default="submission.zip",
        help="Output zip file path (default: submission.zip)",
    )
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    output_zip = Path(args.output).resolve()

    build_zip(submission_dir, output_zip)
    entries = verify_zip(output_zip)

    print(f"[OK] zip generated: {output_zip}")
    print("[OK] zip entries:")
    for name in entries:
        print(name)


if __name__ == "__main__":
    main()
