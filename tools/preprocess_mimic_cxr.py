"""Keep only MIMIC-CXR image files referenced by a JSONL QA file; remove the rest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_IMAGE_DIR = "/home/dida/workspace/phd_work/data/MIMIC-CXR/files"
DEFAULT_QA_PATH = (
    "/home/dida/workspace/phd_work/repos/CARES/data/MIMIC-CXR/mimic_factuality.jsonl"
)


def load_referenced_images(qa_path: str | Path) -> set[str]:
    """Return relative image paths from the JSONL `image` field."""
    referenced: set[str] = set()
    with open(qa_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {qa_path}") from exc
            image = record.get("image")
            if not image:
                raise ValueError(f"Missing `image` field on line {line_no} of {qa_path}")
            referenced.add(os.path.normpath(image))
    return referenced


def collect_files(image_dir: Path) -> list[Path]:
    """Collect all files under image_dir."""
    return [p for p in image_dir.rglob("*") if p.is_file()]


def dirs_that_become_empty(image_dir: Path, to_keep: list[Path]) -> list[Path]:
    """Directories under image_dir that contain no kept files (would become empty)."""
    image_dir = image_dir.resolve()
    kept_ancestor_dirs: set[Path] = set()
    for path in to_keep:
        cur = path.resolve().parent
        while True:
            kept_ancestor_dirs.add(cur)
            if cur == image_dir or cur.parent == cur:
                break
            cur = cur.parent

    candidates = [
        p
        for p in image_dir.rglob("*")
        if p.is_dir() and p.resolve() not in kept_ancestor_dirs
    ]
    return sorted(candidates, key=lambda p: len(p.parts), reverse=True)


def remove_empty_dirs(dirs: list[Path], dry_run: bool, sample_n: int = 10) -> int:
    """Remove (or report) empty directories. `dirs` should be deepest-first."""
    removed = 0
    for d in dirs:
        if dry_run:
            if removed < sample_n:
                print(f"[dry-run] would remove empty dir: {d}")
        else:
            # Only remove if actually empty after file deletions.
            try:
                if any(d.iterdir()):
                    continue
            except FileNotFoundError:
                continue
            d.rmdir()
            if removed < sample_n or (removed + 1) % 1000 == 0:
                print(f"removed empty dir: {d}")
        removed += 1
    if dry_run and removed > sample_n:
        print(f"[dry-run] ... and {removed - sample_n} more empty dirs")
    return removed


def filter_files(
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    qa_path: str | Path = DEFAULT_QA_PATH,
    dry_run: bool = True,
    clean_empty_dirs: bool = True,
) -> None:
    image_dir = Path(image_dir).resolve()
    qa_path = Path(qa_path)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not qa_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {qa_path}")

    referenced = load_referenced_images(qa_path)
    referenced_abs = {str((image_dir / rel).resolve()) for rel in referenced}

    print(f"Referenced images in JSONL: {len(referenced)}")
    missing = [rel for rel in sorted(referenced) if not (image_dir / rel).is_file()]
    if missing:
        print(f"Warning: {len(missing)} referenced images are missing on disk.")
        for rel in missing[:10]:
            print(f"  missing: {rel}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    all_files = collect_files(image_dir)
    to_keep: list[Path] = []
    to_remove: list[Path] = []
    for path in all_files:
        if str(path.resolve()) in referenced_abs:
            to_keep.append(path)
        else:
            to_remove.append(path)

    print(f"Files on disk: {len(all_files)}")
    print(f"Files to keep: {len(to_keep)}")
    print(f"Files to remove: {len(to_remove)}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")

    sample_n = 10
    if dry_run:
        for path in to_remove[:sample_n]:
            print(f"[dry-run] would remove: {path}")
        if len(to_remove) > sample_n:
            print(f"[dry-run] ... and {len(to_remove) - sample_n} more files")
    else:
        for i, path in enumerate(to_remove, start=1):
            path.unlink()
            if i <= sample_n or i % 10000 == 0 or i == len(to_remove):
                print(f"removed ({i}/{len(to_remove)}): {path}")

    empty_dirs = 0
    if clean_empty_dirs:
        dirs = dirs_that_become_empty(image_dir, to_keep)
        empty_dirs = remove_empty_dirs(dirs, dry_run=dry_run, sample_n=sample_n)

    print("---")
    print(
        f"{'Would remove' if dry_run else 'Removed'} "
        f"{len(to_remove)} files and {empty_dirs} empty directories."
    )
    if dry_run:
        print("Re-run with --execute to apply deletions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep MIMIC-CXR files referenced by a JSONL `image` field and "
            "remove unused files. Dry-run by default."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Root directory containing MIMIC-CXR image files.",
    )
    parser.add_argument(
        "--qa-path",
        type=str,
        default=DEFAULT_QA_PATH,
        help="JSONL file with an `image` field per record.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete unused files. Without this flag, only a dry run is performed.",
    )
    parser.add_argument(
        "--keep-empty-dirs",
        action="store_true",
        help="Do not remove empty directories after deleting files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_files(
        image_dir=args.image_dir,
        qa_path=args.qa_path,
        dry_run=not args.execute,
        clean_empty_dirs=not args.keep_empty_dirs,
    )
