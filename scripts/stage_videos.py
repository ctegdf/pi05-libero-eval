#!/usr/bin/env python3
"""Copy the publishable video inventory into an isolated local directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from release_layout import iter_video_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    destination_root = args.destination_root.expanduser().resolve()

    rows = list(iter_video_files(source_root))
    copied = skipped = 0
    total_bytes = 0
    for source, path, _ in rows:
        destination = destination_root / source.source_dir / path.name
        total_bytes += path.stat().st_size
        if destination.is_file() and destination.stat().st_size == path.stat().st_size:
            skipped += 1
            continue
        if args.dry_run:
            copied += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1

    print(
        f"videos={len(rows)} copied={copied} skipped={skipped} "
        f"size_gib={total_bytes / (1024**3):.3f} destination={destination_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
