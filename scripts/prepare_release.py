#!/usr/bin/env python3
"""Create the public GitHub artifact from the immutable evaluation archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from release_layout import METADATA_SOURCES, iter_video_files, public_path_for_record, relative_provenance_path


PRIVATE_TEXT = re.compile(r"(?:/home/[^\s\"']+|\b(?:hhj@)?10\.0\.106\.18\b)")
DROP_FIELDS = {"trace", "video_trace", "video_original"}
SUMMARY_SOURCES = (
    "openpi-libero/official-full",
    "openpi-libero/official-smoke",
    "openpi-libero/base-libero-assets-full",
    "openpi-libero/base-libero-assets-smoke",
    "plus-pro/plus-full-merged",
    "plus-pro/plus-smoke",
    "plus-pro/pro-full",
    "plus-pro/pro-smoke",
    "libero-x/smoke",
    "libero-x/full/level1",
    "libero-x/full/level2",
    "libero-x/full/level3",
    "libero-x/full/level4",
    "libero-x/full/level5",
)
TEXT_SUFFIXES = {".md", ".html", ".py", ".sh", ".json", ".csv", ".svg", ".txt"}


def scrub_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = PRIVATE_TEXT.sub("<private>", value)
    return value


def sanitize_record(record: dict[str, Any], source_root: Path, video_index: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if key in DROP_FIELDS:
            continue
        if key in {"video", "bddl_path", "init_path", "prompt_source_path"}:
            continue
        if isinstance(value, dict):
            output[key] = {str(k): scrub_text(v) for k, v in value.items()}
        else:
            output[key] = scrub_text(value)
    try:
        public_video = public_path_for_record(source_root, record, video_index)
    except FileNotFoundError:
        raw_video = record.get("video_original") or record.get("video")
        output["video"] = None
        output["video_available"] = False
        output["video_filename"] = Path(str(raw_video)).name if raw_video else None
    else:
        output["video"] = public_video
        output["video_available"] = public_video is not None
        if public_video is None:
            raw_video = record.get("video_original") or record.get("video")
            output["video_filename"] = Path(str(raw_video)).name if raw_video else None
    for key in ("bddl_path", "init_path", "prompt_source_path"):
        if key in record:
            output[key] = relative_provenance_path(record[key])
    return dict(sorted(output.items()))


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def sanitize_text_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scrubbed = PRIVATE_TEXT.sub("<private>", text)
        if scrubbed != text:
            path.write_text(scrubbed, encoding="utf-8")


def copy_public_sources(source_root: Path, release_root: Path) -> None:
    copy_tree(source_root / "analysis-report", release_root / "analysis-report")
    copy_tree(source_root / "plus-pro" / "harness", release_root / "harness" / "plus-pro")
    copy_tree(source_root / "libero-x" / "harness", release_root / "harness" / "libero-x")
    # The old finalizer shell wrappers contain machine-specific paths and are
    # not part of the reusable public harness.
    for path in (release_root / "harness").rglob("*.sh"):
        path.unlink()
    sanitize_text_tree(release_root / "analysis-report")
    sanitize_text_tree(release_root / "harness")
    for relative, target in (
        ("openpi-libero/EVAL_REPORT.md", "docs/openpi-libero-EVAL_REPORT.md"),
        ("plus-pro/final-audit/report.md", "docs/plus-pro-final-audit.md"),
        ("libero-x/EVAL_REPORT.md", "docs/libero-x-EVAL_REPORT.md"),
    ):
        destination = release_root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(PRIVATE_TEXT.sub("<private>", (source_root / relative).read_text(encoding="utf-8")), encoding="utf-8")


def copy_summaries(source_root: Path, release_root: Path) -> None:
    results_root = release_root / "results"
    for relative in SUMMARY_SOURCES:
        for suffix in ("summary.json", "summary.csv"):
            source = source_root / relative / suffix
            target = results_root / relative / suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(PRIVATE_TEXT.sub("<private>", source.read_text(encoding="utf-8")), encoding="utf-8")


def prepare_metadata(source_root: Path, release_root: Path) -> dict[str, Any]:
    results_root = release_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    video_index = {}
    for _, path, public_path in iter_video_files(source_root):
        if path.name in video_index and video_index[path.name] != public_path:
            raise ValueError(f"duplicate video filename across sources: {path.name}")
        video_index[path.name] = public_path
    counts = Counter()
    hashes = {}
    missing_videos = {}
    for relative in METADATA_SOURCES:
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = results_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        digest = hashlib.sha256()
        with source.open("r", encoding="utf-8") as input_stream, target.open("w", encoding="utf-8") as output_stream:
            for line in input_stream:
                record = json.loads(line)
                sanitized = sanitize_record(record, source_root, video_index)
                encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                output_stream.write(encoded)
                digest.update(encoded.encode("utf-8"))
                count += 1
                if sanitized.get("video_available") is False:
                    missing_videos[relative] = missing_videos.get(relative, 0) + 1
        counts[relative] = count
        hashes[relative] = digest.hexdigest()
    manifest = {
        "schema_version": 1,
        "source": "completed pi05_libero evaluation archive",
        "metadata_files": dict(counts),
        "metadata_sha256": hashes,
        "missing_video_records": missing_videos,
        "video_path_policy": "paths are relative to the Hugging Face dataset root",
        "private_fields_removed": sorted(DROP_FIELDS),
    }
    (results_root / "release-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--skip-static-sources", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    release_root = args.release_root.expanduser().resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_static_sources:
        copy_public_sources(source_root, release_root)
    copy_summaries(source_root, release_root)
    manifest = prepare_metadata(source_root, release_root)
    print(json.dumps({"metadata_files": len(manifest["metadata_files"]), "episodes": sum(manifest["metadata_files"].values())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
