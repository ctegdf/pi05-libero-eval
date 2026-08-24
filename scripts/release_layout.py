"""Shared source-to-public layout for the pi05 LIBERO release.

The source tree is intentionally not copied or modified.  Both the metadata
preparer and the Hugging Face uploader import this module so that every video
path has one canonical public destination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class VideoSource:
    source_dir: str
    public_prefix: str


VIDEO_SOURCES = (
    VideoSource("openpi-libero/official-full/videos", "videos/official/full"),
    VideoSource("openpi-libero/official-smoke/videos", "videos/official/smoke"),
    VideoSource("openpi-libero/base-libero-assets-full/videos", "videos/base-libero-assets/full"),
    VideoSource("openpi-libero/base-libero-assets-smoke/videos", "videos/base-libero-assets/smoke"),
    # The merged run is the canonical source for plus-full-merged/episodes.jsonl.
    # The three split directories remain in the immutable source archive as
    # intermediate run artifacts, but are not uploaded a second time.
    VideoSource("plus-pro/plus-full-merged/videos", "videos/plus/full"),
    VideoSource("plus-pro/plus-smoke/videos", "videos/plus/smoke"),
    VideoSource("plus-pro/pro-full/videos", "videos/pro/full"),
    VideoSource("plus-pro/pro-smoke/videos", "videos/pro/smoke"),
    VideoSource("libero-x/full/level1/videos", "videos/libero-x/full/level1"),
    VideoSource("libero-x/full/level2/videos", "videos/libero-x/full/level2"),
    VideoSource("libero-x/full/level3/videos", "videos/libero-x/full/level3"),
    VideoSource("libero-x/full/level4/videos", "videos/libero-x/full/level4"),
    VideoSource("libero-x/full/level5/videos", "videos/libero-x/full/level5"),
    VideoSource("libero-x/smoke/videos", "videos/libero-x/smoke"),
)

METADATA_SOURCES = (
    "openpi-libero/official-full/episodes.jsonl",
    "openpi-libero/official-smoke/episodes.jsonl",
    "openpi-libero/base-libero-assets-full/episodes.jsonl",
    "openpi-libero/base-libero-assets-smoke/episodes.jsonl",
    "plus-pro/plus-full-merged/episodes.jsonl",
    "plus-pro/plus-smoke/episodes.jsonl",
    "plus-pro/pro-full/episodes.jsonl",
    "plus-pro/pro-smoke/episodes.jsonl",
    "libero-x/full/level1/episodes.jsonl",
    "libero-x/full/level2/episodes.jsonl",
    "libero-x/full/level3/episodes.jsonl",
    "libero-x/full/level4/episodes.jsonl",
    "libero-x/full/level5/episodes.jsonl",
    "libero-x/smoke/episodes.jsonl",
)


def iter_video_files(source_root: Path) -> Iterable[tuple[VideoSource, Path, str]]:
    """Yield regular MP4s and their public paths, in deterministic order."""

    seen: set[str] = set()
    for source in VIDEO_SOURCES:
        directory = source_root / source.source_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"missing video source directory: {directory}")
        for path in sorted(directory.glob("*.mp4")):
            if not path.is_file():
                continue
            public_path = f"{source.public_prefix}/{path.name}"
            if public_path in seen:
                raise ValueError(f"duplicate public video path: {public_path}")
            seen.add(public_path)
            yield source, path, public_path


def public_path_for_record(
    source_root: Path,
    record: dict,
    video_index: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Map an episode's original video field to the HF-relative path."""

    # Prefer the final rendered path. For LIBERO-Plus, ``video_original``
    # points at an intermediate split run while ``video`` points at the
    # merged file selected for publication.
    candidates = [record.get("video"), record.get("video_original")]
    raw = next((value for value in candidates if isinstance(value, str) and value), None)
    if raw is None:
        return None
    normalized = raw.replace("\\", "/")
    if video_index is not None:
        basename = Path(normalized).name
        public_path = video_index.get(basename)
        if public_path is None:
            raise FileNotFoundError(f"record points to an unknown video filename: {basename}")
        return public_path

    matches = []
    for source in VIDEO_SOURCES:
        marker = f"/{source.source_dir}/"
        if marker in normalized:
            matches.append((len(marker), source))
        elif normalized.startswith(source.source_dir + "/"):
            matches.append((len(source.source_dir) + 1, source))
    if not matches:
        raise ValueError(f"video path does not match a known source: {raw}")
    _, source = max(matches, key=lambda item: item[0])
    basename = Path(normalized).name
    candidate = source_root / source.source_dir / basename
    if not candidate.is_file():
        raise FileNotFoundError(f"record points to missing video: {candidate}")
    return f"{source.public_prefix}/{basename}"


def relative_provenance_path(value: object) -> object:
    """Remove private checkout prefixes while retaining useful provenance."""

    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    for marker in ("benchmarks/LIBERO-plus/", "datasets/LIBERO-Pro/", "vendor/LIBERO-X/"):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return marker + normalized[marker_index + len(marker) :]
    if normalized.startswith("/home/"):
        return "<private-path>"
    return normalized
