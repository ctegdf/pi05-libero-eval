#!/usr/bin/env python3
"""Upload the source archive's videos to a Hugging Face dataset.

Files are passed directly to ``huggingface_hub`` from the source tree.  No
second local video copy is made.  Uploads are grouped into resumable commits;
re-running the command safely skips files already present with the same size
and SHA-256 when ``--resume`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from release_layout import iter_video_files


EXPECTED_VIDEO_COUNT = 56639


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(source_root: Path, manifest_path: Path | None, skip_sha256: bool) -> list[dict[str, Any]]:
    rows = []
    for _, path, public_path in iter_video_files(source_root):
        row: dict[str, Any] = {"path": public_path, "size": path.stat().st_size}
        if not skip_sha256:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    if len(rows) != EXPECTED_VIDEO_COUNT:
        raise RuntimeError(f"expected {EXPECTED_VIDEO_COUNT} MP4 files, found {len(rows)}")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def load_hf():
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise SystemExit("Install the uploader dependency first: python3 -m pip install huggingface_hub") from exc
    return HfApi, CommitOperationAdd


def remote_video_inventory(api: Any, repo_id: str) -> dict[str, dict[str, Any]]:
    """Return remote MP4 metadata needed for content-aware resume."""

    inventory: dict[str, dict[str, Any]] = {}
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        recursive=True,
        expand=True,
    ):
        path = getattr(item, "path", None)
        if not isinstance(path, str) or not path.startswith("videos/") or not path.endswith(".mp4"):
            continue
        lfs = getattr(item, "lfs", None)
        remote_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        inventory[path] = {
            "size": getattr(item, "size", None),
            "sha256": remote_sha256,
        }
    return inventory


def rows_to_upload(
    rows: list[dict[str, Any]],
    remote: dict[str, dict[str, Any]],
    *,
    allow_size_only: bool,
) -> list[dict[str, Any]]:
    """Keep missing, changed, or unverifiable files for the next commit."""

    pending = []
    for row in rows:
        remote_row = remote.get(row["path"])
        if remote_row is None:
            pending.append(row)
            continue
        if remote_row.get("size") != row["size"]:
            pending.append(row)
            continue
        local_sha256 = row.get("sha256")
        remote_sha256 = remote_row.get("sha256")
        if local_sha256 and remote_sha256:
            if local_sha256 != remote_sha256:
                pending.append(row)
            continue
        if allow_size_only:
            continue
        pending.append(row)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="owner/dataset-name")
    parser.add_argument("--manifest", type=Path, default=Path("hf-video-manifest.jsonl"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--commit-message", default="Upload pi05 LIBERO rollout videos")
    parser.add_argument("--skip-sha256", action="store_true", help="faster inventory; not recommended for the final release")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip remote videos only after matching size and SHA-256 (or size with --skip-sha256)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source_root = args.source_root.expanduser().resolve()
    rows = inventory(source_root, args.manifest.expanduser().resolve(), args.skip_sha256)
    print(f"validated {len(rows)} MP4 files; manifest={args.manifest}")
    if args.dry_run:
        print(f"dry-run: would upload {sum(row['size'] for row in rows) / (1024**3):.2f} GiB to dataset {args.repo_id}")
        return 0

    HfApi, CommitOperationAdd = load_hf()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    remote = remote_video_inventory(api, args.repo_id) if args.resume else {}
    source_by_public = {public_path: path for _, path, public_path in iter_video_files(source_root)}
    rows = rows_to_upload(rows, remote, allow_size_only=args.skip_sha256)
    if not rows:
        print("all videos are already present; uploading manifest/card only")
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        operations = []
        for row in batch:
            source = source_by_public[row["path"]]
            operations.append(CommitOperationAdd(path_in_repo=row["path"], path_or_fileobj=str(source)))
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"{args.commit_message} ({start + 1}-{start + len(batch)} of {len(rows)})",
        )
        print(f"uploaded {start + len(batch)}/{len(rows)}", flush=True)
    api.upload_file(
        path_or_fileobj=str(args.manifest.expanduser().resolve()),
        path_in_repo="manifests/video-manifest.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add video inventory manifest",
    )
    card = Path(__file__).resolve().parents[1] / "docs" / "hf-dataset-card.md"
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
