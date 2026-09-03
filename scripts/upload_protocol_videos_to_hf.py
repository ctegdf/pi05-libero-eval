#!/usr/bin/env python3
"""Upload the results/runs/ (MODEL_EVAL_RESULT_PROTOCOL_V1) video tree to a
Hugging Face dataset.

Unlike scripts/upload_videos_to_hf.py (which is hardcoded to pi0.5's
release_layout.VIDEO_SOURCES / EXPECTED_VIDEO_COUNT), this walks whatever is
actually staged under a `videos/` root produced by scripts/export_to_protocol.py
(`videos/<model_id>/<benchmark_id>/<run_id>/episode-NNNNNN.mp4`), so it works
for any current or future model_id without editing this script.

Files are passed directly to `huggingface_hub` from the staged tree. Uploads
are grouped into resumable commits; re-running with --resume safely skips
files already present with the same size and SHA-256.

Runs with more than --shard-size files get an extra `part-NNN/` path segment
so no Hub directory exceeds the file-count guidance (mirrors
upload_videos_to_hf.py's handling of pi0.5's 10,030-file LIBERO-Plus run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_video_files(videos_root: Path):
    """Yield (path, public_path) for every staged MP4, sorted for determinism."""
    for path in sorted(videos_root.rglob("*.mp4")):
        if not path.is_file():
            continue
        public_path = str(path.relative_to(videos_root.parent))  # keeps "videos/" prefix
        yield path, public_path


def sharded_path(public_path: str, run_prefix: str, index_in_run: int, shard_size: int) -> str:
    if shard_size <= 0:
        return public_path
    shard = index_in_run // shard_size
    filename = public_path[len(run_prefix) :]
    return f"{run_prefix}part-{shard:03d}/{filename}"


def inventory(videos_root: Path, shard_size: int, skip_sha256: bool) -> list[dict[str, Any]]:
    run_counts: dict[str, int] = {}
    rows = []
    for path, public_path in iter_video_files(videos_root):
        # public_path looks like videos/<model>/<benchmark>/<run_id>/episode-NNNNNN.mp4
        parts = public_path.split("/")
        run_prefix = "/".join(parts[:4]) + "/"  # videos/<model>/<benchmark>/<run_id>/
        index_in_run = run_counts.get(run_prefix, 0)
        run_counts[run_prefix] = index_in_run + 1
        row: dict[str, Any] = {
            "path": public_path,
            "upload_path": sharded_path(public_path, run_prefix, index_in_run, shard_size) if run_counts[run_prefix] > shard_size else public_path,
            "size": path.stat().st_size,
        }
        if not skip_sha256:
            row["sha256"] = sha256_file(path)
        rows.append((path, row))
    # Second pass: if a run ended up over shard_size, ALL of its rows must be
    # sharded (not just the ones seen after crossing the threshold), so redo
    # upload_path uniformly per run once final counts are known.
    for run_prefix, count in run_counts.items():
        if count <= shard_size:
            continue
        idx = 0
        for path, row in rows:
            if row["path"].startswith(run_prefix):
                row["upload_path"] = sharded_path(row["path"], run_prefix, idx, shard_size)
                idx += 1
    return [row for _, row in rows], {path: row for path, row in rows}


def load_hf():
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise SystemExit("Install the uploader dependency first: python3 -m pip install huggingface_hub") from exc
    return HfApi, CommitOperationAdd


def remote_video_inventory(api: Any, repo_id: str, scopes: list[str], need_sha256: bool) -> dict[str, dict[str, Any]]:
    """List only the given `path_in_repo` prefixes (e.g. "videos/openvla",
    "videos/act"), not the whole dataset tree. A full unscoped listing walks
    every file in the repo on every call -- including unrelated pi0.5 videos
    that have nothing to do with this upload -- which gets slower and more
    fragile against a flaky connection as the dataset grows. A prefix that
    doesn't exist yet on the Hub (first upload) is treated as empty, not an
    error.

    `expand=True` pulls extra LFS metadata (sha256, last_commit, security scan
    results) per file over and above the base listing -- skip it entirely
    when the caller is only going to do a size-only resume comparison anyway
    (`--skip-sha256`), since it roughly doubles the per-page request cost for
    data that will be thrown away."""
    inv: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        try:
            tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=scope, recursive=True, expand=need_sha256)
        except Exception as exc:
            if "not found" in str(exc).lower() or "404" in str(exc):
                continue
            raise
        for item in tree:
            path = getattr(item, "path", None)
            if not isinstance(path, str) or not path.endswith(".mp4"):
                continue
            lfs = getattr(item, "lfs", None)
            inv[path] = {"size": getattr(item, "size", None), "sha256": getattr(lfs, "sha256", None) if lfs is not None else None}
    return inv


def matches_remote(row: dict[str, Any], remote: dict[str, dict[str, Any]], allow_size_only: bool) -> bool:
    remote_row = remote.get(row["upload_path"])
    if remote_row is None:
        return False
    if remote_row.get("size") != row["size"]:
        return False
    local_sha256, remote_sha256 = row.get("sha256"), remote_row.get("sha256")
    if local_sha256 and remote_sha256:
        return local_sha256 == remote_sha256
    return allow_size_only


def write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, required=True, help="the release-root's videos/ directory")
    parser.add_argument("--repo-id", required=True, help="owner/dataset-name")
    parser.add_argument("--manifest", type=Path, default=Path("hf-video-manifest.jsonl"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=5000, help="max files per run before adding part-NNN/ subdirs")
    parser.add_argument("--commit-message", default="Upload OpenVLA/ACT LIBERO rollout videos")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-card", type=Path, default=None, help="markdown file to upload as the dataset README.md")
    args = parser.parse_args()

    videos_root = args.videos_root.expanduser().resolve()
    rows, by_path = inventory(videos_root, args.shard_size, args.skip_sha256)
    print(f"found {len(rows)} MP4 files under {videos_root}")

    if args.dry_run:
        write_manifest(rows, args.manifest)
        total_gib = sum(r["size"] for r in rows) / (1024**3)
        print(f"dry-run: would upload {total_gib:.2f} GiB to dataset {args.repo_id}; manifest={args.manifest}")
        return 0

    HfApi, CommitOperationAdd = load_hf()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    scopes = sorted({row["path"].split("/")[1] for row in rows})
    scopes = [f"videos/{model_id}" for model_id in scopes]
    remote = remote_video_inventory(api, args.repo_id, scopes, need_sha256=not args.skip_sha256) if args.resume else {}
    write_manifest(rows, args.manifest)
    pending = [r for r in rows if not matches_remote(r, remote, args.skip_sha256)]
    if not pending:
        print("all videos already present; uploading manifest/card only")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        operations = [
            CommitOperationAdd(path_in_repo=row["upload_path"], path_or_fileobj=str(videos_root.parent / row["path"]))
            for row in batch
        ]
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"{args.commit_message} ({start + 1}-{start + len(batch)} of {len(pending)})",
        )
        print(f"uploaded {start + len(batch)}/{len(pending)}", flush=True)
    api.upload_file(
        path_or_fileobj=str(args.manifest),
        path_in_repo="manifests/video-manifest.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add video inventory manifest",
    )
    if args.dataset_card is not None:
        api.upload_file(
            path_or_fileobj=str(args.dataset_card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
