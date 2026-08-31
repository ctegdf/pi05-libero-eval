#!/usr/bin/env python3
"""Export one harness run's episodes.jsonl into MODEL_EVAL_RESULT_PROTOCOL_V1 layout.

This is an ADDITIVE publish step, not a replacement for the working harness's
own rich output (episodes.jsonl with pi0.5-style fields, manifests/, runtime/,
logs/). Per docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md §1, that protocol explicitly
does not care about "checkpoint 加载方式/推理框架/GPU 分配方式/benchmark 启动命令" --
it is a minimal public interchange schema, not a mandate for how evaluation
internally works. This script reads the harness's own episodes.jsonl (see
../../harness-refactor/new/eval_client.py's _episode_record shape, or the
existing pi0.5 harness shape -- both are supersets of what the protocol needs)
and re-derives the leaner protocol schema from it, the same way the existing
pi0.5 `prepare_release.py` re-derives a public metadata tree from the
immutable evaluation archive without ever mutating the source.

Writes, under --release-root:
    results/runs/<model_id>/<benchmark_id>/<run_id>/{run.json,episodes.jsonl,summary.json,summary.csv}
    videos/<model_id>/<benchmark_id>/<run_id>/<episode-NNNNNN>.mp4   (renamed, sequential -- see _stage_videos)
and updates results/run-registry.json.

Never touches results/openpi-libero, results/plus-pro, results/libero-x, or
anything under harness/ -- this script is scoped entirely to the new
results/runs/ tree per protocol §12.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
MODEL_IDS = ("act", "openvla")
MODEL_NAMES = {"act": "ACT", "openvla": "OpenVLA"}
BENCHMARK_IDS = ("libero", "libero-plus", "libero-pro")
PHASES = ("smoke", "full")
RUN_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ALLOWED_VIDEO_STATUS = {"written", "not_recorded", "missing", "corrupt", "not_requested"}
PRIVATE_TEXT = re.compile(r"(?:/home/[^\s\"']+|\b(?:hhj@)?10\.0\.106\.18\b)")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSONL at %s:%d: %s" % (path, line_number, exc)) from exc
    return records


def _map_video_status(raw_status: Optional[str], video_available: bool) -> str:
    if video_available:
        return "written"
    # video_available is False from here on -- the protocol's own invariant
    # (video_available=false => video_status != written) means "written"
    # must NEVER be returned below, regardless of what the source harness
    # claimed, e.g. if the source said "written" but the file is missing at
    # export time (moved/deleted/wrong path), that is exactly a "missing"
    # video, not a written one.
    if raw_status in ("not_recorded", None):
        return "not_recorded"
    if raw_status == "not_requested":
        return "not_requested"
    # Covers the harness's "failed" (video render errored) and any other
    # non-"written" status, including a stale "written" claim -- "missing" is
    # the closest honest mapping; we don't have enough signal to distinguish
    # "missing" from "corrupt".
    return "missing"


def _stage_video(
    source_video: Optional[str], source_base: Path, video_index: Mapping[str, Sequence[Path]], video_root: Path, sequence: int
) -> Tuple[Optional[Path], Optional[str]]:
    """Copies the harness's video into videos/<model>/<benchmark>/<run_id>/episode-NNNNNN.mp4.

    Returns (staged_path_on_disk, protocol_relative_path) or (None, None) if
    there is no source video to stage. Renaming to a sequential, content-free
    filename avoids leaking any private naming (source-id hashes, absolute
    path fragments) that the harness's own filenames embed."""
    if not source_video:
        return None, None
    raw_path = Path(source_video).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend((source_base / raw_path, source_base / "videos" / raw_path.name, source_base / raw_path.name))
    source_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source_path is None:
        matches = list(video_index.get(raw_path.name, ()))
        if len(matches) == 1:
            source_path = matches[0]
        elif matches:
            raw_parts = {part.lower() for part in raw_path.parts[:-1] if part not in ("/", "videos")}
            scored = [
                (sum(part.lower() in raw_parts for part in candidate.parts), candidate)
                for candidate in matches
            ]
            best_score = max(score for score, _ in scored)
            best = [candidate for score, candidate in scored if score == best_score and score > 0]
            if len(best) == 1:
                source_path = best[0]
    if source_path is None:
        return None, None
    staged_name = "episode-%06d.mp4" % sequence
    staged_path = video_root / staged_name
    video_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, staged_path)
    return staged_path, staged_name


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return PRIVATE_TEXT.sub("<private>", value)
    if isinstance(value, Mapping):
        return {str(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


_EXTRA_KEYS = (
    "category", "difficulty", "perturbation", "prompt_field", "task_description",
    "stage", "max_steps", "wait_steps", "replan", "gl_backend", "started_at",
    "finished_at", "protocol", "phase", "benchmark", "policy", "attempt", "attempt_id",
)


def _convert_episode(
    record: Mapping[str, Any],
    run_id: str,
    model_id: str,
    benchmark_id: str,
    phase: str,
    source_base: Path,
    video_index: Mapping[str, Sequence[Path]],
    video_root: Path,
    video_public_prefix: str,
    sequence: int,
) -> Dict[str, Any]:
    status = record.get("status")
    if status not in ("success", "failure", "error"):
        raise ValueError("unexpected source status %r for episode %r" % (status, record.get("episode_id")))
    # Protocol requires success to be a strict bool (true/false), never null;
    # an "error" record's source `success` is None -- treat that as false,
    # which still satisfies "success=false => status != success".
    success = bool(record.get("success"))

    staged_path, staged_name = _stage_video(record.get("video"), source_base, video_index, video_root, sequence)
    video_available = staged_path is not None
    video_status = _map_video_status(record.get("video_status"), video_available)
    video_public = "%s/%s" % (video_public_prefix, staged_name) if staged_name else None

    extra = {key: record[key] for key in _EXTRA_KEYS if key in record and record[key] is not None}

    episode = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "benchmark": benchmark_id,
        "phase": phase,
        "episode_id": record["episode_id"],
        "suite": record.get("suite"),
        "task_id": record.get("task_id"),
        "trial": record.get("trial"),
        "seed": record.get("seed"),
        "status": status,
        "success": success,
        "action_steps": record.get("action_steps"),
        "duration_seconds": record.get("duration_seconds"),
        "error_category": record.get("error_category"),
        "error_message": record.get("error"),
        "video": video_public,
        "video_available": video_available,
        "video_status": video_status,
    }
    if extra:
        episode["extra"] = _scrub(extra)
    return episode


def convert_episodes(
    records: Sequence[Mapping[str, Any]],
    run_id: str,
    model_id: str,
    benchmark_id: str,
    phase: str,
    source_base: Path,
    video_index: Mapping[str, Sequence[Path]],
    video_root: Path,
    video_public_prefix: str,
) -> List[Dict[str, Any]]:
    converted = []
    for sequence, record in enumerate(records):
        converted.append(
            _convert_episode(record, run_id, model_id, benchmark_id, phase, source_base, video_index, video_root, video_public_prefix, sequence)
        )
    episode_ids = [item["episode_id"] for item in converted]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("source episodes.jsonl has duplicate episode_id values; resolve before exporting")
    return converted


def build_summary(episodes: Sequence[Mapping[str, Any]], run_id: str, model_id: str, benchmark_id: str, phase: str) -> Dict[str, Any]:
    total = len(episodes)
    successes = sum(1 for e in episodes if e["status"] == "success")
    failures = sum(1 for e in episodes if e["status"] == "failure")
    errors = sum(1 for e in episodes if e["status"] == "error")
    videos_available = sum(1 for e in episodes if e["video_available"])
    videos_missing = total - videos_available
    by_suite: Dict[str, Dict[str, Any]] = {}
    for suite in sorted({e["suite"] for e in episodes if e["suite"]}):
        suite_episodes = [e for e in episodes if e["suite"] == suite]
        s_total = len(suite_episodes)
        s_successes = sum(1 for e in suite_episodes if e["status"] == "success")
        s_failures = sum(1 for e in suite_episodes if e["status"] == "failure")
        s_errors = sum(1 for e in suite_episodes if e["status"] == "error")
        s_videos_available = sum(1 for e in suite_episodes if e["video_available"])
        by_suite[suite] = {
            "total": s_total,
            "successes": s_successes,
            "failures": s_failures,
            "errors": s_errors,
            "success_rate": (s_successes / s_total) if s_total else None,
            "videos_available": s_videos_available,
            "videos_missing": s_total - s_videos_available,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "benchmark": benchmark_id,
        "phase": phase,
        "total": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "success_rate": (successes / total) if total else None,
        "videos_available": videos_available,
        "videos_missing": videos_missing,
        "by_suite": by_suite,
    }


def write_summary_csv(summary: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["suite", "total", "successes", "failures", "errors", "success_rate", "videos_available", "videos_missing"])
        for suite in sorted(summary["by_suite"]):
            stats = summary["by_suite"][suite]
            writer.writerow([
                suite, stats["total"], stats["successes"], stats["failures"], stats["errors"],
                stats["success_rate"], stats["videos_available"], stats["videos_missing"],
            ])


def build_run_json(
    args: argparse.Namespace, summary: Mapping[str, Any], started_at: Optional[str], finished_at: Optional[str]
) -> Dict[str, Any]:
    recorded = summary["total"]
    expected = args.expected_episodes if args.expected_episodes is not None else recorded
    complete = recorded >= expected and summary["errors"] == 0
    status = args.status or ("completed" if complete else "partial")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "model_id": args.model_id,
        "model_name": MODEL_NAMES[args.model_id],
        "benchmark": args.benchmark,
        "phase": args.phase,
        "status": status,
        "checkpoint": {
            "name": args.checkpoint_name,
            "revision": args.checkpoint_revision,
            "sha256": args.checkpoint_sha256,
        },
        "protocol": {
            "seed": args.seed,
            "trial_count": args.trial_count,
            "suite_selection": args.suite_selection,
            "task_selection": args.task_selection,
            "max_steps_policy": args.max_steps_policy,
            "action_horizon": args.action_horizon,
        },
        "implementation": {
            "inference_framework": args.inference_framework,
            "repository": args.repository,
            "repository_revision": args.repository_revision,
            "config_file": args.config_file,
        },
        "provenance": {
            "created_at": _utc_now(),
            "started_at": started_at,
            "finished_at": finished_at,
            "hardware": args.hardware,
            "cuda_version": args.cuda_version,
            "python_version": args.python_version,
        },
        "counts": {
            "expected_episodes": expected,
            "recorded_episodes": recorded,
            "successes": summary["successes"],
            "failures": summary["failures"],
            "errors": summary["errors"],
            "videos_available": summary["videos_available"],
            "videos_missing": summary["videos_missing"],
        },
        "artifacts": {
            "episodes": "episodes.jsonl",
            "summary_json": "summary.json",
            "summary_csv": "summary.csv",
            "video_root": "videos/%s/%s/%s" % (args.model_id, args.benchmark, args.run_id),
        },
        "notes": args.notes,
    }


def update_registry(release_root: Path, run_json: Mapping[str, Any], run_path: str) -> None:
    registry_path = release_root / "results" / "run-registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": SCHEMA_VERSION, "runs": []}
    entry = {
        "run_id": run_json["run_id"],
        "model_id": run_json["model_id"],
        "benchmark": run_json["benchmark"],
        "phase": run_json["phase"],
        "status": run_json["status"],
        "path": run_path,
    }
    runs = [r for r in registry["runs"] if r["run_id"] != entry["run_id"]]
    runs.append(entry)
    registry["runs"] = sorted(runs, key=lambda r: (r["model_id"], r["benchmark"], r["phase"], r["run_id"]))
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episodes", required=True, type=Path, help="path to the harness's own episodes.jsonl")
    parser.add_argument(
        "--video-source-root",
        type=Path,
        default=None,
        help="root containing source MP4s; needed when episode video fields are private absolute paths",
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True, choices=MODEL_IDS)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARK_IDS)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--status", choices=("completed", "partial", "failed", "cancelled"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trial-count", type=int, default=None)
    parser.add_argument("--suite-selection", default=None)
    parser.add_argument("--task-selection", default=None)
    parser.add_argument("--max-steps-policy", default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--inference-framework", default=None)
    parser.add_argument("--repository", default=None)
    parser.add_argument("--repository-revision", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--cuda-version", default=None)
    parser.add_argument("--python-version", default=None)
    parser.add_argument("--notes", default=None)
    return parser


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_RE.match(run_id):
        raise ValueError("--run-id %r must be lowercase ASCII letters/digits/hyphens only (protocol §4)" % run_id)


def export(args: argparse.Namespace) -> Path:
    _check_run_id(args.run_id)
    release_root = args.release_root.expanduser().resolve()
    run_dir = release_root / "results" / "runs" / args.model_id / args.benchmark / args.run_id
    if run_dir.exists():
        raise FileExistsError(
            "run directory already exists: %s (protocol §13.9: use a new run_id, never overwrite)" % run_dir
        )
    video_root = release_root / "videos" / args.model_id / args.benchmark / args.run_id
    video_public_prefix = "videos/%s/%s/%s" % (args.model_id, args.benchmark, args.run_id)

    source_records = _load_jsonl(args.source_episodes.expanduser().resolve())
    source_base = args.source_episodes.expanduser().resolve().parent
    video_source_root = (args.video_source_root or source_base).expanduser().resolve()
    video_index: Dict[str, List[Path]] = {}
    for candidate in video_source_root.rglob("*.mp4"):
        if not candidate.is_file():
            continue
        video_index.setdefault(candidate.name, []).append(candidate)
    episodes = convert_episodes(
        source_records,
        args.run_id,
        args.model_id,
        args.benchmark,
        args.phase,
        source_base,
        video_index,
        video_root,
        video_public_prefix,
    )
    summary = build_summary(episodes, args.run_id, args.model_id, args.benchmark, args.phase)

    started_at = min((r.get("started_at") for r in source_records if r.get("started_at")), default=None)
    finished_at = max((r.get("finished_at") for r in source_records if r.get("finished_at")), default=None)
    run_json = build_run_json(args, summary, started_at, finished_at)

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(summary, run_dir / "summary.csv")

    relative_run_path = str(run_dir.relative_to(release_root))
    update_registry(release_root, run_json, relative_run_path)
    return run_dir


def main() -> int:
    args = build_parser().parse_args()
    run_dir = export(args)
    print(json.dumps({"run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
