"""Merge completed, suite-disjoint LIBERO-Plus shards into one auditable result."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys
import traceback
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pi05_eval_support as support


POLICY_RNG_INITIAL_KEY = 0
POLICY_RNG_SCOPE = "server_process/per-suite"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-repo", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--suite-dir",
        required=True,
        action="append",
        metavar="SUITE=PATH",
        help="authoritative result directory for one suite; provide all four suites",
    )
    return parser


def parse_suite_dirs(values: Sequence[str]) -> Dict[str, pathlib.Path]:
    result: Dict[str, pathlib.Path] = {}
    for value in values:
        suite, separator, raw_path = value.partition("=")
        if not separator or suite not in support.SUITES or not raw_path:
            raise ValueError("--suite-dir must be SUITE=PATH with a canonical suite: %s" % value)
        if suite in result:
            raise ValueError("duplicate authoritative suite directory: %s" % suite)
        path = pathlib.Path(raw_path).expanduser().resolve()
        if not path.is_dir() or not (path / "episodes.jsonl").is_file():
            raise ValueError("suite result is missing episodes.jsonl: %s" % path)
        result[suite] = path
    missing = sorted(set(support.SUITES) - set(result))
    if missing:
        raise ValueError("missing authoritative suite directories: %s" % missing)
    return result


def validate_shard_records(
    matrix: Sequence[support.EpisodeSpec],
    records_by_suite: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    expected_by_suite = {
        suite: {spec.episode_id for spec in matrix if spec.suite == suite}
        for suite in support.SUITES
    }
    if set(records_by_suite) != set(support.SUITES):
        raise ValueError("record inputs must contain exactly the four canonical suites")

    all_records: List[Mapping[str, Any]] = []
    attempt_ids = set()
    episode_sources: Dict[str, str] = {}
    for suite in support.SUITES:
        expected = expected_by_suite[suite]
        for record in records_by_suite[suite]:
            episode_id = str(record.get("episode_id", ""))
            attempt_id = str(record.get("attempt_id", ""))
            if record.get("suite") != suite:
                raise ValueError("record suite does not match authoritative input %s: %s" % (suite, episode_id))
            if not episode_id or episode_id not in expected:
                raise ValueError("unplanned episode in authoritative input %s: %s" % (suite, episode_id))
            owner = episode_sources.setdefault(episode_id, suite)
            if owner != suite:
                raise ValueError("episode appears in multiple authoritative inputs: %s" % episode_id)
            if not attempt_id or attempt_id in attempt_ids:
                raise ValueError("attempt ID is missing or duplicated: %s" % attempt_id)
            attempt_ids.add(attempt_id)
            all_records.append(record)

    latest = support._latest_attempts(all_records)
    planned_ids = {spec.episode_id for spec in matrix}
    missing = planned_ids - set(latest)
    if missing:
        raise ValueError("%d planned episodes have no attempt" % len(missing))
    nonterminal = [
        episode_id
        for episode_id in planned_ids
        if latest[episode_id].get("status") not in ("success", "failure")
    ]
    if nonterminal:
        raise ValueError("%d planned episodes have no policy outcome" % len(nonterminal))
    extras = set(latest) - planned_ids
    if extras:
        raise ValueError("%d unplanned episodes are present" % len(extras))

    grouped: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in all_records:
        grouped[str(record["episode_id"])].append(record)
    ordered: List[Mapping[str, Any]] = []
    for spec in matrix:
        attempts = sorted(grouped[spec.episode_id], key=lambda record: int(record.get("attempt", 0)))
        ordered.extend(attempts)
    return ordered


def _matrix_digest(matrix: Sequence[support.EpisodeSpec]) -> str:
    payload = "".join(spec.episode_id + "\n" for spec in matrix).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _latest_manifest(directory: pathlib.Path, prefix: str) -> Optional[Dict[str, str]]:
    matches = sorted((directory / "manifests").glob(prefix + "-*.json"))
    if not matches:
        return None
    path = matches[-1].resolve()
    return {"path": str(path), "sha256": support.sha256_file(path)}


def _link_videos(
    records: Sequence[Mapping[str, Any]],
    staging_dir: pathlib.Path,
    final_dir: pathlib.Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    staging_videos = staging_dir / "videos"
    final_videos = final_dir / "videos"
    staging_videos.mkdir(parents=True)
    merged: List[Dict[str, Any]] = []
    verification: List[Dict[str, Any]] = []
    used_names = set()
    for record in records:
        item = dict(record)
        check = dict(record)
        if record.get("status") in ("success", "failure"):
            source = pathlib.Path(str(record.get("video", ""))).resolve()
            if record.get("video_status") != "written" or not source.is_file() or source.stat().st_size == 0:
                raise ValueError("policy outcome has no non-empty video: %s" % record.get("attempt_id"))
            name = "%s-%s" % (record["suite"], source.name)
            if name in used_names:
                raise ValueError("merged video filename collision: %s" % name)
            used_names.add(name)
            staging_link = staging_videos / name
            final_link = final_videos / name
            os.symlink(os.path.relpath(str(source), str(final_videos)), str(staging_link))
            item["video_original"] = str(source)
            item["video"] = str(final_link)
            item["video_relative"] = str(pathlib.Path("videos") / name)
            check["video"] = str(staging_link)
        merged.append(item)
        verification.append(check)
    return merged, verification


def merge(args: argparse.Namespace) -> pathlib.Path:
    benchmark_repo = args.benchmark_repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("merged output already exists: %s" % output_dir)
    suite_dirs = parse_suite_dirs(args.suite_dir)
    sources = support.discover_sources("plus", benchmark_repo)
    matrix = support.expand_matrix("plus", "full", sources)
    directory_records: Dict[pathlib.Path, List[Mapping[str, Any]]] = {}
    directory_suites: Dict[pathlib.Path, set] = collections.defaultdict(set)
    for suite, directory in suite_dirs.items():
        directory_suites[directory].add(suite)
        if directory not in directory_records:
            directory_records[directory] = support.load_jsonl(directory / "episodes.jsonl")
    for directory, records in directory_records.items():
        unexpected = [
            str(record.get("episode_id", ""))
            for record in records
            if record.get("suite") not in directory_suites[directory]
        ]
        if unexpected:
            raise ValueError(
                "%s contains %d records outside its authoritative suites %s"
                % (directory, len(unexpected), sorted(directory_suites[directory]))
            )
    records_by_suite = {
        suite: [
            record
            for record in directory_records[directory]
            if record.get("suite") == suite
        ]
        for suite, directory in suite_dirs.items()
    }
    ordered = validate_shard_records(matrix, records_by_suite)

    staging = output_dir.parent / (".%s.tmp-%s" % (output_dir.name, uuid.uuid4().hex[:8]))
    staging.mkdir(parents=True, exist_ok=False)
    try:
        merged, verification = _link_videos(ordered, staging, output_dir)
        episodes_path = staging / "episodes.jsonl"
        with episodes_path.open("x", encoding="utf-8") as stream:
            for record in merged:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        summary = support.aggregate(merged, matrix)
        integrity = support.verify_integrity(verification, matrix, require_videos=True)
        if not summary["total"]["complete"] or not integrity["passed"]:
            raise ValueError("merged result did not pass completion checks: %s" % integrity["issues"])
        summary.update(
            {
                "benchmark": "plus",
                "protocol": "official-checkpoint/multi-server-suite-sharded",
                "phase": "full",
                "seed": 7,
                "policy_rng_initial_key": POLICY_RNG_INITIAL_KEY,
                "policy_rng_scope": POLICY_RNG_SCOPE,
                "integrity": integrity,
            }
        )
        support.write_summaries(summary, staging)
        manifests = staging / "manifests"
        manifests.mkdir()
        source_evidence = {}
        for suite, directory in suite_dirs.items():
            source_evidence[suite] = {
                "directory": str(directory),
                "episodes_jsonl_sha256": support.sha256_file(directory / "episodes.jsonl"),
                "environment_manifest": _latest_manifest(directory, "environment"),
                "checkpoint_manifest": _latest_manifest(directory, "checkpoint-server"),
            }
        support.write_new_json(
            manifests / "merge.json",
            {
                "status": "passed",
                "created_at": _utc_now(),
                "benchmark": "plus",
                "planned_episodes": len(matrix),
                "suite_counts": dict(collections.Counter(spec.suite for spec in matrix)),
                "matrix_episode_ids_sha256": _matrix_digest(matrix),
                "merged_records": len(merged),
                "merged_policy_outcomes": summary["total"]["policy_denominator"],
                "merged_episodes_jsonl_sha256": support.sha256_file(episodes_path),
                "policy_rng_initial_key": POLICY_RNG_INITIAL_KEY,
                "policy_rng_scope": POLICY_RNG_SCOPE,
                "environment_seed": 7,
                "sources": source_evidence,
                "integrity": integrity,
            },
        )
        os.replace(str(staging), str(output_dir))
    except Exception:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = merge(args)
    except Exception as exc:
        print("merge failed: %s: %s" % (exc.__class__.__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
