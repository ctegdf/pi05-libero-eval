#!/usr/bin/env python3
"""Validate one results/runs/<model_id>/<benchmark_id>/<run_id>/ directory
against MODEL_EVAL_RESULT_PROTOCOL_V1 (docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md §11).

Usage:
    python3 scripts/validate_run.py --run-dir results/runs/act/libero/act-libero-full-v1

Exits 0 and prints "VALID\\n..." on success; exits non-zero and prints every
violation found (not just the first) on failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


MODEL_IDS = ("act", "openvla")
BENCHMARK_IDS = ("libero", "libero-plus", "libero-pro")
PHASES = ("smoke", "full")
STATUS_VALUES = ("planned", "running", "completed", "partial", "failed", "cancelled")
EPISODE_STATUS_VALUES = ("success", "failure", "error", "cancelled", "not_recorded")
VIDEO_STATUS_VALUES = ("written", "not_recorded", "missing", "corrupt", "not_requested")
RUN_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PRIVATE_TEXT_RE = re.compile(r"/home/[^\s\"']+|\b(?:[A-Za-z0-9_.-]+@)?\d{1,3}(?:\.\d{1,3}){3}\b")
TOKEN_LIKE_KEYS = {"token", "api_key", "apikey", "secret", "password", "ssh_user", "ssh_username"}


class Violations(list):
    def add(self, message: str) -> None:
        self.append(message)


def _load_json(path: Path, violations: Violations) -> Any:
    if not path.is_file():
        violations.add("missing required file: %s" % path.name)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        violations.add("%s is not valid JSON: %s" % (path.name, exc))
        return None


def _load_jsonl(path: Path, violations: Violations) -> List[Dict[str, Any]]:
    if not path.is_file():
        violations.add("missing required file: %s" % path.name)
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            violations.add("episodes.jsonl:%d is not valid JSON: %s" % (line_number, exc))
            continue
        if not isinstance(record, dict):
            violations.add("episodes.jsonl:%d is not a JSON object" % line_number)
            continue
        records.append(record)
    return records


def _scan_private_text(value: Any, path: str, violations: Violations, seen: set) -> None:
    if isinstance(value, str):
        if PRIVATE_TEXT_RE.search(value):
            violations.add("private-looking text (absolute path or IP) at %s: %r" % (path, value))
        return
    if isinstance(value, Mapping):
        for key, sub in value.items():
            if str(key).lower() in TOKEN_LIKE_KEYS and sub not in (None, "", "<private>"):
                violations.add("possible token/credential field at %s.%s" % (path, key))
            _scan_private_text(sub, "%s.%s" % (path, key), violations, seen)
        return
    if isinstance(value, list):
        for index, sub in enumerate(value):
            _scan_private_text(sub, "%s[%d]" % (path, index), violations, seen)


def validate_run_json(run_json: Mapping[str, Any], run_dir_name: str, violations: Violations) -> None:
    for field in ("schema_version", "run_id", "model_id", "benchmark", "phase", "status"):
        if not run_json.get(field) and run_json.get(field) != 0:
            violations.add("run.json missing required field: %s" % field)
    run_id = run_json.get("run_id")
    if isinstance(run_id, str) and not RUN_ID_RE.match(run_id):
        violations.add("run.json run_id %r violates §4 (lowercase ascii/digits/hyphens only)" % run_id)
    if isinstance(run_id, str) and run_id != run_dir_name:
        violations.add("run.json run_id %r does not match its containing directory name %r" % (run_id, run_dir_name))
    if run_json.get("model_id") not in MODEL_IDS:
        violations.add("run.json model_id %r is not one of %s" % (run_json.get("model_id"), MODEL_IDS))
    if run_json.get("benchmark") not in BENCHMARK_IDS:
        violations.add("run.json benchmark %r is not one of %s" % (run_json.get("benchmark"), BENCHMARK_IDS))
    if run_json.get("phase") not in PHASES:
        violations.add("run.json phase %r is not one of %s" % (run_json.get("phase"), PHASES))
    if run_json.get("status") not in STATUS_VALUES:
        violations.add("run.json status %r is not one of %s" % (run_json.get("status"), STATUS_VALUES))
    if run_json.get("status") == "completed":
        provenance = run_json.get("provenance") or {}
        counts = run_json.get("counts") or {}
        for field in ("started_at", "finished_at"):
            if not provenance.get(field):
                violations.add("run.json status=completed requires provenance.%s" % field)
        for field in ("expected_episodes", "recorded_episodes", "successes", "failures", "errors"):
            if counts.get(field) is None:
                violations.add("run.json status=completed requires counts.%s" % field)


def validate_episodes(
    episodes: Sequence[Mapping[str, Any]], run_json: Mapping[str, Any], violations: Violations
) -> None:
    seen_ids: Dict[str, int] = {}
    for index, episode in enumerate(episodes):
        location = "episodes.jsonl record %d (episode_id=%r)" % (index, episode.get("episode_id"))
        for field in ("schema_version", "run_id", "model_id", "benchmark", "phase", "episode_id", "status"):
            if episode.get(field) in (None, ""):
                violations.add("%s missing required field: %s" % (location, field))
        episode_id = episode.get("episode_id")
        if isinstance(episode_id, str):
            seen_ids[episode_id] = seen_ids.get(episode_id, 0) + 1
        for field in ("run_id", "model_id", "benchmark", "phase"):
            if field in episode and field in run_json and episode[field] != run_json[field]:
                violations.add("%s field %s=%r does not match run.json %s=%r" % (location, field, episode[field], field, run_json[field]))
        status = episode.get("status")
        if status is not None and status not in EPISODE_STATUS_VALUES:
            violations.add("%s has invalid status %r" % (location, status))
        success = episode.get("success")
        if success is not None and not isinstance(success, bool):
            violations.add("%s success must be a bool or null, got %r" % (location, success))
        if success is True and status != "success":
            violations.add("%s violates invariant: success=true but status=%r (must be success)" % (location, status))
        if success is False and status == "success":
            violations.add("%s violates invariant: success=false but status=success" % location)
        video_available = episode.get("video_available")
        video_status = episode.get("video_status")
        if video_status is not None and video_status not in VIDEO_STATUS_VALUES:
            violations.add("%s has invalid video_status %r" % (location, video_status))
        if video_available is True and video_status != "written":
            violations.add("%s violates invariant: video_available=true but video_status=%r (must be written)" % (location, video_status))
        if video_available is False and video_status == "written":
            violations.add("%s violates invariant: video_available=false but video_status=written" % location)
    duplicates = sorted(episode_id for episode_id, count in seen_ids.items() if count > 1)
    if duplicates:
        violations.add("episodes.jsonl has %d duplicate episode_id value(s): %s" % (len(duplicates), duplicates[:10]))


def validate_summary_consistency(
    episodes: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], violations: Violations
) -> None:
    total = len(episodes)
    successes = sum(1 for e in episodes if e.get("status") == "success")
    failures = sum(1 for e in episodes if e.get("status") == "failure")
    errors = sum(1 for e in episodes if e.get("status") == "error")
    videos_available = sum(1 for e in episodes if e.get("video_available") is True)
    videos_missing = total - videos_available
    for field, expected in (
        ("total", total), ("successes", successes), ("failures", failures), ("errors", errors),
        ("videos_available", videos_available), ("videos_missing", videos_missing),
    ):
        if summary.get(field) != expected:
            violations.add(
                "summary.json.%s=%r does not match recomputation from episodes.jsonl (%r) -- §7 requires summary "
                "values be recomputed from episodes.jsonl, never hand-filled" % (field, summary.get(field), expected)
            )
    if total and successes + failures + errors != total:
        violations.add("episodes.jsonl: successes+failures+errors != total (%d+%d+%d != %d)" % (successes, failures, errors, total))


def validate_summary_csv(csv_path: Path, summary: Mapping[str, Any], violations: Violations) -> None:
    if not csv_path.is_file():
        violations.add("missing required file: summary.csv")
        return
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)
    if not rows:
        violations.add("summary.csv is empty")
        return
    expected_header = ["suite", "total", "successes", "failures", "errors", "success_rate", "videos_available", "videos_missing"]
    if rows[0] != expected_header:
        violations.add("summary.csv header %r does not match protocol §8: %r" % (rows[0], expected_header))
    by_suite = summary.get("by_suite") or {}
    csv_suites = {row[0] for row in rows[1:] if row}
    if csv_suites != set(by_suite):
        violations.add("summary.csv suites %r do not match summary.json.by_suite keys %r" % (sorted(csv_suites), sorted(by_suite)))


def validate(run_dir: Path) -> Violations:
    violations = Violations()
    run_json = _load_json(run_dir / "run.json", violations)
    episodes = _load_jsonl(run_dir / "episodes.jsonl", violations)
    summary = _load_json(run_dir / "summary.json", violations)

    if isinstance(run_json, dict):
        validate_run_json(run_json, run_dir.name, violations)
    if episodes and isinstance(run_json, dict):
        validate_episodes(episodes, run_json, violations)
    elif episodes:
        validate_episodes(episodes, {}, violations)
    if episodes and isinstance(summary, dict):
        validate_summary_consistency(episodes, summary, violations)
    if isinstance(summary, dict):
        validate_summary_csv(run_dir / "summary.csv", summary, violations)
    elif not (run_dir / "summary.csv").is_file():
        violations.add("missing required file: summary.csv")

    for path in (run_dir / "run.json", run_dir / "summary.json"):
        if path.is_file():
            payload = _load_json(path, Violations())  # already validated as JSON above
            if isinstance(payload, (dict, list)):
                _scan_private_text(payload, path.name, violations, set())
    for index, episode in enumerate(episodes):
        _scan_private_text(episode, "episodes.jsonl[%d]" % index, violations, set())

    return violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print("FAIL\nrun directory does not exist: %s" % run_dir, file=sys.stderr)
        return 2
    violations = validate(run_dir)
    if violations:
        print("FAIL", file=sys.stderr)
        for violation in violations:
            print("- %s" % violation, file=sys.stderr)
        return 1

    episodes = _load_jsonl(run_dir / "episodes.jsonl", Violations())
    successes = sum(1 for e in episodes if e.get("status") == "success")
    failures = sum(1 for e in episodes if e.get("status") == "failure")
    errors = sum(1 for e in episodes if e.get("status") == "error")
    videos_available = sum(1 for e in episodes if e.get("video_available") is True)
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    print("VALID")
    print("run_id=%s" % run_json.get("run_id"))
    print("episodes=%d" % len(episodes))
    print("successes=%d" % successes)
    print("failures=%d" % failures)
    print("errors=%d" % errors)
    print("videos_available=%d" % videos_available)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
