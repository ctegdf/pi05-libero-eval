"""Audit all LIBERO, LIBERO-Plus and LIBERO-Pro results and write one report."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
import uuid
from typing import Any, Dict, Mapping, Optional, Sequence

import pi05_eval_support as support


OFFICIAL_MIN_SUCCESS_RATE = 0.9385
EXPECTED_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
EXPECTED_PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
EXPECTED_PRO_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
EXPECTED_NORM_STATS_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
EXPECTED_PARAMS_TREE_SHA256 = "b5d2c61bb555413cba73b66b6876c5e895e9f6ea69e6eeb9827ea9ea7339fa45"
EXPECTED_PARAMS_FILES = 15
EXPECTED_PARAMS_BYTES = 12_439_083_567
EXPECTED_BASE_PARAMS_TREE_SHA256 = "7ed18c089c75ccd1b2aa1506045a575177a4b81691a38d4687da0715fb7ba0cb"
EXPECTED_BASE_PARAMS_FILES = 20
EXPECTED_BASE_PARAMS_BYTES = 12_441_721_931
HARNESS_FILES = (
    "PLUS_SHARD_PROTOCOL.md",
    "README.md",
    "audit_final_results.py",
    "finalize_all_results.sh",
    "finalize_plus_shards.sh",
    "merge_plus_shards.py",
    "pi05_eval_client.py",
    "pi05_eval_server.py",
    "pi05_eval_support.py",
    "recover_finalizers.sh",
    "run_full_sequence.sh",
    "run_pi05_eval.py",
    "sync_final_archive.sh",
    "test_pi05_eval.py",
)
_SOURCE_INVENTORY_CACHE: Dict[tuple, Dict[str, Any]] = {}
_CHECKPOINT_INVENTORY_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-eval-root", required=True, type=pathlib.Path)
    parser.add_argument("--plus-repo", required=True, type=pathlib.Path)
    parser.add_argument("--plus-result", required=True, type=pathlib.Path)
    parser.add_argument("--pro-data", required=True, type=pathlib.Path)
    parser.add_argument("--pro-result", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_official_threshold(success_rate: Optional[float]) -> None:
    _require(
        success_rate is not None and success_rate >= OFFICIAL_MIN_SUCCESS_RATE,
        "old official success rate %s is below required %.2f%%"
        % (_format_rate(success_rate), 100.0 * OFFICIAL_MIN_SUCCESS_RATE),
    )


def _digest_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(value + "\n" for value in values).encode("utf-8")).hexdigest()


def _runtime_manifests(result_dir: pathlib.Path) -> tuple:
    directory = result_dir / "manifests"
    environments = sorted(directory.glob("environment-*.json"))
    _require(bool(environments), "%s environment manifest is missing" % result_dir)
    anchor = environments[-1].resolve()
    run_id = anchor.name[len("environment-") : -len(".json")]
    _require(bool(run_id), "%s environment manifest has no run ID" % result_dir)
    paths = {}
    payloads = {}
    for prefix in ("git", "checkpoint-server", "sources", "preflight", "environment"):
        path = (directory / (prefix + "-" + run_id + ".json")).resolve()
        _require(path.is_file(), "%s %s manifest for run %s is missing" % (result_dir, prefix, run_id))
        paths[prefix] = path
        payloads[prefix] = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payloads["environment"].get("run_id") == run_id,
        "%s environment manifest run ID disagrees" % result_dir,
    )
    return run_id, paths, payloads


def _normalized_status(status: Any) -> tuple:
    return tuple(sorted(line.strip() for line in str(status or "").splitlines() if line.strip()))


def _current_git_state(repo: pathlib.Path, untracked_files_all: bool = True) -> Dict[str, Any]:
    values = {}
    status_arguments = ["status", "--porcelain=v1"]
    if untracked_files_all:
        status_arguments.append("--untracked-files=all")
    for key, arguments in (("commit", ["rev-parse", "HEAD"]), ("status", status_arguments)):
        completed = subprocess.run(
            ["git", "-C", str(repo)] + list(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
            timeout=30.0,
        )
        _require(completed.returncode == 0, "%s git %s failed: %s" % (repo, key, completed.stderr.strip()))
        values[key] = completed.stdout.strip()
    return values


def _current_source_inventory(benchmark: str, benchmark_repo: pathlib.Path) -> Dict[str, Any]:
    key = (benchmark, str(benchmark_repo.resolve()))
    if key not in _SOURCE_INVENTORY_CACHE:
        sources = support.discover_sources(
            benchmark, benchmark_repo, allow_missing_pro_cells=benchmark == "pro"
        )
        _SOURCE_INVENTORY_CACHE[key] = support.source_manifest(benchmark, benchmark_repo, sources)
    return _SOURCE_INVENTORY_CACHE[key]


def _checkpoint_inventory(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoint_dir = pathlib.Path(str(checkpoint["checkpoint"])).resolve()
    params_dir = checkpoint_dir / "params"
    norm_stats_path = pathlib.Path(str(checkpoint.get("norm_stats", ""))).resolve()
    cache_key = (str(params_dir), str(norm_stats_path))
    if cache_key in _CHECKPOINT_INVENTORY_CACHE:
        return dict(_CHECKPOINT_INVENTORY_CACHE[cache_key])
    param_files = sorted(path for path in params_dir.rglob("*") if path.is_file())
    rows = []
    for path in param_files:
        rows.append(
            {
                "path": path.relative_to(params_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": support.sha256_file(path),
            }
        )
    tree_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    inventory = {
        "params_dir": str(params_dir),
        "params_source_matches": pathlib.Path(str(checkpoint.get("params_source", ""))).resolve() == params_dir,
        "param_files": len(rows),
        "param_bytes": sum(row["size"] for row in rows),
        "params_tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "norm_stats_exists": norm_stats_path.is_file(),
        "norm_stats_sha256": support.sha256_file(norm_stats_path) if norm_stats_path.is_file() else None,
    }
    _CHECKPOINT_INVENTORY_CACHE[cache_key] = inventory
    return dict(inventory)


def _runtime_provenance(
    result_dir: pathlib.Path,
    benchmark: str,
    benchmark_commit: str,
    source_count: int,
) -> Dict[str, Any]:
    run_id, manifest_paths, manifests = _runtime_manifests(result_dir)
    git_manifest = manifests["git"]
    checkpoint = manifests["checkpoint-server"]
    sources = manifests["sources"]
    preflight = manifests["preflight"]
    environment = manifests["environment"]
    _require(git_manifest["openpi"].get("commit") == EXPECTED_OPENPI_COMMIT, "%s OpenPI commit disagrees" % result_dir)
    _require(git_manifest["benchmark"].get("commit") == benchmark_commit, "%s benchmark commit disagrees" % result_dir)
    for name, expected_commit in (("openpi", EXPECTED_OPENPI_COMMIT), ("benchmark", benchmark_commit)):
        recorded = git_manifest[name]
        current = _current_git_state(pathlib.Path(recorded["path"]))
        _require(current["commit"] == expected_commit, "%s current %s commit disagrees" % (result_dir, name))
        _require(
            _normalized_status(current["status"]) == _normalized_status(recorded.get("status")),
            "%s current %s worktree differs from the run manifest" % (result_dir, name),
        )
    _require(
        checkpoint.get("benchmark") == benchmark
        and checkpoint.get("protocol") == "official"
        and checkpoint.get("config") == "pi05_libero"
        and checkpoint.get("asset_id") == "physical-intelligence/libero"
        and checkpoint.get("bind_host") == "127.0.0.1"
        and checkpoint.get("norm_stats_sha256") == EXPECTED_NORM_STATS_SHA256,
        "%s checkpoint provenance disagrees" % result_dir,
    )
    checkpoint_inventory = _checkpoint_inventory(checkpoint)
    _require(
        checkpoint_inventory["params_source_matches"] is True
        and checkpoint_inventory["param_files"] == EXPECTED_PARAMS_FILES
        and checkpoint_inventory["param_bytes"] == EXPECTED_PARAMS_BYTES
        and checkpoint_inventory["params_tree_sha256"] == EXPECTED_PARAMS_TREE_SHA256
        and checkpoint_inventory["norm_stats_exists"] is True
        and checkpoint_inventory["norm_stats_sha256"] == EXPECTED_NORM_STATS_SHA256,
        "%s checkpoint files are incomplete" % result_dir,
    )
    _require(
        sources.get("benchmark") == benchmark
        and sources.get("source_count") == source_count
        and len(str(sources.get("inventory_sha256", ""))) == 64,
        "%s source inventory disagrees" % result_dir,
    )
    current_inventory = _current_source_inventory(
        benchmark, pathlib.Path(str(sources["benchmark_repo"]))
    )
    _require(
        current_inventory["source_count"] == sources["source_count"]
        and current_inventory["inventory_sha256"] == sources["inventory_sha256"],
        "%s current source inventory differs from the run manifest" % result_dir,
    )
    _require(
        preflight.get("status") == "passed"
        and preflight.get("benchmark") == benchmark
        and preflight.get("finite") is True
        and int(preflight.get("action_steps", 0)) >= 5
        and preflight.get("action_dimension") == 7
        and preflight.get("seed") == 7
        and preflight.get("resize") == 224
        and preflight.get("replan") == 5,
        "%s inference preflight disagrees" % result_dir,
    )
    fixed = environment.get("fixed_environment", {})
    _require(
        environment.get("benchmark") == benchmark
        and environment.get("phase") == "full"
        and fixed.get("MUJOCO_GL") == "egl"
        and fixed.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.75",
        "%s fixed environment disagrees" % result_dir,
    )
    return {
        "result_dir": str(result_dir.resolve()),
        "run_id": run_id,
        "openpi_commit": EXPECTED_OPENPI_COMMIT,
        "benchmark_commit": benchmark_commit,
        "checkpoint_config": checkpoint["config"],
        "checkpoint": checkpoint["checkpoint"],
        "norm_stats_sha256": checkpoint["norm_stats_sha256"],
        "checkpoint_param_files": checkpoint_inventory["param_files"],
        "checkpoint_param_bytes": checkpoint_inventory["param_bytes"],
        "checkpoint_params_tree_sha256": checkpoint_inventory["params_tree_sha256"],
        "source_count": sources["source_count"],
        "source_inventory_sha256": sources["inventory_sha256"],
        "current_source_inventory_sha256": current_inventory["inventory_sha256"],
        "openpi_worktree_status": list(_normalized_status(git_manifest["openpi"].get("status"))),
        "benchmark_worktree_status": list(_normalized_status(git_manifest["benchmark"].get("status"))),
        "preflight_action_steps": preflight["action_steps"],
        "preflight_action_dimension": preflight["action_dimension"],
        "manifests": {
            prefix: {"path": str(path), "sha256": support.sha256_file(path)}
            for prefix, path in manifest_paths.items()
        },
    }


def _harness_hashes() -> Dict[str, str]:
    root = pathlib.Path(__file__).resolve().parent
    paths = {name: root / name for name in HARNESS_FILES}
    _require(all(path.is_file() for path in paths.values()), "final audit harness source is incomplete")
    return {name: support.sha256_file(path) for name, path in paths.items()}


def _exact_video_inventory(
    result_dir: pathlib.Path, records: Sequence[Mapping[str, Any]], planned: int
) -> Dict[str, Any]:
    policy_records = [record for record in records if record.get("status") in ("success", "failure")]
    record_paths = [pathlib.Path(str(record.get("video", ""))).resolve() for record in policy_records]
    inventory_entries = list((result_dir / "videos").glob("*.mp4"))
    inventory_paths = [path.resolve() for path in inventory_entries]
    _require(len(policy_records) == planned, "%s policy video records are incomplete" % result_dir)
    _require(
        all(
            record.get("video_status") == "written"
            and path.is_file()
            and path.stat().st_size > 0
            for record, path in zip(policy_records, record_paths)
        ),
        "%s record videos are missing or empty" % result_dir,
    )
    _require(
        len(inventory_entries) == planned
        and all(path.is_file() and path.stat().st_size > 0 for path in inventory_entries),
        "%s video directory inventory is incomplete" % result_dir,
    )
    _require(
        len(set(record_paths)) == planned and len(set(inventory_paths)) == planned,
        "%s video paths are not unique" % result_dir,
    )
    _require(
        set(record_paths) == set(inventory_paths),
        "%s record video paths do not equal the video inventory" % result_dir,
    )
    return {
        "record_paths": len(record_paths),
        "inventory_entries": len(inventory_entries),
        "unique_resolved_paths": len(set(record_paths)),
        "missing_or_empty": 0,
        "exact_set_match": True,
    }


def _record_audit(result_dir: pathlib.Path, planned: int) -> Dict[str, Any]:
    path = result_dir / "episodes.jsonl"
    records = support.load_jsonl(path)
    episode_ids = [str(record.get("episode_id", "")) for record in records]
    attempt_ids = [str(record.get("attempt_id", "")) for record in records]
    videos = [path.resolve() for path in (result_dir / "videos").glob("*.mp4")]
    record_videos = [pathlib.Path(str(record.get("video", ""))).resolve() for record in records]
    _require(len(records) == planned, "%s has %d records, expected %d" % (result_dir, len(records), planned))
    _require(len(set(episode_ids)) == planned and all(episode_ids), "%s episode IDs are not unique" % result_dir)
    _require(len(set(attempt_ids)) == planned and all(attempt_ids), "%s attempt IDs are not unique" % result_dir)
    _require(all(record.get("status") in ("success", "failure") for record in records), "%s has non-policy outcomes" % result_dir)
    _require(len(videos) == planned and all(video.stat().st_size > 0 for video in videos), "%s videos are incomplete" % result_dir)
    _require(
        all(
            record.get("video_status") == "written"
            and video.is_file()
            and video.stat().st_size > 0
            for record, video in zip(records, record_videos)
        ),
        "%s record video paths are incomplete" % result_dir,
    )
    _require(len(set(record_videos)) == planned, "%s record video paths are not unique" % result_dir)
    _require(set(record_videos) == set(videos), "%s record video paths do not equal the video inventory" % result_dir)
    successes = sum(record.get("status") == "success" for record in records)
    observed_suites = sorted({str(record.get("suite")) for record in records})
    suite_records = {
        suite: [record for record in records if str(record.get("suite")) == suite]
        for suite in observed_suites
    }
    suite_evidence = {
        suite: {
            "planned": len(items),
            "successes": sum(record.get("status") == "success" for record in items),
            "failures": sum(record.get("status") == "failure" for record in items),
            "success_rate": sum(record.get("status") == "success" for record in items) / len(items),
        }
        for suite, items in suite_records.items()
    }
    macro_success_rate = sum(item["success_rate"] for item in suite_evidence.values()) / len(suite_evidence)
    return {
        "records": len(records),
        "unique_episode_ids": len(set(episode_ids)),
        "unique_attempt_ids": len(set(attempt_ids)),
        "videos": len(videos),
        "unique_record_video_paths": len(set(record_videos)),
        "successes": successes,
        "failures": planned - successes,
        "success_rate": float(successes) / planned,
        "macro_suite_success_rate": macro_success_rate,
        "suites": suite_evidence,
        "episodes_jsonl_sha256": support.sha256_file(path),
    }


def _old_matrix_audit(result_dir: pathlib.Path, expected_protocol: str) -> Dict[str, Any]:
    records = support.load_jsonl(result_dir / "episodes.jsonl")
    expected_keys = {
        (suite, task_id, trial)
        for suite in support.SUITES
        for task_id in range(10)
        for trial in range(50)
    }
    actual_keys = []
    descriptions: Dict[tuple, set] = {}
    for record in records:
        suite = str(record.get("suite", ""))
        task_id = record.get("task_id")
        trial = record.get("trial")
        key = (suite, task_id, trial)
        actual_keys.append(key)
        description = str(record.get("task_description", "")).strip()
        descriptions.setdefault((suite, task_id), set()).add(description)
        expected_episode_id = "%s:%s:task-%02d:trial-%02d:seed-7" % (
            expected_protocol,
            suite,
            task_id if isinstance(task_id, int) else -1,
            trial if isinstance(trial, int) else -1,
        )
        _require(
            record.get("protocol") == expected_protocol
            and record.get("phase") == "full"
            and record.get("seed") == 7
            and record.get("resize") == 224
            and record.get("replan") == 5
            and record.get("max_steps") == support.MAX_STEPS.get(suite)
            and record.get("wait_steps") == 10
            and record.get("gl_backend") == "egl"
            and record.get("attempt") == 0
            and record.get("episode_id") == expected_episode_id
            and record.get("attempt_id") == expected_episode_id + ":attempt-00"
            and bool(description)
            and record.get("success") is (record.get("status") == "success")
            and record.get("error") is None
            and record.get("error_category") is None,
            "%s has a record outside the fixed LIBERO protocol" % result_dir,
        )
    _require(
        len(actual_keys) == len(expected_keys)
        and len(set(actual_keys)) == len(expected_keys)
        and set(actual_keys) == expected_keys,
        "%s is not the exact four-suite, ten-task, fifty-trial matrix" % result_dir,
    )
    _require(
        set(descriptions) == {(suite, task_id) for suite in support.SUITES for task_id in range(10)}
        and all(len(values) == 1 and "" not in values for values in descriptions.values()),
        "%s task descriptions are missing or inconsistent across trials" % result_dir,
    )
    return {
        "suite_count": len(support.SUITES),
        "tasks_per_suite": 10,
        "trials_per_task": 50,
        "matrix_keys": len(expected_keys),
        "matrix_keys_sha256": _digest_lines(
            ["%s:%02d:%02d" % key for key in sorted(expected_keys)]
        ),
        "fixed_protocol": expected_protocol,
        "fixed_phase": "full",
        "fixed_seed": 7,
        "fixed_resize": 224,
        "fixed_replan": 5,
        "fixed_wait_steps": 10,
        "fixed_gl_backend": "egl",
        "max_steps_by_suite": dict(support.MAX_STEPS),
    }


def _old_runtime_manifests(result_dir: pathlib.Path) -> tuple:
    directory = result_dir / "manifests"
    environments = sorted(directory.glob("environment-*.json"))
    _require(bool(environments), "%s old environment manifest is missing" % result_dir)
    anchor = environments[-1].resolve()
    run_id = anchor.name[len("environment-") : -len(".json")]
    _require(bool(run_id), "%s old environment manifest has no run ID" % result_dir)
    prefixes = (
        "checkpoint-request",
        "checkpoint-server",
        "environment",
        "git",
        "preflight-egl",
        "result",
        "server-startup",
    )
    paths = {}
    payloads = {}
    for prefix in prefixes:
        path = (directory / (prefix + "-" + run_id + ".json")).resolve()
        _require(path.is_file(), "%s old %s manifest for run %s is missing" % (result_dir, prefix, run_id))
        paths[prefix] = path
        payloads[prefix] = json.loads(path.read_text(encoding="utf-8"))
    for prefix in ("environment", "result", "server-startup"):
        _require(
            payloads[prefix].get("run_id") == run_id,
            "%s old %s manifest run ID disagrees" % (result_dir, prefix),
        )
    return run_id, paths, payloads


def _cli_values(argv: Sequence[Any]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    items = [str(value) for value in argv]
    _require(bool(items) and items[0] == "examples/libero/run_pi05_libero.py", "old launcher argv disagrees")
    index = 1
    while index < len(items):
        option = items[index]
        _require(option.startswith("--"), "old launcher argv has a positional argument")
        if option == "--resume":
            values[option] = "true"
            index += 1
            continue
        _require(index + 1 < len(items), "old launcher argv has a missing value")
        _require(option not in values, "old launcher argv repeats %s" % option)
        values[option] = items[index + 1]
        index += 2
    return values


def _old_runtime_provenance(result_dir: pathlib.Path, expected_protocol: str) -> Dict[str, Any]:
    run_id, manifest_paths, manifests = _old_runtime_manifests(result_dir)
    request = manifests["checkpoint-request"]
    checkpoint = manifests["checkpoint-server"]
    environment = manifests["environment"]
    git_manifest = manifests["git"]
    preflight = manifests["preflight-egl"]
    result = manifests["result"]
    startup = manifests["server-startup"]
    shared_checkpoint_fields = (
        "protocol",
        "config",
        "asset_id",
        "checkpoint",
        "official_checkpoint",
        "base_checkpoint",
        "params_source",
        "assets_source",
        "norm_stats",
        "norm_stats_sha256",
        "view_dir",
    )
    _require(
        all(request.get(key) == checkpoint.get(key) for key in shared_checkpoint_fields),
        "%s old checkpoint request and server manifests disagree" % result_dir,
    )
    official_checkpoint = (result_dir.parent / "checkpoint-staging" / "pi05_libero").resolve()
    norm_stats = official_checkpoint / support.NORM_STATS_RELATIVE
    _require(
        checkpoint.get("protocol") == expected_protocol
        and checkpoint.get("config") == "pi05_libero"
        and checkpoint.get("asset_id") == "physical-intelligence/libero"
        and checkpoint.get("bind_host") == "127.0.0.1"
        and pathlib.Path(str(checkpoint.get("official_checkpoint", ""))).resolve() == official_checkpoint
        and pathlib.Path(str(checkpoint.get("assets_source", ""))).resolve() == (official_checkpoint / "assets")
        and pathlib.Path(str(checkpoint.get("norm_stats", ""))).resolve() == norm_stats
        and checkpoint.get("norm_stats_sha256") == EXPECTED_NORM_STATS_SHA256
        and norm_stats.is_file()
        and support.sha256_file(norm_stats) == EXPECTED_NORM_STATS_SHA256,
        "%s old checkpoint provenance disagrees" % result_dir,
    )
    inventory = _checkpoint_inventory(checkpoint)
    params_link = pathlib.Path(str(checkpoint.get("checkpoint", ""))) / "params"
    assets_link = pathlib.Path(str(checkpoint.get("checkpoint", ""))) / "assets"
    if expected_protocol == "official":
        _require(
            checkpoint.get("base_checkpoint") is None
            and checkpoint.get("view_dir") is None
            and pathlib.Path(str(checkpoint.get("checkpoint", ""))).resolve() == official_checkpoint
            and params_link.resolve() == pathlib.Path(str(checkpoint.get("params_source", ""))).resolve()
            and assets_link.resolve() == pathlib.Path(str(checkpoint.get("assets_source", ""))).resolve()
            and inventory["param_files"] == EXPECTED_PARAMS_FILES
            and inventory["param_bytes"] == EXPECTED_PARAMS_BYTES
            and inventory["params_tree_sha256"] == EXPECTED_PARAMS_TREE_SHA256,
            "%s old official parameters disagree" % result_dir,
        )
    else:
        base_checkpoint = pathlib.Path(str(checkpoint.get("base_checkpoint", ""))).resolve()
        view_dir = pathlib.Path(str(checkpoint.get("view_dir", ""))).resolve()
        view_links = checkpoint.get("view_links", {})
        _require(
            expected_protocol == "base-libero-assets"
            and base_checkpoint.name == "pi05_base"
            and pathlib.Path(str(checkpoint.get("checkpoint", ""))).resolve() == view_dir
            and view_dir.parent == (result_dir / "runtime" / run_id).resolve()
            and params_link.is_symlink()
            and assets_link.is_symlink()
            and params_link.resolve() == (base_checkpoint / "params").resolve()
            and assets_link.resolve() == (official_checkpoint / "assets").resolve()
            and pathlib.Path(str(checkpoint.get("params_source", ""))).resolve() == params_link.resolve()
            and pathlib.Path(str(view_links.get("params", ""))).resolve() == params_link.resolve()
            and pathlib.Path(str(view_links.get("assets", ""))).resolve() == assets_link.resolve()
            and inventory["param_files"] == EXPECTED_BASE_PARAMS_FILES
            and inventory["param_bytes"] == EXPECTED_BASE_PARAMS_BYTES
            and inventory["params_tree_sha256"] == EXPECTED_BASE_PARAMS_TREE_SHA256,
            "%s old Base plus LIBERO-assets checkpoint view disagrees" % result_dir,
        )
    _require(
        git_manifest.get("actual_head") == EXPECTED_OPENPI_COMMIT
        and git_manifest.get("required_upstream_commit") == "15a9616a"
        and git_manifest.get("required_commit_available_locally") is True
        and git_manifest.get("head_error") is None
        and git_manifest.get("status_error") is None
        and isinstance(git_manifest.get("worktree_status"), list),
        "%s old OpenPI git manifest disagrees" % result_dir,
    )
    openpi_repo = result_dir.parents[3]
    # The old launcher recorded default git-status output, which collapses an
    # untracked directory to one entry.  Match using that same representation.
    current_git = _current_git_state(openpi_repo, untracked_files_all=False)
    _require(
        current_git["commit"] == EXPECTED_OPENPI_COMMIT
        and _normalized_status(current_git["status"]) == _normalized_status("\n".join(git_manifest["worktree_status"])),
        "%s current OpenPI tree differs from the old run manifest" % result_dir,
    )
    fixed = environment.get("fixed_environment", {})
    argv = _cli_values(environment.get("argv", []))
    _require(
        environment.get("protocol") == expected_protocol
        and environment.get("phase") == "full"
        and int(environment.get("port")) == int(checkpoint.get("port"))
        and fixed.get("CUDA_VISIBLE_DEVICES") == str(environment.get("gpu_id"))
        and fixed.get("MUJOCO_EGL_DEVICE_ID") == str(environment.get("gpu_id"))
        and fixed.get("MUJOCO_GL") == "egl"
        and fixed.get("PYOPENGL_PLATFORM") == "egl"
        and fixed.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.75"
        and argv.get("--protocol") == expected_protocol
        and argv.get("--phase") == "full"
        and int(argv.get("--port", -1)) == int(environment.get("port"))
        and argv.get("--gpu-id") == str(environment.get("gpu_id"))
        and pathlib.Path(argv.get("--checkpoint-dir", "")).resolve() == official_checkpoint.parent
        and pathlib.Path(argv.get("--output-dir", "")).resolve() == result_dir.resolve(),
        "%s old fixed environment or launcher argv disagrees" % result_dir,
    )
    _require(
        preflight.get("status") == "passed"
        and preflight.get("protocol") == expected_protocol
        and preflight.get("finite") is True
        and int(preflight.get("action_steps", 0)) >= 5
        and preflight.get("action_dimension") == 7,
        "%s old inference preflight disagrees" % result_dir,
    )
    event_status = {event.get("stage"): event for event in result.get("events", [])}
    recorded_preflight = result.get("preflight_manifests", {}).get("egl", {})
    _require(
        result.get("status") == "passed"
        and result.get("exit_code") == 0
        and result.get("phase") == "full"
        and result.get("protocol") == expected_protocol
        and result.get("policy_success_summary_created") is True
        and result.get("error") is None
        and result.get("error_category") is None
        and result.get("trace") is None
        and recorded_preflight.get("exists") is True
        and pathlib.Path(str(recorded_preflight.get("path", ""))).resolve()
        == manifest_paths["preflight-egl"]
        and result.get("preflight_manifests", {}).get("glx", {}).get("exists") is False
        and pathlib.Path(str(result.get("server_startup_manifest", ""))).resolve()
        == manifest_paths["server-startup"]
        and set(event_status) == {"checkpoint_validation", "server_startup", "client_egl"}
        and all(event.get("status") == "passed" for event in event_status.values())
        and event_status["client_egl"].get("exit_code") == 0
        and pathlib.Path(str(event_status["client_egl"].get("preflight_manifest", ""))).resolve()
        == manifest_paths["preflight-egl"],
        "%s old result manifest disagrees" % result_dir,
    )
    _require(
        startup.get("status") == "passed"
        and startup.get("error") is None
        and startup.get("error_category") is None
        and startup.get("diagnostic_manifest") is None,
        "%s old server startup manifest disagrees" % result_dir,
    )
    return {
        "run_id": run_id,
        "openpi_commit": EXPECTED_OPENPI_COMMIT,
        "protocol": expected_protocol,
        "checkpoint_config": checkpoint["config"],
        "checkpoint": checkpoint["checkpoint"],
        "params_source": checkpoint["params_source"],
        "assets_source": checkpoint["assets_source"],
        "norm_stats_sha256": checkpoint["norm_stats_sha256"],
        "checkpoint_param_files": inventory["param_files"],
        "checkpoint_param_bytes": inventory["param_bytes"],
        "checkpoint_params_tree_sha256": inventory["params_tree_sha256"],
        "bind_host": checkpoint["bind_host"],
        "preflight_action_steps": preflight["action_steps"],
        "preflight_action_dimension": preflight["action_dimension"],
        "manifests": {
            prefix: {"path": str(path), "sha256": support.sha256_file(path)}
            for prefix, path in manifest_paths.items()
        },
    }


def _old_full_row(
    eval_root: pathlib.Path, directory_name: str, benchmark: str, protocol: str
) -> Dict[str, Any]:
    result_dir = eval_root / directory_name
    expected_protocol = "official" if protocol == "pi05_libero" else "base-libero-assets"
    evidence = _record_audit(result_dir, 2000)
    matrix_evidence = _old_matrix_audit(result_dir, expected_protocol)
    provenance = _old_runtime_provenance(result_dir, expected_protocol)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    total = summary["total"]
    suites = summary["suites"]
    _require(
        total.get("complete") is True
        and total.get("planned") == 2000
        and total.get("attempted_unique") == 2000
        and total.get("policy_denominator") == 2000
        and total.get("excluded_errors") == 0
        and total.get("unknown_errors") == 0,
        "%s summary is incomplete" % directory_name,
    )
    expected_threshold = {
        "applies": expected_protocol == "official",
        "passed": True if expected_protocol == "official" else None,
        "required": OFFICIAL_MIN_SUCCESS_RATE,
    }
    _require(
        summary.get("protocol") == expected_protocol
        and summary.get("phase") == "full"
        and summary.get("required_commit") == "15a9616a"
        and summary.get("official_full_threshold") == expected_threshold,
        "%s summary protocol provenance disagrees" % directory_name,
    )
    _require(
        total.get("successes") == evidence["successes"]
        and total.get("failures") == evidence["failures"]
        and abs(float(total.get("success_rate")) - evidence["success_rate"]) < 1e-12,
        "%s total summary disagrees with episode records" % directory_name,
    )
    _require(set(suites) == set(support.SUITES), "%s suite summary is incomplete" % directory_name)
    _require(
        set(evidence["suites"]) == set(support.SUITES)
        and all(item["planned"] == 500 for item in evidence["suites"].values()),
        "%s episode suite inventory is not four groups of 500" % directory_name,
    )
    _require(
        all(
            item.get("complete") is True
            and item.get("planned") == 500
            and item.get("attempted_unique") == 500
            and item.get("policy_denominator") == 500
            and item.get("excluded_errors") == 0
            and item.get("unknown_errors") == 0
            for item in suites.values()
        ),
        "%s suite summary has incomplete or infrastructure-error outcomes" % directory_name,
    )
    computed_macro = evidence["macro_suite_success_rate"]
    macro_success_rate = float(total["macro_success_rate"])
    _require(abs(computed_macro - macro_success_rate) < 1e-12, "%s macro success rate disagrees" % directory_name)
    _require(
        all(
            item.get("successes") == evidence["suites"][suite]["successes"]
            and item.get("failures") == evidence["suites"][suite]["failures"]
            and abs(float(item.get("success_rate")) - evidence["suites"][suite]["success_rate"]) < 1e-12
            for suite, item in suites.items()
        ),
        "%s suite summary disagrees with episode records" % directory_name,
    )
    return {
        "benchmark": benchmark,
        "protocol": protocol,
        "status": "complete",
        "protocol_applicable": True,
        "protocol_planned": 2000,
        "evaluated_planned": 2000,
        "successes": evidence["successes"],
        "failures": evidence["failures"],
        "success_rate": evidence["success_rate"],
        "macro_suite_success_rate": macro_success_rate,
        "infrastructure_errors": 0,
        "note": "official threshold passed" if protocol == "pi05_libero" else "no success-rate threshold",
        "result_dir": str(result_dir),
        "evidence": dict(
            evidence,
            summary_suites=suites,
            matrix=matrix_evidence,
            runtime_provenance=provenance,
        ),
    }


def _base_native_row(eval_root: pathlib.Path) -> Dict[str, Any]:
    manifests = sorted((eval_root / "base-native-smoke" / "manifests").glob("result-*.json"))
    _require(bool(manifests), "base-native diagnostic result is missing")
    result = json.loads(manifests[-1].read_text(encoding="utf-8"))
    error = str(result.get("error", ""))
    _require(result.get("error_category") == "checkpoint", "base-native failure is not checkpoint-classified")
    _require("norm_stats" in error and not result.get("policy_success_summary_created"), "base-native N/A evidence is incomplete")
    return {
        "benchmark": "LIBERO",
        "protocol": "pi05_base/native-assets",
        "status": "not_applicable",
        "protocol_applicable": False,
        "protocol_planned": 2000,
        "evaluated_planned": 0,
        "successes": None,
        "failures": None,
        "success_rate": None,
        "infrastructure_errors": 0,
        "note": "required physical-intelligence/libero norm_stats absent; not a zero-success result",
        "result_dir": str(eval_root / "base-native-smoke"),
        "evidence": {"result_manifest": str(manifests[-1]), "error": error},
    }


def _matrix_row(
    benchmark: str,
    protocol: str,
    result_dir: pathlib.Path,
    matrix: Sequence[support.EpisodeSpec],
    protocol_planned: int,
    applicable: bool,
    note: str,
) -> Dict[str, Any]:
    records = support.load_jsonl(result_dir / "episodes.jsonl")
    summary = support.aggregate(records, matrix)
    integrity = support.verify_integrity(records, matrix, require_videos=True)
    total = summary["total"]
    infrastructure_errors = total["excluded_errors"] + total["unknown_errors"]
    historical_error_attempts = sum(record.get("status") == "error" for record in records)
    _require(total["complete"], "%s matrix is incomplete" % benchmark)
    _require(integrity["passed"], "%s integrity failed: %s" % (benchmark, integrity["issues"]))
    _require(total["extra_episode_records"] == 0, "%s has extra episodes" % benchmark)
    video_inventory = _exact_video_inventory(result_dir, records, len(matrix))
    merge_path = result_dir / "manifests" / "merge.json"
    _require(merge_path.is_file(), "%s merge manifest is missing" % benchmark)
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    episodes_sha256 = support.sha256_file(result_dir / "episodes.jsonl")
    matrix_sha256 = _digest_lines([spec.episode_id for spec in matrix])
    _require(
        merge.get("status") == "passed"
        and merge.get("planned_episodes") == len(matrix)
        and merge.get("merged_records") == len(records)
        and merge.get("merged_policy_outcomes") == len(matrix)
        and merge.get("merged_episodes_jsonl_sha256") == episodes_sha256
        and merge.get("matrix_episode_ids_sha256") == matrix_sha256
        and merge.get("suite_counts") == dict(support.PLUS_COUNTS)
        and merge.get("environment_seed") == 7
        and merge.get("policy_rng_initial_key") == 0
        and merge.get("policy_rng_scope") == "server_process/per-suite"
        and merge.get("integrity", {}).get("passed") is True,
        "%s merge provenance disagrees" % benchmark,
    )
    source_entries = merge.get("sources", {})
    _require(set(source_entries) == set(support.SUITES), "%s merge sources are incomplete" % benchmark)
    provenance_by_directory: Dict[str, Any] = {}
    for suite, entry in source_entries.items():
        directory = pathlib.Path(str(entry.get("directory", ""))).resolve()
        _require(directory.is_dir(), "%s source directory is missing: %s" % (suite, directory))
        _require(
            entry.get("episodes_jsonl_sha256") == support.sha256_file(directory / "episodes.jsonl"),
            "%s source episode hash disagrees" % suite,
        )
        key = str(directory)
        if key not in provenance_by_directory:
            provenance_by_directory[key] = _runtime_provenance(
                directory, "plus", EXPECTED_PLUS_COMMIT, support.PLUS_FULL_EPISODES
            )
    return {
        "benchmark": benchmark,
        "protocol": protocol,
        "status": "complete" if applicable else "partial_incompatible",
        "protocol_applicable": applicable,
        "protocol_planned": protocol_planned,
        "evaluated_planned": len(matrix),
        "successes": total["successes"],
        "failures": total["failures"],
        "success_rate": total["success_rate"],
        "infrastructure_errors": infrastructure_errors,
        "note": note,
        "result_dir": str(result_dir),
        "evidence": {
            "integrity": integrity,
            "groups": summary["groups"],
            "historical_error_attempts": historical_error_attempts,
            "video_inventory": video_inventory,
            "episodes_jsonl_sha256": episodes_sha256,
            "matrix_episode_ids_sha256": matrix_sha256,
            "merge_manifest": {"path": str(merge_path), "sha256": support.sha256_file(merge_path)},
            "runtime_provenance": provenance_by_directory,
        },
    }


def _pro_key(value: Any) -> tuple:
    if isinstance(value, support.EpisodeSpec):
        return (str(value.perturbation), value.suite, value.source_id, int(value.trial))
    return (
        str(value.get("perturbation")),
        str(value.get("suite")),
        str(value.get("source_id")),
        int(value.get("trial", -1)),
    )


def _require_pro_summary(
    summary: Mapping[str, Any],
    successes: int,
    failures: int,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    compatibility: Mapping[str, Any],
) -> None:
    planned = successes + failures
    total_summary = summary["total"]
    record_success_rate = float(successes) / planned
    suite_macro_success_rate = sum(
        group["success_rate"] for group in groups["suite"].values()
    ) / len(groups["suite"])
    _require(
        summary.get("benchmark") == "pro"
        and summary.get("protocol") == "official"
        and summary.get("phase") == "full"
        and summary.get("seed") == 7
        and summary.get("resize") == 224
        and summary.get("replan") == 5
        and total_summary.get("planned") == planned
        and total_summary.get("complete") is True
        and total_summary.get("policy_denominator") == planned
        and total_summary.get("successes") == successes
        and total_summary.get("failures") == failures
        and abs(float(total_summary.get("success_rate")) - record_success_rate) < 1e-12
        and abs(float(total_summary.get("macro_suite_success_rate")) - suite_macro_success_rate) < 1e-12
        and total_summary.get("excluded_errors") == 0
        and total_summary.get("unknown_errors") == 0
        and summary.get("integrity", {}).get("passed") is True
        and summary.get("compatibility") == compatibility,
        "PRO summary disagrees with records or compatibility",
    )
    for dimension, expected_groups in groups.items():
        observed_groups = summary.get("groups", {}).get(dimension, {})
        _require(set(observed_groups) == set(expected_groups), "PRO %s groups disagree" % dimension)
        _require(
            all(
                observed_groups[value].get("planned") == expected["planned"]
                and observed_groups[value].get("successes") == expected["successes"]
                and observed_groups[value].get("failures") == expected["failures"]
                and abs(
                    float(observed_groups[value].get("success_rate"))
                    - expected["success_rate"]
                ) < 1e-12
                and observed_groups[value].get("complete") is True
                for value, expected in expected_groups.items()
            ),
            "PRO %s group counts disagree" % dimension,
        )


def _pro_row(
    result_dir: pathlib.Path,
    matrix: Sequence[support.EpisodeSpec],
    compatibility: Mapping[str, Any],
) -> Dict[str, Any]:
    records = support.load_jsonl(result_dir / "episodes.jsonl")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    planned_keys = [_pro_key(spec) for spec in matrix]
    record_keys = [_pro_key(record) for record in records]
    attempt_ids = [str(record.get("attempt_id", "")) for record in records]
    episode_ids = [str(record.get("episode_id", "")) for record in records]
    _require(len(planned_keys) == 8000 and len(set(planned_keys)) == 8000, "PRO canonical matrix keys are not 8000 unique entries")
    _require(len(records) == 8000, "PRO has %d records, expected 8000" % len(records))
    _require(len(set(record_keys)) == 8000, "PRO canonical record keys are not unique")
    _require(set(record_keys) == set(planned_keys), "PRO records do not equal the official available matrix")
    _require(len(set(attempt_ids)) == 8000 and all(attempt_ids), "PRO attempt IDs are not unique")
    _require(len(set(episode_ids)) == 8000 and all(episode_ids), "PRO episode IDs are not unique")
    _require(all(record.get("status") in ("success", "failure") for record in records), "PRO has non-policy outcomes")
    video_inventory = _exact_video_inventory(result_dir, records, 8000)
    successes = sum(record.get("status") == "success" for record in records)
    failures = 8000 - successes
    groups: Dict[str, Dict[str, Any]] = {}
    for dimension in ("suite", "perturbation"):
        values = sorted({str(record.get(dimension)) for record in records})
        groups[dimension] = {}
        for value in values:
            selected = [record for record in records if str(record.get(dimension)) == value]
            group_successes = sum(record.get("status") == "success" for record in selected)
            groups[dimension][value] = {
                "planned": len(selected),
                "successes": group_successes,
                "failures": len(selected) - group_successes,
                "success_rate": float(group_successes) / len(selected),
                "complete": True,
            }
    _require_pro_summary(summary, successes, failures, groups, compatibility)
    runtime_provenance = _runtime_provenance(
        result_dir, "pro", EXPECTED_PRO_COMMIT, len(matrix) // support.PRO_TRIALS
    )
    canonical_keys_sha256 = _digest_lines(
        [json.dumps(key, separators=(",", ":")) for key in sorted(planned_keys)]
    )
    return {
        "benchmark": "LIBERO-Pro",
        "protocol": "pi05_libero",
        "status": "partial_incompatible",
        "protocol_applicable": False,
        "protocol_planned": 10000,
        "evaluated_planned": 8000,
        "successes": successes,
        "failures": failures,
        "success_rate": float(successes) / 8000,
        "infrastructure_errors": 0,
        "note": "8000 available episodes complete; 2000 env episodes N/A because official cells are absent",
        "result_dir": str(result_dir),
        "evidence": {
            "canonical_key": ["perturbation", "suite", "source_id", "trial"],
            "canonical_keys": 8000,
            "unique_attempt_ids": len(set(attempt_ids)),
            "unique_episode_ids": len(set(episode_ids)),
            "videos": 8000,
            "unique_video_paths": video_inventory["unique_resolved_paths"],
            "missing_videos": video_inventory["missing_or_empty"],
            "video_inventory": video_inventory,
            "groups": groups,
            "summary_json_sha256": support.sha256_file(result_dir / "summary.json"),
            "episodes_jsonl_sha256": support.sha256_file(result_dir / "episodes.jsonl"),
            "canonical_keys_sha256": canonical_keys_sha256,
            "runtime_provenance": runtime_provenance,
            "note": "task_id is registry-order metadata and is deliberately excluded from canonical identity",
        },
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    eval_root = args.openpi_eval_root.expanduser().resolve()
    plus_repo = args.plus_repo.expanduser().resolve()
    plus_result = args.plus_result.expanduser().resolve()
    pro_data = args.pro_data.expanduser().resolve()
    pro_result = args.pro_result.expanduser().resolve()

    plus_sources = support.discover_sources("plus", plus_repo)
    plus_matrix = support.expand_matrix("plus", "full", plus_sources)
    pro_sources = support.discover_sources("pro", pro_data, allow_missing_pro_cells=True)
    compatibility = support.pro_compatibility(pro_sources)
    _require(compatibility["available_planned_episodes"] == 8000, "PRO available matrix is not 8000")
    _require(compatibility["unavailable_planned_episodes"] == 2000, "PRO N/A matrix is not 2000")
    _require({cell["perturbation"] for cell in compatibility["unavailable_cells"]} == {"env"}, "PRO missing cells are not exactly env")
    pro_matrix = support.expand_matrix("pro", "full", pro_sources, allow_incompatible_pro=True)

    rows = [
        _old_full_row(eval_root, "official-full", "LIBERO", "pi05_libero"),
        _old_full_row(eval_root, "base-libero-assets-full", "LIBERO", "pi05_base/LIBERO-assets"),
        _base_native_row(eval_root),
        _matrix_row(
            "LIBERO-Plus", "pi05_libero/multi-server-suite-sharded", plus_result,
            plus_matrix, 10030, True,
            "environment seed 7; policy RNG key 0 scoped independently per suite server",
        ),
        _pro_row(pro_result, pro_matrix, compatibility),
    ]
    _require_official_threshold(rows[0]["macro_suite_success_rate"])
    _require(all(row["infrastructure_errors"] == 0 for row in rows), "an evaluated protocol has infrastructure errors")
    return {
        "status": "passed",
        "created_at": _utc_now(),
        "rows": rows,
        "final_audit_time_harness_sources_sha256": _harness_hashes(),
        "pro_compatibility": compatibility,
        "acceptance": {
            "old_official_required_success_rate": OFFICIAL_MIN_SUCCESS_RATE,
            "old_official_actual_success_rate": rows[0]["success_rate"],
            "old_official_macro_suite_success_rate": rows[0]["macro_suite_success_rate"],
            "old_official_passed": rows[0]["macro_suite_success_rate"] >= OFFICIAL_MIN_SUCCESS_RATE,
            "plus_full_matrix_complete": rows[3]["evaluated_planned"] == 10030,
            "pro_available_matrix_complete": rows[4]["evaluated_planned"] == 8000,
            "no_infrastructure_errors": True,
        },
    }


def _format_rate(value: Optional[float]) -> str:
    return "N/A" if value is None else "%.2f%%" % (100.0 * value)


def _format_value(value: Any) -> str:
    return "N/A" if value is None else str(value)


def write_report(report: Mapping[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("audit output already exists: %s" % output_dir)
    staging = output_dir.parent / (".%s.tmp-%s" % (output_dir.name, uuid.uuid4().hex[:8]))
    staging.mkdir(parents=True)
    try:
        support.atomic_write_json(staging / "report.json", report)
        columns = [
            "benchmark", "protocol", "status", "protocol_applicable", "protocol_planned",
            "evaluated_planned", "successes", "failures", "success_rate",
            "macro_suite_success_rate", "infrastructure_errors", "note", "result_dir",
        ]
        with (staging / "report.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["rows"])
        lines = [
            "# OpenPI LIBERO evaluation audit",
            "",
            "Status: **passed**",
            "",
            "| Benchmark | Protocol | Status | Evaluated / Protocol | Successes | Failures | Success rate | Infra errors | Note |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in report["rows"]:
            lines.append(
                "| {benchmark} | {protocol} | {status} | {evaluated_planned} / {protocol_planned} | {successes} | {failures} | {rate} | {infrastructure_errors} | {note} |".format(
                    successes=_format_value(row["successes"]),
                    failures=_format_value(row["failures"]),
                    rate=_format_rate(row["success_rate"]),
                    **{key: value for key, value in row.items() if key not in ("successes", "failures")}
                )
            )
        lines.extend(
            [
                "",
                "Official LIBERO macro-suite acceptance: {actual} >= {required} (**passed**).".format(
                    actual=_format_rate(report["acceptance"]["old_official_macro_suite_success_rate"]),
                    required=_format_rate(report["acceptance"]["old_official_required_success_rate"]),
                ),
                "LIBERO-Pro's unavailable 2,000 environment episodes are N/A and are not converted into failures.",
                "LIBERO-Plus uses environment seed 7; policy sampling uses an independent JAX key-0 stream per suite server.",
                "",
            ]
        )
        (staging / "report.md").write_text("\n".join(lines), encoding="utf-8")
        os.replace(str(staging), str(output_dir))
    except Exception:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args)
        output = write_report(report, args.output_dir)
    except Exception as exc:
        print("audit failed: %s: %s" % (exc.__class__.__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
