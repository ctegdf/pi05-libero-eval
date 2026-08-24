"""Python-3.8-compatible, resumable LIBERO-Plus/Pro rollout client."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import logging
import math
import pathlib
import re
import socket
import sys
import time
import traceback
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pi05_eval_support as support


DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
RESIZE = 224
REPLAN = 5
WAIT_STEPS = 10
SEED = 7
EGL_FAILURE_EXIT = 86


@dataclasses.dataclass
class EvaluationData:
    modules: Dict[str, Any]
    sources: List[support.TaskSource]
    matrix: List[support.EpisodeSpec]
    pending: List[support.EpisodeSpec]
    attempts: Dict[str, int]
    records_path: pathlib.Path


@dataclasses.dataclass
class RolloutOutcome:
    status: str = "failure"
    error_category: Optional[str] = None
    error: Optional[str] = None
    trace: Optional[str] = None
    stage: str = "environment reset"
    action_steps: int = 0


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=support.BENCHMARKS)
    parser.add_argument("--phase", required=True, choices=support.PHASES)
    parser.add_argument("--benchmark-repo", required=True, type=pathlib.Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--suite", action="append", choices=support.SUITES)
    parser.add_argument("--preflight-manifest", required=True, type=pathlib.Path)
    return parser


def _lazy_imports() -> Dict[str, Any]:
    import imageio
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    return {
        "imageio": imageio,
        "np": np,
        "benchmark": benchmark,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "image_tools": image_tools,
        "WebsocketClientPolicy": websocket_client_policy.WebsocketClientPolicy,
    }


def _registry_key(source: support.TaskSource) -> str:
    if source.benchmark == "plus":
        return source.suite
    suffix = {
        "object": "object",
        "swap(position)": "swap",
        "lan(semantic)": "lan",
        "task": "task",
        "env": "env",
    }[str(source.perturbation)]
    return "%s_%s" % (source.suite, suffix)


def _task_bddl_name(task: Any) -> str:
    value = getattr(task, "bddl_file", None)
    if not value:
        raise support.BenchmarkInventoryError("registered benchmark task has no bddl_file")
    return pathlib.Path(str(value)).name


def _verify_and_order_registry(
    sources: Sequence[support.TaskSource], modules: Mapping[str, Any]
) -> List[support.TaskSource]:
    registry = modules["benchmark"].get_benchmark_dict()
    grouped: Dict[str, List[support.TaskSource]] = {}
    for source in sources:
        grouped.setdefault(_registry_key(source), []).append(source)
    ordered: List[support.TaskSource] = []
    for key, items in sorted(grouped.items()):
        if key not in registry:
            raise support.BenchmarkInventoryError("required registered benchmark suite is missing: %s" % key)
        suite = registry[key]()
        if int(suite.n_tasks) != len(items):
            raise support.BenchmarkInventoryError(
                "registered suite %s has %d tasks, inventory has %d" % (key, int(suite.n_tasks), len(items))
            )
        by_name: Dict[str, List[support.TaskSource]] = {}
        for source in items:
            by_name.setdefault(source.bddl_path.name, []).append(source)
        for task_id in range(int(suite.n_tasks)):
            bddl_name = _task_bddl_name(suite.get_task(task_id))
            matches = by_name.get(bddl_name, [])
            if len(matches) != 1:
                raise support.BenchmarkInventoryError(
                    "registered suite %s task id %d BDDL %s maps to %d inventory entries"
                    % (key, task_id, bddl_name, len(matches))
                )
            source = matches[0]
            if source.benchmark == "plus" and source.task_id != task_id + 1:
                raise support.BenchmarkInventoryError(
                    "Plus classification id/order mismatch for %s: JSON id=%d registry index=%d (expected id=%d)"
                    % (source.source_id, source.task_id, task_id, task_id + 1)
                )
            # Keep Plus's official 1-based classification ID in all records;
            # only Pro needs the 0-based registered task index assigned here.
            ordered.append(
                source if source.benchmark == "plus" else dataclasses.replace(source, task_id=task_id)
            )
    allow_missing = sources[0].benchmark == "pro" and not support.pro_compatibility(ordered)["protocol_applicable"]
    support.validate_sources(sources[0].benchmark, ordered, allow_missing_pro_cells=allow_missing)
    return ordered


def _quat2axisangle(quat: Any, np: Any) -> Any:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def _make_env(spec: support.EpisodeSpec, modules: Mapping[str, Any]) -> Any:
    env = modules["OffScreenRenderEnv"](
        bddl_file_name=str(spec.bddl_path),
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
    )
    env.seed(SEED)
    return env


def _policy_observation(
    obs: Mapping[str, Any], prompt: str, modules: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Any]:
    np = modules["np"]
    image_tools = modules["image_tools"]
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"], np), obs["robot0_gripper_qpos"])
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }, image


def _connect(client_class: Any, port: int) -> Any:
    probe = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    probe.close()
    return client_class("127.0.0.1", port)


def _close_client(client: Any) -> str:
    try:
        public = getattr(client, "close", None)
        if callable(public):
            public()
            return "public"
        websocket = getattr(client, "_ws", None)
        private = getattr(websocket, "close", None)
        if callable(private):
            private()
            return "private_websocket"
    except Exception as exc:
        logging.warning("client cleanup failed: %s", exc)
        return "error"
    return "unavailable"


def _validate_metadata(metadata: Any, benchmark: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("server metadata is not a mapping")
    expected = {
        "evaluation_config": support.CONFIG_NAME,
        "evaluation_protocol": "official",
        "evaluation_benchmark": benchmark,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("server metadata %s mismatch: expected %r, got %r" % (key, value, metadata.get(key)))
    provenance = metadata.get("checkpoint_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("asset_id") != support.ASSET_ID:
        raise ValueError("server official checkpoint provenance is missing or invalid")


def _connect_validated(modules: Mapping[str, Any], args: argparse.Namespace) -> Any:
    client = _connect(modules["WebsocketClientPolicy"], args.port)
    try:
        _validate_metadata(client.get_server_metadata(), args.benchmark)
    except Exception:
        _close_client(client)
        raise
    return client


def _recover_client(
    client: Any, modules: Mapping[str, Any], args: argparse.Namespace, category: Optional[str]
) -> Any:
    if category not in ("connection", "policy_runtime"):
        return client
    _close_client(client)
    return _connect_validated(modules, args)


def _unwrap_init_states(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("init_states", "states", "data"):
            if key in value:
                value = value[key]
                break
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return value


def _load_init_states(path: pathlib.Path, np: Any) -> Any:
    errors: List[str] = []
    try:
        import torch

        try:
            value = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(str(path), map_location="cpu")
        value = _unwrap_init_states(value)
        value = _normalize_init_states(value, np)
        if hasattr(value, "__len__"):
            return value
    except Exception as exc:
        errors.append("torch: %s: %s" % (exc.__class__.__name__, exc))
    try:
        value = np.load(str(path), allow_pickle=True)
        if hasattr(value, "files"):
            keys = list(value.files)
            selected = next((key for key in ("init_states", "states", "data") if key in keys), keys[0])
            value = value[selected]
        value = _unwrap_init_states(value)
        value = _normalize_init_states(value, np)
        if hasattr(value, "__len__"):
            return value
    except Exception as exc:
        errors.append("numpy: %s: %s" % (exc.__class__.__name__, exc))
    raise ValueError("unable to load init states %s (%s)" % (path, "; ".join(errors)))


def _normalize_init_states(value: Any, np: Any) -> Any:
    """Treat a flat numeric simulator state as one official init state."""
    if getattr(value, "ndim", None) == 1 and getattr(value, "dtype", None) != np.dtype("O"):
        return np.expand_dims(value, axis=0)
    return value


def _reset_to_trial(env: Any, initial_states: Any, trial: int) -> Any:
    if not hasattr(initial_states, "__len__") or len(initial_states) <= trial:
        actual = len(initial_states) if hasattr(initial_states, "__len__") else "unknown"
        raise ValueError("init state file has %s states; trial %d requested" % (actual, trial))
    env.reset()
    return env.set_init_state(initial_states[trial])


def _preflight(
    client: Any, spec: support.EpisodeSpec, modules: Mapping[str, Any]
) -> Dict[str, Any]:
    initial_states = _load_init_states(spec.init_path, modules["np"])
    env = _make_env(spec, modules)
    try:
        obs = _reset_to_trial(env, initial_states, 0)
        for _ in range(WAIT_STEPS):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        policy_obs, _ = _policy_observation(obs, spec.prompt, modules)
        actions = support.validate_action_result(client.infer(policy_obs), REPLAN, 7)
        return {
            "action_steps": len(actions),
            "action_dimension": 7,
            "finite": True,
            "suite": spec.suite,
            "source_id": spec.source_id,
            "bddl_path": str(spec.bddl_path),
            "prompt_source_path": str(spec.prompt_source_path or spec.bddl_path),
            "prompt": spec.prompt,
            "prompt_field": spec.prompt_field,
        }
    finally:
        env.close()


def _write_preflight(args: argparse.Namespace, status: str, **details: Any) -> None:
    payload = {
        "status": status,
        "benchmark": args.benchmark,
        "protocol": "official",
        "timestamp": _utc_now(),
        "resize": RESIZE,
        "replan": REPLAN,
        "wait_steps": WAIT_STEPS,
        "seed": SEED,
        "suite_filter": args.suite,
    }
    payload.update(details)
    support.write_new_json(args.preflight_manifest, payload)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "episode"


def _emit_video(path: pathlib.Path, frames: Sequence[Any], imageio: Any) -> Dict[str, Any]:
    if not frames:
        return {"video": None, "video_status": "not_recorded", "video_error": None, "video_trace": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimwrite(path, list(frames), fps=10)
    except Exception as exc:
        return {
            "video": None,
            "video_status": "failed",
            "video_error": "%s: %s" % (exc.__class__.__name__, exc),
            "video_trace": traceback.format_exc(),
        }
    return {"video": str(path), "video_status": "written", "video_error": None, "video_trace": None}


def _video_path(args: argparse.Namespace, spec: support.EpisodeSpec, attempt: int, status: str) -> pathlib.Path:
    parts = [
        spec.benchmark,
        spec.suite,
        spec.perturbation or spec.category or "task",
        "task-%04d" % spec.task_id,
        "trial-%02d" % spec.trial,
        "attempt-%02d" % attempt,
        status,
        uuid.uuid4().hex[:10],
    ]
    return args.output_dir / "videos" / (_safe_name("_".join(parts)) + ".mp4")


def _rollout(
    env: Any,
    spec: support.EpisodeSpec,
    initial_states: Any,
    client: Any,
    modules: Mapping[str, Any],
) -> Tuple[RolloutOutcome, List[Any]]:
    outcome = RolloutOutcome()
    frames: List[Any] = []
    actions = collections.deque()
    try:
        obs = _reset_to_trial(env, initial_states, spec.trial)
        for _ in range(WAIT_STEPS):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        for _ in range(spec.max_steps):
            policy_obs, frame = _policy_observation(obs, spec.prompt, modules)
            frames.append(frame)
            if not actions:
                outcome.stage = "policy infer"
                plan = support.validate_action_result(client.infer(policy_obs), REPLAN, 7)
                actions.extend(plan[:REPLAN])
            outcome.stage = "environment step"
            obs, _, done, _ = env.step(actions.popleft())
            outcome.action_steps += 1
            if done:
                outcome.status = "success"
                break
    except Exception as exc:
        outcome.status = "error"
        outcome.error_category = support.classify_error(exc, outcome.stage)
        outcome.error = "%s: %s" % (exc.__class__.__name__, exc)
        outcome.trace = traceback.format_exc()
        if _is_egl_error(outcome.trace):
            raise
    return outcome, frames


def _is_egl_error(text: str) -> bool:
    lowered = text.lower()
    return "egl" in lowered and any(
        token in lowered for token in ("failed", "error", "initialize", "display", "context", "device")
    )


def _episode_record(
    args: argparse.Namespace,
    spec: support.EpisodeSpec,
    attempt: int,
    outcome: RolloutOutcome,
    video: Mapping[str, Any],
    started_at: str,
    duration: float,
) -> Dict[str, Any]:
    record = {
        "episode_id": spec.episode_id,
        "attempt_id": "%s:attempt-%04d" % (spec.episode_id, attempt),
        "attempt": attempt,
        "benchmark": spec.benchmark,
        "protocol": "official",
        "phase": args.phase,
        "suite": spec.suite,
        "category": spec.category,
        "difficulty": spec.difficulty,
        "perturbation": spec.perturbation,
        "task_id": spec.task_id,
        "source_id": spec.source_id,
        "bddl_path": str(spec.bddl_path),
        "prompt_source_path": str(spec.prompt_source_path or spec.bddl_path),
        "init_path": str(spec.init_path),
        "task_description": spec.prompt,
        "prompt_field": spec.prompt_field,
        "trial": spec.trial,
        "seed": spec.seed,
        "max_steps": spec.max_steps,
        "wait_steps": WAIT_STEPS,
        "resize": RESIZE,
        "replan": REPLAN,
        "status": outcome.status,
        "success": outcome.status == "success" if outcome.status != "error" else None,
        "error_category": outcome.error_category,
        "error": outcome.error,
        "trace": outcome.trace,
        "stage": outcome.stage,
        "action_steps": outcome.action_steps,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": duration,
        "gl_backend": "egl",
    }
    record.update(video)
    return record


def _error_record(spec: support.EpisodeSpec, attempt: int, exc: BaseException, stage: str) -> Dict[str, Any]:
    return {
        "episode_id": spec.episode_id,
        "attempt_id": "%s:attempt-%04d" % (spec.episode_id, attempt),
        "attempt": attempt,
        "benchmark": spec.benchmark,
        "protocol": "official",
        "suite": spec.suite,
        "category": spec.category,
        "difficulty": spec.difficulty,
        "perturbation": spec.perturbation,
        "task_id": spec.task_id,
        "source_id": spec.source_id,
        "trial": spec.trial,
        "seed": spec.seed,
        "max_steps": spec.max_steps,
        "status": "error",
        "success": None,
        "error_category": support.classify_error(exc, stage),
        "error": "%s: %s" % (exc.__class__.__name__, exc),
        "trace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "stage": stage,
        "started_at": _utc_now(),
        "finished_at": _utc_now(),
        "video": None,
        "video_status": "not_recorded",
    }


def _record_group_error(
    specs: Sequence[support.EpisodeSpec], attempts: Mapping[str, int], path: pathlib.Path,
    exc: BaseException, stage: str,
) -> None:
    for spec in specs:
        support.append_jsonl(path, _error_record(spec, attempts.get(spec.episode_id, 0), exc, stage))


def _evaluate_group(
    args: argparse.Namespace, data: EvaluationData, specs: Sequence[support.EpisodeSpec], client: Any
) -> Any:
    try:
        initial_states = _load_init_states(specs[0].init_path, data.modules["np"])
    except Exception as exc:
        _record_group_error(specs, data.attempts, data.records_path, exc, "environment init states")
        return client
    if len(initial_states) <= max(spec.trial for spec in specs):
        exc = ValueError(
            "init state file %s contains %d states, needs trial %d"
            % (specs[0].init_path, len(initial_states), max(spec.trial for spec in specs))
        )
        _record_group_error(specs, data.attempts, data.records_path, exc, "environment init states")
        return client
    try:
        env = _make_env(specs[0], data.modules)
    except Exception as exc:
        if _is_egl_error(traceback.format_exc()):
            raise
        _record_group_error(specs, data.attempts, data.records_path, exc, "environment create")
        return client
    try:
        for spec in specs:
            attempt = data.attempts.get(spec.episode_id, 0)
            started = time.monotonic()
            started_at = _utc_now()
            outcome, frames = _rollout(env, spec, initial_states, client, data.modules)
            video = _emit_video(
                _video_path(args, spec, attempt, outcome.status), frames, data.modules["imageio"]
            )
            support.append_jsonl(
                data.records_path,
                _episode_record(args, spec, attempt, outcome, video, started_at, time.monotonic() - started),
            )
            logging.info("%s attempt=%d status=%s", spec.episode_id, attempt, outcome.status)
            client = _recover_client(client, data.modules, args, outcome.error_category)
    finally:
        env.close()
    return client


def _prepare(args: argparse.Namespace) -> EvaluationData:
    modules = _lazy_imports()
    modules["np"].random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "episodes.jsonl"
    existing = support.load_jsonl(records_path)
    sources = support.discover_sources(
        args.benchmark, args.benchmark_repo, allow_missing_pro_cells=args.benchmark == "pro"
    )
    sources = _verify_and_order_registry(sources, modules)
    partial_pro = args.benchmark == "pro" and not support.pro_compatibility(sources)["protocol_applicable"]
    matrix = support.expand_matrix(
        args.benchmark, args.phase, sources, allow_incompatible_pro=partial_pro
    )
    matrix = support.filter_matrix_by_suites(matrix, args.suite)
    pending = [] if args.phase == "preflight" else support.select_pending(matrix, existing, args.resume)
    return EvaluationData(modules, sources, matrix, pending, support.next_attempts(existing), records_path)


def _evaluate_pending(args: argparse.Namespace, data: EvaluationData, client: Any) -> Any:
    grouped: Dict[Tuple[str, str], List[support.EpisodeSpec]] = {}
    for spec in data.pending:
        key = (str(spec.bddl_path), str(spec.init_path))
        grouped.setdefault(key, []).append(spec)
    for specs in grouped.values():
        client = _evaluate_group(args, data, specs, client)
    return client


def _finalize(args: argparse.Namespace, data: EvaluationData) -> int:
    records = support.load_jsonl(data.records_path)
    summary = support.aggregate(records, data.matrix)
    integrity = support.verify_integrity(records, data.matrix, require_videos=True)
    summary.update(
        {
            "benchmark": args.benchmark,
            "protocol": "official",
            "phase": args.phase,
            "seed": SEED,
            "resize": RESIZE,
            "replan": REPLAN,
            "wait_steps": WAIT_STEPS,
            "suite_filter": args.suite,
            "integrity": integrity,
        }
    )
    if args.benchmark == "pro":
        summary["compatibility"] = support.pro_compatibility(data.sources)
    support.write_summaries(summary, args.output_dir)
    if not summary["total"]["complete"] or not integrity["passed"]:
        logging.error("evaluation incomplete; infrastructure/video errors remain retryable with --resume")
        return 2
    return 0


def evaluate(args: argparse.Namespace) -> int:
    data = _prepare(args)
    client = _connect_validated(data.modules, args)
    try:
        evidence = _preflight(client, data.matrix[0], data.modules)
        _write_preflight(args, "passed", **evidence)
        logging.info("preflight validated >=5 finite 7D actions")
        if args.phase == "preflight":
            return 0
        client = _evaluate_pending(args, data, client)
    finally:
        _close_client(client)
    return _finalize(args, data)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args(argv)
    try:
        return evaluate(args)
    except Exception as exc:
        rendered = traceback.format_exc()
        logging.error("client startup/preflight failed:\n%s", rendered)
        if not args.preflight_manifest.exists():
            _write_preflight(
                args,
                "failed",
                error_category=support.classify_error(exc, "client preflight"),
                error="%s: %s" % (exc.__class__.__name__, exc),
                trace=rendered,
            )
        if _is_egl_error(rendered):
            return EGL_FAILURE_EXIT
        return 2


if __name__ == "__main__":
    sys.exit(main())
