"""Resumable pi0.5-on-LIBERO-X zero-shot rollout client.

Behaves like plus-pro/harness/pi05_eval_client.py (resume-safe JSONL
ledger, exact policy-outcome accounting, non-colliding video paths), but
speaks LIBERO-X's environment API, which differs from LIBERO-Plus/Pro's:

- Init-state loading uses `env.reset()` + `env.regenerate_obs_from_state(...)`
  (LIBERO-X's OffScreenRenderEnv method name), not `env.set_init_state(...)`.
- The released eval_template.py performs zero dummy no-op steps after an
  init-mode reset (it sets `t = num_steps_wait` directly). We keep that
  behavior for the released-template protocol result instead of importing
  LIBERO-Plus/Pro's convention of stepping WAIT_STEPS dummy actions, and
  record `executed_wait_steps` explicitly on every record so this choice is
  auditable rather than silently assumed.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import logging
import pathlib
import socket
import sys
import time
import traceback
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import libero_x_support as support

ENV_RESOLUTION = 256
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
    parser.add_argument("--phase", required=True, choices=support.PHASES)
    parser.add_argument("--bddl-root", required=True, type=pathlib.Path)
    parser.add_argument("--init-root", required=True, type=pathlib.Path)
    parser.add_argument("--level5-prompt-root", required=True, type=pathlib.Path)
    parser.add_argument("--level", action="append", choices=support.LEVELS)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-manifest", required=True, type=pathlib.Path)
    return parser


def _lazy_imports() -> Dict[str, Any]:
    import imageio
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    return {
        "imageio": imageio,
        "np": np,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "image_tools": image_tools,
        "WebsocketClientPolicy": websocket_client_policy.WebsocketClientPolicy,
    }


def _quat2axisangle(quat: Any, np: Any) -> Any:
    import math

    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def _make_env(spec: support.EpisodeSpec, modules: Mapping[str, Any]) -> Any:
    # No env.seed() here: one env is shared across every trial of a
    # bddl/init group (see _evaluate_group), and the released template
    # reseeds per-trial with the trial index itself (see _reset_to_trial),
    # not once at construction time.
    return modules["OffScreenRenderEnv"](
        bddl_file_name=str(spec.bddl_path),
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
        horizon=spec.max_steps + support.WAIT_STEPS + 1,
    )


def _policy_observation(obs: Mapping[str, Any], prompt: str, modules: Mapping[str, Any]) -> Tuple[Dict[str, Any], Any]:
    np = modules["np"]
    image_tools = modules["image_tools"]
    if support.FLIP_IMAGES:
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    else:
        image = np.ascontiguousarray(obs["agentview_image"][::-1, :])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, :])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, support.RESIZE, support.RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, support.RESIZE, support.RESIZE))
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


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("server metadata is not a mapping")
    expected = {"evaluation_config": support.CONFIG_NAME, "evaluation_protocol": "official"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("server metadata %s mismatch: expected %r, got %r" % (key, value, metadata.get(key)))
    provenance = metadata.get("checkpoint_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("asset_id") != support.ASSET_ID:
        raise ValueError("server official checkpoint provenance is missing or invalid")


def _connect_validated(modules: Mapping[str, Any], args: argparse.Namespace) -> Any:
    client = _connect(modules["WebsocketClientPolicy"], args.port)
    try:
        _validate_metadata(client.get_server_metadata())
    except Exception:
        _close_client(client)
        raise
    return client


def _recover_client(client: Any, modules: Mapping[str, Any], args: argparse.Namespace, category: Optional[str]) -> Any:
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
    import torch

    try:
        value = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")
    value = _unwrap_init_states(value)
    if not hasattr(value, "__len__"):
        raise ValueError("unable to interpret init states in %s" % path)
    return value


def _reset_to_trial(env: Any, initial_states: Any, trial: int, np: Any) -> Any:
    if len(initial_states) <= trial:
        raise ValueError("init state file has %d states; trial %d requested" % (len(initial_states), trial))
    # Matches the released eval_template.py exactly: env.seed(ep_id) then
    # env.reset(), i.e. the environment RNG is reseeded per-trial with the
    # trial index, not a single fixed seed shared across all trials.
    env.seed(trial)
    env.reset()
    state = initial_states[trial]
    state = state.numpy() if hasattr(state, "numpy") else np.asarray(state)
    # LIBERO-X's env exposes regenerate_obs_from_state (not set_init_state,
    # which is the LIBERO-Plus/Pro fork's method name for the same idea).
    return env.regenerate_obs_from_state(state)


def _preflight(client: Any, spec: support.EpisodeSpec, modules: Mapping[str, Any]) -> Dict[str, Any]:
    initial_states = _load_init_states(spec.init_path, modules["np"])
    env = _make_env(spec, modules)
    try:
        obs = _reset_to_trial(env, initial_states, spec.trial, modules["np"])
        policy_obs, _ = _policy_observation(obs, spec.prompt, modules)
        actions = support.validate_action_result(client.infer(policy_obs), support.REPLAN, 7)
        return {
            "action_steps": len(actions),
            "action_dimension": 7,
            "finite": True,
            "level": spec.level,
            "source_id": "%s/%s" % (spec.level, spec.bddl_path.name),
            "bddl_path": str(spec.bddl_path),
            "prompt": spec.prompt,
            "prompt_field": spec.prompt_field,
            "executed_wait_steps": support.EXECUTED_WAIT_STEPS,
        }
    finally:
        env.close()


def _write_preflight(args: argparse.Namespace, status: str, **details: Any) -> None:
    payload = {
        "status": status,
        "protocol": support.PROTOCOL,
        "timestamp": _utc_now(),
        "resize": support.RESIZE,
        "replan": support.REPLAN,
        "max_steps": support.MAX_STEPS,
        "flip_images": support.FLIP_IMAGES,
        "executed_wait_steps": support.EXECUTED_WAIT_STEPS,
        "seed": support.SEED,
        "level_filter": args.level,
    }
    payload.update(details)
    support.write_new_json(args.preflight_manifest, payload)


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
        spec.level,
        "task-%s" % spec.task_num,
        spec.variant or "novariant",
        spec.prompt_field,
        "trial-%02d" % spec.trial,
        "attempt-%02d" % attempt,
        status,
        uuid.uuid4().hex[:10],
    ]
    return args.output_dir / "videos" / (support._slug("_".join(parts)) + ".mp4")


def _rollout(env: Any, spec: support.EpisodeSpec, initial_states: Any, client: Any, modules: Mapping[str, Any]) -> Tuple[RolloutOutcome, List[Any]]:
    outcome = RolloutOutcome()
    frames: List[Any] = []
    actions: collections.deque = collections.deque()
    try:
        obs = _reset_to_trial(env, initial_states, spec.trial, modules["np"])
        for _ in range(spec.max_steps):
            policy_obs, frame = _policy_observation(obs, spec.prompt, modules)
            frames.append(frame)
            if not actions:
                outcome.stage = "policy infer"
                plan = support.validate_action_result(client.infer(policy_obs), support.REPLAN, 7)
                actions.extend(plan[: support.REPLAN])
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
    return "egl" in lowered and any(token in lowered for token in ("failed", "error", "initialize", "display", "context", "device"))


def _episode_record(args: argparse.Namespace, spec: support.EpisodeSpec, attempt: int, outcome: RolloutOutcome, video: Mapping[str, Any], started_at: str, duration: float) -> Dict[str, Any]:
    record = {
        "episode_id": spec.episode_id,
        "attempt_id": "%s:attempt-%04d" % (spec.episode_id, attempt),
        "attempt": attempt,
        "protocol": support.PROTOCOL,
        "phase": args.phase,
        "level": spec.level,
        "task_num": spec.task_num,
        "variant": spec.variant,
        "bddl_path": str(spec.bddl_path),
        "init_path": str(spec.init_path),
        "task_description": spec.prompt,
        "prompt_field": spec.prompt_field,
        "trial": spec.trial,
        "seed": spec.seed,
        "max_steps": spec.max_steps,
        "executed_wait_steps": support.EXECUTED_WAIT_STEPS,
        "resize": support.RESIZE,
        "replan": support.REPLAN,
        "flip_images": support.FLIP_IMAGES,
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
        "protocol": support.PROTOCOL,
        "level": spec.level,
        "task_num": spec.task_num,
        "variant": spec.variant,
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


def _record_group_error(specs: Sequence[support.EpisodeSpec], attempts: Mapping[str, int], path: pathlib.Path, exc: BaseException, stage: str) -> None:
    for spec in specs:
        support.append_jsonl(path, _error_record(spec, attempts.get(spec.episode_id, 0), exc, stage))


def _evaluate_group(args: argparse.Namespace, data: EvaluationData, specs: Sequence[support.EpisodeSpec], client: Any) -> Any:
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
            video = _emit_video(_video_path(args, spec, attempt, outcome.status), frames, data.modules["imageio"])
            if outcome.status in ("success", "failure") and video.get("video_status") != "written":
                # A written video is required for a terminal outcome (see
                # verify_integrity's require_videos check); demoting a
                # video-write failure to a retryable error here - instead of
                # leaving it as a terminal success/failure with a missing
                # video - is what actually makes --resume able to repair it,
                # since select_pending only ever re-queues non-terminal
                # (error) episodes.
                outcome.status = "error"
                outcome.error_category = "environment"
                outcome.error = "video write failed or produced no frames: %s" % video.get("video_error")
                outcome.stage = "video emit"
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
    modules["np"].random.seed(support.SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "episodes.jsonl"
    existing = support.load_jsonl(records_path)
    if existing and args.phase != "preflight" and not args.resume:
        # Re-running the full matrix into a non-empty ledger without
        # --resume would append a second terminal outcome per episode_id,
        # which verify_integrity correctly rejects as non-unique - but only
        # after burning a full rollout pass. Fail fast instead.
        raise support.InventoryError(
            "%s already has %d record(s); pass --resume to continue it, or use a fresh --output-dir"
            % (records_path, len(existing))
        )
    sources = support.discover_sources(args.bddl_root, args.init_root, args.level5_prompt_root)
    registry_check = support.cross_check_registry(sources)
    logging.info("registry cross-check: %s", registry_check.get("status"))
    levels = args.level or list(support.LEVELS)
    matrix = support.expand_matrix(args.phase, sources, levels=levels)
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
            "protocol": support.PROTOCOL,
            "phase": args.phase,
            "seed": support.SEED,
            "resize": support.RESIZE,
            "replan": support.REPLAN,
            "max_steps": support.MAX_STEPS,
            "flip_images": support.FLIP_IMAGES,
            "executed_wait_steps": support.EXECUTED_WAIT_STEPS,
            "level_filter": args.level,
            "integrity": integrity,
            "note": (
                "released-template protocol (max_steps fixed at 1200 for every task/level); "
                "not a reproduction of the LIBERO-X paper's own per-task horizon protocol, "
                "and pi05_libero was fine-tuned on standard LIBERO only, so this is a "
                "zero-shot transfer evaluation, not the paper's in-domain fine-tuned result."
            ),
        }
    )
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
        logging.info("preflight validated >=%d finite 7D actions", support.REPLAN)
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
