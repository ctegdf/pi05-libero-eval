"""Loopback-only OpenPI policy server for pi0.5-on-LIBERO-X zero-shot evaluation.

Near-identical to the already-validated
plus-pro/harness/pi05_eval_server.py: serves the single official
pi05_libero checkpoint over a loopback websocket, with the checkpoint
identity/provenance and metadata handshake unchanged so
pi05_liberox_client.py can use the exact same server-validation contract.
No behavior here is LIBERO-X-specific; only the harness around it (client,
launcher) differs.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import pathlib
import sys
import traceback
from typing import Any, Dict, Optional, Sequence, Tuple

import libero_x_support as support


class ServerStartupError(RuntimeError):
    def __init__(self, category: str, stage: str, cause: BaseException):
        super().__init__("%s: %s" % (cause.__class__.__name__, cause))
        self.category = category
        self.stage = stage
        self.cause = cause


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--provenance-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--diagnostic-manifest", required=True, type=pathlib.Path)
    return parser


def classify_startup_error(exc: BaseException, stage: str) -> str:
    if isinstance(exc, support.CheckpointError) or stage == "checkpoint_resolution":
        return "checkpoint"
    rendered = "%s %s" % (exc.__class__.__name__, exc)
    lowered = rendered.lower()
    if stage == "imports" or any(token in lowered for token in ("cuda", "cudnn", "jax", "xla", "importerror", "no module named")):
        return "environment"
    if stage == "policy_load" and any(token in lowered for token in ("checkpoint", "params", "restore", "orbax", "not found", "no such file")):
        return "checkpoint"
    return "policy_runtime"


def _raise_startup(exc: BaseException, stage: str) -> None:
    raise ServerStartupError(classify_startup_error(exc, stage), stage, exc) from exc


def _load_policy(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    try:
        resolution = support.resolve_checkpoint(args.checkpoint_dir)
    except Exception as exc:
        _raise_startup(exc, "checkpoint_resolution")
    provenance = resolution.provenance()
    provenance.update(
        {
            "benchmark": "libero-x-zero-shot",
            "bind_host": "127.0.0.1",
            "port": args.port,
            "norm_stats_sha256": support.sha256_file(resolution.norm_stats),
        }
    )
    support.write_new_json(args.provenance_manifest, provenance)
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except Exception as exc:
        _raise_startup(exc, "imports")
    try:
        config = training_config.get_config(support.CONFIG_NAME)
        policy = policy_config.create_trained_policy(config, resolution.checkpoint)
    except Exception as exc:
        _raise_startup(exc, "policy_load")
    metadata = dict(policy.metadata or {})
    metadata.update(
        {
            "evaluation_config": support.CONFIG_NAME,
            "evaluation_protocol": "official",
            "evaluation_benchmark": "libero-x-zero-shot",
            "checkpoint_provenance": provenance,
        }
    )
    return policy, metadata


def serve(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in range 1..65535")
    policy, metadata = _load_policy(args)
    try:
        from openpi.serving import websocket_policy_server
    except Exception as exc:
        _raise_startup(exc, "imports")
    logging.info("Serving official pi05_libero for libero-x-zero-shot on ws://127.0.0.1:%d", args.port)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port, metadata=metadata)
    server.serve_forever()


def _diagnostic(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, ServerStartupError):
        category, stage, cause = exc.category, exc.stage, exc.cause
    else:
        category, stage, cause = classify_startup_error(exc, "server_runtime"), "server_runtime", exc
    return {
        "status": "failed",
        "timestamp": _utc_now(),
        "error_category": category,
        "stage": stage,
        "error": "%s: %s" % (cause.__class__.__name__, cause),
        "trace": traceback.format_exc(),
        "exit_code": 2,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args(argv)
    try:
        serve(args)
    except Exception as exc:
        diagnostic = _diagnostic(exc)
        support.write_new_json(args.diagnostic_manifest, diagnostic)
        logging.error("Server startup failed [%s]: %s", diagnostic["error_category"], diagnostic["error"])
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
