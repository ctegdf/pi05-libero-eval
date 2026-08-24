"""Launch an isolated, auditable pi0.5-on-LIBERO-X zero-shot evaluation.

Mirrors plus-pro/harness/run_pi05_eval.py's proven design: a fresh scratch
$HOME with its own .libero/config.yaml per run (so LIBERO-X's interactive
first-import input() prompt is never hit), a policy server started as an
exact-PID subprocess this launcher alone signals (never the shared
<private> which fuser -k's the port and globally
pkills spawn_main/resource_tracker), and a JSON manifest with a
stage-by-stage events timeline for post-hoc auditing.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple

import libero_x_support as support

HERE = pathlib.Path(__file__).resolve().parent
SERVER = HERE / "pi05_liberox_server.py"
CLIENT = HERE / "pi05_liberox_client.py"
EGL_FAILURE_EXIT = 86
PACKAGE_INVENTORY_SCRIPT = """
import json
from importlib import metadata
packages = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name") or "unknown"
    packages[name] = distribution.version
items = [{"name": name, "version": packages[name]} for name in sorted(packages)]
print(json.dumps({"packages": items[:2000], "truncated": len(items) > 2000}))
"""


@dataclasses.dataclass
class RunContext:
    run_id: str
    started_at: str
    logs_dir: pathlib.Path
    manifests_dir: pathlib.Path
    home_dir: pathlib.Path
    result_manifest: pathlib.Path
    preflight_manifest: pathlib.Path
    events: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ServerHandle:
    process: subprocess.Popen
    log: IO[str]
    log_path: pathlib.Path
    started_at: str


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=support.PHASES)
    parser.add_argument("--vendor-root", required=True, type=pathlib.Path, help="LIBERO-X checkout root (contains eval_template.py)")
    parser.add_argument("--openpi-repo", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--port", type=int, default=8140)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--level", action="append", choices=support.LEVELS)
    return parser


def _event(stage: str, status: str, **details: Any) -> Dict[str, Any]:
    result = {"stage": stage, "status": status, "timestamp": _utc_now()}
    result.update(details)
    return result


def _validate_executable(value: str, label: str) -> str:
    path = pathlib.Path(os.path.abspath(str(pathlib.Path(value).expanduser())))
    if not path.is_file() or not os.access(str(path), os.X_OK):
        raise ValueError("%s is not an executable file: %s" % (label, path))
    return str(path)


def _check_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in range 1..65535")
    if not args.gpu_id.strip() or "," in args.gpu_id or any(char.isspace() for char in args.gpu_id):
        raise ValueError("--gpu-id must identify exactly one GPU")
    args.server_python = _validate_executable(args.server_python, "--server-python")
    args.client_python = _validate_executable(args.client_python, "--client-python")
    for label, path in (("--vendor-root", args.vendor_root), ("--openpi-repo", args.openpi_repo), ("--checkpoint-dir", args.checkpoint_dir)):
        if not path.is_dir():
            raise ValueError("%s does not exist: %s" % (label, path))
    if not (args.openpi_repo / "src" / "openpi").is_dir():
        raise ValueError("--openpi-repo does not contain src/openpi: %s" % args.openpi_repo)
    bddl_files_default = args.vendor_root / "libero" / "libero" / "bddl_files"
    init_files_default = args.vendor_root / "libero" / "libero" / "init_files"
    assets_default = args.vendor_root / "libero" / "libero" / "assets"
    for label, path in (("bddl_files", bddl_files_default), ("init_states", init_files_default), ("assets", assets_default)):
        if not path.is_dir():
            raise ValueError("vendor checkout is missing expected %s directory: %s" % (label, path))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))


def _git_manifest(repo: pathlib.Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(repo)}
    commands = {
        "commit": ["git", "-C", str(repo), "rev-parse", "HEAD"],
        "status": ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False, timeout=30.0)
            result[key] = completed.stdout.strip() if completed.returncode == 0 else None
            if completed.returncode:
                result[key + "_error"] = completed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = None
            result[key + "_error"] = "%s: %s" % (exc.__class__.__name__, exc)
    return result


def _package_inventory(interpreter: str) -> Dict[str, Any]:
    try:
        completed = subprocess.run([interpreter, "-c", PACKAGE_INVENTORY_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False, timeout=60.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "error": "%s: %s" % (exc.__class__.__name__, exc)}
    if completed.returncode:
        return {"status": "error", "exit_code": completed.returncode, "error": completed.stderr[-10000:]}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": "invalid package inventory: %s" % exc}
    return dict({"status": "ok", "interpreter": interpreter}, **payload)


def _nvidia_inventory() -> Dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "error": "nvidia-smi not found"}
    query = "index,name,driver_version,memory.total,memory.used,memory.free"
    try:
        completed = subprocess.run([executable, "--query-gpu=" + query, "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False, timeout=10.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "error": "%s: %s" % (exc.__class__.__name__, exc)}
    return {"status": "ok" if completed.returncode == 0 else "error", "exit_code": completed.returncode, "query": query, "stdout": completed.stdout[-10000:], "stderr": completed.stderr[-10000:]}


def _yaml_scalar(value: pathlib.Path) -> str:
    return json.dumps(str(value.resolve()))


def _create_libero_home(args: argparse.Namespace, context: RunContext) -> Dict[str, str]:
    """Pre-writes .libero/config.yaml so LIBERO-X's import-time input()
    prompt (triggered whenever the config file is missing) is never hit."""
    bddl_root = args.vendor_root / "libero" / "libero" / "bddl_files"
    init_root = args.vendor_root / "libero" / "libero" / "init_files"
    assets_root = args.vendor_root / "libero" / "libero" / "assets"
    datasets_root = args.vendor_root / "libero" / "datasets"
    benchmark_root = args.vendor_root / "libero" / "libero"
    libero_dir = context.home_dir / ".libero"
    libero_dir.mkdir(parents=True, exist_ok=False)
    config_path = libero_dir / "config.yaml"
    lines = [
        "benchmark_root: %s" % _yaml_scalar(benchmark_root),
        "bddl_files: %s" % _yaml_scalar(bddl_root),
        "init_states: %s" % _yaml_scalar(init_root),
        "datasets: %s" % _yaml_scalar(datasets_root),
        "assets: %s" % _yaml_scalar(assets_root),
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"home": str(context.home_dir), "config": str(config_path), "libero_config_path": str(libero_dir)}


def _child_env(args: argparse.Namespace, context: RunContext, client: bool) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(context.home_dir),
            "LIBERO_CONFIG_PATH": str(context.home_dir / ".libero"),
            # LIBERO-X-specific overrides that let eval_template.py-style
            # raw bddl/init directory access bypass config.yaml entirely
            # for these two keys; harmless when unused by other codepaths.
            "LIBERO_X_BDDL_ROOT": str(args.vendor_root / "libero" / "libero_x" / "bddl"),
            "LIBERO_X_INIT_ROOT": str(args.vendor_root / "libero" / "libero_x" / "init"),
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": args.gpu_id if client else "0",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": str(args.vendor_root),
        }
    )
    if not client:
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    return env


def _server_command(args: argparse.Namespace, context: RunContext) -> List[str]:
    return [
        args.server_python,
        str(SERVER),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--port",
        str(args.port),
        "--provenance-manifest",
        str(context.manifests_dir / ("checkpoint-server-%s.json" % context.run_id)),
        "--diagnostic-manifest",
        str(context.manifests_dir / ("server-diagnostic-%s.json" % context.run_id)),
    ]


def _start_server(args: argparse.Namespace, context: RunContext) -> ServerHandle:
    log_path = context.logs_dir / ("server-%s.log" % context.run_id)
    log = log_path.open("x", encoding="utf-8")
    started_at = _utc_now()
    try:
        process = subprocess.Popen(_server_command(args, context), cwd=str(args.openpi_repo), env=_child_env(args, context, client=False), stdout=log, stderr=subprocess.STDOUT)
    except Exception as exc:
        log.close()
        context.events.append(_event("server_startup", "failed", error_category="environment", error="%s: %s" % (exc.__class__.__name__, exc), trace=traceback.format_exc(), log_path=str(log_path)))
        raise
    return ServerHandle(process, log, log_path, started_at)


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError("policy server exited during startup with code %d" % return_code)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("policy server did not become reachable within %.0f seconds" % timeout)


def _wait_server_audited(context: RunContext, server: ServerHandle, port: int) -> None:
    try:
        _wait_for_server(server.process, port)
    except Exception as exc:
        diagnostic_path = context.manifests_dir / ("server-diagnostic-%s.json" % context.run_id)
        diagnostic = {}
        try:
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        category = diagnostic.get("error_category") or "environment"
        context.events.append(_event("server_startup", "failed", exit_code=server.process.poll(), error_category=category, error=diagnostic.get("error") or "%s: %s" % (exc.__class__.__name__, exc), trace=diagnostic.get("trace") or traceback.format_exc(), log_path=str(server.log_path)))
        raise
    context.events.append(_event("server_startup", "passed", log_path=str(server.log_path)))


def _stop_server(server: ServerHandle) -> None:
    # Only the exact child spawned by this run is ever signalled - never
    # fuser -k or a broad pkill against process names.
    if server.process.poll() is None:
        server.process.terminate()
        try:
            server.process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=5.0)
    server.log.close()


def _client_command(args: argparse.Namespace, context: RunContext) -> List[str]:
    command = [
        args.client_python,
        str(CLIENT),
        "--phase",
        args.phase,
        "--bddl-root",
        str(args.vendor_root / "libero" / "libero_x" / "bddl"),
        "--init-root",
        str(args.vendor_root / "libero" / "libero_x" / "init"),
        "--level5-prompt-root",
        str(args.vendor_root / "libero" / "libero_x" / "LEVEL5"),
        "--port",
        str(args.port),
        "--output-dir",
        str(args.output_dir),
        "--preflight-manifest",
        str(context.preflight_manifest),
    ]
    if args.resume:
        command.append("--resume")
    for level in args.level or []:
        command += ["--level", level]
    return command


def _run_client(args: argparse.Namespace, context: RunContext) -> int:
    log_path = context.logs_dir / ("client-egl-%s.log" % context.run_id)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(_client_command(args, context), cwd=str(HERE), env=_child_env(args, context, client=True), stdout=log, stderr=subprocess.STDOUT)
    return process.returncode


def _write_manifests(args: argparse.Namespace, context: RunContext) -> None:
    support.write_new_json(context.manifests_dir / ("environment-%s.json" % context.run_id), {
        "argv": sys.argv,
        "run_id": context.run_id,
        "phase": args.phase,
        "vendor_root": str(args.vendor_root),
        "openpi_repo": str(args.openpi_repo),
        "checkpoint_dir": str(args.checkpoint_dir),
        "client_python": args.client_python,
        "server_python": args.server_python,
        "port": args.port,
        "gpu_id": args.gpu_id,
        "platform": platform.platform(),
        "python": sys.version,
        "level_filter": args.level,
        "nvidia_smi": _nvidia_inventory(),
        "inventories": {
            "client_packages": _package_inventory(args.client_python),
            "server_packages": _package_inventory(args.server_python),
        },
    })
    support.write_new_json(context.manifests_dir / ("git-vendor-%s.json" % context.run_id), _git_manifest(args.vendor_root))
    support.write_new_json(context.manifests_dir / ("git-openpi-%s.json" % context.run_id), _git_manifest(args.openpi_repo))


def orchestrate(args: argparse.Namespace) -> int:
    _check_args(args)
    run_id = _run_id()
    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    manifests_dir = output_dir / "manifests"
    runtime_dir = output_dir / "runtime" / run_id
    home_dir = runtime_dir / "home"
    for path in (logs_dir, manifests_dir, home_dir):
        path.mkdir(parents=True, exist_ok=True)
    context = RunContext(
        run_id=run_id,
        started_at=_utc_now(),
        logs_dir=logs_dir,
        manifests_dir=manifests_dir,
        home_dir=home_dir,
        result_manifest=manifests_dir / ("result-%s.json" % run_id),
        preflight_manifest=manifests_dir / ("preflight-%s.json" % run_id),
    )
    exit_code = 2
    try:
        _create_libero_home(args, context)
        _write_manifests(args, context)
        server = _start_server(args, context)
        try:
            _wait_server_audited(context, server, args.port)
            context.events.append(_event("client_run", "started"))
            exit_code = _run_client(args, context)
            status = "passed" if exit_code == 0 else ("egl_failure" if exit_code == EGL_FAILURE_EXIT else "failed")
            context.events.append(_event("client_run", status, exit_code=exit_code))
        finally:
            _stop_server(server)
    except Exception as exc:
        context.events.append(_event("orchestration", "failed", error="%s: %s" % (exc.__class__.__name__, exc), trace=traceback.format_exc()))
        exit_code = 2
    finally:
        support.write_new_json(context.result_manifest, {
            "run_id": run_id,
            "started_at": context.started_at,
            "finished_at": _utc_now(),
            "phase": args.phase,
            "exit_code": exit_code,
            "events": context.events,
        })
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return orchestrate(args)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
