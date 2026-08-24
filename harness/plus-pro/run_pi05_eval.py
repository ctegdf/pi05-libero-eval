"""Launch an isolated, auditable pi05_libero evaluation for Plus or Pro."""

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
from typing import Any, Dict, IO, List, Mapping, Optional, Sequence, Tuple

import pi05_eval_support as support


HERE = pathlib.Path(__file__).resolve().parent
SERVER = HERE / "pi05_eval_server.py"
CLIENT = HERE / "pi05_eval_client.py"
SHARED_CACHE = HERE / "runtime" / "cache"
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
    runtime_dir: pathlib.Path
    home_dir: pathlib.Path
    result_manifest: pathlib.Path
    preflight_manifest: pathlib.Path
    benchmark_data_root: pathlib.Path
    summary_before: Optional[Tuple[int, int]]
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
    parser.add_argument("--benchmark", required=True, choices=support.BENCHMARKS)
    parser.add_argument("--phase", required=True, choices=support.PHASES)
    parser.add_argument("--benchmark-repo", required=True, type=pathlib.Path)
    parser.add_argument("--openpi-repo", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark-data-dir", type=pathlib.Path)
    parser.add_argument("--assets-dir", type=pathlib.Path)
    parser.add_argument("--imagemagick-runtime", type=pathlib.Path)
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--suite", action="append", choices=support.SUITES)
    return parser


def _event(stage: str, status: str, **details: Any) -> Dict[str, Any]:
    result = {"stage": stage, "status": status, "timestamp": _utc_now()}
    result.update(details)
    return result


def _file_token(path: pathlib.Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _read_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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
    if args.suite and len(args.suite) != len(set(args.suite)):
        raise ValueError("--suite must not contain duplicates")
    args.server_python = _validate_executable(args.server_python, "--server-python")
    args.client_python = _validate_executable(args.client_python, "--client-python")
    args.benchmark_repo = support._require_dir(args.benchmark_repo, "benchmark repository")
    args.openpi_repo = support._require_dir(args.openpi_repo, "OpenPI repository")
    if not (args.openpi_repo / "src" / "openpi").is_dir():
        raise ValueError("--openpi-repo does not contain src/openpi: %s" % args.openpi_repo)
    if args.benchmark_data_dir is None:
        args.benchmark_data_dir = (
            args.benchmark_repo
            if args.benchmark == "plus"
            else HERE / "datasets" / "LIBERO-Pro"
        )
    args.benchmark_data_dir = support._require_dir(args.benchmark_data_dir, "benchmark data")
    if args.assets_dir is None:
        # The simulation assets are the common LIBERO assets staged outside
        # both clones.  Pro reuses this read-only copy.
        args.assets_dir = HERE / "data" / "libero-plus" / "assets"
    args.assets_dir = support._require_dir(args.assets_dir, "independent benchmark assets")
    if args.benchmark == "plus":
        if args.imagemagick_runtime is None:
            args.imagemagick_runtime = args.benchmark_repo / "runtime" / "imagemagick"
        args.imagemagick_runtime = support._require_dir(args.imagemagick_runtime, "Plus ImageMagick runtime")
        _imagemagick_coder_dir(args.imagemagick_runtime)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))


def _imagemagick_coder_dir(runtime: pathlib.Path) -> pathlib.Path:
    root = runtime / "root" / "usr" / "lib"
    matches = sorted(path for path in root.glob("*/ImageMagick-*/modules-*/coders") if path.is_dir())
    if len(matches) != 1:
        raise ValueError("expected one ImageMagick coder directory below %s, found %d" % (root, len(matches)))
    return matches[0].resolve()


def _git_manifest(repo: pathlib.Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(repo)}
    commands = {
        "commit": ["git", "-C", str(repo), "rev-parse", "HEAD"],
        "status": ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        "remote": ["git", "-C", str(repo), "remote", "get-url", "origin"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False,
                timeout=30.0,
            )
            result[key] = completed.stdout.strip() if completed.returncode == 0 else None
            if completed.returncode:
                result[key + "_error"] = completed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = None
            result[key + "_error"] = "%s: %s" % (exc.__class__.__name__, exc)
    return result


def _package_inventory(interpreter: str) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            [interpreter, "-c", PACKAGE_INVENTORY_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
            timeout=60.0,
        )
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
        completed = subprocess.run(
            [executable, "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "error": "%s: %s" % (exc.__class__.__name__, exc)}
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "exit_code": completed.returncode,
        "query": query,
        "stdout": completed.stdout[-10000:],
        "stderr": completed.stderr[-10000:],
    }


def _find_package_root(repo: pathlib.Path, bddl_root: pathlib.Path) -> pathlib.Path:
    candidates = [repo / "libero" / "libero", repo / "libero", bddl_root.parent]
    for path in candidates:
        if path.is_dir() and (path / "bddl_files").is_dir():
            return path.resolve()
    return bddl_root.parent.resolve()


def _yaml_scalar(value: pathlib.Path) -> str:
    # JSON strings are valid YAML scalars and correctly quote spaces/colons.
    return json.dumps(str(value.resolve()))


def _create_libero_home(
    args: argparse.Namespace, context: RunContext, data_root: pathlib.Path
) -> Dict[str, str]:
    bddl_root = support._find_one(data_root, ("bddl_files",), "bddl_files")
    init_root = support._find_one(data_root, ("init_files", "init_states"), "init_files")
    assets_root = args.assets_dir.resolve()
    datasets_root = data_root.resolve()
    benchmark_root = _find_package_root(args.benchmark_repo, bddl_root)
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
    return {
        "home": str(context.home_dir),
        "config": str(config_path),
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(bddl_root),
        "init_states": str(init_root),
        "datasets": str(datasets_root),
        "assets": str(assets_root),
    }


def _pro_registry_dir(source: support.TaskSource) -> str:
    suffix = {
        "object": "object",
        "swap(position)": "swap",
        "lan(semantic)": "lan",
        "task": "task",
        "env": "env",
    }[str(source.perturbation)]
    return "%s_%s" % (source.suite, suffix)


def _symlink_new(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("runtime view destination already exists: %s" % destination)
    os.symlink(str(source.resolve()), str(destination))


def _build_pro_view(
    context: RunContext, sources: Sequence[support.TaskSource], assets_dir: pathlib.Path
) -> pathlib.Path:
    """Build a per-run symlink view; never mutate the clone or HF snapshot."""
    view = context.runtime_dir / "libero-pro-view"
    (view / "bddl_files").mkdir(parents=True)
    (view / "init_files").mkdir()
    for source in sources:
        cell = _pro_registry_dir(source)
        _symlink_new(source.bddl_path, view / "bddl_files" / cell / source.bddl_path.name)
        _symlink_new(source.init_path, view / "init_files" / cell / source.init_path.name)
    _symlink_new(assets_dir, view / "assets")
    return view


def _prepare_run(args: argparse.Namespace) -> Tuple[RunContext, List[support.TaskSource]]:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    runtime_dir = args.output_dir / "runtime" / run_id
    logs_dir = args.output_dir / "logs"
    manifests_dir = args.output_dir / "manifests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=False)
    # Keep configuration and outputs isolated per run, while reusing downloaded
    # model support files so a slow network cannot repeatedly block startup.
    SHARED_CACHE.mkdir(parents=True, exist_ok=True)
    home_dir = runtime_dir / "home"
    home_dir.mkdir()
    context = RunContext(
        run_id=run_id,
        started_at=_utc_now(),
        logs_dir=logs_dir,
        manifests_dir=manifests_dir,
        runtime_dir=runtime_dir,
        home_dir=home_dir,
        result_manifest=manifests_dir / ("result-%s.json" % run_id),
        preflight_manifest=manifests_dir / ("preflight-%s.json" % run_id),
        benchmark_data_root=args.benchmark_data_dir,
        summary_before=_file_token(args.output_dir / "summary.json"),
    )
    try:
        sources = support.discover_sources(
            args.benchmark,
            args.benchmark_data_dir,
            allow_missing_pro_cells=args.benchmark == "pro",
        )
        partial_pro = (
            args.benchmark == "pro" and not support.pro_compatibility(sources)["protocol_applicable"]
        )
        matrix = support.expand_matrix(
            args.benchmark,
            args.phase,
            sources,
            allow_incompatible_pro=partial_pro,
        )
        matrix = support.filter_matrix_by_suites(matrix, args.suite)
        if args.benchmark == "pro":
            context.benchmark_data_root = _build_pro_view(context, sources, args.assets_dir)
        else:
            context.benchmark_data_root = args.benchmark_data_dir
        config = _create_libero_home(args, context, context.benchmark_data_root)
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "status": "failed",
            "benchmark": args.benchmark,
            "phase": args.phase,
            "error_category": "environment",
            "stage": "benchmark_precheck",
            "error": "%s: %s" % (exc.__class__.__name__, exc),
            "trace": traceback.format_exc(),
            "policy_success_summary_created": False,
        }
        support.write_new_json(manifests_dir / ("benchmark-error-%s.json" % run_id), failure)
        support.write_new_json(context.result_manifest, failure)
        raise
    if args.benchmark == "pro":
        compatibility = support.pro_compatibility(sources)
        support.write_new_json(
            manifests_dir / ("compatibility-%s.json" % run_id), compatibility
        )
        context.events.append(
            _event(
                "benchmark_compatibility",
                "passed" if compatibility["protocol_applicable"] else "partial_incompatible",
                protocol_applicable=compatibility["protocol_applicable"],
                available_cells=compatibility["available_cells"],
                unavailable_cells=compatibility["unavailable_cells"],
            )
        )
    support.write_new_json(
        manifests_dir / ("sources-%s.json" % run_id),
        support.source_manifest(args.benchmark, args.benchmark_data_dir, sources),
    )
    support.write_new_json(
        manifests_dir / ("git-%s.json" % run_id),
        {
            "harness": _git_manifest(HERE),
            "benchmark": _git_manifest(args.benchmark_repo),
            "benchmark_data": _git_manifest(args.benchmark_data_dir),
            "openpi": _git_manifest(args.openpi_repo),
        },
    )
    support.write_new_json(
        manifests_dir / ("environment-%s.json" % run_id),
        {
            "run_id": run_id,
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "benchmark": args.benchmark,
            "phase": args.phase,
            "matrix_episodes": len(matrix),
            "suite_filter": args.suite,
            "port": args.port,
            "gpu_id": args.gpu_id,
            "server_python": args.server_python,
            "client_python": args.client_python,
            "benchmark_code_repo": str(args.benchmark_repo),
            "benchmark_data_dir": str(args.benchmark_data_dir),
            "benchmark_runtime_view": str(context.benchmark_data_root),
            "assets_dir": str(args.assets_dir),
            "libero_config": config,
            "fixed_environment": {
                "CUDA_VISIBLE_DEVICES": args.gpu_id,
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": args.gpu_id,
                "HOME": str(home_dir),
                "XDG_CACHE_HOME": str(SHARED_CACHE),
                "MAGICK_HOME": (
                    str(args.imagemagick_runtime / "prefix") if args.benchmark == "plus" else None
                ),
                "MAGICK_CONFIGURE_PATH": (
                    str(args.imagemagick_runtime / "root" / "etc" / "ImageMagick-6")
                    if args.benchmark == "plus"
                    else None
                ),
                "MAGICK_CODER_MODULE_PATH": (
                    str(_imagemagick_coder_dir(args.imagemagick_runtime))
                    if args.benchmark == "plus"
                    else None
                ),
            },
            "inventories": {
                "server_packages": _package_inventory(args.server_python),
                "client_packages": _package_inventory(args.client_python),
                "nvidia_smi": _nvidia_inventory(),
            },
        },
    )
    return context, sources


def _python_paths(args: argparse.Namespace, client: bool) -> List[pathlib.Path]:
    paths = [HERE, args.openpi_repo / "src"]
    if client:
        paths.extend(
            [
                args.benchmark_repo,
                args.openpi_repo / "packages" / "openpi-client" / "src",
            ]
        )
    return [path.resolve() for path in paths if path.exists()]


def _child_env(
    args: argparse.Namespace,
    context: RunContext,
    client: bool,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    env.update(
        {
            "HOME": str(context.home_dir),
            "XDG_CACHE_HOME": str(SHARED_CACHE),
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": args.gpu_id,
        }
    )
    if args.benchmark == "plus":
        prefix = args.imagemagick_runtime / "prefix"
        env["MAGICK_HOME"] = str(prefix)
        env["MAGICK_CONFIGURE_PATH"] = str(
            args.imagemagick_runtime / "root" / "etc" / "ImageMagick-6"
        )
        env["MAGICK_CODER_MODULE_PATH"] = str(_imagemagick_coder_dir(args.imagemagick_runtime))
        library = str(prefix / "lib")
        env["LD_LIBRARY_PATH"] = (
            library + os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else library
        )
    entries = [str(path) for path in _python_paths(args, client)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _validate_checkpoint(args: argparse.Namespace, context: RunContext) -> None:
    try:
        resolution = support.resolve_checkpoint(args.checkpoint_dir)
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_category": "checkpoint",
            "error": "%s: %s" % (exc.__class__.__name__, exc),
            "trace": traceback.format_exc(),
            "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
            "policy_success_summary_created": False,
        }
        support.write_new_json(context.manifests_dir / ("checkpoint-error-%s.json" % context.run_id), payload)
        context.events.append(_event("checkpoint_validation", "failed", **payload))
        raise
    provenance = resolution.provenance()
    provenance["norm_stats_sha256"] = support.sha256_file(resolution.norm_stats)
    support.write_new_json(
        context.manifests_dir / ("checkpoint-request-%s.json" % context.run_id), provenance
    )
    context.events.append(_event("checkpoint_validation", "passed"))


def _server_command(args: argparse.Namespace, context: RunContext) -> List[str]:
    return [
        args.server_python,
        str(SERVER),
        "--benchmark",
        args.benchmark,
        "--checkpoint-dir",
        str(args.checkpoint_dir.expanduser().resolve()),
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
        process = subprocess.Popen(
            _server_command(args, context),
            cwd=str(args.openpi_repo),
            env=_child_env(args, context, client=False),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        log.close()
        context.events.append(
            _event(
                "server_startup",
                "failed",
                error_category="environment",
                error="%s: %s" % (exc.__class__.__name__, exc),
                trace=traceback.format_exc(),
                log_path=str(log_path),
            )
        )
        raise
    return ServerHandle(process, log, log_path, started_at)


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    url = "http://127.0.0.1:%d/healthz" % port
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError("policy server exited during startup with code %d" % return_code)
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise TimeoutError("policy server did not become healthy within %.0f seconds" % timeout)


def _wait_server_audited(args: argparse.Namespace, context: RunContext, server: ServerHandle) -> None:
    try:
        _wait_for_server(server.process, args.port)
    except Exception as exc:
        diagnostic_path = context.manifests_dir / ("server-diagnostic-%s.json" % context.run_id)
        diagnostic = _read_json(diagnostic_path) or {}
        category = diagnostic.get("error_category") or "environment"
        context.events.append(
            _event(
                "server_startup",
                "failed",
                exit_code=server.process.poll(),
                error_category=category,
                error=diagnostic.get("error") or "%s: %s" % (exc.__class__.__name__, exc),
                trace=diagnostic.get("trace") or traceback.format_exc(),
                log_path=str(server.log_path),
                diagnostic_manifest=str(diagnostic_path) if diagnostic else None,
            )
        )
        raise
    context.events.append(_event("server_startup", "passed", log_path=str(server.log_path)))


def _stop_server(server: ServerHandle) -> None:
    # Only the exact child spawned by this run is ever signalled.
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
        "--benchmark",
        args.benchmark,
        "--phase",
        args.phase,
        "--benchmark-repo",
        str(context.benchmark_data_root),
        "--port",
        str(args.port),
        "--output-dir",
        str(args.output_dir),
        "--preflight-manifest",
        str(context.preflight_manifest),
    ]
    if args.resume:
        command.append("--resume")
    for suite in args.suite or []:
        command.extend(["--suite", suite])
    return command


def _run_client(args: argparse.Namespace, context: RunContext) -> int:
    log_path = context.logs_dir / ("client-egl-%s.log" % context.run_id)
    started_at = _utc_now()
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            _client_command(args, context),
            cwd=str(args.benchmark_repo),
            env=_child_env(args, context, client=True),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    category = None
    if completed.returncode == EGL_FAILURE_EXIT:
        category = "environment"
    elif completed.returncode:
        preflight = _read_json(context.preflight_manifest) or {}
        category = preflight.get("error_category") or "environment"
    context.events.append(
        _event(
            "client_egl",
            "passed" if completed.returncode == 0 else "failed",
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=completed.returncode,
            error_category=category,
            log_path=str(log_path),
            preflight_manifest=str(context.preflight_manifest),
        )
    )
    return completed.returncode


def _run_evaluation(args: argparse.Namespace, context: RunContext) -> int:
    server = _start_server(args, context)
    try:
        _wait_server_audited(args, context, server)
        return _run_client(args, context)
    finally:
        _stop_server(server)


def _result_payload(
    args: argparse.Namespace,
    context: RunContext,
    exit_code: int,
    terminal_error: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary_after = _file_token(args.output_dir / "summary.json")
    summary_created = summary_after is not None and summary_after != context.summary_before
    failed = [event for event in context.events if event.get("status") == "failed"]
    event_category = next(
        (event.get("error_category") for event in reversed(failed) if event.get("error_category")), None
    )
    category = terminal_error.get("error_category") if terminal_error else event_category
    partial = any(event.get("status") == "partial_incompatible" for event in context.events)
    return {
        "run_id": context.run_id,
        "benchmark": args.benchmark,
        "protocol": "official",
        "phase": args.phase,
        "suite_filter": args.suite,
        "started_at": context.started_at,
        "finished_at": _utc_now(),
        "status": ("partial_incompatible" if partial and exit_code == 0 else ("passed" if exit_code == 0 else "failed")),
        "exit_code": exit_code,
        "error_category": category,
        "error": terminal_error.get("error") if terminal_error else None,
        "trace": terminal_error.get("trace") if terminal_error else None,
        "preflight_manifest": str(context.preflight_manifest) if context.preflight_manifest.exists() else None,
        "policy_success_summary_created": summary_created,
        "summary_required": args.phase != "preflight",
        "protocol_applicable": not partial,
        "events": context.events,
    }


def orchestrate(args: argparse.Namespace) -> int:
    _check_args(args)
    context: Optional[RunContext] = None
    exit_code = 2
    terminal_error: Optional[Dict[str, Any]] = None
    try:
        context, _ = _prepare_run(args)
        _validate_checkpoint(args, context)
        exit_code = _run_evaluation(args, context)
        return exit_code
    except Exception as exc:
        category = support.classify_error(exc, "orchestration")
        terminal_error = {
            "error_category": category,
            "error": "%s: %s" % (exc.__class__.__name__, exc),
            "trace": traceback.format_exc(),
        }
        if context is not None:
            context.events.append(_event("orchestration", "failed", exit_code=exit_code, **terminal_error))
        raise
    finally:
        if context is not None:
            support.write_new_json(
                context.result_manifest,
                _result_payload(args, context, exit_code, terminal_error),
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return orchestrate(args)
    except Exception as exc:
        print("orchestration failed: %s: %s" % (exc.__class__.__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
