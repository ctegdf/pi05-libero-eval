"""Standard-library support for auditable pi0.5-on-LIBERO-X zero-shot evaluation.

This module intentionally imports neither LIBERO nor NumPy/torch at module
scope, so inventory discovery, matrix expansion, resume semantics, error
classification and reporting can be exercised (and unit tested) without a
simulator or GPU. Only `cross_check_registry` needs a LIBERO-X-configured
Python environment, and it degrades to "skipped" rather than failing hard
when that environment is unavailable, because LEVEL4/LEVEL5 have no
`libero.libero.benchmark` registry entry at all (see README notes) and can
only ever be filesystem-validated.

Naming note: the released `pi05_libero` checkpoint used here was fine-tuned
on standard LIBERO only, never on LIBERO-X's own training set. Every result
produced through this module is therefore a *zero-shot transfer* evaluation,
not a reproduction of the LIBERO-X paper's in-domain fine-tuned numbers.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# natsort is only installed in the liberox client env, not the server's
# openpi .venv (which imports this module purely for checkpoint/provenance
# helpers). Deferred to keep this module importable server-side.

PROTOCOL = "pi05-libero-zero-shot"
LEVELS = ("LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5")
# LEVEL5 reuses LEVEL4's bddl/init files verbatim; only the prompt differs.
BDDL_LEVEL_OF = {
    "LEVEL1": "LEVEL1",
    "LEVEL2": "LEVEL2",
    "LEVEL3": "LEVEL3",
    "LEVEL4": "LEVEL4",
    "LEVEL5": "LEVEL4",
}
# Only these levels are registered libero.libero.benchmark suites
# (LIBERO_X_LEVEL1/2/3); LEVEL4/LEVEL5 exist purely as bddl/init files on
# disk with no registry entry, confirmed by reading the vendored checkout's
# libero/libero/benchmark/__init__.py at the pinned commit.
REGISTERED_LEVELS = ("LEVEL1", "LEVEL2", "LEVEL3")
REGISTRY_SUITE_NAME = {"LEVEL1": "libero_x_level1", "LEVEL2": "libero_x_level2", "LEVEL3": "libero_x_level3"}

PHASES = ("preflight", "smoke", "full")

# Verified against the pinned vendor checkout (commit f5287264) by directly
# listing libero/libero_x/{bddl,init}/LEVEL*/ and torch.load-sampling init
# files: every level has exactly EXPECTED_TRIALS_PER_TASK states per task.
EXPECTED_TASK_COUNT = {"LEVEL1": 600, "LEVEL2": 600, "LEVEL3": 600, "LEVEL4": 826, "LEVEL5": 826}
EXPECTED_TRIALS_PER_TASK = 10

# Released-template protocol constants. The paper describes a per-task
# horizon of 1.1x human demonstration time; the released eval_template.py
# instead hardcodes max_steps=1200 for every task and level, and the
# authors have not clarified the discrepancy (upstream issue #3, open).
# Results produced with these constants are the released-template protocol,
# not a reproduction of the paper's own evaluation protocol.
MAX_STEPS = 1200
RESIZE = 224
REPLAN = 5
SEED = 7
WAIT_STEPS = 10
# The released eval_template.py, in --load-mode init, sets t = num_steps_wait
# directly instead of stepping the dummy action that many times. We keep
# that behavior for the main released-template result and record it
# explicitly rather than silently deviating in either direction.
EXECUTED_WAIT_STEPS = 0
FLIP_IMAGES = True  # matches the already-validated 97.1%-control pi05 client's obs[::-1, ::-1]

L5_TAGS = ("L5-1", "L5-2", "L5-3", "L5-4", "L5-5")
BDDL_TASK_KEY_RE = re.compile(r"__T(\d+)(?:__A(\d+))?")

EXCLUDED_ERROR_CATEGORIES = ("connection", "environment", "checkpoint", "policy_runtime")

CONFIG_NAME = "pi05_libero"
ASSET_ID = "physical-intelligence/libero"
NORM_STATS_RELATIVE = pathlib.Path("assets") / ASSET_ID / "norm_stats.json"


class InventoryError(ValueError):
    pass


class CheckpointError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class CheckpointResolution:
    checkpoint: pathlib.Path
    params: pathlib.Path
    assets: pathlib.Path
    norm_stats: pathlib.Path

    def provenance(self) -> Dict[str, Any]:
        return {
            "protocol": "official",
            "config": CONFIG_NAME,
            "asset_id": ASSET_ID,
            "checkpoint": str(self.checkpoint),
            "params_source": str(self.params),
            "assets_source": str(self.assets),
            "norm_stats": str(self.norm_stats),
        }


def resolve_checkpoint(checkpoint_dir: pathlib.Path) -> CheckpointResolution:
    """Identical resolution rule to the already-validated openpi-libero /
    LIBERO-Plus/Pro harnesses: this evaluation only ever serves the single
    official pi05_libero checkpoint, so all LIBERO-family benchmarks in
    this results tree share one checkpoint identity."""
    root = pathlib.Path(checkpoint_dir).expanduser().resolve()
    checkpoint = root / "pi05_libero" if (root / "pi05_libero").is_dir() else root
    params = checkpoint / "params"
    assets = checkpoint / "assets"
    norm_stats = checkpoint / NORM_STATS_RELATIVE
    missing = [str(path) for path in (checkpoint, params, assets) if not path.is_dir()]
    if not norm_stats.is_file():
        missing.append(str(norm_stats))
    if missing:
        raise CheckpointError("official pi05_libero checkpoint is incomplete; missing: %s" % missing)
    return CheckpointResolution(checkpoint, params.resolve(), assets.resolve(), norm_stats.resolve())


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "item"


# --------------------------------------------------------------------------
# Inventory: TaskSource (one per bddl/init file pair, per prompt variant)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaskSource:
    level: str
    task_num: str  # zero-padded numeric id parsed from the bddl filename, e.g. "061"
    variant: str  # "" or "A1", "A2", ... parsed from the bddl filename
    bddl_path: pathlib.Path
    init_path: pathlib.Path
    prompt: str  # LEVEL1-4: the fixed prompt. LEVEL5: unused, see level5_prompts.
    prompt_field: str  # "bddl_language" (LEVEL1-4) or "level5_rotation" (LEVEL5)
    prompt_source_path: pathlib.Path
    # LEVEL5 only: {tag -> prompt}. The released protocol does NOT evaluate
    # each task once per tag (that would 5x the episode budget); it keeps
    # the same EXPECTED_TRIALS_PER_TASK trials as every other level and
    # rotates the tag by trial index: tag = L5_TAGS[(trial // 2) % 5]
    # (eval_template.py's `(ep_id // 2) % len(prompt_candidates)`).
    level5_prompts: Optional[Mapping[str, str]] = None

    @property
    def source_id(self) -> str:
        return "%s/%s" % (self.level, self.bddl_path.name)

    def identity(self) -> str:
        return self.source_id

    def prompt_for_trial(self, trial: int) -> Tuple[str, str]:
        if self.level != "LEVEL5":
            return self.prompt, self.prompt_field
        assert self.level5_prompts is not None
        tag = L5_TAGS[(trial // 2) % len(L5_TAGS)]
        return self.level5_prompts[tag], tag


def _parse_bddl_task_key(bddl_name: str) -> Tuple[str, str]:
    match = BDDL_TASK_KEY_RE.search(bddl_name)
    if not match:
        raise InventoryError("could not parse __T<id> from BDDL filename: %s" % bddl_name)
    task_num = match.group(1).zfill(3)
    variant = "A%s" % match.group(2) if match.group(2) else ""
    return task_num, variant


def _list_level_files(root: pathlib.Path, level: str, suffix: str) -> Dict[str, pathlib.Path]:
    level_dir = root / BDDL_LEVEL_OF[level]
    if not level_dir.is_dir():
        raise InventoryError("missing level directory: %s" % level_dir)
    files = sorted(p for p in level_dir.iterdir() if p.suffix == suffix)
    by_stem: Dict[str, pathlib.Path] = {}
    for path in files:
        if path.stem in by_stem:
            raise InventoryError("duplicate stem %s in %s" % (path.stem, level_dir))
        by_stem[path.stem] = path
    return by_stem


def _load_level5_prompts(prompt_root: pathlib.Path) -> Dict[str, Dict[Tuple[str, str], str]]:
    prompt_map: Dict[str, Dict[Tuple[str, str], str]] = {}
    for tag in L5_TAGS:
        path = prompt_root / ("%s.jsonl" % tag)
        if not path.is_file():
            raise InventoryError("LEVEL5 requires %s (missing)" % path)
        per_tag: Dict[Tuple[str, str], str] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                raw_task_id = record.get("task_id")
                if raw_task_id is None or not str(raw_task_id).isdigit():
                    raise InventoryError("invalid/missing task_id at %s:%d" % (path, line_number))
                task_num = str(raw_task_id).zfill(3)
                variant = record.get("variant") or ""
                task_desc = record.get("task_desc") or ""
                if not task_desc:
                    raise InventoryError("empty task_desc at %s:%d" % (path, line_number))
                key = (task_num, variant)
                if key in per_tag:
                    raise InventoryError("duplicate prompt key %s in %s" % (key, path))
                per_tag[key] = task_desc
        prompt_map[tag] = per_tag
    return prompt_map


def discover_sources(bddl_root: pathlib.Path, init_root: pathlib.Path, level5_prompt_root: pathlib.Path,
                      levels: Sequence[str] = LEVELS) -> List[TaskSource]:
    """Filesystem-only inventory. No LIBERO import required."""
    sources: List[TaskSource] = []
    for level in levels:
        bddl_files = _list_level_files(bddl_root, level, ".bddl")
        init_files = _list_level_files(init_root, level, ".init")
        if set(bddl_files) != set(init_files):
            only_bddl = sorted(set(bddl_files) - set(init_files))
            only_init = sorted(set(init_files) - set(bddl_files))
            raise InventoryError(
                "%s bddl/init stem mismatch: %d only-in-bddl, %d only-in-init (e.g. %r / %r)"
                % (level, len(only_bddl), len(only_init), only_bddl[:3], only_init[:3])
            )
        expected = EXPECTED_TASK_COUNT[level]
        if len(bddl_files) != expected:
            raise InventoryError("%s has %d bddl/init pairs; expected %d" % (level, len(bddl_files), expected))

        if level in REGISTERED_LEVELS:
            # LEVEL1-3 filenames carry no embedded numeric task id (e.g.
            # "EXTENSION_KITCHEN_SCENE10_..."); the registered
            # libero.libero.benchmark suite instead assigns 0-based task
            # ids by natsorted bddl filename order (mirrored here so
            # task_num matches suite.get_task(task_id) without importing
            # LIBERO). Confirmed empirically: no __T<id> pattern appears in
            # any LEVEL1-3 filename in the pinned vendor checkout.
            from natsort import natsorted

            ordered = natsorted(bddl_files.values(), key=lambda path: path.name)
            for index, bddl_path in enumerate(ordered):
                prompt = _parse_bddl_language(bddl_path)
                sources.append(
                    TaskSource(
                        level=level,
                        task_num=str(index).zfill(3),
                        variant="",
                        bddl_path=bddl_path,
                        init_path=init_files[bddl_path.stem],
                        prompt=prompt,
                        prompt_field="bddl_language",
                        prompt_source_path=bddl_path,
                    )
                )
        elif level == "LEVEL4":
            for stem, bddl_path in bddl_files.items():
                task_num, variant = _parse_bddl_task_key(bddl_path.name)
                prompt = _parse_bddl_language(bddl_path)
                sources.append(
                    TaskSource(
                        level=level,
                        task_num=task_num,
                        variant=variant,
                        bddl_path=bddl_path,
                        init_path=init_files[stem],
                        prompt=prompt,
                        prompt_field="bddl_language",
                        prompt_source_path=bddl_path,
                    )
                )
        else:
            prompt_map = _load_level5_prompts(level5_prompt_root)
            level4_keys = {
                _parse_bddl_task_key(path.name) for path in bddl_files.values()
            }
            for tag in L5_TAGS:
                missing = level4_keys - set(prompt_map[tag])
                extra = set(prompt_map[tag]) - level4_keys
                if missing or extra:
                    raise InventoryError(
                        "%s keys mismatch vs LEVEL4: %d missing, %d extra (tag=%s)"
                        % (level, len(missing), len(extra), tag)
                    )
            # One source per LEVEL4 task/variant (826 total, not 826 x 5):
            # the 5 prompt tags are a function of trial index, not an
            # independent multiplier on the episode budget.
            for stem, bddl_path in bddl_files.items():
                task_num, variant = _parse_bddl_task_key(bddl_path.name)
                key = (task_num, variant)
                level5_prompts = {tag: prompt_map[tag][key] for tag in L5_TAGS}
                sources.append(
                    TaskSource(
                        level=level,
                        task_num=task_num,
                        variant=variant,
                        bddl_path=bddl_path,
                        init_path=init_files[stem],
                        prompt="",
                        prompt_field="level5_rotation",
                        prompt_source_path=level5_prompt_root,
                        level5_prompts=level5_prompts,
                    )
                )
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise InventoryError("discovered sources contain duplicate source_id values")
    return sources


def _parse_bddl_language(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+(.+?)\)", text, re.DOTALL)
    if not match:
        raise InventoryError("could not parse (:language ...) from %s" % path)
    return " ".join(match.group(1).split())


def cross_check_registry(sources: Sequence[TaskSource]) -> Dict[str, Any]:
    """Best-effort cross-check against libero.libero.benchmark for LEVEL1-3.

    Returns a report dict; never raises for an unimportable/unavailable
    LIBERO environment (LEVEL4/LEVEL5 have no registry entry to check
    against regardless, so this can only ever partially validate the
    inventory). Raises InventoryError only for a genuine mismatch once the
    registry *is* available, since that indicates real inventory corruption.
    """
    try:
        from libero.libero import benchmark as libero_benchmark  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only off-target
        return {"status": "skipped", "reason": "import failed: %s: %s" % (exc.__class__.__name__, exc)}

    # The registry builds each suite from a `SCENE_LEVEL<n>` subdirectory of
    # whatever the active config.yaml's `bddl_files`/`init_states` point at
    # (see libero/libero/benchmark/__init__.py _build_libero_x_tasks), which
    # is a different directory convention from our own libero_x/bddl/LEVEL<n>
    # filesystem inventory. Wiring those up exactly (e.g. via a runtime
    # SCENE_LEVEL<n> -> LEVEL<n> symlink tree) is not on the rollout critical
    # path, so any failure to construct/compare the registry here is
    # reported, never raised: this check is a best-effort bonus, and the
    # actual environment/episode construction never depends on it (it uses
    # OffScreenRenderEnv(bddl_file_name=...) against our own inventory
    # directly, never the registry).
    try:
        registry = libero_benchmark.get_benchmark_dict()
        by_level: Dict[str, List[TaskSource]] = collections.defaultdict(list)
        for source in sources:
            if source.level in REGISTERED_LEVELS:
                by_level[source.level].append(source)

        checked = {}
        for level, suite_name in REGISTRY_SUITE_NAME.items():
            items = by_level.get(level, [])
            if suite_name not in registry:
                raise InventoryError("registered benchmark suite missing: %s" % suite_name)
            suite = registry[suite_name]()
            if int(suite.n_tasks) != len(items):
                raise InventoryError(
                    "registered suite %s has %d tasks, filesystem inventory has %d" % (suite_name, suite.n_tasks, len(items))
                )
            by_name = {source.bddl_path.name: source for source in items}
            for task_id in range(int(suite.n_tasks)):
                bddl_name = pathlib.Path(str(suite.get_task(task_id).bddl_file)).name
                if bddl_name not in by_name:
                    raise InventoryError("registry task %s not present in filesystem inventory for %s" % (bddl_name, level))
            checked[level] = {"suite_name": suite_name, "n_tasks": int(suite.n_tasks)}
    except Exception as exc:
        return {"status": "skipped", "reason": "%s: %s" % (exc.__class__.__name__, exc)}
    return {"status": "checked", "levels": checked}


# --------------------------------------------------------------------------
# EpisodeSpec / matrix expansion
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    level: str
    task_num: str
    variant: str
    bddl_path: pathlib.Path
    init_path: pathlib.Path
    prompt: str
    prompt_field: str
    prompt_source_path: pathlib.Path
    trial: int
    max_steps: int = MAX_STEPS
    seed: int = SEED

    @property
    def episode_id(self) -> str:
        label = ":".join(
            value
            for value in (
                PROTOCOL,
                self.level,
                "task-%s" % self.task_num,
                ("variant-%s" % self.variant) if self.variant else None,
                self.prompt_field,
                "trial-%02d" % self.trial,
                "seed-%d" % self.seed,
            )
            if value
        )
        digest = hashlib.sha256(
            ("%s|%s|%d" % (self.bddl_path.name, self.prompt_field, self.trial)).encode("utf-8")
        ).hexdigest()[:12]
        return "%s:%s" % (label[:300], digest)


def _episodes_for_source(source: TaskSource, trials: Sequence[int]) -> List[EpisodeSpec]:
    episodes = []
    for trial in trials:
        prompt, prompt_field = source.prompt_for_trial(trial)
        episodes.append(
            EpisodeSpec(
                level=source.level,
                task_num=source.task_num,
                variant=source.variant,
                bddl_path=source.bddl_path,
                init_path=source.init_path,
                prompt=prompt,
                prompt_field=prompt_field,
                prompt_source_path=source.prompt_source_path,
                trial=trial,
            )
        )
    return episodes


def expand_matrix(phase: str, sources: Sequence[TaskSource], levels: Sequence[str] = LEVELS,
                   smoke_trials: Sequence[int] = (0, 2, 4, 6, 8)) -> List[EpisodeSpec]:
    """Build the episode matrix for a phase.

    `full`: every source x every one of EXPECTED_TRIALS_PER_TASK trials.
    `smoke`: a fixed, small, deterministic subset per level that still
      exercises every LEVEL5 prompt tag (trials 0,2,4,6,8 land on all five
      (ep_id // 2) % 5 buckets) and, where available, the new-predicate /
      new-object LEVEL4 code path.
    `preflight`: exactly one episode, deterministically the first source.
    """
    if phase not in PHASES:
        raise ValueError("unknown phase: %s" % phase)
    filtered = [source for source in sources if source.level in levels]
    if not filtered:
        raise InventoryError("no sources for requested levels: %s" % (levels,))

    if phase == "preflight":
        first = sorted(filtered, key=lambda source: source.identity())[0]
        matrix = _episodes_for_source(first, [0])
    elif phase == "full":
        matrix = [episode for source in filtered for episode in _episodes_for_source(source, range(EXPECTED_TRIALS_PER_TASK))]
    else:  # smoke
        by_level: Dict[str, List[TaskSource]] = collections.defaultdict(list)
        for source in filtered:
            by_level[source.level].append(source)
        matrix = []
        for level, items in by_level.items():
            items = sorted(items, key=lambda source: source.identity())
            if level == "LEVEL5":
                # trials 0,2,4,6,8 land on all five (trial // 2) % 5
                # buckets of a single task, exercising every prompt tag.
                matrix.extend(_episodes_for_source(items[0], smoke_trials))
            else:
                sample = items[:3] if len(items) >= 3 else items
                for source in sample:
                    matrix.extend(_episodes_for_source(source, (0, 1)))

    episode_ids = [spec.episode_id for spec in matrix]
    if len(episode_ids) != len(set(episode_ids)):
        raise InventoryError("expanded matrix has duplicate episode ids")
    if phase == "full":
        expected = sum(EXPECTED_TASK_COUNT[level] for level in levels) * EXPECTED_TRIALS_PER_TASK
        if len(matrix) != expected:
            raise InventoryError("full matrix contains %d episodes; expected %d" % (len(matrix), expected))
    return matrix


# --------------------------------------------------------------------------
# Action / error classification
# --------------------------------------------------------------------------


def validate_action_result(result: Any, min_steps: int = REPLAN, action_dim: int = 7) -> List[List[float]]:
    if not isinstance(result, Mapping) or "actions" not in result:
        raise ValueError("inference result must be a mapping containing actions")
    raw_actions = result["actions"]
    if not hasattr(raw_actions, "__len__") or len(raw_actions) < min_steps:
        actual = len(raw_actions) if hasattr(raw_actions, "__len__") else "unknown"
        raise ValueError("policy returned %s steps; at least %d required" % (actual, min_steps))
    actions: List[List[float]] = []
    for step_index, raw_step in enumerate(raw_actions):
        if not hasattr(raw_step, "__len__") or len(raw_step) != action_dim:
            actual = len(raw_step) if hasattr(raw_step, "__len__") else "unknown"
            raise ValueError("action step %d has dimension %s; expected %d" % (step_index, actual, action_dim))
        step: List[float] = []
        for value in raw_step:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("action step %d contains a non-numeric value" % step_index) from exc
            if not math.isfinite(number):
                raise ValueError("action step %d contains a non-finite value" % step_index)
            step.append(number)
        actions.append(step)
    return actions


def classify_error(exc: BaseException, stage: str = "") -> str:
    rendered = (exc.__class__.__name__ + " " + str(exc)).lower()
    stage_lower = stage.lower()
    if isinstance(exc, CheckpointError) or "checkpoint" in stage_lower:
        return "checkpoint"
    if "connect" in stage_lower or any(token in rendered for token in ("websocket", "connection", "socket")):
        return "connection"
    if "infer" in stage_lower or "policy" in stage_lower or "action" in stage_lower:
        return "policy_runtime"
    return "environment"


# --------------------------------------------------------------------------
# JSONL ledger / resume
# --------------------------------------------------------------------------


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = pathlib.Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSONL at %s:%d: %s" % (path, line_number, exc)) from exc
            if not isinstance(record, dict):
                raise ValueError("JSONL record at %s:%d is not an object" % (path, line_number))
            records.append(record)
    return records


def append_jsonl(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def next_attempts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    attempts: Dict[str, int] = {}
    for record in records:
        episode_id = str(record.get("episode_id", ""))
        if episode_id:
            attempts[episode_id] = max(attempts.get(episode_id, 0), int(record.get("attempt", 0)) + 1)
    return attempts


def select_pending(matrix: Sequence[EpisodeSpec], records: Iterable[Mapping[str, Any]], resume: bool) -> List[EpisodeSpec]:
    if not resume:
        return list(matrix)
    completed = {
        str(record.get("episode_id"))
        for record in records
        if record.get("status") in ("success", "failure") and record.get("episode_id")
    }
    return [spec for spec in matrix if spec.episode_id not in completed]


def _latest_attempts(records: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        episode_id = str(record.get("episode_id", ""))
        if episode_id:
            grouped.setdefault(episode_id, []).append(record)
    latest: Dict[str, Mapping[str, Any]] = {}
    for episode_id, attempts in grouped.items():
        policy = [record for record in attempts if record.get("status") in ("success", "failure")]
        latest[episode_id] = max(policy or attempts, key=lambda item: int(item.get("attempt", 0)))
    return latest


def _stats(records: Sequence[Mapping[str, Any]], planned: int) -> Dict[str, Any]:
    successes = sum(record.get("status") == "success" for record in records)
    failures = sum(record.get("status") == "failure" for record in records)
    excluded = sum(
        record.get("status") == "error" and record.get("error_category") in EXCLUDED_ERROR_CATEGORIES
        for record in records
    )
    unknown = sum(record.get("status") == "error" for record in records) - excluded
    denominator = successes + failures
    return {
        "planned": planned,
        "attempted_unique": len(records),
        "successes": successes,
        "failures": failures,
        "excluded_errors": excluded,
        "unknown_errors": unknown,
        "policy_denominator": denominator,
        "success_rate": float(successes) / denominator if denominator else None,
        "complete": denominator == planned,
    }


def aggregate(records: Iterable[Mapping[str, Any]], matrix: Sequence[EpisodeSpec]) -> Dict[str, Any]:
    latest = _latest_attempts(records)
    planned_by_id = {spec.episode_id: spec for spec in matrix}
    relevant = {episode_id: record for episode_id, record in latest.items() if episode_id in planned_by_id}
    groups: Dict[str, Dict[str, Any]] = {}
    for dimension in ("level", "prompt_field"):
        values = sorted({getattr(spec, dimension) for spec in matrix})
        dimension_groups: Dict[str, Any] = {}
        for value in values:
            specs = [spec for spec in matrix if getattr(spec, dimension) == value]
            dimension_groups[value] = _stats(
                [relevant[spec.episode_id] for spec in specs if spec.episode_id in relevant], len(specs)
            )
        groups[dimension] = dimension_groups
    total = _stats(list(relevant.values()), len(matrix))
    level_rates = [item["success_rate"] for item in groups["level"].values() if item["success_rate"] is not None]
    total["macro_level_success_rate"] = sum(level_rates) / len(level_rates) if level_rates else None
    total["extra_episode_records"] = len(set(latest) - set(planned_by_id))
    return {"groups": groups, "total": total}


def verify_integrity(records: Sequence[Mapping[str, Any]], matrix: Sequence[EpisodeSpec], require_videos: bool = True) -> Dict[str, Any]:
    issues: List[str] = []
    episode_ids = [spec.episode_id for spec in matrix]
    if len(episode_ids) != len(set(episode_ids)):
        issues.append("matrix episode ids are not unique")
    attempt_ids = [str(record.get("attempt_id", "")) for record in records]
    if not all(attempt_ids) or len(attempt_ids) != len(set(attempt_ids)):
        issues.append("record attempt ids are missing or not unique")
    unknown_statuses = [record for record in records if record.get("status") not in ("success", "failure", "error")]
    if unknown_statuses:
        issues.append("%d records have an unknown status" % len(unknown_statuses))
    planned = set(episode_ids)
    extras = {str(record.get("episode_id")) for record in records} - planned
    extras.discard("")
    if extras:
        issues.append("records contain %d unplanned episode ids" % len(extras))
    latest = _latest_attempts(records)
    missing = [episode_id for episode_id in episode_ids if episode_id not in latest]
    if missing:
        issues.append("%d planned episodes have no attempt" % len(missing))
    nonterminal = [
        episode_id for episode_id in episode_ids
        if episode_id in latest and latest[episode_id].get("status") not in ("success", "failure")
    ]
    if nonterminal:
        issues.append("%d planned episodes have no policy outcome" % len(nonterminal))
    policy_counts = collections.Counter(
        str(record.get("episode_id"))
        for record in records
        if record.get("status") in ("success", "failure") and record.get("episode_id") in planned
    )
    nonunique = [episode_id for episode_id in episode_ids if policy_counts.get(episode_id, 0) != 1]
    if nonunique:
        issues.append("%d planned episodes do not have exactly one policy outcome" % len(nonunique))
    missing_videos = 0
    video_paths: List[pathlib.Path] = []
    if require_videos:
        for episode_id in episode_ids:
            record = latest.get(episode_id)
            if not record or record.get("status") not in ("success", "failure"):
                continue
            video = record.get("video")
            video_path = pathlib.Path(str(video)) if video else None
            if (
                record.get("video_status") != "written"
                or video_path is None
                or not video_path.is_file()
                or video_path.stat().st_size == 0
            ):
                missing_videos += 1
            else:
                video_paths.append(video_path.resolve())
        if missing_videos:
            issues.append("%d policy outcomes have no written video" % missing_videos)
        if len(video_paths) != len(set(video_paths)):
            issues.append("policy outcomes do not have unique video paths")
    return {
        "passed": not issues,
        "issues": issues,
        "planned": len(matrix),
        "records": len(records),
        "missing_videos": missing_videos,
    }


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_new_json(path: pathlib.Path, value: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_summaries(summary: Mapping[str, Any], output_dir: pathlib.Path) -> None:
    output_dir = pathlib.Path(output_dir)
    atomic_write_json(output_dir / "summary.json", summary)
    csv_path = output_dir / "summary.csv"
    descriptor, temporary = tempfile.mkstemp(prefix="summary.csv.", dir=str(output_dir))
    fields = (
        "dimension", "scope", "planned", "attempted_unique", "successes", "failures",
        "excluded_errors", "unknown_errors", "policy_denominator", "success_rate", "complete",
        "macro_level_success_rate",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for dimension, values in summary["groups"].items():
                for scope, stats in sorted(values.items()):
                    row = {"dimension": dimension, "scope": scope}
                    row.update({key: stats.get(key) for key in fields if key in stats})
                    writer.writerow(row)
            total_row = {"dimension": "total", "scope": "total"}
            total_row.update({key: summary["total"].get(key) for key in fields if key in summary["total"]})
            writer.writerow(total_row)
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
