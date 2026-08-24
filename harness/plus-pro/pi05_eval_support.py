"""Standard-library support for auditable pi0.5 LIBERO-Plus/Pro evaluation.

The module intentionally imports neither LIBERO nor NumPy.  Repository
inventory, protocol expansion, resume semantics, error classification and
reporting can therefore be tested with the Python 3.8 client interpreter
before any simulator is started.
"""

from __future__ import annotations

import ast
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


BENCHMARKS: Tuple[str, ...] = ("plus", "pro")
PHASES: Tuple[str, ...] = ("preflight", "smoke", "full")
SUITES: Tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS: Mapping[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
PLUS_COUNTS: Mapping[str, int] = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
PLUS_FULL_EPISODES = 10030
PRO_PERTURBATIONS: Tuple[str, ...] = (
    "object",
    "swap(position)",
    "lan(semantic)",
    "task",
    "env",
)
PRO_TASKS_PER_CELL = 10
PRO_TRIALS = 50
PRO_FULL_EPISODES = 10000
CONFIG_NAME = "pi05_libero"
ASSET_ID = "physical-intelligence/libero"
NORM_STATS_RELATIVE = pathlib.Path("assets") / ASSET_ID / "norm_stats.json"
EXCLUDED_ERROR_CATEGORIES: Set[str] = {
    "environment",
    "connection",
    "policy_runtime",
    "checkpoint",
}


class BenchmarkInventoryError(ValueError):
    """The checked-out benchmark does not implement the required protocol."""


class CheckpointError(ValueError):
    """The official pi05_libero checkpoint is missing or inconsistent."""


@dataclasses.dataclass(frozen=True)
class TaskSource:
    benchmark: str
    suite: str
    task_id: int
    source_id: str
    bddl_path: pathlib.Path
    init_path: pathlib.Path
    prompt: str
    prompt_field: str = "language_instruction"
    category: Optional[str] = None
    difficulty: Optional[str] = None
    perturbation: Optional[str] = None
    prompt_source_path: Optional[pathlib.Path] = None

    def identity(self) -> str:
        return "%s/%s/%s" % (self.benchmark, self.suite, self.source_id)


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    benchmark: str
    suite: str
    task_id: int
    source_id: str
    bddl_path: pathlib.Path
    init_path: pathlib.Path
    prompt: str
    prompt_field: str
    trial: int
    max_steps: int
    seed: int = 7
    category: Optional[str] = None
    difficulty: Optional[str] = None
    perturbation: Optional[str] = None
    prompt_source_path: Optional[pathlib.Path] = None

    @property
    def episode_id(self) -> str:
        label = ":".join(
            value
            for value in (
                self.benchmark,
                self.suite,
                self.perturbation,
                self.category,
                self.difficulty,
                "task-%04d" % self.task_id,
                "source-%s" % _slug(self.source_id),
                "trial-%02d" % self.trial,
                "seed-%d" % self.seed,
            )
            if value
        )
        # Long natural-language task names remain readable while the digest
        # guarantees that truncation cannot collide.
        digest = hashlib.sha256(self.source_id.encode("utf-8")).hexdigest()[:12]
        return "%s:%s" % (label[:300], digest)

    def dimensions(self) -> Dict[str, Optional[str]]:
        return {
            "suite": self.suite,
            "category": self.category,
            "difficulty": self.difficulty,
            "perturbation": self.perturbation,
        }


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


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "item"


def _require_dir(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_dir():
        raise BenchmarkInventoryError("%s directory is missing: %s" % (label, path))
    return path


def _find_one(root: pathlib.Path, names: Sequence[str], label: str) -> pathlib.Path:
    direct = [root / name for name in names if (root / name).is_dir()]
    if direct:
        return min(direct, key=lambda path: len(path.parts)).resolve()
    matches = [path for path in root.rglob("*") if path.is_dir() and path.name in names]
    if not matches:
        raise BenchmarkInventoryError("%s not found below %s (names=%s)" % (label, root, list(names)))
    return min(matches, key=lambda path: (len(path.parts), str(path))).resolve()


def canonical_suite(value: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if "libero_spatial" in normalized or normalized == "spatial":
        return "libero_spatial"
    if "libero_object" in normalized or normalized == "object":
        return "libero_object"
    if "libero_goal" in normalized or normalized == "goal":
        return "libero_goal"
    if "libero_10" in normalized or normalized in ("libero10", "long", "long_horizon"):
        return "libero_10"
    return None


def canonical_perturbation(value: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "object": {"object", "object_perturbation", "object_perturbations"},
        "swap(position)": {"swap", "position", "swap_position", "position_perturbation"},
        "lan(semantic)": {"lan", "language", "semantic", "lan_semantic", "semantic_perturbation"},
        "task": {"task", "task_perturbation", "task_perturbations"},
        "env": {"env", "environment", "env_perturbation", "environment_perturbation"},
    }
    for canonical, values in aliases.items():
        if normalized in values:
            return canonical
    return None


def _extract_bddl_field(text: str, field: str) -> Optional[str]:
    # BDDL fields are shallow s-expressions.  Prompts do not contain closing
    # parentheses in either official benchmark; accepting quoted or bare text
    # avoids deriving semantics from filenames.
    pattern = r"\(\s*:?%s\b\s*(.*?)\)" % re.escape(field)
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw = " ".join(match.group(1).strip().split())
    if not raw:
        return None
    if raw[0:1] in ("\"", "'"):
        try:
            decoded = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            decoded = raw.strip("\"'")
        raw = str(decoded)
    return raw.strip()


def parse_bddl_prompt(path: pathlib.Path) -> Tuple[str, str]:
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    prompt = _extract_bddl_field(text, "language_instruction")
    field = "language_instruction"
    if prompt is None:
        # Some upstream LIBERO revisions call the exact same in-file field
        # ``language``.  This is still BDDL-derived and never a filename
        # reconstruction; the manifest records which field was used.
        prompt = _extract_bddl_field(text, "language")
        field = "language"
    if prompt is None:
        raise BenchmarkInventoryError("BDDL has no language_instruction/language field: %s" % path)
    return prompt, field


def _index_files(root: pathlib.Path, suffixes: Sequence[str]) -> List[pathlib.Path]:
    wanted = tuple(value.lower() for value in suffixes)
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.name.lower().endswith(wanted)),
        key=str,
    )


def prefer_official_init_files(paths: Sequence[pathlib.Path]) -> List[pathlib.Path]:
    """Select only official ``.pruned_init`` evaluation states.

    The Pro HF snapshot intentionally ships both ``.init`` and
    ``.pruned_init`` for each task.  LIBERO's evaluation API consumes the
    latter; treating ``.init`` as an equivalent fallback would silently
    change the evaluation distribution.  Both current Plus and Pro releases
    provide pruned artifacts for every runnable task, so absence is a hard
    inventory error rather than a fallback opportunity.
    """
    selected = sorted(
        {path.resolve() for path in paths if path.name.lower().endswith(".pruned_init")}, key=str
    )
    if not selected:
        raise BenchmarkInventoryError("no official .pruned_init files were found")
    return selected


def _suite_from_path(path: pathlib.Path) -> Optional[str]:
    for component in reversed(path.parts):
        suite = canonical_suite(component)
        if suite is not None:
            return suite
    return canonical_suite(path.stem)


def _stem_variants(path: pathlib.Path) -> Set[str]:
    name = path.name
    for suffix in (".pruned_init", ".init", ".npy", ".npz", ".pt", ".pth", ".bddl"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return {name, name.lower(), _slug(name).lower()}


def _build_file_index(paths: Sequence[pathlib.Path]) -> Dict[Tuple[Optional[str], str], List[pathlib.Path]]:
    index: Dict[Tuple[Optional[str], str], List[pathlib.Path]] = {}
    for path in paths:
        suite = _suite_from_path(path)
        for stem in _stem_variants(path):
            index.setdefault((suite, stem), []).append(path)
            index.setdefault((None, stem), []).append(path)
    return index


def _lookup_named_file(
    index: Mapping[Tuple[Optional[str], str], Sequence[pathlib.Path]], suite: str, name: str, label: str
) -> pathlib.Path:
    raw = pathlib.Path(name).name
    variants = _stem_variants(pathlib.Path(raw))
    candidates: List[pathlib.Path] = []
    for stem in variants:
        candidates.extend(index.get((suite, stem), ()))
    candidates = sorted(set(candidates), key=str)
    if not candidates:
        for stem in variants:
            candidates.extend(index.get((None, stem), ()))
        candidates = sorted(set(candidates), key=str)
    if len(candidates) != 1:
        raise BenchmarkInventoryError(
            "%s lookup for suite=%s name=%s yielded %d files: %s"
            % (label, suite, name, len(candidates), [str(path) for path in candidates[:20]])
        )
    return candidates[0]


_PLUS_NEWOBJ_MARKERS = ("_add_", "_level")


def _plus_original_init_stem(task_name: str) -> Optional[str]:
    """Mirror LIBERO-Plus' mapping from a variant task to its base init file.

    The upstream benchmark gives language and camera variants compound suffixes,
    while table, texture-background and light variants end in a numbered
    suffix.  Matching the numbered variants only at the end is important:
    several unperturbed task names contain ordinary text such as
    ``from_table_center``.
    """
    stem = pathlib.Path(task_name).stem
    lowered = stem.lower()
    for marker in ("_language_", "_view_"):
        position = lowered.find(marker)
        if position >= 0:
            return stem[:position]
    match = re.search(r"_(?:table|tb)_\d+$", lowered)
    if match:
        return stem[: match.start()]
    position = lowered.find("_light_")
    if position >= 0:
        return stem[:position]
    return None


def _resolve_plus_init(
    init_index: Mapping[Tuple[Optional[str], str], Sequence[pathlib.Path]],
    suite: str,
    task_name: str,
) -> pathlib.Path:
    try:
        return _lookup_named_file(init_index, suite, task_name, "Plus init state")
    except BenchmarkInventoryError:
        pass
    lowered = pathlib.Path(task_name).stem.lower()
    if any(marker in lowered for marker in _PLUS_NEWOBJ_MARKERS):
        raise BenchmarkInventoryError(
            "Plus new-object init state is missing for %s/%s" % (suite, task_name)
        )
    base = _plus_original_init_stem(task_name)
    if base is None:
        raise BenchmarkInventoryError("Plus init state cannot be derived for %s/%s" % (suite, task_name))
    candidates: List[pathlib.Path] = []
    for path in init_index.get((suite, base.lower()), ()):
        in_newobj = "libero_newobj" in [part.lower() for part in path.parts]
        if not in_newobj:
            candidates.append(path)
    candidates = sorted(set(candidates), key=str)
    if len(candidates) != 1:
        raise BenchmarkInventoryError(
            "Plus fallback init state for %s/%s base=%s yielded %d files: %s"
            % (suite, task_name, base, len(candidates), [str(path) for path in candidates[:20]])
        )
    return candidates[0]


def _classification_path(repo: pathlib.Path) -> pathlib.Path:
    candidates = [
        repo / "libero" / "libero" / "benchmark" / "task_classification.json",
        repo / "libero" / "benchmark" / "task_classification.json",
        repo / "task_classification.json",
    ]
    candidates.extend(repo.rglob("task_classification.json"))
    existing = sorted(set(path.resolve() for path in candidates if path.is_file()), key=str)
    if len(existing) != 1:
        raise BenchmarkInventoryError(
            "expected exactly one task_classification.json below %s, found %s" % (repo, existing)
        )
    return existing[0]


def _plus_bddl_paths(
    bddl_root: pathlib.Path, suite: str, task_name: str
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Resolve Plus' logical environment path and concrete prompt BDDL.

    Camera, robot-initial-state and sensor-noise variants are registered as
    ``*_view_*_initstate_*`` paths without duplicate files.  The upstream
    environment parses that suffix before opening the base BDDL, so the
    logical path must remain intact even though it does not exist on disk.
    """
    suite_root = bddl_root / suite
    logical = (suite_root / (task_name + ".bddl")).resolve()
    if logical.is_file():
        return logical, logical
    marker = task_name.lower().find("_view_")
    if marker < 0:
        raise BenchmarkInventoryError("Plus classified BDDL is missing: %s" % logical)
    prompt_source = (suite_root / (task_name[:marker] + ".bddl")).resolve()
    if not prompt_source.is_file():
        raise BenchmarkInventoryError(
            "Plus view task base BDDL is missing: logical=%s base=%s" % (logical, prompt_source)
        )
    return logical, prompt_source


def discover_plus_sources(benchmark_repo: pathlib.Path) -> List[TaskSource]:
    repo = _require_dir(benchmark_repo, "LIBERO-Plus repository")
    classification_path = _classification_path(repo)
    try:
        payload = json.loads(classification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkInventoryError("invalid Plus task classification: %s" % exc) from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkInventoryError("Plus task_classification.json must contain an object")

    bddl_root = _find_one(repo, ("bddl_files",), "Plus bddl_files")
    init_root = _find_one(repo, ("init_files", "init_states"), "Plus init_files")
    init_files = prefer_official_init_files(
        _index_files(init_root, (".pruned_init", ".init", ".npy", ".npz", ".pt", ".pth"))
    )
    init_index = _build_file_index(init_files)
    sources: List[TaskSource] = []
    for raw_suite, entries in payload.items():
        suite = canonical_suite(str(raw_suite))
        if suite is None:
            continue
        if not isinstance(entries, list):
            raise BenchmarkInventoryError("Plus classification suite %s is not a list" % raw_suite)
        ids: List[int] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise BenchmarkInventoryError("Plus classification entry in %s is not an object" % suite)
            missing = [key for key in ("id", "name", "category", "difficulty_level") if key not in entry]
            if missing:
                raise BenchmarkInventoryError("Plus classification entry is missing %s: %s" % (missing, entry))
            task_id = int(entry["id"])
            task_name = str(entry["name"])
            ids.append(task_id)
            bddl, prompt_source = _plus_bddl_paths(bddl_root, suite, task_name)
            init_path = _resolve_plus_init(init_index, suite, task_name)
            prompt, prompt_field = parse_bddl_prompt(prompt_source)
            sources.append(
                TaskSource(
                    "plus", suite, task_id, "%06d:%s" % (task_id, task_name), bddl, init_path, prompt, prompt_field,
                    category=str(entry["category"]), difficulty=str(entry["difficulty_level"]),
                    prompt_source_path=prompt_source,
                )
            )
        if len(entries) != PLUS_COUNTS[suite]:
            raise BenchmarkInventoryError(
                "Plus %s classification count is %d; expected %d" % (suite, len(entries), PLUS_COUNTS[suite])
            )
        if len(ids) != len(set(ids)):
            raise BenchmarkInventoryError("Plus %s classification ids are not unique" % suite)
        if ids != list(range(1, len(entries) + 1)):
            raise BenchmarkInventoryError(
                "Plus %s classification ids must be the official ordered 1..N sequence" % suite
            )
    validate_sources("plus", sources)
    return sorted(sources, key=lambda item: (SUITES.index(item.suite), item.task_id, item.source_id))


def _perturbation_from_path(path: pathlib.Path) -> Optional[str]:
    for component in reversed(path.parts[:-1]):
        perturbation = canonical_perturbation(component)
        if perturbation is not None:
            return perturbation
        normalized = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
        # LIBERO-Pro stores pre-generated suites in combined directories such
        # as ``libero_goal_lan`` and ``libero_object_env``.
        for suffix, canonical in (
            ("_object", "object"),
            ("_swap", "swap(position)"),
            ("_lan", "lan(semantic)"),
            ("_task", "task"),
            ("_env", "env"),
        ):
            if normalized.startswith("libero_") and normalized.endswith(suffix):
                base = normalized[: -len(suffix)]
                if canonical_suite(base) is not None:
                    return canonical
    return None


def discover_pro_sources(
    benchmark_repo: pathlib.Path, allow_missing_cells: bool = False
) -> List[TaskSource]:
    repo = _require_dir(benchmark_repo, "LIBERO-Pro repository")
    bddls = _index_files(repo, (".bddl",))
    init_files = prefer_official_init_files(
        _index_files(repo, (".pruned_init", ".init", ".npy", ".npz", ".pt", ".pth"))
    )
    grouped: Dict[Tuple[str, str], List[pathlib.Path]] = {}
    for path in bddls:
        suite = _suite_from_path(path)
        perturbation = _perturbation_from_path(path)
        if suite is not None and perturbation is not None:
            grouped.setdefault((perturbation, suite), []).append(path)
    sources: List[TaskSource] = []
    for perturbation in PRO_PERTURBATIONS:
        for suite in SUITES:
            paths = sorted(grouped.get((perturbation, suite), ()), key=str)
            if not paths and allow_missing_cells:
                continue
            if len(paths) != PRO_TASKS_PER_CELL:
                raise BenchmarkInventoryError(
                    "Pro cell %s/%s contains %d BDDL tasks; expected %d"
                    % (perturbation, suite, len(paths), PRO_TASKS_PER_CELL)
                )
            for task_id, bddl in enumerate(paths):
                init_candidates = [
                    path
                    for path in init_files
                    if _suite_from_path(path) == suite
                    and _perturbation_from_path(path) == perturbation
                    and bool(_stem_variants(path) & _stem_variants(bddl))
                ]
                if len(init_candidates) != 1:
                    raise BenchmarkInventoryError(
                        "Pro init state lookup for %s/%s/%s yielded %d files: %s"
                        % (
                            perturbation,
                            suite,
                            bddl.name,
                            len(init_candidates),
                            [str(path) for path in init_candidates[:20]],
                        )
                    )
                init_path = init_candidates[0]
                prompt, prompt_field = parse_bddl_prompt(bddl)
                # Runtime views contain symlinks to the immutable HF
                # snapshot.  ``_index_files`` resolves them deliberately for
                # provenance, so a filesystem-relative ID is not stable.
                # Protocol coordinates plus the official filename are.
                relative = "%s/%s/%s" % (perturbation, suite, bddl.name)
                sources.append(
                    TaskSource(
                        "pro", suite, task_id, relative, bddl, init_path, prompt, prompt_field,
                        perturbation=perturbation,
                    )
                )
    validate_sources("pro", sources, allow_missing_pro_cells=allow_missing_cells)
    return sources


def discover_sources(
    benchmark: str, benchmark_repo: pathlib.Path, allow_missing_pro_cells: bool = False
) -> List[TaskSource]:
    if benchmark == "plus":
        return discover_plus_sources(benchmark_repo)
    if benchmark == "pro":
        return discover_pro_sources(benchmark_repo, allow_missing_cells=allow_missing_pro_cells)
    raise BenchmarkInventoryError("unknown benchmark: %s" % benchmark)


def validate_sources(
    benchmark: str, sources: Sequence[TaskSource], allow_missing_pro_cells: bool = False
) -> None:
    if any(source.benchmark != benchmark for source in sources):
        raise BenchmarkInventoryError("inventory mixes benchmark identities")
    identities = [source.identity() for source in sources]
    if len(identities) != len(set(identities)):
        raise BenchmarkInventoryError("inventory contains duplicate source identities")
    if benchmark == "plus":
        counts = {suite: sum(source.suite == suite for source in sources) for suite in SUITES}
        if counts != dict(PLUS_COUNTS) or len(sources) != PLUS_FULL_EPISODES:
            raise BenchmarkInventoryError(
                "Plus inventory count mismatch: %s total=%d expected=%s total=%d"
                % (counts, len(sources), dict(PLUS_COUNTS), PLUS_FULL_EPISODES)
            )
    elif benchmark == "pro":
        expected = len(PRO_PERTURBATIONS) * len(SUITES) * PRO_TASKS_PER_CELL
        if len(sources) != expected and not allow_missing_pro_cells:
            raise BenchmarkInventoryError("Pro source inventory has %d tasks; expected %d" % (len(sources), expected))
        for perturbation in PRO_PERTURBATIONS:
            for suite in SUITES:
                count = sum(
                    source.perturbation == perturbation and source.suite == suite for source in sources
                )
                if count == 0 and allow_missing_pro_cells:
                    continue
                if count != PRO_TASKS_PER_CELL:
                    raise BenchmarkInventoryError(
                        "Pro source cell %s/%s has %d tasks; expected %d"
                        % (perturbation, suite, count, PRO_TASKS_PER_CELL)
                    )


def _episode(source: TaskSource, trial: int) -> EpisodeSpec:
    return EpisodeSpec(
        source.benchmark,
        source.suite,
        source.task_id,
        source.source_id,
        source.bddl_path,
        source.init_path,
        source.prompt,
        source.prompt_field,
        trial,
        MAX_STEPS[source.suite],
        category=source.category,
        difficulty=source.difficulty,
        perturbation=source.perturbation,
        prompt_source_path=source.prompt_source_path,
    )


def pro_compatibility(sources: Sequence[TaskSource]) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for perturbation in PRO_PERTURBATIONS:
        for suite in SUITES:
            count = sum(
                source.perturbation == perturbation and source.suite == suite for source in sources
            )
            row = {
                "perturbation": perturbation,
                "suite": suite,
                "source_tasks": count,
                "planned_episodes": PRO_TASKS_PER_CELL * PRO_TRIALS,
                "applicability": "available" if count == PRO_TASKS_PER_CELL else "N/A",
                "reason": None if count == PRO_TASKS_PER_CELL else "pre-generated cell is absent",
            }
            cells.append(row)
            if row["applicability"] == "N/A":
                missing.append(row)
    return {
        "status": "compatible" if not missing else "partial_incompatible",
        "protocol_applicable": not missing,
        "required_cells": len(PRO_PERTURBATIONS) * len(SUITES),
        "available_cells": len(cells) - len(missing),
        "unavailable_cells": missing,
        "protocol_planned_episodes": PRO_FULL_EPISODES,
        "available_planned_episodes": (len(cells) - len(missing)) * PRO_TASKS_PER_CELL * PRO_TRIALS,
        "unavailable_planned_episodes": len(missing) * PRO_TASKS_PER_CELL * PRO_TRIALS,
        "cells": cells,
    }


def expand_matrix(
    benchmark: str,
    phase: str,
    sources: Sequence[TaskSource],
    allow_incompatible_pro: bool = False,
) -> List[EpisodeSpec]:
    if benchmark not in BENCHMARKS:
        raise ValueError("unknown benchmark: %s" % benchmark)
    if phase not in PHASES:
        raise ValueError("unknown phase: %s" % phase)
    validate_sources(benchmark, sources, allow_missing_pro_cells=allow_incompatible_pro)
    if phase == "preflight":
        return [_episode(sorted(sources, key=lambda source: source.identity())[0], 0)]
    if benchmark == "plus":
        if phase == "full":
            matrix = [_episode(source, 0) for source in sources]
        else:
            grouped: Dict[Tuple[str, str], List[TaskSource]] = {}
            for source in sources:
                if source.category is not None:
                    grouped.setdefault((source.suite, source.category), []).append(source)
            matrix = [
                _episode(min(items, key=lambda item: (item.task_id, item.source_id)), 0)
                for _, items in sorted(grouped.items())
            ]
            category_counts = {
                suite: len({source.category for source in sources if source.suite == suite and source.category})
                for suite in SUITES
            }
            if any(count > 7 for count in category_counts.values()):
                raise BenchmarkInventoryError("Plus has more than seven categories in a suite: %s" % category_counts)
    else:
        if phase == "full":
            matrix = [_episode(source, trial) for source in sources for trial in range(PRO_TRIALS)]
        else:
            grouped_pro: Dict[Tuple[str, str], List[TaskSource]] = {}
            for source in sources:
                grouped_pro.setdefault((str(source.perturbation), source.suite), []).append(source)
            matrix = [
                _episode(min(items, key=lambda item: (item.task_id, item.source_id)), 0)
                for _, items in sorted(grouped_pro.items())
            ]
    episode_ids = [spec.episode_id for spec in matrix]
    if len(episode_ids) != len(set(episode_ids)):
        raise BenchmarkInventoryError("expanded matrix has duplicate episode ids")
    if phase == "full":
        expected = (
            PLUS_FULL_EPISODES
            if benchmark == "plus"
            else (len(sources) * PRO_TRIALS if allow_incompatible_pro else PRO_FULL_EPISODES)
        )
        if len(matrix) != expected:
            raise BenchmarkInventoryError(
                "%s full matrix contains %d episodes; expected %d" % (benchmark, len(matrix), expected)
            )
    return matrix


def filter_matrix_by_suites(
    matrix: Sequence[EpisodeSpec], suites: Optional[Sequence[str]]
) -> List[EpisodeSpec]:
    if not suites:
        return list(matrix)
    requested = list(suites)
    if len(requested) != len(set(requested)):
        raise ValueError("suite filter contains duplicates: %s" % requested)
    unknown = sorted(set(requested) - set(SUITES))
    if unknown:
        raise ValueError("unknown suites in filter: %s" % unknown)
    selected = [spec for spec in matrix if spec.suite in set(requested)]
    if not selected:
        raise BenchmarkInventoryError("suite filter selected no episodes: %s" % requested)
    return selected


def resolve_checkpoint(checkpoint_dir: pathlib.Path) -> CheckpointResolution:
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


def validate_action_result(result: Any, min_steps: int = 5, action_dim: int = 7) -> List[List[float]]:
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


def select_pending(
    matrix: Sequence[EpisodeSpec], records: Iterable[Mapping[str, Any]], resume: bool
) -> List[EpisodeSpec]:
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
    for dimension in ("suite", "category", "difficulty", "perturbation"):
        values = sorted(
            {str(getattr(spec, dimension)) for spec in matrix if getattr(spec, dimension) is not None}
        )
        dimension_groups: Dict[str, Any] = {}
        for value in values:
            specs = [spec for spec in matrix if str(getattr(spec, dimension)) == value]
            dimension_groups[value] = _stats(
                [relevant[spec.episode_id] for spec in specs if spec.episode_id in relevant], len(specs)
            )
        groups[dimension] = dimension_groups
    total = _stats(list(relevant.values()), len(matrix))
    suite_rates = [item["success_rate"] for item in groups["suite"].values() if item["success_rate"] is not None]
    total["macro_suite_success_rate"] = sum(suite_rates) / len(suite_rates) if suite_rates else None
    total["extra_episode_records"] = len(set(latest) - set(planned_by_id))
    return {"groups": groups, "total": total}


def verify_integrity(
    records: Sequence[Mapping[str, Any]], matrix: Sequence[EpisodeSpec], require_videos: bool = True
) -> Dict[str, Any]:
    issues: List[str] = []
    episode_ids = [spec.episode_id for spec in matrix]
    if len(episode_ids) != len(set(episode_ids)):
        issues.append("matrix episode ids are not unique")
    attempt_ids = [str(record.get("attempt_id", "")) for record in records]
    if not all(attempt_ids) or len(attempt_ids) != len(set(attempt_ids)):
        issues.append("record attempt ids are missing or not unique")
    unknown_statuses = [
        record for record in records if record.get("status") not in ("success", "failure", "error")
    ]
    if unknown_statuses:
        issues.append("%d records have an unknown status" % len(unknown_statuses))
    unclassified_errors = [
        record
        for record in records
        if record.get("status") == "error"
        and record.get("error_category") not in EXCLUDED_ERROR_CATEGORIES
    ]
    if unclassified_errors:
        issues.append("%d error records have an unknown error category" % len(unclassified_errors))
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
        episode_id
        for episode_id in episode_ids
        if episode_id in latest and latest[episode_id].get("status") not in ("success", "failure")
    ]
    if nonterminal:
        issues.append("%d planned episodes have no policy outcome" % len(nonterminal))
    policy_counts = collections.Counter(
        str(record.get("episode_id"))
        for record in records
        if record.get("status") in ("success", "failure") and record.get("episode_id") in planned
    )
    nonunique_policy_outcomes = [
        episode_id for episode_id in episode_ids if policy_counts.get(episode_id, 0) != 1
    ]
    if nonunique_policy_outcomes:
        issues.append(
            "%d planned episodes do not have exactly one policy outcome"
            % len(nonunique_policy_outcomes)
        )
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
        "unique_attempt_ids": len(set(attempt_ids)),
        "unique_policy_outcomes": sum(count == 1 for count in policy_counts.values()),
        "unique_video_paths": len(set(video_paths)),
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
        "macro_suite_success_rate", "extra_episode_records", "applicability", "reason",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for dimension, values in summary["groups"].items():
                for scope, stats in values.items():
                    writer.writerow(dict({"dimension": dimension, "scope": scope}, **stats))
            compatibility = summary.get("compatibility")
            if isinstance(compatibility, Mapping):
                for cell in compatibility.get("unavailable_cells", []):
                    writer.writerow(
                        {
                            "dimension": "compatibility_cell",
                            "scope": "%s/%s" % (cell["perturbation"], cell["suite"]),
                            "planned": cell["planned_episodes"],
                            "applicability": "N/A",
                            "reason": cell["reason"],
                            "complete": False,
                        }
                    )
            writer.writerow(dict({"dimension": "total", "scope": "total"}, **summary["total"]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(
    benchmark: str, benchmark_repo: pathlib.Path, sources: Sequence[TaskSource]
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    file_hashes: Dict[pathlib.Path, str] = {}

    def cached_sha(path: pathlib.Path) -> str:
        resolved = path.resolve()
        if resolved not in file_hashes:
            file_hashes[resolved] = sha256_file(resolved)
        return file_hashes[resolved]

    for source in sources:
        row = {
            "identity": source.identity(),
            "suite": source.suite,
            "task_id": source.task_id,
            "source_id": source.source_id,
            "category": source.category,
            "difficulty": source.difficulty,
            "perturbation": source.perturbation,
            "bddl_path": str(source.bddl_path),
            "bddl_path_exists": source.bddl_path.is_file(),
            "prompt_source_path": str(source.prompt_source_path or source.bddl_path),
            "prompt_source_sha256": cached_sha(source.prompt_source_path or source.bddl_path),
            "init_path": str(source.init_path),
            "init_sha256": cached_sha(source.init_path),
            "prompt": source.prompt,
            "prompt_field": source.prompt_field,
        }
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(encoded)
        rows.append(row)
    return {
        "benchmark": benchmark,
        "benchmark_repo": str(pathlib.Path(benchmark_repo).expanduser().resolve()),
        "source_count": len(rows),
        "unique_source_files": len(file_hashes),
        "inventory_sha256": digest.hexdigest(),
        "sources": rows,
    }
