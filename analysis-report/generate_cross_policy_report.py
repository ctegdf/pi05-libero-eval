#!/usr/bin/env python3
"""Cross-policy (pi0.5 vs OpenVLA vs ACT) comparison report generator.

Imports pure utility functions (read_jsonl/wilson/aggregate/grouped/LABELS/
SUITES/...) from generate_report.py rather than modifying it -- that script
is a hardcoded, already-audited pi0.5-only narrative generator (per project
convention in CLAUDE.md/the eval plan, it is only meant to be imported as a
utility library, not extended). This is a new, source-list-driven sibling
covering the OpenVLA/ACT campaigns added later, structurally symmetric to
pi0.5's 3-benchmark shape (libero / plus / pro) but pulling from 4 per-suite
episodes.jsonl files instead of one pre-merged file, since OpenVLA/ACT run
each LIBERO suite under its own per-suite checkpoint/server process.

Gracefully skips a (policy, benchmark) combination whose source files are not
all present, instead of hard-asserting like generate_report.py does, so this
can be re-run incrementally as campaigns complete.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_report import (  # noqa: E402
    LABELS,
    PLUS_CATEGORIES,
    PRO_PERTURBATIONS,
    SUITES,
    aggregate,
    grouped,
    read_jsonl,
    wilson,
)

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

POLICIES = ["pi05", "openvla", "act"]
POLICY_LABELS = {
    "pi05": "π0.5 (pi05_libero)",
    "openvla": "OpenVLA-7B（按子集微调）",
    "act": "ACT（按子集从头训练）",
}
BENCHMARKS = ["libero", "plus", "pro"]
BENCHMARK_LABELS = {"libero": "LIBERO（标准四套件）", "plus": "LIBERO-Plus", "pro": "LIBERO-Pro"}

_SUFFIX = {"libero_spatial": "spatial", "libero_object": "object", "libero_goal": "goal", "libero_10": "10"}


def _suite_paths(root: str, prefix: str) -> list[Path]:
    return [ARCHIVE_ROOT / root / f"{prefix}-libero_{suffix}" / "episodes.jsonl" for suffix in _SUFFIX.values()]


SOURCES: dict[str, dict[str, list[Path]]] = {
    "pi05": {
        "libero": [ARCHIVE_ROOT / "openpi-libero/official-full/episodes.jsonl"],
        "plus": [ARCHIVE_ROOT / "plus-pro/plus-full-merged/episodes.jsonl"],
        "pro": [ARCHIVE_ROOT / "plus-pro/pro-full/episodes.jsonl"],
    },
    "openvla": {
        "libero": _suite_paths("openvla-libero", "libero-full"),
        "plus": _suite_paths("openvla-plus-pro", "plus-full"),
        "pro": _suite_paths("openvla-plus-pro", "pro-full"),
    },
    "act": {
        "libero": _suite_paths("act-libero-full", "libero-full"),
        "plus": _suite_paths("act-plus-pro", "plus-full"),
        "pro": _suite_paths("act-plus-pro", "pro-full"),
    },
}


def load_rows(paths: list[Path]) -> list[dict] | None:
    """Load and dedupe to one terminal record per episode_id.

    Mirrors eval_support._latest_attempts(): among records with a terminal
    (success/failure) status for a given episode_id, keep the one with the
    highest `attempt` number. This matters because OpenVLA/ACT's per-suite
    episodes.jsonl files are raw harness ledgers (unlike pi0.5's pre-merged
    files), which can contain more than one terminal row per episode_id --
    either a legitimate retry (attempt=1, 2, ...) or a leftover duplicate
    from a since-fixed concurrent-write incident -- and naively counting
    every row would inflate totals above the true planned episode count.
    """
    rows: list[dict] = []
    for path in paths:
        if not path.is_file():
            return None
        rows.extend(read_jsonl(path))
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        episode_id = row.get("episode_id")
        if episode_id:
            by_episode.setdefault(episode_id, []).append(row)
    latest: list[dict] = []
    for attempts in by_episode.values():
        terminal = [r for r in attempts if r.get("status") in ("success", "failure")]
        if not terminal:
            continue
        latest.append(max(terminal, key=lambda r: int(r.get("attempt", 0))))
    return latest


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_rate(rate: float | None) -> str:
    return "" if rate is None else f"{rate * 100:.2f}%"


def fmt_ci(low: float, high: float) -> str:
    import math

    if math.isnan(low):
        return ""
    return f"[{low * 100:.1f}%, {high * 100:.1f}%]"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    total_rows: list[dict] = []
    suite_rows: list[dict] = []
    category_rows: list[dict] = []
    perturbation_rows: list[dict] = []

    for benchmark in BENCHMARKS:
        for policy in POLICIES:
            paths = SOURCES[policy][benchmark]
            rows = load_rows(paths)
            if rows is None:
                print(f"SKIP {policy}/{benchmark}: source files not all present")
                continue

            stats = aggregate(rows)
            ci_low, ci_high = stats["ci95_low"], stats["ci95_high"]
            total_rows.append(
                {
                    "benchmark": benchmark,
                    "policy": policy,
                    "planned_or_attempted": stats["total"],
                    "successes": stats["successes"],
                    "failures": stats["failures"],
                    "success_rate": stats["rate"],
                    "success_rate_pct": fmt_rate(stats["rate"]),
                    "ci95": fmt_ci(ci_low, ci_high),
                }
            )
            print(
                f"{benchmark:7s} {policy:8s} total={stats['total']:6d} "
                f"successes={stats['successes']:6d} rate={fmt_rate(stats['rate']):>8s} "
                f"ci95={fmt_ci(ci_low, ci_high)}"
            )

            for g in grouped(rows, ["suite"]):
                suite_rows.append(
                    {
                        "benchmark": benchmark,
                        "policy": policy,
                        "suite": g["suite"],
                        "suite_zh": LABELS.get(g["suite"], g["suite"]),
                        "total": g["total"],
                        "successes": g["successes"],
                        "success_rate": g["rate"],
                        "success_rate_pct": fmt_rate(g["rate"]),
                    }
                )

            if benchmark == "plus":
                for g in grouped(rows, ["category"]):
                    if g["category"] not in PLUS_CATEGORIES:
                        continue
                    category_rows.append(
                        {
                            "benchmark": benchmark,
                            "policy": policy,
                            "category": g["category"],
                            "category_zh": LABELS.get(g["category"], g["category"]),
                            "total": g["total"],
                            "successes": g["successes"],
                            "success_rate": g["rate"],
                            "success_rate_pct": fmt_rate(g["rate"]),
                        }
                    )

            if benchmark == "pro":
                for g in grouped(rows, ["perturbation"]):
                    if g["perturbation"] not in PRO_PERTURBATIONS:
                        continue
                    perturbation_rows.append(
                        {
                            "benchmark": benchmark,
                            "policy": policy,
                            "perturbation": g["perturbation"],
                            "perturbation_zh": LABELS.get(g["perturbation"], g["perturbation"]),
                            "total": g["total"],
                            "successes": g["successes"],
                            "success_rate": g["rate"],
                            "success_rate_pct": fmt_rate(g["rate"]),
                        }
                    )

    write_csv(
        DATA_DIR / "cross_policy_total.csv",
        total_rows,
        ["benchmark", "policy", "planned_or_attempted", "successes", "failures", "success_rate_pct", "ci95"],
    )
    write_csv(
        DATA_DIR / "cross_policy_by_suite.csv",
        suite_rows,
        ["benchmark", "policy", "suite", "suite_zh", "total", "successes", "success_rate_pct"],
    )
    write_csv(
        DATA_DIR / "cross_policy_by_category.csv",
        category_rows,
        ["benchmark", "policy", "category", "category_zh", "total", "successes", "success_rate_pct"],
    )
    write_csv(
        DATA_DIR / "cross_policy_by_perturbation.csv",
        perturbation_rows,
        ["benchmark", "policy", "perturbation", "perturbation_zh", "total", "successes", "success_rate_pct"],
    )
    print(f"\nwrote CSVs under {DATA_DIR}")


if __name__ == "__main__":
    main()
