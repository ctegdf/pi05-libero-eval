#!/usr/bin/env python3
"""Generate a data-rich PPT report comparing pi05_libero and pi05_base."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uno
from com.sun.star.awt import Point, Size


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
OFFICIAL_JSONL = ARCHIVE / "openpi-libero/official-full/episodes.jsonl"
BASE_JSONL = ARCHIVE / "openpi-libero/base-libero-assets-full/episodes.jsonl"
OFFICIAL_VIDEO_DIR = ARCHIVE / "openpi-libero/official-full/videos"
BASE_VIDEO_DIR = ARCHIVE / "openpi-libero/base-libero-assets-full/videos"
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PREVIEW_DIR = HERE / "preview"

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
SUITE_COLORS = {
    "libero_spatial": "#2E7D6E",
    "libero_object": "#3D7CC9",
    "libero_goal": "#9A6DD7",
    "libero_10": "#D8802B",
}
OFFICIAL_COLOR = "#176B5B"
BASE_COLOR = "#C74B50"
INK = "#142238"
MUTED = "#657287"
GRID = "#DCE3EA"
AMBER = "#D98B2B"
BG = "#F6F8FB"

FONT_FAMILY = "Noto Sans CJK JP"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(values: Sequence[float], probability: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), probability * 100))


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    failures = total - successes
    ci_low, ci_high = wilson(successes, total)
    durations = [float(row["duration_seconds"]) for row in rows]
    steps = [float(row["action_steps"]) for row in rows]
    return {
        "total": total,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / total,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "total_action_steps": int(sum(steps)),
        "mean_action_steps": statistics.mean(steps),
        "median_action_steps": statistics.median(steps),
        "mean_budget_utilization": statistics.mean(
            row["action_steps"] / row["max_steps"] for row in rows
        ),
        "total_episode_duration_hours": sum(durations) / 3600,
        "mean_duration_seconds": statistics.mean(durations),
        "median_duration_seconds": statistics.median(durations),
        "p90_duration_seconds": percentile(durations, 0.90),
        "max_step_failures": sum(
            (not row["success"]) and row["action_steps"] == row["max_steps"] for row in rows
        ),
        "infrastructure_errors": sum(row.get("error_category") is not None for row in rows),
    }


def group_rows(rows: Sequence[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, members in groups.items():
        item = {key: value for key, value in zip(keys, values)}
        item.update(aggregate(members))
        if "task_description" in keys:
            item["short_task"] = short_task(str(item["task_description"]))
        output.append(item)
    return output


def short_task(text: str, limit: int = 50) -> str:
    replacements = {
        "pick up the ": "pick ",
        " and place it ": " → ",
        "put the ": "put ",
        "place it ": "→ ",
        " of the cabinet": "",
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result if len(result) <= limit else result[: limit - 1] + "…"


def assert_integrity(official: Sequence[dict[str, Any]], base: Sequence[dict[str, Any]]) -> None:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return row["suite"], row["task_id"], row["trial"], row["seed"]

    assert len(official) == len(base) == 2000
    official_by_key = {key(row): row for row in official}
    base_by_key = {key(row): row for row in base}
    assert len(official_by_key) == len(base_by_key) == 2000
    assert set(official_by_key) == set(base_by_key)
    assert all(
        official_by_key[item]["task_description"] == base_by_key[item]["task_description"]
        for item in official_by_key
    )
    assert all(
        official_by_key[item]["max_steps"] == base_by_key[item]["max_steps"]
        for item in official_by_key
    )
    for name, rows, video_dir in [
        ("official", official, OFFICIAL_VIDEO_DIR),
        ("base", base, BASE_VIDEO_DIR),
    ]:
        assert len({row["episode_id"] for row in rows}) == 2000, name
        assert len({row["attempt_id"] for row in rows}) == 2000, name
        assert all(row["status"] in {"success", "failure"} for row in rows), name
        assert all(row.get("error_category") is None for row in rows), name
        assert all(row["video_status"] == "written" for row in rows), name
        videos = list(video_dir.glob("*.mp4"))
        assert len(videos) == 2000, name
        assert all(path.is_file() and path.stat().st_size > 0 for path in videos), name


def build_metrics() -> dict[str, Any]:
    official = read_jsonl(OFFICIAL_JSONL)
    base = read_jsonl(BASE_JSONL)
    assert_integrity(official, base)
    pair_key = lambda row: (row["suite"], row["task_id"], row["trial"], row["seed"])
    official_by_key = {pair_key(row): row for row in official}
    base_by_key = {pair_key(row): row for row in base}
    paired_outcomes = Counter(
        (bool(official_by_key[key]["success"]), bool(base_by_key[key]["success"]))
        for key in official_by_key
    )
    overall = {"official": aggregate(official), "base": aggregate(base)}
    suite = {}
    for protocol, rows in [("official", official), ("base", base)]:
        suite[protocol] = {
            row["suite"]: row for row in group_rows(rows, "suite")
        }
    task_official = group_rows(official, "suite", "task_id", "task_description")
    task_base = group_rows(base, "suite", "task_id", "task_description")
    task_base_by_key = {(row["suite"], row["task_id"]): row for row in task_base}
    task = []
    for row in task_official:
        base_row = task_base_by_key[(row["suite"], row["task_id"])]
        task.append(
            {
                "suite": row["suite"],
                "suite_label": SUITE_LABELS[row["suite"]],
                "task_id": row["task_id"],
                "task_description": row["task_description"],
                "short_task": row["short_task"],
                "official_successes": row["successes"],
                "official_total": row["total"],
                "official_success_rate": row["success_rate"],
                "official_ci95_low": row["ci95_low"],
                "official_ci95_high": row["ci95_high"],
                "official_mean_action_steps": row["mean_action_steps"],
                "official_median_action_steps": row["median_action_steps"],
                "official_budget_utilization": row["mean_budget_utilization"],
                "official_median_duration_seconds": row["median_duration_seconds"],
                "base_successes": base_row["successes"],
                "base_success_rate": base_row["success_rate"],
                "base_mean_action_steps": base_row["mean_action_steps"],
                "base_budget_utilization": base_row["mean_budget_utilization"],
                "base_median_duration_seconds": base_row["median_duration_seconds"],
                "failures": row["failures"],
            }
        )
    task.sort(key=lambda row: (SUITES.index(row["suite"]), row["task_id"]))
    worst_tasks = sorted(task, key=lambda row: (-row["failures"], row["suite"], row["task_id"]))
    cumulative = 0
    for rank, row in enumerate(worst_tasks, start=1):
        cumulative += row["failures"]
        row["failure_rank"] = rank
        row["cumulative_failures"] = cumulative
        row["cumulative_failure_share"] = cumulative / overall["official"]["failures"]
    total_base_steps = overall["base"]["total_action_steps"]
    saved_steps = total_base_steps - overall["official"]["total_action_steps"]
    task_rate_counts = Counter(round(row["official_success_rate"] * 100) for row in task)
    paired_difference_pp = 100 * (
        overall["official"]["success_rate"] - overall["base"]["success_rate"]
    )
    task_differences_pp = np.asarray(
        [100 * (row["official_success_rate"] - row["base_success_rate"]) for row in task]
    )
    bootstrap_rng = np.random.default_rng(7)
    bootstrap_indices = bootstrap_rng.integers(0, len(task), size=(100_000, len(task)))
    bootstrap_means = task_differences_pp[bootstrap_indices].mean(axis=1)
    bootstrap_ci_low, bootstrap_ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    suite_rows = []
    for suite_name in SUITES:
        official_row = suite["official"][suite_name]
        base_row = suite["base"][suite_name]
        suite_rows.append(
            {
                "suite": suite_name,
                "suite_label": SUITE_LABELS[suite_name],
                "official_successes": official_row["successes"],
                "base_successes": base_row["successes"],
                "total": official_row["total"],
                "official_success_rate": official_row["success_rate"],
                "official_ci95_low": official_row["ci95_low"],
                "official_ci95_high": official_row["ci95_high"],
                "base_success_rate": base_row["success_rate"],
                "base_ci95_low": base_row["ci95_low"],
                "base_ci95_high": base_row["ci95_high"],
                "official_mean_action_steps": official_row["mean_action_steps"],
                "base_mean_action_steps": base_row["mean_action_steps"],
                "official_budget_utilization": official_row["mean_budget_utilization"],
                "base_budget_utilization": base_row["mean_budget_utilization"],
                "official_duration_hours": official_row["total_episode_duration_hours"],
                "base_duration_hours": base_row["total_episode_duration_hours"],
                "official_median_duration_seconds": official_row["median_duration_seconds"],
                "base_median_duration_seconds": base_row["median_duration_seconds"],
            }
        )
    return {
        "schema_version": 1,
        "comparison": {
            "official_protocol": "pi05_libero checkpoint",
            "base_protocol": "pi05_base params + official LIBERO assets/norm stats",
            "matrix": "4 suites × 10 tasks × 50 trials",
            "episodes_per_protocol": 2000,
            "paired_matrix_exact": True,
            "seed": 7,
            "resize": 224,
            "replan_steps": 5,
            "wait_steps": 10,
            "max_steps": {
                "libero_spatial": 220,
                "libero_object": 280,
                "libero_goal": 300,
                "libero_10": 520,
            },
        },
        "overall": overall,
        "paired_outcomes": {
            "official_success_base_success": paired_outcomes[(True, True)],
            "official_success_base_failure": paired_outcomes[(True, False)],
            "official_failure_base_success": paired_outcomes[(False, True)],
            "official_failure_base_failure": paired_outcomes[(False, False)],
        },
        "suite": suite_rows,
        "task": task,
        "worst_tasks": worst_tasks,
        "derived": {
            "observed_success_rate_gap_pp": paired_difference_pp,
            "task_paired_bootstrap_replicates": 100_000,
            "task_paired_bootstrap_ci95_low_pp": float(bootstrap_ci_low),
            "task_paired_bootstrap_ci95_high_pp": float(bootstrap_ci_high),
            "saved_action_steps": saved_steps,
            "saved_action_step_share": saved_steps / total_base_steps,
            "duration_hours_saved": overall["base"]["total_episode_duration_hours"]
            - overall["official"]["total_episode_duration_hours"],
            "base_to_official_duration_ratio": overall["base"]["total_episode_duration_hours"]
            / overall["official"]["total_episode_duration_hours"],
            "tasks_at_100pct": task_rate_counts[100],
            "tasks_at_least_98pct": sum(count for rate, count in task_rate_counts.items() if rate >= 98),
            "tasks_at_least_94pct": sum(count for rate, count in task_rate_counts.items() if rate >= 94),
            "task_success_rate_distribution": dict(sorted(task_rate_counts.items(), reverse=True)),
            "top3_failure_share": sum(row["failures"] for row in worst_tasks[:3])
            / overall["official"]["failures"],
            "top5_failure_share": sum(row["failures"] for row in worst_tasks[:5])
            / overall["official"]["failures"],
            "top10_failure_share": sum(row["failures"] for row in worst_tasks[:10])
            / overall["official"]["failures"],
        },
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#AAB4C0",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars: Any, fmt: str = "{:.1f}%", offset: float = 1.0) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )


def chart_overall(metrics: dict[str, Any]) -> None:
    official = metrics["overall"]["official"]
    base = metrics["overall"]["base"]
    values = [official["success_rate"] * 100, base["success_rate"] * 100]
    lows = [official["ci95_low"] * 100, base["ci95_low"] * 100]
    highs = [official["ci95_high"] * 100, base["ci95_high"] * 100]
    errors = np.array([[value - low for value, low in zip(values, lows)], [high - value for value, high in zip(values, highs)]])
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar(
        [0, 1], values, width=0.55, color=[OFFICIAL_COLOR, BASE_COLOR],
        yerr=errors, capsize=7, ecolor=INK,
    )
    ax.axhline(93.85, color=AMBER, linestyle="--", linewidth=2, label="官方协议验收线 93.85%")
    ax.set_ylim(0, 106)
    ax.set_xticks([0, 1], ["官方 pi05_libero\n微调权重", "pi05_base 参数\n+ LIBERO stats/assets"])
    ax.set_ylabel("Episode 成功率")
    ax.set_yticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    add_bar_labels(ax, [bars[0]])
    ax.text(0, values[0] - 8, "1942 / 2000", ha="center", color="white", fontsize=13, fontweight="bold")
    ax.text(1, 3.5, "0.0% · 0 / 2000", ha="center", color=BASE_COLOR, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("同一套 LIBERO 试卷上，任务微调权重与 Base 权重出现决定性差距", fontsize=17, fontweight="bold", pad=16)
    ax.text(
        0.5, -0.18,
        f"Wilson 95% CI：官方 {official['ci95_low']*100:.2f}%–{official['ci95_high']*100:.2f}%；Base 0.00%–{base['ci95_high']*100:.2f}%",
        transform=ax.transAxes, ha="center", color=MUTED, fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, "01_overall_success")


def chart_suite(metrics: dict[str, Any]) -> None:
    rows = metrics["suite"]
    x = np.arange(len(rows))
    official = [row["official_success_rate"] * 100 for row in rows]
    base = [row["base_success_rate"] * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    width = 0.34
    official_bars = ax.bar(x - width / 2, official, width, color=OFFICIAL_COLOR, label="pi05_libero")
    base_bars = ax.bar(x + width / 2, base, width, color=BASE_COLOR, label="pi05_base + stats/assets")
    ax.set_xticks(x, [row["suite_label"] for row in rows])
    ax.set_ylim(0, 106)
    ax.set_yticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.set_ylabel("成功率")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    add_bar_labels(ax, official_bars)
    for index, row in enumerate(rows):
        ax.text(index - width / 2, official[index] - 8, f"{row['official_successes']}/500", ha="center", color="white", fontsize=10, fontweight="bold")
        ax.text(index + width / 2, 2.8, "0.0% · 0/500", ha="center", color=BASE_COLOR, fontsize=8.5, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_title("四个任务套件全部拉开差距；LIBERO-10 是微调权重的相对短板", fontsize=17, fontweight="bold", pad=16)
    fig.tight_layout()
    save_figure(fig, "02_suite_success")


def chart_paired_outcomes(metrics: dict[str, Any]) -> None:
    paired = metrics["paired_outcomes"]
    matrix = np.array(
        [
            [paired["official_success_base_success"], paired["official_success_base_failure"]],
            [paired["official_failure_base_success"], paired["official_failure_base_failure"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    image = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=2000, aspect="auto")
    ax.set_xticks([0, 1], ["Base 成功", "Base 失败"])
    ax.set_yticks([0, 1], ["官方成功", "官方失败"])
    ax.set_xlabel("pi05_base + LIBERO stats/assets")
    ax.set_ylabel("官方 pi05_libero")
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > 1000 else INK
            ax.text(col, row - 0.05, f"{value:,}", ha="center", va="center", fontsize=28, fontweight="bold", color=color)
            ax.text(col, row + 0.20, f"{value / 20:.1f}%", ha="center", va="center", fontsize=12, color=color)
    ax.set_title("逐 episode 配对：1,942 次官方胜出，Base 反超为 0", fontsize=17, fontweight="bold", pad=16)
    ax.text(0.5, -0.18, "配对键：suite × task × trial × seed；两边都失败 58 次", transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)
    ax.tick_params(length=0, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label("episode 数")
    fig.tight_layout()
    save_figure(fig, "09_paired_outcomes")


def chart_task_heatmap(metrics: dict[str, Any]) -> None:
    task_by_key = {(row["suite"], row["task_id"]): row for row in metrics["task"]}
    official = np.array(
        [[task_by_key[(suite, task_id)]["official_success_rate"] * 100 for task_id in range(10)] for suite in SUITES]
    )
    base = np.array(
        [[task_by_key[(suite, task_id)]["base_success_rate"] * 100 for task_id in range(10)] for suite in SUITES]
    )
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 5.8), gridspec_kw={"hspace": 0.45})
    for ax, data, label in zip(axes, [official, base], ["官方 pi05_libero 微调权重", "pi05_base + LIBERO stats/assets"]):
        image = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(10), [f"T{task_id}" for task_id in range(10)])
        ax.set_yticks(range(4), [SUITE_LABELS[suite] for suite in SUITES])
        ax.set_title(label, loc="left", fontsize=13, fontweight="bold", pad=7)
        for row_index in range(4):
            for col_index in range(10):
                value = data[row_index, col_index]
                color = "white" if value < 75 else INK
                ax.text(col_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, orientation="vertical", fraction=0.022, pad=0.02)
    colorbar.set_label("任务成功率 (%)")
    fig.suptitle("40 个任务 × 50 trials：微调权重多数任务接近满分，Base 在每个任务上均为 0", fontsize=17, fontweight="bold", y=1.02)
    save_figure(fig, "03_task_heatmap")


def chart_task_distribution(metrics: dict[str, Any]) -> None:
    distribution = {int(rate): int(count) for rate, count in metrics["derived"]["task_success_rate_distribution"].items()}
    rates = sorted(distribution)
    counts = [distribution[rate] for rate in rates]
    fig, ax = plt.subplots(figsize=(11.7, 6.2))
    colors = [BASE_COLOR if rate < 90 else AMBER if rate < 98 else OFFICIAL_COLOR for rate in rates]
    bars = ax.bar([str(rate) for rate in rates], counts, color=colors, width=0.65)
    ax.set_xlabel("任务成功率 (%)")
    ax.set_ylabel("任务数量（共 40 个）")
    ax.set_ylim(0, max(counts) + 4)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("官方微调权重的任务级稳定性：29/40 个任务 ≥98%，19 个任务满分", fontsize=17, fontweight="bold", pad=16)
    ax.text(
        0.98, 0.93,
        "Base 对照：40/40 个任务均为 0%",
        transform=ax.transAxes, ha="right", va="top", fontsize=12,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FBEAEC", "edgecolor": BASE_COLOR},
    )
    fig.tight_layout()
    save_figure(fig, "04_task_distribution")


def chart_failure_pareto(metrics: dict[str, Any]) -> None:
    rows = [row for row in metrics["worst_tasks"] if row["failures"] > 0]
    failures = [row["failures"] for row in rows]
    cumulative = np.cumsum(failures) / sum(failures) * 100
    labels = [f"{row['suite_label']} T{row['task_id']}" for row in rows]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x = np.arange(len(rows))
    bars = ax.bar(x, failures, color=[SUITE_COLORS[row["suite"]] for row in rows], alpha=0.9)
    ax.set_ylabel("失败 episode 数")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top"]].set_visible(False)
    for bar, count in zip(bars, failures):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.35, str(count), ha="center", fontsize=9, fontweight="bold")
    second = ax.twinx()
    second.plot(x, cumulative, color=BASE_COLOR, marker="o", linewidth=2.3, label="累计失败占比")
    second.set_ylim(0, 108)
    second.set_ylabel("累计失败占比")
    second.set_yticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    second.spines[["top"]].set_visible(False)
    second.axhline(50, color="#9AA5B1", linestyle="--", linewidth=1)
    second.legend(loc="center right", frameon=False)
    ax.set_title("58 次失败并非平均分布：最难 3 个任务贡献 51.7% 失败", fontsize=17, fontweight="bold", pad=16)
    fig.tight_layout()
    save_figure(fig, "05_failure_pareto")


def chart_action_budget(metrics: dict[str, Any]) -> None:
    rows = metrics["suite"]
    x = np.arange(len(rows))
    official = [row["official_budget_utilization"] * 100 for row in rows]
    base = [row["base_budget_utilization"] * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(12.3, 6.2))
    width = 0.34
    official_bars = ax.bar(x - width / 2, official, width, color=OFFICIAL_COLOR, label="pi05_libero")
    base_bars = ax.bar(x + width / 2, base, width, color=BASE_COLOR, label="pi05_base + stats/assets")
    ax.set_xticks(x, [row["suite_label"] for row in rows])
    ax.set_ylabel("平均动作预算使用率")
    ax.set_yticks(np.arange(0, 121, 20), [f"{value}%" for value in range(0, 121, 20)])
    ax.set_ylim(0, 118)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    add_bar_labels(ax, official_bars)
    add_bar_labels(ax, base_bars)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Base 在每个 suite 都耗尽 100% 动作预算；微调权重平均使用约一半", fontsize=17, fontweight="bold", pad=16)
    ax.text(
        0.02, 0.92,
        f"全矩阵少执行 {metrics['derived']['saved_action_steps']:,} 步（节省 {metrics['derived']['saved_action_step_share']*100:.1f}%）",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E9F5F1", "edgecolor": OFFICIAL_COLOR},
    )
    fig.tight_layout()
    save_figure(fig, "06_action_budget")


def chart_duration(metrics: dict[str, Any]) -> None:
    rows = metrics["suite"]
    x = np.arange(len(rows))
    official = [row["official_duration_hours"] for row in rows]
    base = [row["base_duration_hours"] for row in rows]
    fig, ax = plt.subplots(figsize=(12.3, 6.2))
    width = 0.34
    official_bars = ax.bar(x - width / 2, official, width, color=OFFICIAL_COLOR, label="pi05_libero")
    base_bars = ax.bar(x + width / 2, base, width, color=BASE_COLOR, label="pi05_base + stats/assets")
    ax.set_xticks(x, [row["suite_label"] for row in rows])
    ax.set_ylabel("500 个 episodes 的记录时长总和（小时）")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in [official_bars, base_bars]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{bar.get_height():.2f}h", ha="center", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Base 因持续运行至超时，记录的 episode-duration 总和达到微调权重的 1.90 倍", fontsize=17, fontweight="bold", pad=16)
    ax.text(
        0.99, 0.92,
        f"全矩阵：官方 {metrics['overall']['official']['total_episode_duration_hours']:.2f}h  /  Base {metrics['overall']['base']['total_episode_duration_hours']:.2f}h",
        transform=ax.transAxes, ha="right", fontsize=12, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF4E5", "edgecolor": AMBER},
    )
    fig.tight_layout()
    save_figure(fig, "07_duration_cost")


def chart_task_efficiency_scatter(metrics: dict[str, Any]) -> None:
    rows = metrics["task"]
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    for suite in SUITES:
        members = [row for row in rows if row["suite"] == suite]
        ax.scatter(
            [row["official_budget_utilization"] * 100 for row in members],
            [row["official_success_rate"] * 100 for row in members],
            s=75, color=SUITE_COLORS[suite], label=SUITE_LABELS[suite], alpha=0.9, edgecolor="white", linewidth=0.8,
        )
    worst = metrics["worst_tasks"][0]
    ax.annotate(
        "最难任务：两只 moka pot\n60% 成功，87% 动作预算",
        xy=(worst["official_budget_utilization"] * 100, worst["official_success_rate"] * 100),
        xytext=(67, 72), arrowprops={"arrowstyle": "->", "color": BASE_COLOR},
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FBEAEC", "edgecolor": BASE_COLOR}, fontsize=10,
    )
    ax.set_xlim(20, 100)
    ax.set_ylim(55, 102)
    ax.set_xlabel("平均动作预算使用率")
    ax.set_ylabel("任务成功率")
    ax.set_xticks(np.arange(20, 101, 20), [f"{value}%" for value in range(20, 101, 20)])
    ax.set_yticks(np.arange(60, 101, 10), [f"{value}%" for value in range(60, 101, 10)])
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="lower left")
    ax.set_title("探索性观察：更高动作预算使用通常伴随更低任务成功率", fontsize=17, fontweight="bold", pad=16)
    ax.text(0.99, 0.04, "40 个任务；相关性仅用于定位长尾，不构成因果结论", transform=ax.transAxes, ha="right", color=MUTED, fontsize=10)
    fig.tight_layout()
    save_figure(fig, "08_task_efficiency_scatter")


def generate_charts(metrics: dict[str, Any]) -> None:
    configure_plotting()
    chart_overall(metrics)
    chart_suite(metrics)
    chart_task_heatmap(metrics)
    chart_task_distribution(metrics)
    chart_failure_pareto(metrics)
    chart_action_budget(metrics)
    chart_duration(metrics)
    chart_task_efficiency_scatter(metrics)
    chart_paired_outcomes(metrics)


def write_artifacts(metrics: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    overall_rows = []
    for protocol, values in metrics["overall"].items():
        overall_rows.append({"protocol": protocol, **values})
    write_csv(DATA_DIR / "overall.csv", overall_rows)
    write_csv(DATA_DIR / "suite_comparison.csv", metrics["suite"])
    write_csv(DATA_DIR / "task_comparison.csv", metrics["task"])
    write_csv(DATA_DIR / "worst_tasks.csv", metrics["worst_tasks"])
    generate_charts(metrics)
    (HERE / "PPT分析报告.md").write_text(build_markdown(metrics), encoding="utf-8")
    (HERE / "README.md").write_text(build_readme(), encoding="utf-8")


def build_markdown(metrics: dict[str, Any]) -> str:
    official = metrics["overall"]["official"]
    base = metrics["overall"]["base"]
    derived = metrics["derived"]
    worst = metrics["worst_tasks"][0]
    return f"""# 官方 pi05_libero vs pi05_base：LIBERO 本体对比 PPT 分析稿

## 使用说明

本文按 PPT 页组织，可直接复制为汇报讲稿。可编辑演示文稿见 `OpenPI_pi05_LIBERO_权重对比分析.pptx`，所有高清图表位于 `figures/`，精确数据位于 `data/`。

## 第 1 页｜封面

**标题：** OpenPI π0.5 权重在 LIBERO 本体上的对比评测  
**副标题：** 官方 `pi05_libero` 微调权重 vs `pi05_base` 参数 + 官方 LIBERO stats/assets

## 第 2 页｜为什么这是一组公平对照

两份权重面对完全相同的 **4 suites × 10 tasks × 50 trials = 2000 episodes**。每个 `(suite, task, trial, seed)` 一一对应，任务描述和最大步数完全相同。固定参数为 seed 7、224×224 图像、每次重规划执行 5 步动作、10 步初始等待、EGL。

Base 对照不是 Base-native：它使用 `pi05_base` 参数，但刻意搭配官方 LIBERO assets 与 norm stats，目的是排除“输入尺度或资源不匹配”这一解释。Base-native 因缺少 LIBERO norm stats 为 N/A，不纳入本页 0% 对照。

## 第 3 页｜总体结果：97.10% vs 0%

- 官方微调权重：**{official['successes']}/{official['total']} = {official['success_rate']*100:.2f}%**，Wilson 95% CI {official['ci95_low']*100:.2f}%–{official['ci95_high']*100:.2f}%。
- Base+stats/assets：**{base['successes']}/{base['total']} = 0%**，Wilson 95% CI 0.00%–{base['ci95_high']*100:.2f}%。
- 观察差距：**{derived['observed_success_rate_gap_pp']:.1f} 个百分点**。
- 按 40 个任务配对 bootstrap（100,000 次）估计，差距的 95% 区间为 **{derived['task_paired_bootstrap_ci95_low_pp']:.2f}–{derived['task_paired_bootstrap_ci95_high_pp']:.2f} 个百分点**。
- 官方权重超过项目预设 93.85% 验收线。

![总体成功率](figures/01_overall_success.png)

## 第 4 页｜逐 episode 配对结果

| | Base 成功 | Base 失败 |
|---|---:|---:|
| 官方成功 | 0 | 1942 |
| 官方失败 | 0 | 58 |

![Paired outcomes](figures/09_paired_outcomes.png)

2000 个完全对齐的 episode 中，官方权重胜出 1942 次，Base 胜出 0 次；两边一致的 58 次全部是共同失败。换句话说，Base 没有在任何一个 trial 上反超官方权重。

## 第 5 页｜Suite 分解：差距覆盖全部任务类型

| Suite | 官方成功/总数 | 官方成功率 | Base 成功/总数 |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {row['suite_label']} | {row['official_successes']}/{row['total']} | {row['official_success_rate']*100:.2f}% | {row['base_successes']}/{row['total']} |"
        for row in metrics["suite"]
    ) + f"""

![Suite](figures/02_suite_success.png)

LIBERO-10 为微调权重相对短板（93.20%），反映长时序、多物体和多阶段任务更难；但 Base 在四个 suite 均为 0/500，因此并非单一 suite 的偶发问题。

## 第 6 页｜40 个任务热力图

![Task heatmap](figures/03_task_heatmap.png)

官方权重在多数任务上接近满分；Base 的 40 个任务全部为 0%。同题同 trial 的配对矩阵，使结果不会被任务组成差异解释。

## 第 7 页｜任务级稳定性

- 19/40 个任务为 50/50，即 100%；
- {derived['tasks_at_least_98pct']}/40 个任务 ≥98%；
- {derived['tasks_at_least_94pct']}/40 个任务 ≥94%；
- 只有 3 个任务低于 94%，其中 1 个明显长尾任务为 60%。

![Task distribution](figures/04_task_distribution.png)

## 第 8 页｜失败长尾与 Pareto 分布

官方权重只有 {official['failures']} 次失败，但前 3 个困难任务贡献 {derived['top3_failure_share']*100:.1f}% 失败，前 5 个贡献 {derived['top5_failure_share']*100:.1f}%，前 10 个贡献 {derived['top10_failure_share']*100:.1f}%。最难任务为：

> `{worst['task_description']}`：{worst['official_successes']}/{worst['official_total']} = {worst['official_success_rate']*100:.0f}%

![Pareto](figures/05_failure_pareto.png)

这说明“总体 97.10%”不等于所有任务同样可靠，剩余风险集中在少数长时序任务。

## 第 9 页｜动作预算：Base 不是很快失败，而是持续尝试后超时

- Base 的 2000 次失败全部执行到 suite 最大步数，四个 suite 的平均动作预算使用率均为 100%；
- 微调权重全矩阵共执行 {official['total_action_steps']:,} 步，Base 执行 {base['total_action_steps']:,} 步；
- 微调权重少执行 **{derived['saved_action_steps']:,} 步，即 {derived['saved_action_step_share']*100:.1f}%**。

![Action budget](figures/06_action_budget.png)

## 第 10 页｜运行成本：Base 失败会消耗更多评测时间

- 官方权重记录的 episode-duration 总和：{official['total_episode_duration_hours']:.2f} 小时；
- Base：{base['total_episode_duration_hours']:.2f} 小时，是前者的 {derived['base_to_official_duration_ratio']:.2f} 倍；
- 官方 episode 时长中位数 {official['median_duration_seconds']:.2f}s，Base 为 {base['median_duration_seconds']:.2f}s。

![Duration](figures/07_duration_cost.png)

注意：这是 episode 记录时长总和，适合比较评测成本，不应解释为真实机器人生产节拍。

## 第 11 页｜任务效率的探索性观察

![Efficiency](figures/08_task_efficiency_scatter.png)

更高动作预算使用通常伴随更低任务成功率，最难的两只 moka pot 任务尤其突出。该关系用于定位困难任务，不构成“动作多必然导致失败”的因果结论。

## 第 12 页｜为什么这些结果可信

- 两份权重均为 2000 个唯一 episode、2000 个唯一 attempt；
- 矩阵均严格覆盖 4 suites × 10 tasks × 50 trials；
- 每份结果均有 2000 个非空录像与记录一一对应；
- 基础设施错误为 0；
- 成功率从原始 JSONL 重新计算，而非只信任 summary；
- checkpoint 参数树、norm stats、OpenPI commit、环境和日志均有 manifest/hash 证据；
- 最终审计状态为 `passed`。

## 第 13 页｜结论与边界

### 能说明什么

1. 在固定原始 LIBERO 协议下，官方任务微调权重具备强任务能力；
2. `pi05_base` 即使获得相同的 LIBERO assets/norm stats，也无法完成任务；
3. 因而 norm stats 只能匹配数据尺度，不能替代任务微调形成的行为能力；
4. 微调权重的剩余风险集中在少数长时序任务，可针对性优化。

### 不能说明什么

1. 不能把仿真结果直接等同于真实机器人安全或生产可用性；
2. 不能证明 Base 模型在其他任务或经过其他适配后必然失败；
3. 本次环境 seed 固定为 7，不能代表跨 seed 方差；
4. 97.1 个百分点是当前配对协议下的观察差距，不是对“所有微调方法”的普遍因果估计。

### 汇报用一句话

> **同一套 2000-episode LIBERO 试卷中，官方 pi05_libero 微调权重达到 97.10%，而 pi05_base 即使搭配官方 LIBERO stats/assets 仍为 0%；这证明任务微调权重而非仅数据归一化，是获得 LIBERO 操作能力的决定性因素。**

## 附录｜数据与复现

- `data/metrics.json`：机器可读全指标；
- `data/suite_comparison.csv`：suite 对比；
- `data/task_comparison.csv`：40 个任务对比；
- `data/worst_tasks.csv`：失败 Pareto；
- `figures/*.png`：PPT 高清图；
- `figures/*.svg`：矢量图；
- `generate_ppt_report.py`：可复现生成脚本。
"""


def build_readme() -> str:
    return """# LIBERO official-vs-base PPT report

This directory contains a reproducible, data-rich comparison of the official
`pi05_libero` fine-tuned checkpoint and `pi05_base` parameters paired with the
official LIBERO assets and normalization statistics.

Outputs:

- `OpenPI_pi05_LIBERO_权重对比分析.pptx`: PowerPoint deck;
- `OpenPI_pi05_LIBERO_权重对比分析.pdf`: fixed-layout review copy;
- `PPT分析报告.md`: slide-by-slide Chinese analysis and speaker notes;
- `figures/`: PNG and SVG chart assets;
- `data/`: JSON and CSV source tables;
- `preview/`: rendered slide previews and contact sheet.

Regenerate:

```bash
python3 generate_ppt_report.py
```

The generator validates the paired 2,000-episode matrices, unique IDs,
terminal outcomes, infrastructure-error separation, and 1:1 non-empty video
inventory before writing any report.
"""


def uno_prop(name: str, value: Any) -> Any:
    prop = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    prop.Name = name
    prop.Value = value
    return prop


class DeckBuilder:
    WIDTH = 33867
    HEIGHT = 19050

    def __init__(self, ctx: Any, output_dir: Path):
        self.ctx = ctx
        self.smgr = ctx.ServiceManager
        self.desktop = self.smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        self.doc = self.desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
        self.pages = self.doc.getDrawPages()
        self.provider = self.smgr.createInstanceWithContext("com.sun.star.graphic.GraphicProvider", ctx)
        self.output_dir = output_dir
        first = self.pages.getByIndex(0)
        self._clear_page(first)
        first.Width = self.WIDTH
        first.Height = self.HEIGHT
        self.slide_count = 0

    def _clear_page(self, page: Any) -> None:
        while page.getCount():
            page.remove(page.getByIndex(0))

    def _new_page(self) -> Any:
        if self.slide_count == 0:
            page = self.pages.getByIndex(0)
        else:
            page = self.pages.insertNewByIndex(self.pages.getCount())
        page.Width = self.WIDTH
        page.Height = self.HEIGHT
        self._clear_page(page)
        self.slide_count += 1
        self.rect(page, 0, 0, 1, 1, 0xF6F8FB, line_color=0xF6F8FB)
        return page

    @staticmethod
    def _color(value: str | int) -> int:
        if isinstance(value, int):
            return value
        return int(value.lstrip("#"), 16)

    def text(
        self,
        page: Any,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        size: float = 18,
        color: str | int = INK,
        bold: bool = False,
        align: int = 0,
    ) -> Any:
        shape = self.doc.createInstance("com.sun.star.drawing.TextShape")
        shape.Position = Point(int(self.WIDTH * x), int(self.HEIGHT * y))
        shape.Size = Size(int(self.WIDTH * w), int(self.HEIGHT * h))
        page.add(shape)
        shape.setString(text)
        shape.CharFontName = FONT_FAMILY
        shape.CharHeight = float(size)
        shape.CharColor = self._color(color)
        shape.CharWeight = 150.0 if bold else 100.0
        shape.ParaAdjust = align
        shape.TextAutoGrowHeight = False
        shape.TextWordWrap = True
        return shape

    def rect(
        self,
        page: Any,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | int,
        line_color: str | int = "#DCE3EA",
        radius: int = 180,
    ) -> Any:
        shape = self.doc.createInstance("com.sun.star.drawing.RectangleShape")
        shape.Position = Point(int(self.WIDTH * x), int(self.HEIGHT * y))
        shape.Size = Size(int(self.WIDTH * w), int(self.HEIGHT * h))
        shape.FillColor = self._color(fill)
        shape.LineColor = self._color(line_color)
        try:
            shape.CornerRadius = radius
        except Exception:
            pass
        page.add(shape)
        return shape

    def image(self, page: Any, path: Path, x: float, y: float, w: float, h: float) -> Any:
        graphic = self.provider.queryGraphic((uno_prop("URL", uno.systemPathToFileUrl(str(path.resolve()))),))
        shape = self.doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
        shape.Position = Point(int(self.WIDTH * x), int(self.HEIGHT * y))
        shape.Size = Size(int(self.WIDTH * w), int(self.HEIGHT * h))
        shape.Graphic = graphic
        page.add(shape)
        return shape

    def title(self, page: Any, title: str, subtitle: str | None = None) -> None:
        self.text(page, 0.035, 0.035, 0.89, 0.085, title, 25, INK, True)
        self.rect(page, 0.035, 0.128, 0.07, 0.007, OFFICIAL_COLOR, line_color=OFFICIAL_COLOR)
        if subtitle:
            self.text(page, 0.12, 0.112, 0.82, 0.045, subtitle, 10.5, MUTED)

    def footer(self, page: Any, page_number: int, source: str = "本项目 episode 级评测数据") -> None:
        self.text(page, 0.035, 0.955, 0.75, 0.025, source, 8.5, MUTED)
        self.text(page, 0.92, 0.955, 0.045, 0.025, f"{page_number:02d}", 8.5, MUTED, align=1)

    def chart_slide(self, title: str, chart: Path, takeaway: str, body: str, page_number: int) -> None:
        page = self._new_page()
        self.title(page, title)
        self.rect(page, 0.025, 0.165, 0.69, 0.75, 0xFFFFFF)
        self.image(page, chart, 0.045, 0.19, 0.65, 0.68)
        self.rect(page, 0.735, 0.165, 0.24, 0.75, 0xFFFFFF)
        self.text(page, 0.76, 0.205, 0.19, 0.055, "一句话结论", 12, OFFICIAL_COLOR, True)
        self.text(page, 0.76, 0.27, 0.19, 0.18, takeaway, 17, INK, True)
        self.rect(page, 0.76, 0.48, 0.15, 0.006, AMBER, line_color=AMBER)
        self.text(page, 0.76, 0.52, 0.19, 0.30, body, 12.3, MUTED)
        self.footer(page, page_number)

    def save(self, stem: str) -> tuple[Path, Path, Path]:
        odp = self.output_dir / f"{stem}.odp"
        pptx = self.output_dir / f"{stem}.pptx"
        pdf = self.output_dir / f"{stem}.pdf"
        self.doc.storeAsURL(
            uno.systemPathToFileUrl(str(odp.resolve())),
            (uno_prop("FilterName", "impress8"), uno_prop("Overwrite", True)),
        )
        self.doc.storeToURL(
            uno.systemPathToFileUrl(str(pptx.resolve())),
            (uno_prop("FilterName", "Impress MS PowerPoint 2007 XML"), uno_prop("Overwrite", True)),
        )
        self.doc.storeToURL(
            uno.systemPathToFileUrl(str(pdf.resolve())),
            (uno_prop("FilterName", "impress_pdf_Export"), uno_prop("Overwrite", True)),
        )
        self.doc.close(True)
        return odp, pptx, pdf


def create_deck(ctx: Any, metrics: dict[str, Any]) -> tuple[Path, Path, Path]:
    deck = DeckBuilder(ctx, HERE)
    official = metrics["overall"]["official"]
    base = metrics["overall"]["base"]
    derived = metrics["derived"]
    worst = metrics["worst_tasks"][0]

    page = deck._new_page()
    deck.rect(page, 0, 0, 1, 1, 0x13243A, line_color=0x13243A)
    deck.rect(page, 0.035, 0.08, 0.012, 0.62, OFFICIAL_COLOR, line_color=OFFICIAL_COLOR)
    deck.text(page, 0.08, 0.13, 0.82, 0.20, "OpenPI π0.5 权重\nLIBERO 本体对比评测", 34, 0xFFFFFF, True)
    deck.text(page, 0.08, 0.39, 0.82, 0.10, "官方 pi05_libero 微调权重  vs  pi05_base 基准权重", 18, 0xD9E5EF)
    deck.text(page, 0.08, 0.57, 0.22, 0.12, "97.10%", 38, 0x75D6B2, True)
    deck.text(page, 0.30, 0.59, 0.24, 0.10, "vs", 22, 0x8CA0B5, True, align=1)
    deck.text(page, 0.54, 0.57, 0.22, 0.12, "0.00%", 38, 0xF08C91, True)
    deck.text(page, 0.08, 0.80, 0.82, 0.06, "同一套 4 suites × 10 tasks × 50 trials 配对矩阵", 14, 0xD9E5EF)
    deck.footer(page, 1, "OpenPI π0.5 × LIBERO · 评测结果归档")

    page = deck._new_page()
    deck.title(page, "先确认可比性：同一考场、同一试卷，仅参数能力不同")
    cards = [
        ("任务矩阵", "4 suites × 10 tasks ×\n50 trials = 2000 episodes"),
        ("输入与推理", "224×224 图像 · seed 7\nreplan steps = 5"),
        ("成功判定", "同一环境与任务描述\n固定 suite 最大步数"),
        ("Base 对照", "pi05_base 参数\n+ 官方 LIBERO stats/assets"),
    ]
    for index, (heading, body) in enumerate(cards):
        x = 0.04 + index * 0.24
        deck.rect(page, x, 0.20, 0.21, 0.25, 0xFFFFFF)
        deck.text(page, x + 0.02, 0.23, 0.17, 0.05, heading, 14, OFFICIAL_COLOR, True)
        deck.text(page, x + 0.02, 0.30, 0.17, 0.11, body, 11, INK)
    deck.rect(page, 0.04, 0.51, 0.92, 0.32, 0xEAF4F1, line_color=OFFICIAL_COLOR)
    deck.text(page, 0.07, 0.56, 0.86, 0.08, "为什么这个对照重要？", 18, OFFICIAL_COLOR, True)
    deck.text(
        page, 0.07, 0.65, 0.86, 0.13,
        "如果 Base 仍然失败，就不能再把原因简单归结为“缺少 LIBERO 的单位换算表或环境资源”。\n两组差异反映的是当前固定协议下，任务微调权重是否真正形成了 LIBERO 操作能力。",
        16, INK,
    )
    deck.footer(page, 2)

    deck.chart_slide(
        "总体结果：同一 2000-episode 矩阵上为 97.10% vs 0%",
        FIGURE_DIR / "01_overall_success.png",
        "任务微调权重是获得 LIBERO 操作能力的决定性因素。",
        f"观察差距 {derived['observed_success_rate_gap_pp']:.1f} 个百分点；任务配对 bootstrap 95% CI 为 {derived['task_paired_bootstrap_ci95_low_pp']:.2f}–{derived['task_paired_bootstrap_ci95_high_pp']:.2f}。\n\n官方权重通过 93.85% 验收线；Base 95% CI 上界仅 {base['ci95_high']*100:.2f}%。",
        3,
    )
    deck.chart_slide(
        "逐 episode 配对：Base 没有任何一次反超",
        FIGURE_DIR / "09_paired_outcomes.png",
        "1,942 次差异全部指向官方权重成功、Base 失败。",
        "官方胜出 1,942 次；Base 胜出 0 次。\n\n两边共同失败 58 次；配对一致率 2.9%，且全部是共同失败。",
        4,
    )
    deck.chart_slide(
        "Suite 分解：差距覆盖空间、物体、目标与长时序任务",
        FIGURE_DIR / "02_suite_success.png",
        "四个 suite 全部拉开差距，不是单一题型的偶发现象。",
        "微调权重：Spatial 98.6%、Object 98.4%、Goal 98.2%、LIBERO-10 93.2%。\n\nBase 四套均为 0/500。",
        5,
    )

    page = deck._new_page()
    deck.title(page, "40 个任务热力图：微调权重多数接近满分，Base 全部为 0")
    deck.rect(page, 0.025, 0.16, 0.95, 0.77, 0xFFFFFF)
    deck.image(page, FIGURE_DIR / "03_task_heatmap.png", 0.055, 0.20, 0.89, 0.64)
    deck.text(page, 0.06, 0.865, 0.88, 0.04, "每个格子包含 50 次 trial；颜色和数字均为任务级成功率。", 10.5, MUTED, align=1)
    deck.footer(page, 6)

    deck.chart_slide(
        "任务级稳定性：29/40 个任务达到至少 98%",
        FIGURE_DIR / "04_task_distribution.png",
        "高分并非来自少数任务拉高平均：多数任务本身就接近满分。",
        f"19 个任务 100%；{derived['tasks_at_least_98pct']} 个任务 ≥98%；{derived['tasks_at_least_94pct']} 个任务 ≥94%。\n\nBase：40/40 个任务均为 0%。",
        7,
    )
    deck.chart_slide(
        "失败长尾：58 次失败的一半集中在 3 个任务",
        FIGURE_DIR / "05_failure_pareto.png",
        "总体 97.10% 仍掩盖少数长时序任务的明显风险。",
        f"最难任务：两只 moka pot 放到炉灶，{worst['official_successes']}/50 = {worst['official_success_rate']*100:.0f}%。\n\nTop-3 占 {derived['top3_failure_share']*100:.1f}% 失败；Top-5 占 {derived['top5_failure_share']*100:.1f}%。",
        8,
    )
    deck.chart_slide(
        "动作预算：Base 不是快速报错，而是持续尝试后全部超时",
        FIGURE_DIR / "06_action_budget.png",
        "Base 的 2000 次失败都跑满最大步数，属于系统性策略失败。",
        f"官方总动作 {official['total_action_steps']:,} 步；Base {base['total_action_steps']:,} 步。\n\n微调权重少执行 {derived['saved_action_steps']:,} 步，减少 {derived['saved_action_step_share']*100:.1f}%。",
        9,
    )
    deck.chart_slide(
        "评测成本：Base 的 episode-duration 总和达到 1.90 倍",
        FIGURE_DIR / "07_duration_cost.png",
        "不会做的权重不仅得分低，也会持续消耗动作与评测时间。",
        f"官方 {official['total_episode_duration_hours']:.2f}h；Base {base['total_episode_duration_hours']:.2f}h。\n\n中位 episode 时长：{official['median_duration_seconds']:.2f}s vs {base['median_duration_seconds']:.2f}s。",
        10,
    )
    deck.chart_slide(
        "探索性观察：高动作预算使用与困难任务相伴",
        FIGURE_DIR / "08_task_efficiency_scatter.png",
        "最难任务既成功率最低，也最接近耗尽动作预算。",
        "该图用于定位困难任务和长尾，不构成因果结论。\n\n建议后续针对多物体、长时序、状态恢复任务做专门回归。",
        11,
    )

    page = deck._new_page()
    deck.title(page, "为什么可以相信这些数字：结果能力与证据质量分开验收")
    evidence = [
        ("2000 / 2000", "每份权重矩阵完整"),
        ("2000", "唯一 episode + attempt"),
        ("2000", "非空录像一一对应"),
        ("0", "基础设施错误"),
        ("Exact", "任务/试次配对一致"),
        ("Passed", "最终独立审计"),
    ]
    for index, (value, label) in enumerate(evidence):
        row, col = divmod(index, 3)
        x, y = 0.06 + col * 0.31, 0.21 + row * 0.28
        deck.rect(page, x, y, 0.27, 0.22, 0xFFFFFF)
        deck.text(page, x + 0.025, y + 0.035, 0.22, 0.07, value, 27, OFFICIAL_COLOR, True, align=1)
        deck.text(page, x + 0.025, y + 0.125, 0.22, 0.045, label, 12, MUTED, align=1)
    deck.text(page, 0.08, 0.80, 0.84, 0.07, "每个分数都能回溯到 JSONL、录像、日志、manifest、Git commit 与 checkpoint 哈希。", 14, INK, True, align=1)
    deck.footer(page, 12)

    page = deck._new_page()
    deck.title(page, "结论：微调权重形成了任务能力，但仍需关注长时序尾部风险")
    deck.rect(page, 0.04, 0.19, 0.43, 0.58, 0xEAF4F1, line_color=OFFICIAL_COLOR)
    deck.text(page, 0.07, 0.23, 0.36, 0.06, "本次数据能够证明", 17, OFFICIAL_COLOR, True)
    deck.text(
        page, 0.07, 0.32, 0.36, 0.38,
        "1. 官方 pi05_libero 在原始 LIBERO 上达到 97.10%。\n\n2. Base 即使搭配相同 stats/assets 仍为 0%。\n\n3. norm stats 不能替代任务微调能力。\n\n4. 剩余失败集中在少数长时序任务。",
        14, INK,
    )
    deck.rect(page, 0.53, 0.19, 0.43, 0.58, 0xFFF3E5, line_color=AMBER)
    deck.text(page, 0.56, 0.23, 0.36, 0.06, "本次数据不能直接证明", 17, AMBER, True)
    deck.text(
        page, 0.56, 0.32, 0.36, 0.38,
        "1. 真实机器人部署安全。\n\n2. 其他任务或其他适配下 Base 必然失败。\n\n3. 跨随机 seed 的性能方差。\n\n4. 所有微调方法都能产生同样增益。",
        14, INK,
    )
    deck.rect(page, 0.08, 0.81, 0.84, 0.10, 0x13243A, line_color=0x13243A)
    deck.text(page, 0.11, 0.835, 0.78, 0.055, "结论：同一试卷 97.10% vs 0%，任务微调权重而非仅数据归一化决定了 LIBERO 能力。", 12.5, 0xFFFFFF, True, align=1)
    deck.footer(page, 13)
    return deck.save("OpenPI_pi05_LIBERO_权重对比分析")


def launch_office() -> tuple[subprocess.Popen[str], Any, tempfile.TemporaryDirectory[str]]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    profile = tempfile.TemporaryDirectory(prefix="pi05-ppt-libreoffice-")
    profile_url = uno.systemPathToFileUrl(profile.name)
    command = [
        shutil.which("soffice") or "soffice",
        "--headless",
        "--norestore",
        "--nofirststartwizard",
        f"-env:UserInstallation={profile_url}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    deadline = time.monotonic() + 20
    while True:
        try:
            ctx = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
            )
            return process, ctx, profile
        except Exception:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=3)
                profile.cleanup()
                raise RuntimeError(f"LibreOffice exited early: {stdout}\n{stderr}")
            if time.monotonic() >= deadline:
                process.terminate()
                profile.cleanup()
                raise TimeoutError("LibreOffice UNO listener did not become ready")
            time.sleep(0.25)


def render_previews(pdf: Path) -> None:
    for path in PREVIEW_DIR.glob("slide-*.png"):
        path.unlink()
    subprocess.run(
        ["pdftoppm", "-png", "-r", "110", str(pdf), str(PREVIEW_DIR / "slide")],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    generated = sorted(PREVIEW_DIR.glob("slide-*.png"))
    if not generated:
        generated = sorted(PREVIEW_DIR.glob("slide-*.png"))
    # pdftoppm creates slide-01.png style names. Build a contact sheet.
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in generated]
    if not images:
        raise RuntimeError("no slide previews generated")
    thumb_width = 480
    thumb_height = round(images[0].height * thumb_width / images[0].width)
    columns = 3
    rows = math.ceil(len(images) / columns)
    gap = 18
    sheet = Image.new(
        "RGB",
        (columns * thumb_width + (columns + 1) * gap, rows * (thumb_height + 28) + (rows + 1) * gap),
        "#E8EDF3",
    )
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        x = gap + col * (thumb_width + gap)
        y = gap + row * (thumb_height + 28 + gap)
        resampling = getattr(Image, "Resampling", Image)
        thumbnail = image.resize((thumb_width, thumb_height), resampling.LANCZOS)
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 5, y + thumb_height + 4), f"Slide {index + 1}", fill="#142238")
    sheet.save(PREVIEW_DIR / "contact_sheet.png")
    for image in images:
        image.close()


def main() -> None:
    metrics = build_metrics()
    write_artifacts(metrics)
    process, ctx, profile = launch_office()
    try:
        odp, pptx, pdf = create_deck(ctx, metrics)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        profile.cleanup()
    render_previews(pdf)
    print(f"official={metrics['overall']['official']['successes']}/2000")
    print(f"base={metrics['overall']['base']['successes']}/2000")
    print(f"pptx={pptx}")
    print(f"pdf={pdf}")
    print(f"slides=13")


if __name__ == "__main__":
    main()
