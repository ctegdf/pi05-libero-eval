#!/usr/bin/env python3
"""Generate a reproducible, dependency-free analysis report for the archived evals."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPORT_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = REPORT_DIR.parent
DATA_DIR = REPORT_DIR / "data"
FIGURE_DIR = REPORT_DIR / "figures"

SOURCES = {
    "official": ARCHIVE_ROOT / "openpi-libero/official-full/episodes.jsonl",
    "base_libero_assets": ARCHIVE_ROOT
    / "openpi-libero/base-libero-assets-full/episodes.jsonl",
    "plus": ARCHIVE_ROOT / "plus-pro/plus-full-merged/episodes.jsonl",
    "pro": ARCHIVE_ROOT / "plus-pro/pro-full/episodes.jsonl",
}
VIDEO_DIRS = {
    "official": ARCHIVE_ROOT / "openpi-libero/official-full/videos",
    "base_libero_assets": ARCHIVE_ROOT / "openpi-libero/base-libero-assets-full/videos",
    "plus": ARCHIVE_ROOT / "plus-pro/plus-full-merged/videos",
    "pro": ARCHIVE_ROOT / "plus-pro/pro-full/videos",
}
FINAL_AUDIT = ARCHIVE_ROOT / "plus-pro/final-audit/report.json"

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PLUS_CATEGORIES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]
PLUS_DIFFICULTIES = ["1", "2", "3", "4", "5", "None"]
PRO_PERTURBATIONS = ["lan(semantic)", "object", "swap(position)", "task"]

LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
    "Background Textures": "背景纹理",
    "Camera Viewpoints": "相机视角",
    "Language Instructions": "语言指令",
    "Light Conditions": "光照条件",
    "Objects Layout": "物体布局",
    "Robot Initial States": "机器人初态",
    "Sensor Noise": "传感器噪声",
    "lan(semantic)": "语言语义",
    "object": "物体替换",
    "swap(position)": "位置交换",
    "task": "任务替换",
    "env": "环境扰动（N/A）",
    "None": "未标注",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return (center - margin, center + margin)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    failures = sum(row["status"] == "failure" for row in rows)
    errors = sum(row["status"] == "error" or row.get("error_category") is not None for row in rows)
    ci_low, ci_high = wilson(successes, total)
    success_steps = [float(row["action_steps"]) for row in rows if row["success"]]
    failure_steps = [float(row["action_steps"]) for row in rows if not row["success"]]
    durations = [float(row["duration_seconds"]) for row in rows]
    return {
        "total": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "rate": successes / total if total else None,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median_action_steps_success": statistics.median(success_steps) if success_steps else None,
        "median_action_steps_failure": statistics.median(failure_steps) if failure_steps else None,
        "failure_horizon_exhausted": sum(
            (not row["success"]) and row["action_steps"] == row["max_steps"] for row in rows
        ),
        "median_duration_seconds": statistics.median(durations) if durations else None,
        "mean_duration_seconds": statistics.mean(durations) if durations else None,
        "p90_duration_seconds": percentile(durations, 0.90) if durations else None,
        "p99_duration_seconds": percentile(durations, 0.99) if durations else None,
        "sum_episode_compute_hours": sum(durations) / 3600,
    }


def grouped(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field)) for field in keys)
        buckets[key].append(row)
    result = []
    for key, bucket in buckets.items():
        item = {field: value for field, value in zip(keys, key)}
        item.update(aggregate(bucket))
        result.append(item)
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    selected = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{100 * value:.{digits}f}%"


def svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_document(width: int, height: int, body: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{svg_escape(title)}">
<rect width="100%" height="100%" fill="#ffffff" rx="18"/>
<style>
text {{ font-family: system-ui, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; fill: #172033; }}
.title {{ font-size: 23px; font-weight: 700; }}
.label {{ font-size: 16px; }}
.small {{ font-size: 13px; fill: #596579; }}
.value {{ font-size: 15px; font-weight: 700; }}
</style>
{body}
</svg>
"""


def rate_color(rate: float) -> str:
    if rate >= 0.9:
        return "#159a6b"
    if rate >= 0.75:
        return "#3484d4"
    if rate >= 0.5:
        return "#e39b22"
    return "#d9534f"


def bar_chart(
    path: Path,
    title: str,
    rows: Sequence[dict[str, Any]],
    label_key: str = "label",
    note: str | None = None,
    threshold: float | None = None,
) -> None:
    width = 1080
    left, right, top, row_height = 260, 220, 90, 58
    plot_width = width - left - right
    height = top + row_height * len(rows) + (65 if note else 35)
    parts = [f'<text class="title" x="36" y="42">{svg_escape(title)}</text>']
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        parts.append(f'<line x1="{x:.1f}" y1="64" x2="{x:.1f}" y2="{height - 45}" stroke="#e7ebf1"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="78" text-anchor="middle">{tick}%</text>')
    if threshold is not None:
        x = left + plot_width * threshold
        parts.append(f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="{height - 42}" stroke="#a52f4b" stroke-width="2" stroke-dasharray="6 5"/>')
        parts.append(f'<text class="small" x="{x - 5:.1f}" y="{height - 47}" text-anchor="end" style="fill:#a52f4b">验收线 {threshold*100:.2f}%</text>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        rate = float(row["rate"])
        ci_low, ci_high = float(row["ci95_low"]), float(row["ci95_high"])
        label = str(row[label_key])
        parts.append(f'<text class="label" x="{left - 16}" y="{y + 25}" text-anchor="end">{svg_escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{plot_width}" height="27" fill="#f2f4f7" rx="7"/>')
        parts.append(
            f'<rect x="{left}" y="{y + 6}" width="{plot_width * rate:.1f}" height="27" '
            f'fill="{rate_color(rate)}" rx="7"/>'
        )
        low_x = left + plot_width * ci_low
        high_x = left + plot_width * ci_high
        mid_y = y + 39
        parts.append(f'<line x1="{low_x:.1f}" y1="{mid_y}" x2="{high_x:.1f}" y2="{mid_y}" stroke="#172033" stroke-width="2"/>')
        parts.append(f'<line x1="{low_x:.1f}" y1="{mid_y - 4}" x2="{low_x:.1f}" y2="{mid_y + 4}" stroke="#172033" stroke-width="2"/>')
        parts.append(f'<line x1="{high_x:.1f}" y1="{mid_y - 4}" x2="{high_x:.1f}" y2="{mid_y + 4}" stroke="#172033" stroke-width="2"/>')
        parts.append(
            f'<text class="value" x="{left + plot_width + 12}" y="{y + 25}">{pct(rate)} '
            f'<tspan class="small">({row["successes"]}/{row["total"]})</tspan></text>'
        )
    if note:
        parts.append(f'<text class="small" x="36" y="{height - 18}">{svg_escape(note)}</text>')
    path.write_text(svg_document(width, height, "\n".join(parts), title), encoding="utf-8")


def line_chart(path: Path, title: str, rows: Sequence[dict[str, Any]], note: str | None = None) -> None:
    width, height = 1080, 480
    left, right, top, bottom = 90, 70, 80, 95
    plot_width, plot_height = width - left - right, height - top - bottom
    parts = [f'<text class="title" x="36" y="42">{svg_escape(title)}</text>']
    for tick in range(0, 101, 20):
        y = top + plot_height * (1 - tick / 100)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e7ebf1"/>')
        parts.append(f'<text class="small" x="{left - 14}" y="{y + 5:.1f}" text-anchor="end">{tick}%</text>')
    points = []
    for index, row in enumerate(rows):
        x = left + plot_width * (index / max(1, len(rows) - 1))
        y = top + plot_height * (1 - float(row["rate"]))
        points.append((x, y))
    parts.append('<polyline points="{}" fill="none" stroke="#3484d4" stroke-width="4"/>'.format(" ".join(f"{x:.1f},{y:.1f}" for x, y in points)))
    for (x, y), row in zip(points, rows):
        low_y = top + plot_height * (1 - float(row["ci95_low"]))
        high_y = top + plot_height * (1 - float(row["ci95_high"]))
        parts.append(f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" stroke="#172033" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{rate_color(float(row["rate"]))}" stroke="#fff" stroke-width="2"/>')
        parts.append(f'<text class="value" x="{x:.1f}" y="{y - 15:.1f}" text-anchor="middle">{pct(float(row["rate"]))}</text>')
        parts.append(f'<text class="label" x="{x:.1f}" y="{height - 57}" text-anchor="middle">{svg_escape(row["label"])}</text>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{height - 34}" text-anchor="middle">n={row["total"]}</text>')
    if note:
        parts.append(f'<text class="small" x="36" y="{height - 10}">{svg_escape(note)}</text>')
    path.write_text(svg_document(width, height, "\n".join(parts), title), encoding="utf-8")


def heat_color(rate: float) -> str:
    red = (217, 83, 79)
    yellow = (245, 190, 74)
    green = (21, 154, 107)
    if rate < 0.5:
        ratio = rate / 0.5
        start, end = red, yellow
    else:
        ratio = (rate - 0.5) / 0.5
        start, end = yellow, green
    rgb = tuple(round(start[i] + (end[i] - start[i]) * ratio) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def heatmap_chart(
    path: Path,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cells: dict[tuple[str, str], dict[str, Any]],
    note: str | None = None,
) -> None:
    width = 1080
    left, top, cell_width, cell_height = 235, 105, 190, 72
    height = top + cell_height * len(row_labels) + (80 if note else 45)
    parts = [f'<text class="title" x="36" y="42">{svg_escape(title)}</text>']
    for col_index, col in enumerate(col_labels):
        x = left + col_index * cell_width + cell_width / 2
        parts.append(f'<text class="label" x="{x:.1f}" y="82" text-anchor="middle">{svg_escape(LABELS.get(col, col))}</text>')
    for row_index, row in enumerate(row_labels):
        y = top + row_index * cell_height
        parts.append(f'<text class="label" x="{left - 16}" y="{y + 31}" text-anchor="end">{svg_escape(LABELS.get(row, row))}</text>')
        for col_index, col in enumerate(col_labels):
            x = left + col_index * cell_width
            cell = cells.get((row, col))
            if cell is None:
                fill, value, detail = "#e8ebf0", "N/A", ""
            else:
                rate = float(cell["rate"])
                fill, value = heat_color(rate), pct(rate)
                detail = f'{cell["successes"]}/{cell["total"]}'
            parts.append(f'<rect x="{x + 4}" y="{y + 4}" width="{cell_width - 8}" height="{cell_height - 8}" rx="9" fill="{fill}"/>')
            parts.append(f'<text x="{x + cell_width / 2:.1f}" y="{y + 29}" text-anchor="middle" style="font-size:17px;font-weight:700;fill:#172033">{svg_escape(value)}</text>')
            if detail:
                parts.append(f'<text class="small" x="{x + cell_width / 2:.1f}" y="{y + 50}" text-anchor="middle">{svg_escape(detail)}</text>')
    if note:
        parts.append(f'<text class="small" x="36" y="{height - 20}">{svg_escape(note)}</text>')
    path.write_text(svg_document(width, height, "\n".join(parts), title), encoding="utf-8")


def exposure_failure_chart(path: Path, title: str, rows: Sequence[dict[str, Any]]) -> None:
    width = 1080
    left, right, top, row_height = 260, 100, 90, 64
    plot_width = width - left - right
    height = top + len(rows) * row_height + 70
    parts = [f'<text class="title" x="36" y="42">{svg_escape(title)}</text>']
    max_share = max(max(float(row["episode_share"]), float(row["failure_share"])) for row in rows)
    axis_max = math.ceil(max_share * 10) / 10
    for tick in range(0, 6):
        share = axis_max * tick / 5
        x = left + plot_width * share / axis_max
        parts.append(f'<line x1="{x:.1f}" y1="65" x2="{x:.1f}" y2="{height - 55}" stroke="#e7ebf1"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="78" text-anchor="middle">{share * 100:.0f}%</text>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        parts.append(f'<text class="label" x="{left - 16}" y="{y + 28}" text-anchor="end">{svg_escape(row["label"])}</text>')
        exposure_width = plot_width * float(row["episode_share"]) / axis_max
        failure_width = plot_width * float(row["failure_share"]) / axis_max
        parts.append(f'<rect x="{left}" y="{y + 7}" width="{exposure_width:.1f}" height="17" rx="5" fill="#93a1b5"/>')
        parts.append(f'<rect x="{left}" y="{y + 29}" width="{failure_width:.1f}" height="17" rx="5" fill="#d9534f"/>')
        parts.append(f'<text class="small" x="{left + exposure_width + 8:.1f}" y="{y + 20}">{float(row["episode_share"]) * 100:.1f}% episodes</text>')
        parts.append(f'<text class="small" x="{left + failure_width + 8:.1f}" y="{y + 43}">{float(row["failure_share"]) * 100:.1f}% failures</text>')
    parts.append(f'<rect x="36" y="{height - 34}" width="14" height="14" fill="#93a1b5"/><text class="small" x="58" y="{height - 22}">评测占比</text>')
    parts.append(f'<rect x="150" y="{height - 34}" width="14" height="14" fill="#d9534f"/><text class="small" x="172" y="{height - 22}">失败占比</text>')
    path.write_text(svg_document(width, height, "\n".join(parts), title), encoding="utf-8")


def source_distribution_chart(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    title = "LIBERO-Pro：source 级成功率呈明显两极分化"
    width, height = 1080, 430
    left, right, top, bar_height = 230, 70, 95, 50
    plot_width = width - left - right
    bins = [
        ("0–20%", 0.0, 0.2, True),
        (">20–50%", 0.2, 0.5, False),
        (">50–80%", 0.5, 0.8, False),
        (">80–<100%", 0.8, 1.0, False),
        ("100%", 1.0, 1.0, True),
    ]
    colors = ["#d9534f", "#ee9a64", "#f1c75b", "#78b98b", "#159a6b"]
    parts = [f'<text class="title" x="36" y="42">{svg_escape(title)}</text>']
    legend_x = 220
    for (label, _, _, _), color in zip(bins, colors):
        parts.append(f'<rect x="{legend_x}" y="58" width="13" height="13" fill="{color}"/><text class="small" x="{legend_x + 20}" y="70">{svg_escape(label)}</text>')
        legend_x += 145
    grouped_sources: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped_sources[str(row["perturbation"])].append(float(row["rate"]))
    for index, perturbation in enumerate(PRO_PERTURBATIONS):
        rates = grouped_sources[perturbation]
        counts = []
        for label, low, high, inclusive_low in bins:
            if label == "0–20%":
                count = sum(rate <= high for rate in rates)
            elif label == "100%":
                count = sum(rate == 1 for rate in rates)
            elif label == ">80–<100%":
                count = sum(low < rate < high for rate in rates)
            else:
                count = sum(low < rate <= high for rate in rates)
            counts.append(count)
        y = top + index * 75
        parts.append(f'<text class="label" x="{left - 16}" y="{y + 31}" text-anchor="end">{svg_escape(LABELS[perturbation])}</text>')
        x = left
        for count, color in zip(counts, colors):
            segment_width = plot_width * count / len(rates)
            if segment_width > 0:
                parts.append(f'<rect x="{x:.1f}" y="{y + 5}" width="{segment_width:.1f}" height="39" fill="{color}"/>')
                if segment_width >= 28:
                    parts.append(f'<text x="{x + segment_width / 2:.1f}" y="{y + 31}" text-anchor="middle" style="font-size:14px;font-weight:700;fill:#172033">{count}</text>')
            x += segment_width
        parts.append(f'<text class="small" x="{width - right + 10}" y="{y + 31}">40 sources</text>')
    parts.append(f'<text class="small" x="36" y="{height - 20}">每个 source 含 50 trials；数字为落在该成功率区间的 source 数。</text>')
    path.write_text(svg_document(width, height, "\n".join(parts), title), encoding="utf-8")


def duration_range_chart(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    title = "LIBERO-Plus：各扰动 episode 时长中位数与 P90"
    width = 1080
    left, right, top, row_height = 260, 105, 90, 58
    plot_width = width - left - right
    axis_max = math.ceil(max(float(row["p90_duration_seconds"]) for row in rows) / 30) * 30
    height = top + len(rows) * row_height + 70
    parts = [f'<text class="title" x="36" y="42">{svg_escape(title)}</text>']
    for tick in range(0, 7):
        seconds = axis_max * tick / 6
        x = left + plot_width * seconds / axis_max
        parts.append(f'<line x1="{x:.1f}" y1="65" x2="{x:.1f}" y2="{height - 55}" stroke="#e7ebf1"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="78" text-anchor="middle">{seconds:.0f}s</text>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        median_x = left + plot_width * float(row["median_duration_seconds"]) / axis_max
        p90_x = left + plot_width * float(row["p90_duration_seconds"]) / axis_max
        parts.append(f'<text class="label" x="{left - 16}" y="{y + 28}" text-anchor="end">{svg_escape(row["label"])}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 22}" x2="{p90_x:.1f}" y2="{y + 22}" stroke="#c8d1dc" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{median_x:.1f}" cy="{y + 22}" r="7" fill="#3484d4"/>')
        parts.append(f'<circle cx="{p90_x:.1f}" cy="{y + 22}" r="7" fill="#e39b22"/>')
        parts.append(f'<text class="small" x="{median_x:.1f}" y="{y + 47}" text-anchor="middle">M {float(row["median_duration_seconds"]):.1f}s</text>')
        parts.append(f'<text class="small" x="{p90_x:.1f}" y="{y + 47}" text-anchor="middle">P90 {float(row["p90_duration_seconds"]):.1f}s</text>')
    parts.append(f'<circle cx="40" cy="{height - 27}" r="6" fill="#3484d4"/><text class="small" x="55" y="{height - 22}">中位数</text>')
    parts.append(f'<circle cx="145" cy="{height - 27}" r="6" fill="#e39b22"/><text class="small" x="160" y="{height - 22}">P90</text>')
    path.write_text(svg_document(width, height, "\n".join(parts), title), encoding="utf-8")


def cell_map(rows: Sequence[dict[str, Any]], row_key: str, col_key: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row[row_key]), str(row[col_key])): row for row in rows}


def assert_terminal_integrity(name: str, rows: Sequence[dict[str, Any]], expected: int) -> None:
    assert len(rows) == expected, (name, len(rows), expected)
    assert len({row["episode_id"] for row in rows}) == expected, name
    assert len({row["attempt_id"] for row in rows}) == expected, name
    assert all(row["status"] in {"success", "failure"} for row in rows), name
    assert all(row.get("error_category") is None for row in rows), name
    assert all(row.get("video_status") == "written" for row in rows), name
    actual_videos = {path.name: path for path in VIDEO_DIRS[name].glob("*.mp4")}
    recorded_videos = {
        Path(row.get("video_relative") or row["video"]).name
        for row in rows
    }
    assert len(actual_videos) == expected, (name, len(actual_videos), expected)
    assert recorded_videos == set(actual_videos), name
    assert all(path.is_file() and path.stat().st_size > 0 for path in actual_videos.values()), name


def ordered(rows: list[dict[str, Any]], key: str, values: Sequence[str]) -> list[dict[str, Any]]:
    index = {value: position for position, value in enumerate(values)}
    return sorted(rows, key=lambda row: index[str(row[key])])


def matching_row(rows: Sequence[dict[str, Any]], **criteria: str) -> dict[str, Any]:
    matches = [row for row in rows if all(str(row[key]) == str(value) for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def narrative_values(metrics: dict[str, Any]) -> dict[str, Any]:
    robot_cells = [
        row for row in metrics["plus_category_suite"] if row["category"] == "Robot Initial States"
    ]
    semantic_cells = [
        row for row in metrics["pro_perturbation_suite"] if row["perturbation"] == "lan(semantic)"
    ]
    object_cells = [
        row for row in metrics["pro_perturbation_suite"] if row["perturbation"] == "object"
    ]
    worst_official = metrics["findings"]["official_worst_task"]
    return {
        "official_macro_rate": statistics.mean(float(row["rate"]) for row in metrics["official_suite"]),
        "official_worst_task_rate": float(worst_official["rate"]),
        "official_worst_task_successes": int(worst_official["successes"]),
        "official_worst_task_total": int(worst_official["total"]),
        "official_worst_task_failures": int(worst_official["total"] - worst_official["successes"]),
        "plus_difficulty_1_rate": float(matching_row(metrics["plus_difficulty"], difficulty="1")["rate"]),
        "plus_difficulty_5_rate": float(matching_row(metrics["plus_difficulty"], difficulty="5")["rate"]),
        "plus_unlabelled_difficulty_rate": float(matching_row(metrics["plus_difficulty"], difficulty="None")["rate"]),
        "plus_unlabelled_difficulty_total": int(matching_row(metrics["plus_difficulty"], difficulty="None")["total"]),
        "plus_background_object_rate": float(matching_row(metrics["plus_category_suite"], category="Background Textures", suite="libero_object")["rate"]),
        "plus_camera_libero10_rate": float(matching_row(metrics["plus_category_suite"], category="Camera Viewpoints", suite="libero_10")["rate"]),
        "plus_robot_min_rate": min(float(row["rate"]) for row in robot_cells),
        "plus_robot_max_rate": max(float(row["rate"]) for row in robot_cells),
        "pro_semantic_rate": float(matching_row(metrics["pro_perturbation"], perturbation="lan(semantic)")["rate"]),
        "pro_object_rate": float(matching_row(metrics["pro_perturbation"], perturbation="object")["rate"]),
        "pro_swap_rate": float(matching_row(metrics["pro_perturbation"], perturbation="swap(position)")["rate"]),
        "pro_task_rate": float(matching_row(metrics["pro_perturbation"], perturbation="task")["rate"]),
        "pro_semantic_min_rate": min(float(row["rate"]) for row in semantic_cells),
        "pro_semantic_max_rate": max(float(row["rate"]) for row in semantic_cells),
        "pro_object_min_rate": min(float(row["rate"]) for row in object_cells),
        "pro_object_max_rate": max(float(row["rate"]) for row in object_cells),
        "pro_swap_libero10_rate": float(matching_row(metrics["pro_perturbation_suite"], perturbation="swap(position)", suite="libero_10")["rate"]),
        "pro_task_object_rate": float(matching_row(metrics["pro_perturbation_suite"], perturbation="task", suite="libero_object")["rate"]),
        "pro_swap_zero_sources": int(matching_row(metrics["pro_source_summary"], perturbation="swap(position)")["zero_sources"]),
        "pro_task_zero_sources": int(matching_row(metrics["pro_source_summary"], perturbation="task")["zero_sources"]),
        "pro_swap_median_source_rate": float(matching_row(metrics["pro_source_summary"], perturbation="swap(position)")["median_rate"]),
        "pro_task_median_source_rate": float(matching_row(metrics["pro_source_summary"], perturbation="task")["median_rate"]),
    }


def make_analysis() -> dict[str, Any]:
    audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    datasets = {name: read_jsonl(path) for name, path in SOURCES.items()}
    assert_terminal_integrity("official", datasets["official"], 2000)
    assert_terminal_integrity("base_libero_assets", datasets["base_libero_assets"], 2000)
    assert_terminal_integrity("plus", datasets["plus"], 10030)
    assert_terminal_integrity("pro", datasets["pro"], 8000)

    overview = []
    overview_specs = [
        ("official", "LIBERO official", "pi05_libero", 2000),
        ("base_libero_assets", "LIBERO Base+stats", "pi05_base + LIBERO assets", 2000),
        ("plus", "LIBERO-Plus", "pi05_libero", 10030),
        ("pro", "LIBERO-Pro 可用矩阵", "pi05_libero", 10000),
    ]
    for key, label, protocol, protocol_total in overview_specs:
        row = {"key": key, "label": label, "protocol": protocol, "protocol_total": protocol_total}
        row.update(aggregate(datasets[key]))
        overview.append(row)

    official_suite = ordered(grouped(datasets["official"], ["suite"]), "suite", SUITES)
    plus_suite = ordered(grouped(datasets["plus"], ["suite"]), "suite", SUITES)
    plus_category = ordered(grouped(datasets["plus"], ["category"]), "category", PLUS_CATEGORIES)
    plus_difficulty = ordered(grouped(datasets["plus"], ["difficulty"]), "difficulty", PLUS_DIFFICULTIES)
    plus_category_suite = grouped(datasets["plus"], ["category", "suite"])
    plus_difficulty_category = grouped(datasets["plus"], ["difficulty", "category"])
    pro_perturbation = ordered(grouped(datasets["pro"], ["perturbation"]), "perturbation", PRO_PERTURBATIONS)
    pro_suite = ordered(grouped(datasets["pro"], ["suite"]), "suite", SUITES)
    pro_perturbation_suite = grouped(datasets["pro"], ["perturbation", "suite"])

    official_task = grouped(datasets["official"], ["suite", "task_id", "task_description"])
    official_task.sort(key=lambda row: (row["rate"], row["suite"], int(row["task_id"])))
    pro_source = grouped(datasets["pro"], ["perturbation", "suite", "source_id", "task_description"])
    pro_source.sort(key=lambda row: (row["rate"], row["perturbation"], row["suite"], row["source_id"]))

    plus_failures = sum(not row["success"] for row in datasets["plus"])
    plus_category_failure = []
    for row in plus_category:
        failures = row["total"] - row["successes"]
        plus_category_failure.append(
            {
                "category": row["category"],
                "label": LABELS[row["category"]],
                "total": row["total"],
                "failures": failures,
                "episode_share": row["total"] / len(datasets["plus"]),
                "failure_share": failures / plus_failures,
            }
        )
    plus_category_failure.sort(key=lambda row: row["failure_share"], reverse=True)

    plus_difficulty_failure = []
    for row in plus_difficulty:
        failures = row["total"] - row["successes"]
        plus_difficulty_failure.append(
            {
                "difficulty": row["difficulty"],
                "label": LABELS.get(row["difficulty"], f'难度 {row["difficulty"]}'),
                "total": row["total"],
                "failures": failures,
                "episode_share": row["total"] / len(datasets["plus"]),
                "failure_share": failures / plus_failures,
            }
        )

    pro_failures = sum(not row["success"] for row in datasets["pro"])
    pro_failure_concentration = []
    for row in pro_perturbation:
        failures = row["total"] - row["successes"]
        pro_failure_concentration.append(
            {
                "perturbation": row["perturbation"],
                "label": LABELS[row["perturbation"]],
                "total": row["total"],
                "failures": failures,
                "episode_share": row["total"] / len(datasets["pro"]),
                "failure_share": failures / pro_failures,
            }
        )
    pro_failure_concentration.sort(key=lambda row: row["failure_share"], reverse=True)

    source_summary = []
    by_perturbation: dict[str, list[float]] = defaultdict(list)
    for row in pro_source:
        by_perturbation[row["perturbation"]].append(float(row["rate"]))
    for perturbation in PRO_PERTURBATIONS:
        rates = sorted(by_perturbation[perturbation])
        source_summary.append(
            {
                "perturbation": perturbation,
                "sources": len(rates),
                "mean_rate": statistics.mean(rates),
                "median_rate": statistics.median(rates),
                "q1_rate": percentile(rates, 0.25),
                "q3_rate": percentile(rates, 0.75),
                "min_rate": min(rates),
                "max_rate": max(rates),
                "zero_sources": sum(rate == 0 for rate in rates),
                "perfect_sources": sum(rate == 1 for rate in rates),
                "low_sources_le_20pct": sum(rate <= 0.2 for rate in rates),
                "high_sources_ge_80pct": sum(rate >= 0.8 for rate in rates),
            }
        )

    official_failures = overview[0]["failures"]
    official_top3_failures = sum(row["total"] - row["successes"] for row in official_task[:3])
    plus_top2_category_failures = sum(row["failures"] for row in plus_category_failure[:2])
    difficulty_45 = [row for row in plus_difficulty_failure if row["difficulty"] in {"4", "5"}]
    pro_position_task = [
        row for row in pro_failure_concentration if row["perturbation"] in {"swap(position)", "task"}
    ]
    plus_sensor_rows = [row for row in datasets["plus"] if row["category"] == "Sensor Noise"]
    plus_sensor_durations = [float(row["duration_seconds"]) for row in plus_sensor_rows]
    plus_duration_sum = sum(float(row["duration_seconds"]) for row in datasets["plus"])

    findings = {
        "official_vs_base_gap_pp": 100 * (overview[0]["rate"] - overview[1]["rate"]),
        "official_worst_task": official_task[0],
        "official_top3_failure_share": official_top3_failures / official_failures,
        "plus_best_category": max(plus_category, key=lambda row: row["rate"]),
        "plus_worst_category": min(plus_category, key=lambda row: row["rate"]),
        "plus_top2_category_failure_share": plus_top2_category_failures / plus_failures,
        "plus_top2_category_episode_share": sum(row["total"] for row in plus_category_failure[:2])
        / len(datasets["plus"]),
        "plus_difficulty_1_to_5_gap_pp": 100
        * (next(row for row in plus_difficulty if row["difficulty"] == "1")["rate"]
           - next(row for row in plus_difficulty if row["difficulty"] == "5")["rate"]),
        "plus_difficulty_45_failure_share": sum(row["failures"] for row in difficulty_45) / plus_failures,
        "plus_difficulty_45_episode_share": sum(row["total"] for row in difficulty_45) / len(datasets["plus"]),
        "plus_macro_suite_rate": statistics.mean(float(row["rate"]) for row in plus_suite),
        "pro_position_task_failure_share": sum(row["failures"] for row in pro_position_task) / pro_failures,
        "pro_position_task_episode_share": sum(row["total"] for row in pro_position_task) / len(datasets["pro"]),
        "plus_sensor_episode_share": len(plus_sensor_rows) / len(datasets["plus"]),
        "plus_sensor_duration_share": sum(plus_sensor_durations) / plus_duration_sum,
        "plus_sensor_mean_duration_seconds": statistics.mean(plus_sensor_durations),
        "plus_sensor_median_duration_seconds": statistics.median(plus_sensor_durations),
        "plus_sensor_p90_duration_seconds": percentile(plus_sensor_durations, 0.90),
        "plus_sensor_p99_duration_seconds": percentile(plus_sensor_durations, 0.99),
        "plus_sensor_over_60_seconds": sum(value > 60 for value in plus_sensor_durations),
        "plus_sensor_over_120_seconds": sum(value > 120 for value in plus_sensor_durations),
        "all_policy_failures_exhaust_horizon": all(
            row["action_steps"] == row["max_steps"]
            for dataset in datasets.values()
            for row in dataset
            if not row["success"]
        ),
    }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_final_audit": str(FINAL_AUDIT.relative_to(ARCHIVE_ROOT)),
        "source_final_audit_status": audit["status"],
        "overview": overview,
        "official_suite": official_suite,
        "official_task": official_task,
        "plus_suite": plus_suite,
        "plus_category": plus_category,
        "plus_difficulty": plus_difficulty,
        "plus_category_suite": plus_category_suite,
        "plus_difficulty_category": plus_difficulty_category,
        "plus_category_failure_concentration": plus_category_failure,
        "plus_difficulty_failure_concentration": plus_difficulty_failure,
        "pro_suite": pro_suite,
        "pro_perturbation": pro_perturbation,
        "pro_perturbation_suite": pro_perturbation_suite,
        "pro_source": pro_source,
        "pro_source_summary": source_summary,
        "pro_failure_concentration": pro_failure_concentration,
        "findings": findings,
        "limitations": [
            "Wilson 95% intervals describe episode-level binomial uncertainty and do not remove within-task/source correlation.",
            "LIBERO-Plus contains one trial per generated variant; its interval is over the evaluated variant population.",
            "LIBERO-Pro contains 50 trials per source; source-level distributions are reported alongside episode-level rates.",
            "Cross-benchmark overall rates have different task compositions and denominators and are descriptive, not paired causal estimates.",
            "LIBERO-Plus was suite-sharded across independent policy-server RNG streams seeded from JAX key 0.",
            "LIBERO-Pro has 2,000 unavailable env episodes recorded as N/A, not failures.",
        ],
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], aligns: Sequence[str] | None = None) -> str:
    align_row = aligns or ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align_row) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_markdown(metrics: dict[str, Any]) -> str:
    overview = metrics["overview"]
    official, base, plus, pro = overview
    findings = metrics["findings"]
    narrative = narrative_values(metrics)
    official_suite_rows = [
        [LABELS[row["suite"]], f'{row["successes"]}/{row["total"]}', pct(row["rate"]), f'{pct(row["ci95_low"])}–{pct(row["ci95_high"])}']
        for row in metrics["official_suite"]
    ]
    plus_category_rows = [
        [LABELS[row["category"]], f'{row["successes"]}/{row["total"]}', pct(row["rate"]), row["total"] - row["successes"]]
        for row in metrics["plus_category"]
    ]
    pro_rows = [
        [LABELS[row["perturbation"]], f'{row["successes"]}/{row["total"]}', pct(row["rate"]), row["total"] - row["successes"]]
        for row in metrics["pro_perturbation"]
    ]
    overview_rows = [
        [
            row["label"],
            row["protocol"],
            f'{row["total"]}/{row["protocol_total"]}',
            f'{row["successes"]}/{row["total"]}',
            pct(row["rate"], 2),
            f'{pct(row["ci95_low"], 2)}–{pct(row["ci95_high"], 2)}',
            row["errors"],
        ]
        for row in overview
    ]
    overview_rows.insert(2, ["LIBERO Base-native", "pi05_base native assets", "0/2000 (N/A)", "N/A", "N/A", "N/A", 0])
    return f"""# OpenPI π0.5 × LIBERO 数据分析报告

> 基于 22,030 个已完成、可审计的 episode 终态重新统计。最终审计状态：**passed**；基础设施错误：**0**。

## 执行摘要

- **官方 LIBERO 微调权重达到 {pct(official['rate'], 2)}**（{official['successes']}/{official['total']}），四个 suite 宏平均为 {pct(narrative['official_macro_rate'], 2)}，高于 93.85% 验收线。
- **Base 参数即使搭配官方 LIBERO norm stats/assets，仍为 0/2000**。与官方协议的差距是 {findings['official_vs_base_gap_pp']:.1f} 个百分点，表明 norm stats 无法替代任务微调权重。
- **LIBERO-Plus 为 {pct(plus['rate'], 2)}**（{plus['successes']}/{plus['total']}）。相机视角和机器人初态只占 {findings['plus_top2_category_episode_share']*100:.1f}% 评测量，却贡献 {findings['plus_top2_category_failure_share']*100:.1f}% 失败。
- **Plus 难度 1 到 5 从 {pct(narrative['plus_difficulty_1_rate'], 2)} 降至 {pct(narrative['plus_difficulty_5_rate'], 2)}**，净下降 {findings['plus_difficulty_1_to_5_gap_pp']:.1f} 个百分点；难度 4–5 占 {findings['plus_difficulty_45_episode_share']*100:.1f}% episodes，却占 {findings['plus_difficulty_45_failure_share']*100:.1f}% 失败。
- **LIBERO-Pro 可用矩阵为 {pct(pro['rate'], 2)}**（{pro['successes']}/{pro['total']}）。语义改写为 {pct(narrative['pro_semantic_rate'], 2)}，位置交换和任务替换分别仅 {pct(narrative['pro_swap_rate'], 2)} / {pct(narrative['pro_task_rate'], 2)}，两者贡献 {findings['pro_position_task_failure_share']*100:.1f}% 失败。
- **Sensor Noise 是运行时间长尾的核心来源**：只占 {findings['plus_sensor_episode_share']*100:.1f}% Plus episodes，却占 {findings['plus_sensor_duration_share']*100:.1f}% episode-duration 总和；中位数 {findings['plus_sensor_median_duration_seconds']:.1f}s，P90 {findings['plus_sensor_p90_duration_seconds']:.1f}s。
- 全部普通策略失败都精确走满 episode 步数上限，无异常中断或连接/EGL 错误被混入失败率。

![总体成功率](figures/overview.svg)

## 1. 总体结果

{markdown_table(['Benchmark', '协议', '已评/协议', '成功/已评', '成功率', 'Wilson 95% CI', 'Infra'], overview_rows, ['---','---','---:','---:','---:','---:','---:'])}

上表的置信区间是 episode 级 Wilson 区间。它表达当前 episode 样本的二项不确定性，不消除同一 task/source 内的相关性。因 benchmark 的任务构成不同，不应把 Plus 和 Pro 的总成功率差直接解释为单一扰动的因果效应。

## 2. 原始 LIBERO：微调权重的作用

{markdown_table(['Suite', '成功/总数', '成功率', 'Wilson 95% CI'], official_suite_rows, ['---','---:','---:','---:'])}

![Official suite](figures/official_suite.svg)

官方权重的 {official['failures']} 次失败并非均匀分布：最难的 `put both moka pots on the stove` 任务为 {narrative['official_worst_task_successes']}/{narrative['official_worst_task_total']}，单任务贡献 {narrative['official_worst_task_failures']} 次失败；失败最多的前三个任务共占 {findings['official_top3_failure_share']*100:.1f}% 全部失败。这说明 {pct(official['rate'], 2)} 的余下风险高度集中在少数长时序/多步任务，而不是全面退化。

在同样的 4 suites × 10 tasks × 50 trials 矩阵上，`pi05_base + LIBERO assets` 为 0/2000，且 2000 次均走满步数上限。这是一个强对照：预处理统计匹配了输入尺度，但没有赋予 Base 模型 LIBERO 任务能力。Base-native 则因缺少 `physical-intelligence/libero/norm_stats.json` 记为 N/A，不是 0%。

## 3. LIBERO-Plus：主要短板是视角、初态和高难度

Plus episode 微平均为 {pct(plus['rate'], 2)}，四个 suite 等权宏平均为 {pct(findings['plus_macro_suite_rate'], 2)}。两者接近，但 category/difficulty 的分母不均匀，不应把不同维度的平均数混为同一指标。

![Plus category](figures/plus_category.svg)

{markdown_table(['扰动类别', '成功/总数', '成功率', '失败数'], plus_category_rows, ['---','---:','---:','---:'])}

![Plus category-suite heatmap](figures/plus_category_suite_heatmap.svg)

类别平均会掩盖明显的 suite 交互：背景纹理在 Object 上达 {pct(narrative['plus_background_object_rate'], 2)}，而相机视角在 LIBERO-10 上只有 {pct(narrative['plus_camera_libero10_rate'], 2)}；机器人初态在四个 suite 都处于 {pct(narrative['plus_robot_min_rate'], 2)}–{pct(narrative['plus_robot_max_rate'], 2)} 低位，表明它是更普遍的鲁棒性短板。

![Plus difficulty](figures/plus_difficulty.svg)

![Plus failure concentration](figures/plus_failure_concentration.svg)

难度曲线单调下降，且失败不成比例地集中在难度 4–5。从研发优先级看，先聚焦「相机视角 × 难度 5」和「机器人初态 × 难度 4–5」，比对全部类别平均用力更有可能快速降低失败数。

## 4. LIBERO-Pro：语义稳健，几何与任务转移是主要瓶颈

{markdown_table(['扰动', '成功/总数', '成功率', '失败数'], pro_rows, ['---','---:','---:','---:'])}

![Pro perturbation](figures/pro_perturbation.svg)

![Pro heatmap](figures/pro_perturbation_suite_heatmap.svg)

语义改写在四个 suite 上都保持 {pct(narrative['pro_semantic_min_rate'], 1)}–{pct(narrative['pro_semantic_max_rate'], 1)}，物体替换也仍有 {pct(narrative['pro_object_min_rate'], 1)}–{pct(narrative['pro_object_max_rate'], 1)}。相比之下，位置交换在 LIBERO-10 仅 {pct(narrative['pro_swap_libero10_rate'], 1)}，任务替换在 Object 仅 {pct(narrative['pro_task_object_rate'], 1)}。这表明模型对语言表达形式较稳健，但对「物体—位置—操作目标」关系的重组明显更脆弱。

![Pro source distribution](figures/pro_source_distribution.svg)

Pro 的每个 perturbation 含 40 个 sources，每 source 50 trials。位置交换有 {narrative['pro_swap_zero_sources']}/40 个 source 完全失败，任务替换为 {narrative['pro_task_zero_sources']}/40；两者的 source 成功率中位数分别只有 {pct(narrative['pro_swap_median_source_rate'], 0)} 和 {pct(narrative['pro_task_median_source_rate'], 0)}。因此低均值不是「所有任务都小幅变差」，而是大量 source 接近彻底失效、少数 source 仍高成功的两极分化。

Pro 快照中缺少 4 个 `env` cells，各 500 episodes，共 2000。本报告的 58.79% 只以 8000 个可运行 episodes 为分母；缺失部分明确是 N/A，不是策略失败。

## 5. 运行成本：Sensor Noise 造成明显时间长尾

![Plus duration range](figures/plus_duration_range.svg)

Sensor Noise 的 1601 episodes 只占 Plus 矩阵 {findings['plus_sensor_episode_share']*100:.1f}%，但占 episode-duration 总和 {findings['plus_sensor_duration_share']*100:.1f}%。其平均时长 {findings['plus_sensor_mean_duration_seconds']:.1f}s、中位数 {findings['plus_sensor_median_duration_seconds']:.1f}s、P90 {findings['plus_sensor_p90_duration_seconds']:.1f}s、P99 {findings['plus_sensor_p99_duration_seconds']:.1f}s；有 {findings['plus_sensor_over_60_seconds']} 条超过 60s，{findings['plus_sensor_over_120_seconds']} 条超过 120s。这解释了 Plus 运行后半段的时间长尾。

注意：这里的 duration 是各 episode 记录时长，并行 GPU/server 下可以重叠，因此其总和不是真实墙钟时间；它适合用于定位类别级运行成本。

## 6. 结论与优先级

1. **微调权重是必需项。** Base+stats 对照证明数据归一化不能替代 LIBERO 任务微调。
2. **首优先修复几何泛化。** Plus 的视角/初态、Pro 的位置/任务重组是失败最集中的方向。
3. **训练和回归集应按「扰动 × suite × 难度」分层。** 仅看总成功率会漏掉如 Camera×LIBERO-10={pct(narrative['plus_camera_libero10_rate'], 2)} 这类高价值短板。
4. **保留 source/task 级指标。** Pro 的两极分化说明后续不应只汇报 episode micro-average，还应跟踪零成功 source 数和 source 中位数。

## 7. 方法和可审计文件

- 分析对象：`official-full` 2000 + `base-libero-assets-full` 2000 + Plus 10030 + Pro 8000 = **22,030 episodes**。
- 所有记录均为唯一 terminal episode/attempt，`error_category=null`，录像集与记录精确匹配。
- 派生数据：[`data/`](data/)；机器可读总指标：[`metrics.json`](data/metrics.json)。
- 可复现生成脚本：[`generate_report.py`](generate_report.py)。脚本只使用 Python 标准库，并对已知矩阵大小、唯一 ID、终态、错误分类和录像状态设置硬校验。
- 完整最终审计：[`../plus-pro/final-audit/report.json`](../plus-pro/final-audit/report.json)。

### 解读边界

- Plus 每个生成变体只有 1 次 trial；其区间表达变体 episode 集合的不确定性，不是单一变体的重复成功率。
- Pro 每 source 有 50 trials，同 source 内并非完全独立；因此同时报告 source 分布。
- Plus 为了完成大矩阵，按 suite 使用独立 policy server 分片；环境 seed=7，policy JAX RNG key=0 按 server 独立作用。
- 本报告比较是当前固定 checkpoint 和协议下的描述性评测，不是训练方法间的因果实验。
"""


def html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build_html(metrics: dict[str, Any]) -> str:
    overview = metrics["overview"]
    official, base, plus, pro = overview
    findings = metrics["findings"]
    narrative = narrative_values(metrics)
    overview_rows = [
        [row["label"], row["protocol"], f'{row["total"]}/{row["protocol_total"]}', f'{row["successes"]}/{row["total"]}', pct(row["rate"], 2), f'{pct(row["ci95_low"], 2)} – {pct(row["ci95_high"], 2)}', row["errors"]]
        for row in overview
    ]
    overview_rows.insert(2, ["LIBERO Base-native", "pi05_base native assets", "0/2000 (N/A)", "N/A", "N/A", "N/A", "0"])
    plus_category_rows = [
        [LABELS[row["category"]], f'{row["successes"]}/{row["total"]}', pct(row["rate"]), row["total"] - row["successes"]]
        for row in metrics["plus_category"]
    ]
    pro_rows = [
        [LABELS[row["perturbation"]], f'{row["successes"]}/{row["total"]}', pct(row["rate"]), row["total"] - row["successes"]]
        for row in metrics["pro_perturbation"]
    ]
    official_rows = [
        [LABELS[row["suite"]], f'{row["successes"]}/{row["total"]}', pct(row["rate"]), f'{pct(row["ci95_low"])} – {pct(row["ci95_high"])}']
        for row in metrics["official_suite"]
    ]
    generated = metrics["generated_at"].replace("+00:00", "Z")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenPI π0.5 × LIBERO 数据分析报告</title>
<style>
:root {{ --ink:#172033; --muted:#617086; --line:#e4e9f0; --card:#fff; --green:#159a6b; --red:#d9534f; --blue:#3484d4; --amber:#e39b22; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#f4f6f9; font-family:system-ui,'Noto Sans CJK SC','Microsoft YaHei',sans-serif; line-height:1.7; }}
.hero {{ background:linear-gradient(130deg,#14243a,#214e69 60%,#167c65); color:white; padding:64px max(28px,calc((100vw - 1180px)/2)); }}
.hero h1 {{ margin:0 0 10px; font-size:clamp(30px,5vw,54px); line-height:1.15; }}
.hero p {{ max-width:850px; color:#dce9f4; font-size:18px; margin:0; }}
.badge {{ display:inline-block; margin-bottom:20px; padding:5px 12px; border:1px solid #7de0ba; border-radius:999px; color:#9ff0cf; font-weight:700; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 22px 70px; }}
h2 {{ margin:48px 0 18px; font-size:30px; }}
h3 {{ margin:30px 0 12px; font-size:21px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px; box-shadow:0 5px 18px rgba(18,38,63,.055); }}
.metric {{ font-size:34px; font-weight:800; line-height:1.15; }}
.metric.green {{ color:var(--green); }} .metric.red {{ color:var(--red); }} .metric.blue {{ color:var(--blue); }} .metric.amber {{ color:var(--amber); }}
.eyebrow {{ color:var(--muted); font-size:14px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }}
.sub {{ color:var(--muted); font-size:14px; margin-top:6px; }}
.insight {{ border-left:5px solid var(--blue); background:#edf5fd; padding:16px 20px; border-radius:8px; margin:18px 0; }}
.warning {{ border-left-color:var(--amber); background:#fff7e6; }}
.figure {{ background:white; border:1px solid var(--line); border-radius:16px; padding:10px; margin:22px 0; overflow:auto; }}
.figure img {{ display:block; width:100%; min-width:760px; height:auto; }}
.caption {{ color:var(--muted); font-size:14px; padding:3px 14px 10px; }}
.table-wrap {{ overflow:auto; background:white; border:1px solid var(--line); border-radius:14px; margin:15px 0 24px; }}
table {{ border-collapse:collapse; width:100%; min-width:680px; }}
th,td {{ padding:12px 14px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
th {{ background:#f7f9fb; font-size:14px; }} tr:last-child td {{ border-bottom:0; }}
code {{ background:#edf0f4; border-radius:5px; padding:2px 5px; }}
.toc a {{ color:#236fa1; text-decoration:none; margin-right:18px; white-space:nowrap; }}
.foot {{ color:var(--muted); font-size:14px; }}
@media print {{ body {{ background:white; }} .hero {{ padding:30px; }} main {{ padding:20px; }} .card,.figure,.table-wrap {{ box-shadow:none; break-inside:avoid; }} }}
</style>
</head>
<body>
<header class="hero">
  <span class="badge">最终审计 PASSED · INFRA ERROR 0</span>
  <h1>OpenPI π0.5 × LIBERO<br>数据分析报告</h1>
  <p>从 22,030 个可审计 episode 中重新统计总体效果、扰动鲁棒性、难度趋势、source 两极分化和失败集中度。生成时间：{html.escape(generated)}</p>
</header>
<main>
  <nav class="card toc"><strong>导航：</strong><a href="#overview">总体</a><a href="#libero">LIBERO</a><a href="#plus">Plus</a><a href="#pro">Pro</a><a href="#runtime">运行成本</a><a href="#recommendations">结论</a><a href="#method">方法</a></nav>

  <section class="grid" style="margin-top:18px">
    <div class="card"><div class="eyebrow">LIBERO official</div><div class="metric green">{pct(official['rate'],2)}</div><div class="sub">{official['successes']} / {official['total']}，通过 93.85% 验收线</div></div>
    <div class="card"><div class="eyebrow">LIBERO-Plus</div><div class="metric blue">{pct(plus['rate'],2)}</div><div class="sub">{plus['successes']} / {plus['total']}，0 基础设施错误</div></div>
    <div class="card"><div class="eyebrow">LIBERO-Pro 可用矩阵</div><div class="metric amber">{pct(pro['rate'],2)}</div><div class="sub">{pro['successes']} / {pro['total']}；另 2000 env episodes=N/A</div></div>
    <div class="card"><div class="eyebrow">Base + LIBERO stats</div><div class="metric red">0 / 2000</div><div class="sub">均为走满步数上限的真实策略失败</div></div>
  </section>

  <section id="overview">
    <h2>1. 一页结论</h2>
    <div class="grid">
      <div class="card"><strong>微调不可替代</strong><p>Base 权重即使搭配 LIBERO norm stats/assets 仍为 0%，与微调权重差 {findings['official_vs_base_gap_pp']:.1f} 个百分点。</p></div>
      <div class="card"><strong>Plus 失败高度可定位</strong><p>相机视角+机器人初态占 {findings['plus_top2_category_failure_share']*100:.1f}% 失败；难度 4–5 占 {findings['plus_difficulty_45_failure_share']*100:.1f}% 失败。</p></div>
      <div class="card"><strong>Pro 的几何转移是主瓶颈</strong><p>位置交换+任务替换只占 50% 评测量，却贡献 {findings['pro_position_task_failure_share']*100:.1f}% 失败。</p></div>
      <div class="card"><strong>失败分类干净</strong><p>全部普通策略失败均走满 max_steps；评测连接、EGL 和 checkpoint 错误没有混入失败率。</p></div>
    </div>
    <div class="figure"><img src="figures/overview.svg" alt="总体成功率"><div class="caption">细线为 episode 级 Wilson 95% 置信区间。Pro 分母为 8000 个可运行 episodes。</div></div>
    {html_table(['Benchmark','协议','已评/协议','成功/已评','成功率','Wilson 95% CI','Infra'], overview_rows)}
  </section>

  <section id="libero">
    <h2>2. 原始 LIBERO：{pct(official['rate'], 2)} 通过验收，余下失败集中在少数任务</h2>
    <div class="figure"><img src="figures/official_suite.svg" alt="Official LIBERO suite 成功率"></div>
    {html_table(['Suite','成功/总数','成功率','Wilson 95% CI'], official_rows)}
    <div class="insight"><strong>任务集中性：</strong> `put both moka pots on the stove` 为 {narrative['official_worst_task_successes']}/{narrative['official_worst_task_total']}，单独贡献 {narrative['official_worst_task_failures']}/{official['failures']} 次官方协议失败；失败最多的前三个任务合计占 {findings['official_top3_failure_share']*100:.1f}%。总体 {pct(official['rate'], 2)} 不代表所有任务同样稳定。</div>
    <p>在同样的 4 suites × 10 tasks × 50 trials 矩阵上，<code>pi05_base + LIBERO assets</code> 为 0/2000。由于所有 Base 失败都走满步数上限，这是能力对照，而不是服务崩溃。Base-native 缺少 LIBERO norm stats，所以严格记为 N/A。</p>
  </section>

  <section id="plus">
    <h2>3. LIBERO-Plus：视角、初态和高难度是主要失败源</h2>
    <p>episode 微平均为 {pct(plus['rate'], 2)}，四个 suite 等权宏平均为 {pct(findings['plus_macro_suite_rate'], 2)}。两者很接近，但 category/difficulty 的分母不均匀，不应把不同维度的平均数混为同一指标。</p>
    <div class="figure"><img src="figures/plus_category.svg" alt="Plus 分类成功率"></div>
    {html_table(['扰动类别','成功/总数','成功率','失败数'], plus_category_rows)}
    <div class="figure"><img src="figures/plus_category_suite_heatmap.svg" alt="Plus 类别与 suite 热力图"><div class="caption">同一类别在 suite 间差异很大，因此不能只看类别平均。</div></div>
    <div class="insight"><strong>交互效应：</strong>背景纹理×Object={pct(narrative['plus_background_object_rate'], 2)}，相机视角×LIBERO-10={pct(narrative['plus_camera_libero10_rate'], 2)}。机器人初态在四个 suite 都仅有 {pct(narrative['plus_robot_min_rate'], 2)}–{pct(narrative['plus_robot_max_rate'], 2)}，是更普遍的短板。</div>
    <div class="figure"><img src="figures/plus_difficulty.svg" alt="Plus 难度趋势"></div>
    <div class="figure"><img src="figures/plus_failure_concentration.svg" alt="Plus 失败集中度"><div class="caption">红色失败占比明显高于灰色评测占比的类别，是优先修复候选。</div></div>
    <p>难度 1–5 从 {pct(narrative['plus_difficulty_1_rate'], 2)} 单调下降到 {pct(narrative['plus_difficulty_5_rate'], 2)}，净降 {findings['plus_difficulty_1_to_5_gap_pp']:.1f} 个百分点。难度 4–5 仅占 {findings['plus_difficulty_45_episode_share']*100:.1f}% episodes，但占 {findings['plus_difficulty_45_failure_share']*100:.1f}% 失败。建议优先建立「相机视角×难度5」和「机器人初态×难度4–5」定向回归集。</p>
  </section>

  <section id="pro">
    <h2>4. LIBERO-Pro：语义稳健，几何/任务重组显著脆弱</h2>
    {html_table(['扰动','成功/总数','成功率','失败数'], pro_rows)}
    <div class="figure"><img src="figures/pro_perturbation.svg" alt="Pro 扰动成功率"></div>
    <div class="figure"><img src="figures/pro_perturbation_suite_heatmap.svg" alt="Pro 扰动与 suite 热力图"></div>
    <div class="insight"><strong>差异不是小波动：</strong>语义改写在四个 suite 为 {pct(narrative['pro_semantic_min_rate'], 1)}–{pct(narrative['pro_semantic_max_rate'], 1)}；位置交换在 LIBERO-10 只有 {pct(narrative['pro_swap_libero10_rate'], 1)}，任务替换在 Object 只有 {pct(narrative['pro_task_object_rate'], 1)}。模型更能容忍语言表达变化，但对空间关系和操作目标重组敏感。</div>
    <div class="figure"><img src="figures/pro_source_distribution.svg" alt="Pro source 级分布"><div class="caption">每个 perturbation 有 40 sources，每 source 有 50 trials。</div></div>
    <p>位置交换有 {narrative['pro_swap_zero_sources']}/40 个 source 完全失败，任务替换为 {narrative['pro_task_zero_sources']}/40；source 成功率中位数仅 {pct(narrative['pro_swap_median_source_rate'], 0)} / {pct(narrative['pro_task_median_source_rate'], 0)}。这是「大量 source 接近彻底失效，少数 source 仍表现很好」的两极分化，所以后续应同时跟踪 micro-average、source 中位数和零成功 source 数。</p>
    <div class="insight warning"><strong>分母边界：</strong>Pro 快照缺少 4 个 <code>env</code> cells，合计 2000 episodes。报告的 58.79% 仅以 8000 个可运行 episodes 为分母；缺失部分是 N/A，不是失败。</div>
  </section>

  <section id="runtime">
    <h2>5. 运行成本：Sensor Noise 是 Plus 时间长尾的核心来源</h2>
    <div class="figure"><img src="figures/plus_duration_range.svg" alt="Plus 各类别 episode 时长"><div class="caption">蓝点为中位数，黄点为 P90；使用分位数避免均值被极端长尾误导。</div></div>
    <div class="insight warning"><strong>时间长尾：</strong>Sensor Noise 只占 {findings['plus_sensor_episode_share']*100:.1f}% Plus episodes，却占 {findings['plus_sensor_duration_share']*100:.1f}% episode-duration 总和。其中位数 {findings['plus_sensor_median_duration_seconds']:.1f}s，P90 {findings['plus_sensor_p90_duration_seconds']:.1f}s，P99 {findings['plus_sensor_p99_duration_seconds']:.1f}s；{findings['plus_sensor_over_60_seconds']} 条超过 60s，{findings['plus_sensor_over_120_seconds']} 条超过 120s。</div>
    <p>episode duration 在多 GPU/server 下可以并行重叠，所以时长总和不等于实际墙钟时间。该指标适合用于定位类别级运行成本，不应被解释为模型成功率或单 GPU 吞吐率。</p>
  </section>

  <section id="recommendations">
    <h2>6. 建议的研发优先级</h2>
    <div class="grid">
      <div class="card"><div class="eyebrow">P0</div><h3>增强几何关系泛化</h3><p>针对位置交换、机器人初态和相机视角扩充训练/回归数据；这些是失败最集中的方向。</p></div>
      <div class="card"><div class="eyebrow">P1</div><h3>单独治理高难度长时序任务</h3><p>Plus 难度5为 {pct(narrative['plus_difficulty_5_rate'], 2)}；原始 LIBERO 最难 moka-pot 任务为 {pct(narrative['official_worst_task_rate'], 0)}。对长序列、多物体和状态恢复做专项回归。</p></div>
      <div class="card"><div class="eyebrow">P1</div><h3>将 source 级指标纳入验收</h3><p>除总成功率外，固定报告 source 中位数、零成功 source 数和扰动×suite 热力图，防止均值掩盖两极分化。</p></div>
    </div>
  </section>

  <section id="method">
    <h2>7. 方法、复现与边界</h2>
    <ul>
      <li>数据集：official 2000 + Base+assets 2000 + Plus 10030 + Pro 8000 = <strong>22,030 terminal episodes</strong>。</li>
      <li>派生统计：<a href="data/metrics.json">metrics.json</a> 和 <a href="data/">CSV 表格</a>；复现脚本：<a href="generate_report.py">generate_report.py</a>。</li>
      <li>脚本仅使用 Python 标准库，并硬校验矩阵大小、唯一 ID、terminal 状态、错误分类和录像写入状态。</li>
      <li>Wilson 95% CI 是 episode 级描述，不消除同 task/source 内相关性。Plus 每变体 1 trial；Pro 每 source 50 trials，因此另报 source 分布。</li>
      <li>Plus 按 suite 使用独立 policy server 分片；环境 seed=7，policy JAX RNG key=0 按 server 独立作用。</li>
      <li>这是固定 checkpoint/协议下的描述性评测，不是训练方法的因果实验。</li>
    </ul>
    <p class="foot">完整最终审计：<a href="../plus-pro/final-audit/report.json">report.json</a> · Markdown 版：<a href="report_zh.md">report_zh.md</a></p>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = make_analysis()
    narrative = narrative_values(metrics)
    (DATA_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table_names = [
        "overview",
        "official_suite",
        "official_task",
        "plus_suite",
        "plus_category",
        "plus_difficulty",
        "plus_category_suite",
        "plus_difficulty_category",
        "plus_category_failure_concentration",
        "plus_difficulty_failure_concentration",
        "pro_suite",
        "pro_perturbation",
        "pro_perturbation_suite",
        "pro_source",
        "pro_source_summary",
        "pro_failure_concentration",
    ]
    for name in table_names:
        write_csv(DATA_DIR / f"{name}.csv", metrics[name])

    overview_chart_rows = [dict(row) for row in metrics["overview"]]
    bar_chart(
        FIGURE_DIR / "overview.svg",
        "总体成功率（不同 benchmark 的分母/构成不同）",
        overview_chart_rows,
        note="细线为 Wilson 95% CI；Pro 只以 8000 个可运行 episodes 为分母，另有 2000 env episodes=N/A。",
    )
    official_chart_rows = [dict(row, label=LABELS[row["suite"]]) for row in metrics["official_suite"]]
    bar_chart(
        FIGURE_DIR / "official_suite.svg",
        "官方 pi05_libero：各 suite 成功率",
        official_chart_rows,
        threshold=0.9385,
    )
    plus_category_chart = [
        dict(row, label=LABELS[row["category"]])
        for row in sorted(metrics["plus_category"], key=lambda item: item["rate"])
    ]
    bar_chart(
        FIGURE_DIR / "plus_category.svg",
        "LIBERO-Plus：扰动类别成功率",
        plus_category_chart,
        note="相机视角与机器人初态是成功率最低的两类扰动。",
    )
    difficulty_chart = [
        dict(row, label=LABELS.get(row["difficulty"], f'难度 {row["difficulty"]}'))
        for row in metrics["plus_difficulty"]
        if row["difficulty"] != "None"
    ]
    line_chart(
        FIGURE_DIR / "plus_difficulty.svg",
        "LIBERO-Plus：难度升高时成功率单调下降",
        difficulty_chart,
        note=f'未标注难度的 {narrative["plus_unlabelled_difficulty_total"]} episodes（成功率 {pct(narrative["plus_unlabelled_difficulty_rate"], 2)}，占比 {narrative["plus_unlabelled_difficulty_total"]/metrics["overview"][2]["total"]*100:.2f}%）不纳入 1–5 趋势线。',
    )
    heatmap_chart(
        FIGURE_DIR / "plus_category_suite_heatmap.svg",
        "LIBERO-Plus：扰动类别 × Suite 成功率",
        PLUS_CATEGORIES,
        SUITES,
        cell_map(metrics["plus_category_suite"], "category", "suite"),
        note="每格显示成功率和成功/总数；红色越深表示风险越高。",
    )
    exposure_failure_chart(
        FIGURE_DIR / "plus_failure_concentration.svg",
        "LIBERO-Plus：各扰动的评测占比 vs 失败占比",
        metrics["plus_category_failure_concentration"],
    )
    pro_chart = [dict(row, label=LABELS[row["perturbation"]]) for row in metrics["pro_perturbation"]]
    bar_chart(
        FIGURE_DIR / "pro_perturbation.svg",
        "LIBERO-Pro：扰动类型成功率",
        pro_chart,
        note=f'语义改写保持 {pct(narrative["pro_semantic_rate"], 1)}；位置交换和任务替换下降到 {pct(narrative["pro_swap_rate"], 1)} / {pct(narrative["pro_task_rate"], 1)}。',
    )
    heatmap_chart(
        FIGURE_DIR / "pro_perturbation_suite_heatmap.svg",
        "LIBERO-Pro：扰动类型 × Suite 成功率",
        PRO_PERTURBATIONS + ["env"],
        SUITES,
        cell_map(metrics["pro_perturbation_suite"], "perturbation", "suite"),
        note="每格固定 500 episodes；位置/任务重组在多个 suite 出现接近彻底失效。",
    )
    source_distribution_chart(FIGURE_DIR / "pro_source_distribution.svg", metrics["pro_source"])
    duration_rows = [
        dict(row, label=LABELS[row["category"]])
        for row in sorted(metrics["plus_category"], key=lambda item: item["p90_duration_seconds"])
    ]
    duration_range_chart(FIGURE_DIR / "plus_duration_range.svg", duration_rows)

    (REPORT_DIR / "report_zh.md").write_text(build_markdown(metrics) + "\n", encoding="utf-8")
    (REPORT_DIR / "report_zh.html").write_text(build_html(metrics), encoding="utf-8")
    print(f"generated {REPORT_DIR / 'report_zh.html'}")
    print(f"analyzed episodes={sum(row['total'] for row in metrics['overview'])}")
    print(f"final audit={metrics['source_final_audit_status']}")


if __name__ == "__main__":
    main()
