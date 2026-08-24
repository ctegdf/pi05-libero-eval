from __future__ import annotations

import subprocess
import unittest
import zipfile
from pathlib import Path

import generate_ppt_report as report


class ReportDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = report.build_metrics()

    def test_paired_matrix_and_overall_results(self) -> None:
        self.assertEqual(self.metrics["overall"]["official"]["successes"], 1942)
        self.assertEqual(self.metrics["overall"]["base"]["successes"], 0)
        self.assertEqual(self.metrics["overall"]["official"]["total"], 2000)
        self.assertEqual(self.metrics["overall"]["base"]["total"], 2000)
        self.assertEqual(
            self.metrics["paired_outcomes"],
            {
                "official_success_base_success": 0,
                "official_success_base_failure": 1942,
                "official_failure_base_success": 0,
                "official_failure_base_failure": 58,
            },
        )

    def test_suite_and_task_breakdowns(self) -> None:
        expected = {
            "libero_spatial": 493,
            "libero_object": 492,
            "libero_goal": 491,
            "libero_10": 466,
        }
        self.assertEqual(
            {row["suite"]: row["official_successes"] for row in self.metrics["suite"]},
            expected,
        )
        self.assertTrue(all(row["base_successes"] == 0 for row in self.metrics["suite"]))
        self.assertEqual(len(self.metrics["task"]), 40)
        self.assertEqual(self.metrics["derived"]["tasks_at_100pct"], 19)
        self.assertEqual(self.metrics["derived"]["tasks_at_least_98pct"], 29)
        self.assertEqual(self.metrics["derived"]["tasks_at_least_94pct"], 37)

    def test_failure_and_efficiency_metrics(self) -> None:
        official = self.metrics["overall"]["official"]
        base = self.metrics["overall"]["base"]
        self.assertEqual(official["failures"], 58)
        self.assertEqual(official["max_step_failures"], 58)
        self.assertEqual(base["failures"], 2000)
        self.assertEqual(base["max_step_failures"], 2000)
        self.assertEqual(official["infrastructure_errors"], 0)
        self.assertEqual(base["infrastructure_errors"], 0)
        self.assertEqual(self.metrics["derived"]["saved_action_steps"], 341336)
        self.assertAlmostEqual(self.metrics["derived"]["top3_failure_share"], 30 / 58)
        self.assertAlmostEqual(self.metrics["derived"]["top10_failure_share"], 46 / 58)
        self.assertGreater(self.metrics["derived"]["task_paired_bootstrap_ci95_low_pp"], 94)
        self.assertLess(self.metrics["derived"]["task_paired_bootstrap_ci95_high_pp"], 99)


class GeneratedArtifactTest(unittest.TestCase):
    def test_all_chart_formats_exist(self) -> None:
        for index, stem in enumerate(
            [
                "overall_success",
                "suite_success",
                "task_heatmap",
                "task_distribution",
                "failure_pareto",
                "action_budget",
                "duration_cost",
                "task_efficiency_scatter",
                "paired_outcomes",
            ],
            start=1,
        ):
            chart_index = 9 if stem == "paired_outcomes" else index
            for suffix in ("png", "svg"):
                path = report.FIGURE_DIR / f"{chart_index:02d}_{stem}.{suffix}"
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 1000, path)

    def test_pptx_is_valid_and_contains_thirteen_slides(self) -> None:
        path = report.HERE / "OpenPI_pi05_LIBERO_权重对比分析.pptx"
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            slides = [
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ]
            media = [name for name in archive.namelist() if name.startswith("ppt/media/")]
            self.assertEqual(len(slides), 13)
            self.assertEqual(len(media), 9)
            combined = b"".join(archive.read(name) for name in slides)
            self.assertIn("97.10%".encode(), combined)
            self.assertIn("1,942".encode(), combined)

    def test_pdf_and_preview_are_complete(self) -> None:
        pdf = report.HERE / "OpenPI_pi05_LIBERO_权重对比分析.pdf"
        info = subprocess.run(
            ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
        ).stdout
        self.assertIn("Pages:           13", info)
        previews = sorted(report.PREVIEW_DIR.glob("slide-*.png"))
        self.assertEqual(len(previews), 13)
        self.assertTrue(all(path.stat().st_size > 1000 for path in previews))
        self.assertGreater((report.PREVIEW_DIR / "contact_sheet.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
