import importlib.util
import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("generate_report", HERE / "generate_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


class AnalysisReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        REPORT.main()
        cls.metrics = json.loads((HERE / "data/metrics.json").read_text(encoding="utf-8"))

    def test_expected_overall_counts(self):
        actual = {
            row["key"]: (row["total"], row["successes"], row["failures"], row["errors"])
            for row in self.metrics["overview"]
        }
        self.assertEqual(actual["official"], (2000, 1942, 58, 0))
        self.assertEqual(actual["base_libero_assets"], (2000, 0, 2000, 0))
        self.assertEqual(actual["plus"], (10030, 8390, 1640, 0))
        self.assertEqual(actual["pro"], (8000, 4703, 3297, 0))

    def test_wilson_interval_known_value(self):
        low, high = REPORT.wilson(1942, 2000)
        self.assertAlmostEqual(low, 0.9626945430771895, places=12)
        self.assertAlmostEqual(high, 0.97749959838387, places=12)
        zero_low, zero_high = REPORT.wilson(0, 2000)
        self.assertAlmostEqual(zero_low, 0.0, places=15)
        self.assertAlmostEqual(zero_high, 0.001917, places=6)

    def test_plus_difficulty_is_monotonic(self):
        rows = [row for row in self.metrics["plus_difficulty"] if row["difficulty"] != "None"]
        rates = [row["rate"] for row in rows]
        self.assertEqual([row["difficulty"] for row in rows], ["1", "2", "3", "4", "5"])
        self.assertTrue(all(left > right for left, right in zip(rates, rates[1:])))
        self.assertAlmostEqual(rates[-1], 1281 / 2083)

    def test_pro_matrix_is_exact(self):
        rows = self.metrics["pro_perturbation_suite"]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["total"] == 500 for row in rows))
        self.assertEqual(sum(row["successes"] for row in rows), 4703)

    def test_policy_failures_exhaust_horizon(self):
        self.assertTrue(self.metrics["findings"]["all_policy_failures_exhaust_horizon"])

    def test_sensor_noise_duration_long_tail(self):
        findings = self.metrics["findings"]
        self.assertAlmostEqual(findings["plus_sensor_episode_share"], 1601 / 10030)
        self.assertGreater(findings["plus_sensor_duration_share"], 0.5)
        self.assertGreater(findings["plus_sensor_p90_duration_seconds"], 170)

    def test_narrative_numbers_are_derived_from_metrics(self):
        narrative = REPORT.narrative_values(self.metrics)
        markdown = (HERE / "report_zh.md").read_text(encoding="utf-8")
        html = (HERE / "report_zh.html").read_text(encoding="utf-8")
        interaction = (
            f'背景纹理×Object={REPORT.pct(narrative["plus_background_object_rate"], 2)}'
            f'，相机视角×LIBERO-10={REPORT.pct(narrative["plus_camera_libero10_rate"], 2)}'
        )
        self.assertIn(interaction, html)
        self.assertIn(f'{narrative["pro_swap_zero_sources"]}/40', markdown)
        difficulty_note = (
            f'{narrative["plus_unlabelled_difficulty_total"]} episodes'
            f'（成功率 {REPORT.pct(narrative["plus_unlabelled_difficulty_rate"], 2)}，占比 1.21%）'
        )
        self.assertIn(difficulty_note, (HERE / "figures/plus_difficulty.svg").read_text(encoding="utf-8"))

    def test_html_references_exist_and_svgs_parse(self):
        report = (HERE / "report_zh.html").read_text(encoding="utf-8")
        self.assertIn("最终审计 PASSED", report)
        self.assertIn("22,030", report)
        for svg_path in sorted((HERE / "figures").glob("*.svg")):
            ET.parse(svg_path)
            self.assertIn(svg_path.name, report)
        self.assertEqual(len(list((HERE / "figures").glob("*.svg"))), 10)


if __name__ == "__main__":
    unittest.main()
