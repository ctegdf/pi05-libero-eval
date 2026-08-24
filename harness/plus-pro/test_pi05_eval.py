"""Unit tests for the standard-library and orchestration evaluation contract."""

import json
import math
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import pi05_eval_client as client
import pi05_eval_support as support
import merge_plus_shards as merger
import audit_final_results as final_audit
import run_pi05_eval as runner


def plus_sources():
    result = []
    for suite in support.SUITES:
        for registry_index in range(support.PLUS_COUNTS[suite]):
            task_id = registry_index + 1
            result.append(
                support.TaskSource(
                    benchmark="plus",
                    suite=suite,
                    task_id=task_id,
                    source_id="%s-%05d" % (suite, task_id),
                    bddl_path=pathlib.Path("/%s/%05d.bddl" % (suite, task_id)),
                    init_path=pathlib.Path("/%s/%05d.pruned_init" % (suite, task_id)),
                    prompt="do task %d" % task_id,
                    category="category-%d" % (registry_index % 7),
                    difficulty="level-%d" % (registry_index % 3),
                )
            )
    return result


def pro_sources():
    result = []
    for perturbation in support.PRO_PERTURBATIONS:
        for suite in support.SUITES:
            for task_id in range(support.PRO_TASKS_PER_CELL):
                source_id = "%s/%s/%02d" % (perturbation, suite, task_id)
                result.append(
                    support.TaskSource(
                        benchmark="pro",
                        suite=suite,
                        task_id=task_id,
                        source_id=source_id,
                        bddl_path=pathlib.Path("/%s/%s/%02d.bddl" % (perturbation, suite, task_id)),
                        init_path=pathlib.Path("/%s/%s/%02d.pruned_init" % (perturbation, suite, task_id)),
                        prompt="do pro task %d" % task_id,
                        perturbation=perturbation,
                    )
                )
    return result


class MatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plus = plus_sources()
        cls.pro = pro_sources()

    def test_plus_full_is_exactly_10030_one_episode_per_task(self):
        matrix = support.expand_matrix("plus", "full", self.plus)
        self.assertEqual(len(matrix), 10030)
        self.assertEqual({spec.trial for spec in matrix}, {0})
        self.assertEqual(len({spec.episode_id for spec in matrix}), 10030)
        counts = {suite: sum(spec.suite == suite for spec in matrix) for suite in support.SUITES}
        self.assertEqual(counts, dict(support.PLUS_COUNTS))

    def test_plus_smoke_uses_first_task_in_each_suite_category(self):
        matrix = support.expand_matrix("plus", "smoke", self.plus)
        self.assertEqual(len(matrix), 4 * 7)
        self.assertEqual(len({(spec.suite, spec.category) for spec in matrix}), 28)
        self.assertTrue(all(spec.trial == 0 for spec in matrix))

    def test_suite_filter_partitions_plus_without_overlap(self):
        matrix = support.expand_matrix("plus", "full", self.plus)
        partitions = [
            support.filter_matrix_by_suites(matrix, ["libero_10", "libero_spatial"]),
            support.filter_matrix_by_suites(matrix, ["libero_goal"]),
            support.filter_matrix_by_suites(matrix, ["libero_object"]),
        ]
        ids = [{spec.episode_id for spec in partition} for partition in partitions]
        self.assertEqual([len(partition) for partition in partitions], [4921, 2591, 2518])
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set().union(*ids), {spec.episode_id for spec in matrix})

    def test_suite_filter_rejects_duplicates(self):
        matrix = support.expand_matrix("plus", "smoke", self.plus)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            support.filter_matrix_by_suites(matrix, ["libero_goal", "libero_goal"])

    def test_pro_full_is_exactly_10000(self):
        matrix = support.expand_matrix("pro", "full", self.pro)
        self.assertEqual(len(matrix), 5 * 4 * 10 * 50)
        self.assertEqual(len({spec.episode_id for spec in matrix}), 10000)
        self.assertEqual(set(spec.trial for spec in matrix), set(range(50)))

    def test_pro_smoke_is_one_task_per_cell(self):
        matrix = support.expand_matrix("pro", "smoke", self.pro)
        self.assertEqual(len(matrix), 5 * 4)
        self.assertEqual(len({(spec.perturbation, spec.suite) for spec in matrix}), 20)
        self.assertTrue(all(spec.task_id == 0 and spec.trial == 0 for spec in matrix))

    def test_pro_missing_env_is_machine_readable_na_and_other_cells_run(self):
        available = [source for source in self.pro if source.perturbation != "env"]
        compatibility = support.pro_compatibility(available)
        matrix = support.expand_matrix(
            "pro", "full", available, allow_incompatible_pro=True
        )
        self.assertEqual(compatibility["status"], "partial_incompatible")
        self.assertFalse(compatibility["protocol_applicable"])
        self.assertEqual(compatibility["available_cells"], 16)
        self.assertEqual(len(compatibility["unavailable_cells"]), 4)
        self.assertEqual(compatibility["unavailable_planned_episodes"], 2000)
        self.assertTrue(all(cell["applicability"] == "N/A" for cell in compatibility["unavailable_cells"]))
        self.assertEqual(len(matrix), 8000)

    def test_na_cells_are_written_without_zero_success_rate(self):
        available = [source for source in self.pro if source.perturbation != "env"]
        matrix = support.expand_matrix("pro", "smoke", available, allow_incompatible_pro=True)
        summary = support.aggregate([], matrix)
        summary["compatibility"] = support.pro_compatibility(available)
        with tempfile.TemporaryDirectory() as temporary:
            support.write_summaries(summary, pathlib.Path(temporary))
            csv_text = (pathlib.Path(temporary) / "summary.csv").read_text(encoding="utf-8")
            payload = json.loads((pathlib.Path(temporary) / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("compatibility_cell", csv_text)
        self.assertIn("N/A", csv_text)
        self.assertEqual(payload["compatibility"]["unavailable_planned_episodes"], 2000)
        self.assertNotIn("success_rate", payload["compatibility"]["unavailable_cells"][0])

    def test_max_steps_match_suite_protocol(self):
        matrix = support.expand_matrix("pro", "smoke", self.pro)
        self.assertTrue(all(spec.max_steps == support.MAX_STEPS[spec.suite] for spec in matrix))


class PromptTest(unittest.TestCase):
    def test_prompt_comes_from_language_instruction_not_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "misleading_filename.bddl"
            path.write_text(
                '(define (problem x)\n  (:language_instruction "put the red mug on the plate")\n)\n',
                encoding="utf-8",
            )
            prompt, field = support.parse_bddl_prompt(path)
        self.assertEqual(prompt, "put the red mug on the plate")
        self.assertEqual(field, "language_instruction")
        self.assertNotIn("misleading", prompt)

    def test_multiline_bare_prompt_and_legacy_bddl_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            modern = root / "modern.bddl"
            modern.write_text("(:language_instruction move the bowl\n onto the stove)\n", encoding="utf-8")
            legacy = root / "legacy.bddl"
            legacy.write_text("(:language open the drawer)\n", encoding="utf-8")
            self.assertEqual(support.parse_bddl_prompt(modern), ("move the bowl onto the stove", "language_instruction"))
            self.assertEqual(support.parse_bddl_prompt(legacy), ("open the drawer", "language"))


class ResumeAggregationIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = support.expand_matrix("pro", "smoke", pro_sources())

    def test_resume_skips_only_policy_outcomes(self):
        records = [
            {"episode_id": self.matrix[0].episode_id, "attempt": 0, "status": "success"},
            {
                "episode_id": self.matrix[1].episode_id,
                "attempt": 0,
                "status": "error",
                "error_category": "connection",
            },
            {"episode_id": self.matrix[2].episode_id, "attempt": 0, "status": "failure"},
        ]
        pending = support.select_pending(self.matrix, records, resume=True)
        pending_ids = {spec.episode_id for spec in pending}
        self.assertNotIn(self.matrix[0].episode_id, pending_ids)
        self.assertIn(self.matrix[1].episode_id, pending_ids)
        self.assertNotIn(self.matrix[2].episode_id, pending_ids)
        self.assertEqual(support.next_attempts(records)[self.matrix[1].episode_id], 1)

    def test_aggregation_excludes_infrastructure_errors_and_groups(self):
        specs = self.matrix[:4]
        records = [
            {"episode_id": specs[0].episode_id, "attempt": 0, "status": "success"},
            {"episode_id": specs[1].episode_id, "attempt": 0, "status": "failure"},
            {
                "episode_id": specs[2].episode_id,
                "attempt": 0,
                "status": "error",
                "error_category": "environment",
            },
        ]
        summary = support.aggregate(records, specs)
        self.assertEqual(summary["total"]["policy_denominator"], 2)
        self.assertEqual(summary["total"]["success_rate"], 0.5)
        self.assertEqual(summary["total"]["excluded_errors"], 1)
        self.assertFalse(summary["total"]["complete"])
        self.assertIn("suite", summary["groups"])
        self.assertIn("perturbation", summary["groups"])

    def test_later_error_does_not_erase_a_policy_outcome(self):
        spec = self.matrix[0]
        summary = support.aggregate(
            [
                {"episode_id": spec.episode_id, "attempt": 0, "status": "success"},
                {
                    "episode_id": spec.episode_id,
                    "attempt": 1,
                    "status": "error",
                    "error_category": "connection",
                },
            ],
            [spec],
        )
        self.assertTrue(summary["total"]["complete"])
        self.assertEqual(summary["total"]["successes"], 1)

    def test_integrity_checks_attempt_ids_and_videos(self):
        spec = self.matrix[0]
        with tempfile.TemporaryDirectory() as temporary:
            video = pathlib.Path(temporary) / "episode.mp4"
            video.write_bytes(b"video")
            record = {
                "episode_id": spec.episode_id,
                "attempt_id": spec.episode_id + ":attempt-0000",
                "attempt": 0,
                "status": "failure",
                "video": str(video),
                "video_status": "written",
            }
            passed = support.verify_integrity([record], [spec])
            video.write_bytes(b"")
            empty = support.verify_integrity([record], [spec])
            video.unlink()
            failed = support.verify_integrity([record, record], [spec])
        self.assertTrue(passed["passed"])
        self.assertFalse(empty["passed"])
        self.assertEqual(empty["missing_videos"], 1)
        self.assertFalse(failed["passed"])
        self.assertTrue(any("attempt ids" in issue for issue in failed["issues"]))
        self.assertEqual(failed["missing_videos"], 1)

    def test_integrity_rejects_multiple_policy_outcomes_and_reused_video(self):
        specs = self.matrix[:2]
        with tempfile.TemporaryDirectory() as temporary:
            video = pathlib.Path(temporary) / "shared.mp4"
            video.write_bytes(b"video")
            records = [
                {
                    "episode_id": specs[0].episode_id,
                    "attempt_id": "attempt-0",
                    "attempt": 0,
                    "status": "failure",
                    "video": str(video),
                    "video_status": "written",
                },
                {
                    "episode_id": specs[0].episode_id,
                    "attempt_id": "attempt-1",
                    "attempt": 1,
                    "status": "success",
                    "video": str(video),
                    "video_status": "written",
                },
                {
                    "episode_id": specs[1].episode_id,
                    "attempt_id": "attempt-2",
                    "attempt": 0,
                    "status": "success",
                    "video": str(video),
                    "video_status": "written",
                },
            ]
            result = support.verify_integrity(records, specs)
        self.assertFalse(result["passed"])
        self.assertTrue(any("exactly one policy outcome" in issue for issue in result["issues"]))
        self.assertTrue(any("unique video paths" in issue for issue in result["issues"]))

    def test_integrity_rejects_unknown_attempt_status(self):
        spec = self.matrix[0]
        with tempfile.TemporaryDirectory() as temporary:
            video = pathlib.Path(temporary) / "episode.mp4"
            video.write_bytes(b"video")
            records = [
                {
                    "episode_id": spec.episode_id,
                    "attempt_id": "terminal",
                    "attempt": 0,
                    "status": "success",
                    "video": str(video),
                    "video_status": "written",
                },
                {
                    "episode_id": spec.episode_id,
                    "attempt_id": "mystery",
                    "attempt": 1,
                    "status": "mystery",
                },
            ]
            result = support.verify_integrity(records, [spec])
        self.assertFalse(result["passed"])
        self.assertTrue(any("unknown status" in issue for issue in result["issues"]))

    def test_integrity_rejects_unclassified_error_attempt(self):
        spec = self.matrix[0]
        with tempfile.TemporaryDirectory() as temporary:
            video = pathlib.Path(temporary) / "episode.mp4"
            video.write_bytes(b"video")
            records = [
                {
                    "episode_id": spec.episode_id,
                    "attempt_id": "error",
                    "attempt": 0,
                    "status": "error",
                    "error_category": "unexpected",
                },
                {
                    "episode_id": spec.episode_id,
                    "attempt_id": "terminal",
                    "attempt": 1,
                    "status": "success",
                    "video": str(video),
                    "video_status": "written",
                },
            ]
            result = support.verify_integrity(records, [spec])
        self.assertFalse(result["passed"])
        self.assertTrue(any("unknown error category" in issue for issue in result["issues"]))

    def test_final_record_audit_requires_record_video_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary)
            videos = result_dir / "videos"
            videos.mkdir()
            actual_video = videos / "episode.mp4"
            actual_video.write_bytes(b"video")
            record = {
                "episode_id": "episode-1",
                "attempt_id": "attempt-1",
                "status": "success",
                "video": str(actual_video),
                "video_status": "written",
            }
            (result_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            evidence = final_audit._record_audit(result_dir, 1)
            self.assertEqual(evidence["unique_record_video_paths"], 1)
            record["video"] = str(videos / "missing.mp4")
            (result_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record video paths"):
                final_audit._record_audit(result_dir, 1)

    def test_final_record_audit_rejects_external_video_plus_unrelated_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary) / "result"
            videos = result_dir / "videos"
            videos.mkdir(parents=True)
            (videos / "unrelated.mp4").write_bytes(b"unrelated")
            external = pathlib.Path(temporary) / "external.mp4"
            external.write_bytes(b"external")
            record = {
                "episode_id": "episode-1",
                "attempt_id": "attempt-1",
                "suite": "libero_goal",
                "status": "success",
                "video": str(external),
                "video_status": "written",
            }
            (result_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not equal the video inventory"):
                final_audit._record_audit(result_dir, 1)

    def test_shard_merge_validation_accepts_disjoint_authorities(self):
        specs = []
        records = {}
        for index, suite in enumerate(support.SUITES):
            source = support.TaskSource(
                "plus", suite, index + 1, "source-%d" % index,
                pathlib.Path("/%s/task.bddl" % suite),
                pathlib.Path("/%s/task.pruned_init" % suite), "prompt"
            )
            spec = support._episode(source, 0)
            specs.append(spec)
            records[suite] = [{
                "episode_id": spec.episode_id,
                "attempt_id": spec.episode_id + ":attempt-0000",
                "attempt": 0,
                "suite": suite,
                "status": "success",
            }]
        ordered = merger.validate_shard_records(specs, records)
        self.assertEqual([record["suite"] for record in ordered], list(support.SUITES))

    def test_shard_merge_rejects_duplicate_attempt_ids(self):
        specs = []
        records = {}
        for index, suite in enumerate(support.SUITES):
            source = support.TaskSource(
                "plus", suite, index + 1, "source-%d" % index,
                pathlib.Path("/%s/task.bddl" % suite),
                pathlib.Path("/%s/task.pruned_init" % suite), "prompt"
            )
            spec = support._episode(source, 0)
            specs.append(spec)
            records[suite] = [{
                "episode_id": spec.episode_id,
                "attempt_id": "duplicate",
                "attempt": 0,
                "suite": suite,
                "status": "success",
            }]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            merger.validate_shard_records(specs, records)

    def test_shard_merge_rejects_missing_policy_outcome(self):
        specs = []
        records = {}
        for index, suite in enumerate(support.SUITES):
            source = support.TaskSource(
                "plus", suite, index + 1, "source-%d" % index,
                pathlib.Path("/%s/task.bddl" % suite),
                pathlib.Path("/%s/task.pruned_init" % suite), "prompt"
            )
            spec = support._episode(source, 0)
            specs.append(spec)
            records[suite] = [{
                "episode_id": spec.episode_id,
                "attempt_id": spec.episode_id + ":attempt-0000",
                "attempt": 0,
                "suite": suite,
                "status": "error" if index == 0 else "success",
            }]
        with self.assertRaisesRegex(ValueError, "no policy outcome"):
            merger.validate_shard_records(specs, records)

    def test_parse_suite_dirs_allows_one_authoritative_dir_for_two_suites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            combined = root / "combined"
            goal = root / "goal"
            object_dir = root / "object"
            for directory in (combined, goal, object_dir):
                directory.mkdir()
                (directory / "episodes.jsonl").write_text("", encoding="utf-8")
            parsed = merger.parse_suite_dirs([
                "libero_10=%s" % combined,
                "libero_spatial=%s" % combined,
                "libero_goal=%s" % goal,
                "libero_object=%s" % object_dir,
            ])
        self.assertEqual(parsed["libero_10"], parsed["libero_spatial"])


class ErrorAndActionTest(unittest.TestCase):
    def test_final_audit_rate_format_distinguishes_na(self):
        self.assertEqual(final_audit._format_rate(None), "N/A")
        self.assertEqual(final_audit._format_rate(0.971), "97.10%")
        self.assertEqual(final_audit._format_value(None), "N/A")
        self.assertEqual(final_audit._format_value(0), "0")

    def test_final_audit_official_threshold_is_a_hard_gate(self):
        final_audit._require_official_threshold(0.9385)
        with self.assertRaisesRegex(ValueError, "below required 93.85%"):
            final_audit._require_official_threshold(0.93849)
        with self.assertRaisesRegex(ValueError, "below required 93.85%"):
            final_audit._require_official_threshold(None)

    def test_final_audit_report_renders_na_and_macro_acceptance(self):
        base_row = {
            "benchmark": "LIBERO",
            "protocol": "pi05_base/native-assets",
            "status": "not_applicable",
            "protocol_applicable": False,
            "protocol_planned": 2000,
            "evaluated_planned": 0,
            "successes": None,
            "failures": None,
            "success_rate": None,
            "infrastructure_errors": 0,
            "note": "missing norm stats",
            "result_dir": "/result",
        }
        report = {
            "rows": [base_row],
            "acceptance": {
                "old_official_macro_suite_success_rate": 0.971,
                "old_official_required_success_rate": 0.9385,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "audit"
            final_audit.write_report(report, output)
            markdown = (output / "report.md").read_text(encoding="utf-8")
            csv_text = (output / "report.csv").read_text(encoding="utf-8")
        self.assertIn("| N/A | N/A | N/A |", markdown)
        self.assertIn("97.10% >= 93.85%", markdown)
        self.assertIn("macro_suite_success_rate", csv_text.splitlines()[0])

    def test_final_audit_runtime_provenance_is_a_hard_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary)
            manifests = result_dir / "manifests"
            manifests.mkdir()
            payloads = {
                "git": {
                    "openpi": {
                        "path": "/openpi",
                        "commit": final_audit.EXPECTED_OPENPI_COMMIT,
                        "status": " M existing.py",
                    },
                    "benchmark": {
                        "path": "/benchmark",
                        "commit": final_audit.EXPECTED_PLUS_COMMIT,
                        "status": "",
                    },
                },
                "checkpoint-server": {
                    "benchmark": "plus",
                    "protocol": "official",
                    "config": "pi05_libero",
                    "asset_id": "physical-intelligence/libero",
                    "bind_host": "127.0.0.1",
                    "norm_stats_sha256": final_audit.EXPECTED_NORM_STATS_SHA256,
                    "checkpoint": "/checkpoint/pi05_libero",
                },
                "sources": {
                    "benchmark": "plus",
                    "benchmark_repo": "/benchmark-data",
                    "source_count": support.PLUS_FULL_EPISODES,
                    "inventory_sha256": "a" * 64,
                },
                "preflight": {
                    "status": "passed",
                    "benchmark": "plus",
                    "finite": True,
                    "action_steps": 10,
                    "action_dimension": 7,
                    "seed": 7,
                    "resize": 224,
                    "replan": 5,
                },
                "environment": {
                    "run_id": "run",
                    "benchmark": "plus",
                    "phase": "full",
                    "fixed_environment": {
                        "MUJOCO_GL": "egl",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
                    },
                },
            }
            for prefix, payload in payloads.items():
                (manifests / (prefix + "-run.json")).write_text(json.dumps(payload), encoding="utf-8")
            def git_state(path):
                return {
                    "commit": (
                        final_audit.EXPECTED_OPENPI_COMMIT
                        if str(path) == "/openpi"
                        else final_audit.EXPECTED_PLUS_COMMIT
                    ),
                    "status": " M existing.py" if str(path) == "/openpi" else "",
                }

            current_inventory = {
                "source_count": support.PLUS_FULL_EPISODES,
                "inventory_sha256": "a" * 64,
            }
            checkpoint_inventory = {
                "params_source_matches": True,
                "param_files": final_audit.EXPECTED_PARAMS_FILES,
                "param_bytes": final_audit.EXPECTED_PARAMS_BYTES,
                "params_tree_sha256": final_audit.EXPECTED_PARAMS_TREE_SHA256,
                "norm_stats_exists": True,
                "norm_stats_sha256": final_audit.EXPECTED_NORM_STATS_SHA256,
            }
            with mock.patch.object(final_audit, "_current_git_state", side_effect=git_state), mock.patch.object(
                final_audit, "_current_source_inventory", return_value=current_inventory
            ), mock.patch.object(
                final_audit, "_checkpoint_inventory", return_value=checkpoint_inventory
            ):
                provenance = final_audit._runtime_provenance(
                    result_dir, "plus", final_audit.EXPECTED_PLUS_COMMIT, support.PLUS_FULL_EPISODES
                )
            self.assertEqual(provenance["openpi_commit"], final_audit.EXPECTED_OPENPI_COMMIT)
            payloads["checkpoint-server"]["norm_stats_sha256"] = "bad"
            (manifests / "checkpoint-server-run.json").write_text(
                json.dumps(payloads["checkpoint-server"]), encoding="utf-8"
            )
            with mock.patch.object(final_audit, "_current_git_state", side_effect=git_state), mock.patch.object(
                final_audit, "_current_source_inventory", return_value=current_inventory
            ), mock.patch.object(
                final_audit, "_checkpoint_inventory", return_value=checkpoint_inventory
            ), self.assertRaisesRegex(ValueError, "checkpoint provenance disagrees"):
                final_audit._runtime_provenance(
                    result_dir, "plus", final_audit.EXPECTED_PLUS_COMMIT, support.PLUS_FULL_EPISODES
                )

    def test_final_audit_runtime_manifests_must_share_one_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary)
            manifests = result_dir / "manifests"
            manifests.mkdir()
            (manifests / "environment-new.json").write_text(
                json.dumps({"run_id": "new"}), encoding="utf-8"
            )
            for prefix in ("git", "checkpoint-server", "sources", "preflight"):
                (manifests / (prefix + "-old.json")).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest for run new is missing"):
                final_audit._runtime_manifests(result_dir)

    def test_final_audit_pro_summary_rates_and_compatibility_are_hard_gates(self):
        groups = {
            "suite": {
                "libero_goal": {
                    "planned": 2,
                    "successes": 1,
                    "failures": 1,
                    "success_rate": 0.5,
                    "complete": True,
                },
            },
            "perturbation": {
                "task": {
                    "planned": 2,
                    "successes": 1,
                    "failures": 1,
                    "success_rate": 0.5,
                    "complete": True,
                },
            },
        }
        compatibility = {"status": "partial_incompatible", "unavailable_cells": ["env"]}
        summary = {
            "benchmark": "pro",
            "protocol": "official",
            "phase": "full",
            "seed": 7,
            "resize": 224,
            "replan": 5,
            "total": {
                "planned": 2,
                "complete": True,
                "policy_denominator": 2,
                "successes": 1,
                "failures": 1,
                "success_rate": 0.5,
                "macro_suite_success_rate": 0.5,
                "excluded_errors": 0,
                "unknown_errors": 0,
            },
            "integrity": {"passed": True},
            "compatibility": compatibility,
            "groups": groups,
        }
        final_audit._require_pro_summary(summary, 1, 1, groups, compatibility)
        wrong_rate = json.loads(json.dumps(summary))
        wrong_rate["total"]["success_rate"] = 1.0
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            final_audit._require_pro_summary(wrong_rate, 1, 1, groups, compatibility)
        wrong_compatibility = json.loads(json.dumps(summary))
        wrong_compatibility["compatibility"] = {"status": "complete"}
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            final_audit._require_pro_summary(
                wrong_compatibility, 1, 1, groups, compatibility
            )

    def test_final_audit_old_row_executes_without_unbound_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result_dir = root / "official-full"
            result_dir.mkdir()
            suite_rates = dict(zip(support.SUITES, (0.986, 0.984, 0.982, 0.932)))
            suite_successes = {suite: int(rate * 500) for suite, rate in suite_rates.items()}
            (result_dir / "summary.json").write_text(
                json.dumps({
                    "protocol": "official",
                    "phase": "full",
                    "required_commit": "15a9616a",
                    "official_full_threshold": {
                        "applies": True,
                        "passed": True,
                        "required": 0.9385,
                    },
                    "total": {
                        "complete": True,
                        "planned": 2000,
                        "attempted_unique": 2000,
                        "policy_denominator": 2000,
                        "successes": 1942,
                        "failures": 58,
                        "success_rate": 0.971,
                        "macro_success_rate": 0.971,
                        "excluded_errors": 0,
                        "unknown_errors": 0,
                    },
                    "suites": {
                        suite: {
                            "complete": True,
                            "planned": 500,
                            "attempted_unique": 500,
                            "policy_denominator": 500,
                            "excluded_errors": 0,
                            "unknown_errors": 0,
                            "successes": suite_successes[suite],
                            "failures": 500 - suite_successes[suite],
                            "success_rate": rate,
                        }
                        for suite, rate in suite_rates.items()
                    },
                }),
                encoding="utf-8",
            )
            evidence = {
                "successes": 1942,
                "failures": 58,
                "success_rate": 0.971,
                "macro_suite_success_rate": 0.971,
                "suites": {
                    suite: {
                        "planned": 500,
                        "successes": suite_successes[suite],
                        "failures": 500 - suite_successes[suite],
                        "success_rate": rate,
                    }
                    for suite, rate in suite_rates.items()
                },
            }
            with mock.patch.object(final_audit, "_record_audit", return_value=evidence), mock.patch.object(
                final_audit, "_old_matrix_audit", return_value={"matrix_keys": 2000}
            ), mock.patch.object(
                final_audit, "_old_runtime_provenance", return_value={"run_id": "fixture"}
            ):
                row = final_audit._old_full_row(root, "official-full", "LIBERO", "pi05_libero")
        self.assertEqual(row["successes"], 1942)
        self.assertEqual(row["macro_suite_success_rate"], 0.971)
        self.assertEqual(row["infrastructure_errors"], 0)

    def test_final_audit_old_row_rejects_summary_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result_dir = root / "official-full"
            result_dir.mkdir()
            summary = {
                "protocol": "wrong",
                "phase": "full",
                "required_commit": "15a9616a",
                "official_full_threshold": {
                    "applies": True,
                    "passed": True,
                    "required": 0.9385,
                },
                "total": {
                    "complete": True,
                    "planned": 2000,
                    "attempted_unique": 2000,
                    "policy_denominator": 2000,
                    "successes": 2000,
                    "failures": 0,
                    "success_rate": 1.0,
                    "macro_success_rate": 1.0,
                    "excluded_errors": 0,
                    "unknown_errors": 0,
                },
                "suites": {},
            }
            (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            evidence = {
                "successes": 2000,
                "failures": 0,
                "success_rate": 1.0,
                "macro_suite_success_rate": 1.0,
                "suites": {},
            }
            with mock.patch.object(final_audit, "_record_audit", return_value=evidence), mock.patch.object(
                final_audit, "_old_matrix_audit", return_value={"matrix_keys": 2000}
            ), mock.patch.object(
                final_audit, "_old_runtime_provenance", return_value={"run_id": "fixture"}
            ), self.assertRaisesRegex(ValueError, "summary protocol provenance"):
                final_audit._old_full_row(root, "official-full", "LIBERO", "pi05_libero")

    def test_final_audit_old_matrix_requires_exact_task_trial_grid_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary)
            records = []
            for suite in support.SUITES:
                for task_id in range(10):
                    for trial in range(50):
                        episode_id = "official:%s:task-%02d:trial-%02d:seed-7" % (
                            suite,
                            task_id,
                            trial,
                        )
                        records.append({
                            "episode_id": episode_id,
                            "attempt_id": episode_id + ":attempt-00",
                            "attempt": 0,
                            "protocol": "official",
                            "phase": "full",
                            "suite": suite,
                            "task_id": task_id,
                            "trial": trial,
                            "seed": 7,
                            "resize": 224,
                            "replan": 5,
                            "wait_steps": 10,
                            "max_steps": support.MAX_STEPS[suite],
                            "gl_backend": "egl",
                            "task_description": "%s task %d" % (suite, task_id),
                            "status": "success",
                            "success": True,
                            "error": None,
                            "error_category": None,
                        })
            episode_path = result_dir / "episodes.jsonl"
            episode_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            evidence = final_audit._old_matrix_audit(result_dir, "official")
            self.assertEqual(evidence["matrix_keys"], 2000)
            records[-1]["task_id"] = 8
            episode_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fixed LIBERO protocol|exact four-suite"):
                final_audit._old_matrix_audit(result_dir, "official")

    def test_final_audit_old_manifests_must_share_environment_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary)
            manifests = result_dir / "manifests"
            manifests.mkdir()
            run_id = "one"
            for prefix in (
                "checkpoint-request",
                "checkpoint-server",
                "environment",
                "git",
                "preflight-egl",
                "result",
                "server-startup",
            ):
                payload = {"run_id": run_id} if prefix in ("environment", "result", "server-startup") else {}
                (manifests / (prefix + "-" + run_id + ".json")).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            loaded_run_id, paths, _ = final_audit._old_runtime_manifests(result_dir)
            self.assertEqual(loaded_run_id, run_id)
            self.assertEqual(len(paths), 7)
            (manifests / ("result-" + run_id + ".json")).unlink()
            with self.assertRaisesRegex(ValueError, "result manifest"):
                final_audit._old_runtime_manifests(result_dir)

    def test_final_audit_matrix_row_counts_only_unresolved_infrastructure_errors(self):
        source = support.TaskSource(
            "plus", "libero_goal", 1, "source-1",
            pathlib.Path("/task.bddl"), pathlib.Path("/task.pruned_init"), "prompt",
        )
        spec = support._episode(source, 0)
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary)
            videos = result_dir / "videos"
            videos.mkdir()
            video = videos / "video.mp4"
            video.write_bytes(b"video")
            records = [
                {
                    "episode_id": spec.episode_id,
                    "attempt_id": spec.episode_id + ":attempt-0000",
                    "attempt": 0,
                    "suite": spec.suite,
                    "status": "error",
                    "error_category": "environment",
                },
                {
                    "episode_id": spec.episode_id,
                    "attempt_id": spec.episode_id + ":attempt-0001",
                    "attempt": 1,
                    "suite": spec.suite,
                    "status": "success",
                    "video": str(video),
                    "video_status": "written",
                },
            ]
            (result_dir / "episodes.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            episodes_sha256 = support.sha256_file(result_dir / "episodes.jsonl")
            manifests = result_dir / "manifests"
            manifests.mkdir()
            (manifests / "merge.json").write_text(
                json.dumps({
                    "status": "passed",
                    "planned_episodes": 1,
                    "merged_records": 2,
                    "merged_policy_outcomes": 1,
                    "merged_episodes_jsonl_sha256": episodes_sha256,
                    "matrix_episode_ids_sha256": final_audit._digest_lines([spec.episode_id]),
                    "suite_counts": dict(support.PLUS_COUNTS),
                    "environment_seed": 7,
                    "policy_rng_initial_key": 0,
                    "policy_rng_scope": "server_process/per-suite",
                    "integrity": {"passed": True},
                    "sources": {
                        suite: {
                            "directory": str(result_dir),
                            "episodes_jsonl_sha256": episodes_sha256,
                        }
                        for suite in support.SUITES
                    },
                }),
                encoding="utf-8",
            )
            with mock.patch.object(final_audit, "_runtime_provenance", return_value={"passed": True}):
                row = final_audit._matrix_row(
                    "LIBERO-Plus", "pi05_libero", result_dir, [spec], 1, True, "test"
                )
        self.assertEqual(row["infrastructure_errors"], 0)
        self.assertEqual(row["evidence"]["historical_error_attempts"], 1)

    def test_pro_audit_key_ignores_registry_task_id_but_keeps_trial(self):
        source = support.TaskSource(
            "pro", "libero_goal", 9, "task/libero_goal/task.bddl",
            pathlib.Path("/task.bddl"), pathlib.Path("/task.pruned_init"), "prompt",
            perturbation="task",
        )
        spec = support._episode(source, 17)
        record = {
            "perturbation": "task", "suite": "libero_goal",
            "source_id": source.source_id, "task_id": 2, "trial": 17,
        }
        self.assertEqual(final_audit._pro_key(spec), final_audit._pro_key(record))
        record["trial"] = 18
        self.assertNotEqual(final_audit._pro_key(spec), final_audit._pro_key(record))

    def test_four_error_categories(self):
        self.assertEqual(support.classify_error(RuntimeError("bad checkpoint"), "checkpoint load"), "checkpoint")
        self.assertEqual(support.classify_error(ConnectionError("refused"), "connect"), "connection")
        self.assertEqual(support.classify_error(ValueError("bad actions"), "policy infer"), "policy_runtime")
        self.assertEqual(support.classify_error(RuntimeError("EGL failed"), "environment create"), "environment")

    def test_action_shape_and_finiteness(self):
        self.assertEqual(len(support.validate_action_result({"actions": [[0] * 7 for _ in range(5)]})), 5)
        with self.assertRaisesRegex(ValueError, "at least 5"):
            support.validate_action_result({"actions": [[0] * 7 for _ in range(4)]})
        with self.assertRaisesRegex(ValueError, "dimension"):
            support.validate_action_result({"actions": [[0] * 6 for _ in range(5)]})
        actions = [[0] * 7 for _ in range(5)]
        actions[1][2] = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            support.validate_action_result({"actions": actions})


class RegistryAndEnvironmentTest(unittest.TestCase):
    def test_flat_numeric_init_is_one_state_but_object_sequence_is_preserved(self):
        flat = np.arange(47, dtype=np.float64)
        normalized = client._normalize_init_states(flat, np)
        self.assertEqual(normalized.shape, (1, 47))
        matrix = np.arange(94, dtype=np.float64).reshape(2, 47)
        self.assertIs(client._normalize_init_states(matrix, np), matrix)
        object_states = np.asarray([np.arange(3), np.arange(4)], dtype=object)
        self.assertIs(client._normalize_init_states(object_states, np), object_states)

    def test_pro_discovery_accepts_16_complete_cells_and_marks_env_na(self):
        suffixes = {
            "lan(semantic)": "lan",
            "object": "object",
            "swap(position)": "swap",
            "task": "task",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for perturbation, suffix in suffixes.items():
                for suite in support.SUITES:
                    cell = root / ("%s_%s" % (suite, suffix))
                    for task_id in range(10):
                        stem = "task_%02d" % task_id
                        bddl = cell / "bddl_files" / (stem + ".bddl")
                        raw = cell / "init_files" / (stem + ".init")
                        pruned = cell / "init_files" / (stem + ".pruned_init")
                        bddl.parent.mkdir(parents=True, exist_ok=True)
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        bddl.write_text(
                            "(:language_instruction do %s %s %d)\n" % (perturbation, suite, task_id),
                            encoding="utf-8",
                        )
                        raw.write_bytes(b"raw")
                        pruned.write_bytes(b"pruned")
            sources = support.discover_pro_sources(root, allow_missing_cells=True)
        self.assertEqual(len(sources), 160)
        self.assertTrue(all(source.init_path.name.endswith(".pruned_init") for source in sources))
        compatibility = support.pro_compatibility(sources)
        self.assertEqual(compatibility["available_cells"], 16)
        self.assertEqual({cell["perturbation"] for cell in compatibility["unavailable_cells"]}, {"env"})

    def test_official_init_inventory_uses_only_pruned_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            raw = root / "task.init"
            pruned = root / "task.pruned_init"
            raw.write_bytes(b"raw")
            pruned.write_bytes(b"pruned")
            selected = support.prefer_official_init_files([raw, pruned])
        self.assertEqual(selected, [pruned.resolve()])

    def test_missing_pruned_init_is_not_silently_replaced_by_raw_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = pathlib.Path(temporary) / "task.init"
            raw.write_bytes(b"raw")
            with self.assertRaisesRegex(support.BenchmarkInventoryError, "no official .pruned_init"):
                support.prefer_official_init_files([raw])

    def test_plus_init_suffix_does_not_truncate_table_in_task_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            base = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
            pruned = root / "libero_spatial" / (base + ".pruned_init")
            pruned.parent.mkdir(parents=True)
            pruned.write_bytes(b"pruned")
            index = support._build_file_index([pruned.resolve()])
            resolved = support._resolve_plus_init(index, "libero_spatial", base + "_table_11")
        self.assertEqual(resolved, pruned.resolve())

    def test_plus_compound_language_view_suffix_uses_base_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            base = "pick_up_the_bowl"
            pruned = root / "libero_goal" / (base + ".pruned_init")
            pruned.parent.mkdir(parents=True)
            pruned.write_bytes(b"pruned")
            index = support._build_file_index([pruned.resolve()])
            task = base + "_language_2_view_0_0_100_0_0_initstate_0"
            resolved = support._resolve_plus_init(index, "libero_goal", task)
        self.assertEqual(resolved, pruned.resolve())

    def test_plus_view_bddl_keeps_logical_suffix_and_audits_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            suite = "libero_goal"
            base = root / suite / "open_the_drawer.bddl"
            base.parent.mkdir(parents=True)
            base.write_text("(:language open the drawer)\n", encoding="utf-8")
            name = "open_the_drawer_view_0_0_100_0_0_initstate_7"
            logical, prompt_source = support._plus_bddl_paths(root, suite, name)
        self.assertFalse(logical.exists())
        self.assertEqual(logical.name, name + ".bddl")
        self.assertEqual(prompt_source, base.resolve())

    def test_pro_combined_directory_names_are_recognized(self):
        for suffix, expected in (
            ("lan", "lan(semantic)"),
            ("object", "object"),
            ("swap", "swap(position)"),
            ("task", "task"),
            ("env", "env"),
        ):
            path = pathlib.Path("/data/libero_goal_%s/task.bddl" % suffix)
            self.assertEqual(support._perturbation_from_path(path), expected)

    def test_plus_registry_id_order_is_checked(self):
        sources = [
            support.TaskSource(
                "plus", "libero_spatial", 2, "bad", pathlib.Path("bad.bddl"),
                pathlib.Path("bad.pruned_init"), "prompt"
            )
        ]

        class FakeSuite:
            n_tasks = 1

            @staticmethod
            def get_task(task_id):
                return types.SimpleNamespace(bddl_file="bad.bddl")

        modules = {"benchmark": types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite})}
        with mock.patch.object(support, "validate_sources", return_value=None):
            with self.assertRaisesRegex(support.BenchmarkInventoryError, "id/order mismatch"):
                client._verify_and_order_registry(sources, modules)

    def test_plus_registry_accepts_one_based_id_and_preserves_it(self):
        source = support.TaskSource(
            "plus", "libero_spatial", 1, "good", pathlib.Path("good.bddl"),
            pathlib.Path("good.pruned_init"), "prompt"
        )

        class FakeSuite:
            n_tasks = 1

            @staticmethod
            def get_task(task_id):
                return types.SimpleNamespace(bddl_file="good.bddl")

        modules = {"benchmark": types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite})}
        with mock.patch.object(support, "validate_sources", return_value=None):
            ordered = client._verify_and_order_registry([source], modules)
        self.assertEqual(ordered[0].task_id, 1)

    def test_child_environment_is_loopback_run_isolated_and_plus_magick_aware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            imagemagick = root / "runtime" / "imagemagick"
            (imagemagick / "prefix" / "lib").mkdir(parents=True)
            coder_dir = imagemagick / "root" / "usr" / "lib" / "x86_64-linux-gnu" / "ImageMagick-6.9.11" / "modules-Q16" / "coders"
            coder_dir.mkdir(parents=True)
            args = types.SimpleNamespace(
                benchmark="plus",
                benchmark_repo=root,
                openpi_repo=root,
                imagemagick_runtime=imagemagick,
                gpu_id="4",
            )
            context = types.SimpleNamespace(home_dir=root / "isolated-home")
            env = runner._child_env(args, context, client=True, environ={"LD_LIBRARY_PATH": "/old"})
        self.assertEqual(env["HOME"], str(context.home_dir))
        self.assertEqual(env["XDG_CACHE_HOME"], str(runner.SHARED_CACHE))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(env["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.75")
        self.assertEqual(env["MUJOCO_GL"], "egl")
        self.assertEqual(env["MAGICK_HOME"], str(imagemagick / "prefix"))
        self.assertEqual(env["MAGICK_CODER_MODULE_PATH"], str(coder_dir.resolve()))
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith(str(imagemagick / "prefix" / "lib") + os.pathsep))

    def test_cli_exposes_required_interface(self):
        parser = runner.build_parser()
        args = parser.parse_args(
            [
                "--benchmark", "plus", "--phase", "smoke", "--benchmark-repo", "benchmark",
                "--openpi-repo", "openpi", "--checkpoint-dir", "checkpoint",
                "--client-python", "client-python", "--server-python", "server-python",
                "--port", "8125", "--gpu-id", "4", "--output-dir", "output", "--resume",
                "--suite", "libero_10", "--suite", "libero_spatial",
            ]
        )
        self.assertTrue(args.resume)
        self.assertEqual(args.port, 8125)
        self.assertEqual(args.gpu_id, "4")
        self.assertEqual(args.suite, ["libero_10", "libero_spatial"])
        context = types.SimpleNamespace(
            benchmark_data_root=pathlib.Path("benchmark"),
            preflight_manifest=pathlib.Path("preflight.json"),
        )
        command = runner._client_command(args, context)
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--suite"],
            ["libero_10", "libero_spatial"],
        )


if __name__ == "__main__":
    unittest.main()
