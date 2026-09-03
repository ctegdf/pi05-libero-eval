# LIBERO evaluation archive: π0.5 vs OpenVLA vs ACT

[Rollout videos](https://huggingface.co/datasets/ctegdf/pi05-libero-rollouts) |
[π0.5 analysis report](analysis-report/report_zh.md) |
[Cross-policy report](analysis-report/cross_policy_report_zh.md) |
[Result protocol](docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md) |
[Release notes](docs/RELEASE_NOTES.md)

Reproducible evaluation harnesses, aggregate results, and sanitized episode
metadata for three policies on LIBERO / LIBERO-Plus / LIBERO-Pro, evaluated
under the same protocol: **π0.5** (`pi05_libero`, one unified checkpoint,
also covers LIBERO-X), **OpenVLA-7B**, and **ACT** (OpenVLA and ACT each use
4 independently fine-tuned/from-scratch-trained per-suite checkpoints, no
single unified weight). Evaluation code and metadata live in this
repository; rollout videos for all three policies live in the companion
Hugging Face dataset.

## Evaluation results

Every policy's raw numbers, in the same table shape, so the three are
directly comparable without cross-referencing separate documents. "Scope"
records whether the run is same-distribution (checkpoint trained/fine-tuned
on that exact benchmark) or zero-shot transfer (checkpoint never trained on
that benchmark's perturbations), plus any N/A-cell exclusions.

### π0.5 (`pi05_libero`, single unified checkpoint)

| Benchmark | Successes | Evaluated episodes | Success rate | Scope |
| --- | ---: | ---: | ---: | --- |
| LIBERO | 1,942 | 2,000 | 97.10% | Same-distribution, official four-suite protocol |
| LIBERO-Plus | 8,390 | 10,030 | 83.65% | Zero-shot; full generated matrix |
| LIBERO-Pro | 4,703 | 8,000 | 58.79% | Zero-shot; 16/20 cells available, 2,000 episodes N/A (`env` perturbation absent) |
| LIBERO-X | 2,711 | 34,520 | 7.85% | Zero-shot; not a paper-protocol reproduction |

### OpenVLA-7B (4 per-suite fine-tuned checkpoints, no unified weight)

| Benchmark | Successes | Evaluated episodes | Success rate | Scope |
| --- | ---: | ---: | ---: | --- |
| LIBERO | 1,383 | 2,000 | 69.15% | Same-distribution — each checkpoint evaluated only on the suite it was fine-tuned on |
| LIBERO-Plus | 2,534 | 10,030 | 25.26% | Zero-shot; full generated matrix |
| LIBERO-Pro | 2,707 | 8,000 | 33.84% | Zero-shot; 16/20 cells available, 2,000 episodes N/A (same absent `env` cells as π0.5) |

### ACT (4 per-suite checkpoints, trained from scratch, no pretrained backbone)

| Benchmark | Successes | Evaluated episodes | Success rate | Scope |
| --- | ---: | ---: | ---: | --- |
| LIBERO | 498 | 2,000 | 24.90% | Same-distribution — each checkpoint evaluated only on the suite it was trained on |
| LIBERO-Plus | 1,310 | 10,030 | 13.06% | Zero-shot; full generated matrix |
| LIBERO-Pro | 889 | 8,000 | 11.11% | Zero-shot; 16/20 cells available, 2,000 episodes N/A (same absent `env` cells as π0.5/OpenVLA) |

LIBERO-X is π0.5-only (OpenVLA/ACT were never evaluated on it — out of scope
for this comparison campaign, not a missing result).

### Cross-policy comparison

| Benchmark | π0.5 | OpenVLA | ACT |
| --- | ---: | ---: | ---: |
| LIBERO (2,000 ep) | **97.10%** (1,942/2,000) | 69.15% (1,383/2,000) | 24.90% (498/2,000) |
| LIBERO-Plus (10,030 ep) | **83.65%** (8,390/10,030) | 25.26% (2,534/10,030) | 13.06% (1,310/10,030) |
| LIBERO-Pro (8,000 ep) | **58.79%** (4,703/8,000) | 33.84% (2,707/8,000) | 11.11% (889/8,000) |

The ranking (π0.5 ≫ OpenVLA > ACT) holds on every benchmark, but the gap is
not uniform: π0.5's LIBERO-Plus robustness margin over OpenVLA (−13.5pp from
its own same-distribution baseline vs OpenVLA's −43.9pp) is much larger than
its LIBERO-Pro margin (−38.3pp vs OpenVLA's −35.3pp) — i.e. π0.5's advantage
comes mostly from robustness to LIBERO-Plus's visual/environment
perturbations, not from a uniformly stronger grip on LIBERO-Pro's structural
perturbations. OpenVLA fails 0/2,000 (100%) on LIBERO-Pro's `swap(position)`
perturbation — verified as a genuine policy failure (no infra errors, every
episode ran to `max_steps`), not a bug. ACT has no language conditioning by
design, so its `Language Instructions`/`lan(semantic)` perturbation scores
do not indicate language robustness. Full per-suite, per-perturbation
breakdowns and these findings' full derivation are in
[`analysis-report/cross_policy_report_zh.md`](analysis-report/cross_policy_report_zh.md)
(Chinese) and the four per-model `docs/*-EVAL_REPORT.md` files linked below.
**This is not an architecture-fair comparison** — π0.5 uses one unified
checkpoint vs. 4 per-suite checkpoints for OpenVLA/ACT, different action
horizons (`replan=5` / `1` / `100`), different model scales, and ACT has no
language conditioning; see the report's caveats section before citing these
numbers as an architecture verdict.

## Video inventory

The complete MP4 archive for all three policies is published in the same
[Hugging Face dataset](https://huggingface.co/datasets/ctegdf/pi05-libero-rollouts),
under `videos/{official,base-libero-assets,plus,pro,libero-x}/` for π0.5 and
`videos/{openvla,act}/<benchmark>/<run_id>/` for OpenVLA/ACT (protocol v1
layout — see [Result protocol](docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md) §3).

### π0.5

| Benchmark | Full | Smoke | MP4s |
| --- | ---: | ---: | ---: |
| LIBERO official | 2,000 | 8 | 2,008 |
| LIBERO Base assets | 2,000 | 8 | 2,008 |
| LIBERO-Plus | 10,030 | 28 | 10,058 |
| LIBERO-Pro | 8,000 | 16 | 8,016 |
| LIBERO-X | 34,520 | 29 | 34,549 |
| **Subtotal** | **56,550** | **89** | **56,639** |

| Inventory item | Count | Reconciliation |
| --- | ---: | --- |
| Episode metadata records | 56,647 | All retained episode records |
| Available MP4 files | 56,639 | Every file is indexed by size and SHA-256 |
| Metadata-only episodes | 8 | LIBERO-Plus smoke, `video_status=not_recorded` |

### OpenVLA and ACT

Full phase only (protocol v1 does not require smoke-phase publication);
every episode in every run has a matched, non-empty video — 0 missing.

| Policy | LIBERO | LIBERO-Plus | LIBERO-Pro | MP4s |
| --- | ---: | ---: | ---: | ---: |
| OpenVLA | 2,000 | 10,030 | 8,000 | 20,030 |
| ACT | 2,000 | 10,030 | 8,000 | 20,030 |
| **Subtotal** | **4,000** | **20,060** | **16,000** | **40,060** |

### Grand total

| Source | MP4s |
| --- | ---: |
| π0.5 | 56,639 |
| OpenVLA | 20,030 |
| ACT | 20,030 |
| **Total** | **96,699** |

The GitHub repository intentionally contains no checkpoints, videos, runtime
directories, remote logs, or simulator assets — for every policy.

## Contents

| Path | Purpose |
| --- | --- |
| [`harness/`](harness) | Evaluation launchers, inventory checks, integrity checks, and unit tests |
| [`results/`](results) | Aggregate summaries and sanitized episode metadata with relative video paths |
| [`analysis-report/`](analysis-report) | Derived tables, figures, Chinese report, and rendered HTML report |
| [`docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md`](docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md) | Stable result contract for ACT and OpenVLA evaluations |
| [`scripts/export_to_protocol.py`](scripts/export_to_protocol.py) | Export a new model run into the shared result layout |
| [`scripts/validate_run.py`](scripts/validate_run.py) | Validate a published run and report every protocol violation |
| [`scripts/stage_videos.py`](scripts/stage_videos.py) | Build an isolated video tree without modifying the source archive (π0.5 only) |
| [`scripts/upload_videos_to_hf.py`](scripts/upload_videos_to_hf.py) | Validate and upload π0.5's video inventory in resumable batches |
| [`scripts/upload_protocol_videos_to_hf.py`](scripts/upload_protocol_videos_to_hf.py) | Same, generalized for any `results/runs/`-protocol model (OpenVLA, ACT, future models) |

## OpenVLA and ACT runs

The versioned [result protocol](docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md) defines
the layout for ACT and OpenVLA runs on LIBERO, LIBERO-Plus, and LIBERO-Pro:

```text
results/runs/<model_id>/<benchmark_id>/<run_id>/
```

Six `full`-phase runs are published under this layout (`results/run-registry.json`
lists all of them):

```text
results/runs/openvla/libero/openvla-libero-full-v1/
results/runs/openvla/libero-plus/openvla-libero-plus-full-v1/
results/runs/openvla/libero-pro/openvla-libero-pro-full-v1/
results/runs/act/libero/act-libero-full-v1/
results/runs/act/libero-plus/act-libero-plus-full-v1/
results/runs/act/libero-pro/act-libero-pro-full-v1/
```

Each was produced by `scripts/export_to_protocol.py` from the working
harness's own `episodes.jsonl` (OpenVLA/ACT run each LIBERO suite under its
own per-suite checkpoint/server process, so the source is a 4-suite merge —
see each run's `notes` field in `run.json` for exact checkpoint identities)
and validated with:

```bash
python3 scripts/validate_run.py --run-dir results/runs/openvla/libero/openvla-libero-full-v1
python3 scripts/validate_run.py --run-dir results/runs/openvla/libero-plus/openvla-libero-plus-full-v1
python3 scripts/validate_run.py --run-dir results/runs/openvla/libero-pro/openvla-libero-pro-full-v1
python3 scripts/validate_run.py --run-dir results/runs/act/libero/act-libero-full-v1
python3 scripts/validate_run.py --run-dir results/runs/act/libero-plus/act-libero-plus-full-v1
python3 scripts/validate_run.py --run-dir results/runs/act/libero-pro/act-libero-pro-full-v1
```

Per-model narrative reports: [`docs/openvla-libero-EVAL_REPORT.md`](docs/openvla-libero-EVAL_REPORT.md),
[`docs/openvla-plus-pro-EVAL_REPORT.md`](docs/openvla-plus-pro-EVAL_REPORT.md),
[`docs/act-libero-EVAL_REPORT.md`](docs/act-libero-EVAL_REPORT.md),
[`docs/act-plus-pro-EVAL_REPORT.md`](docs/act-plus-pro-EVAL_REPORT.md).

## Reproduce the analysis

```bash
python3 -m unittest discover -s analysis-report -p 'test_*.py' -v
python3 analysis-report/generate_report.py            # pi0.5-only narrative report
python3 -m unittest discover -s harness/plus-pro -p 'test_*.py' -v
python3 -m py_compile harness/libero-x/*.py
python3 analysis-report/generate_cross_policy_report.py  # pi0.5 vs OpenVLA vs ACT
```

Both checked-in reports are already generated. Both generators require their
respective original evaluation archive mounted beside this checkout —
`generate_report.py` needs the π0.5 archive, `generate_cross_policy_report.py`
additionally needs the OpenVLA/ACT archive (it currently reads each policy's
per-suite `episodes.jsonl` from the private working archive, not from the
sanitized `results/runs/` copies checked into this repo) — because both
validate raw episode/video paths; the GitHub checkout intentionally does not
redistribute those large source directories. The already-computed output
(`analysis-report/cross_policy_report_zh.md` and its `data/cross_policy_*.csv`)
is what's actually checked in and citable.

Running a simulator evaluation requires the pinned upstream repositories,
benchmark assets, a compatible CUDA/MuJoCo environment, and the checkpoint.
The benchmark projects and checkpoint are not redistributed here.

## Video archive

Both upload paths require a local source archive and an authenticated
Hugging Face account, and both publish to the same dataset repository
(`ctegdf/pi05-libero-rollouts` — a dataset repository, not a model
repository):

```bash
python3 -m pip install huggingface_hub
hf auth login
```

**π0.5** (`videos/{official,base-libero-assets,plus,pro,libero-x}/`):

```bash
python3 scripts/upload_videos_to_hf.py \
  --source-root /path/to/evaluation-archive \
  --repo-id YOUR_ACCOUNT/pi05-libero-rollouts \
  --commit-message 'Upload complete evaluation rollouts'
```

The script validates the expected 56,639 MP4 files. To keep the GitHub
project and large files separate, stage the videos into a second directory
first: `python3 scripts/stage_videos.py --source-root /path/to/evaluation-archive --destination-root /path/to/pi05-libero-video-release`.

**OpenVLA / ACT** (`videos/{openvla,act}/<benchmark>/<run_id>/`, generalized
for any `results/runs/`-protocol model, no hardcoded file count):

```bash
python3 scripts/upload_protocol_videos_to_hf.py \
  --videos-root videos \
  --repo-id ctegdf/pi05-libero-rollouts \
  --resume \
  --dataset-card docs/hf-dataset-card.md
```

Runs whose video count exceeds `--shard-size` (default 5,000) get an extra
`part-NNN/` path segment — both `openvla-libero-plus-full-v1` and
`act-libero-plus-full-v1` (10,030 videos each) are sharded this way. Both
uploaders skip files already present with a matching size and SHA-256 when
`--resume` is passed, and record a manifest under
`manifests/video-manifest.jsonl` in the dataset.

The π0.5 source archive contains 6,000 LIBERO-X LEVEL1 MP4s and all 6,000
LEVEL1 episode records resolve to non-empty files. Eight LIBERO-Plus smoke
records have `video_status=not_recorded` and are retained as unavailable
metadata — the uploadable π0.5 inventory therefore contains 56,639 real MP4s.
Every OpenVLA/ACT `full`-phase episode has a matched video (0 missing across
all 6 runs, per `run.json.counts.videos_missing`). Neither uploader ever
fabricates a placeholder video for a missing episode.

## Interpretation limits

- LIBERO-Pro has 2,000 unavailable `env`-perturbation episodes **per
  policy** (same absent pre-generated cells for π0.5, OpenVLA, and ACT).
  They are N/A, not failures, and are excluded from each policy's LIBERO-Pro
  success-rate denominator.
- LIBERO-X (π0.5 only) evaluates a standard-LIBERO-fine-tuned checkpoint
  zero-shot on new tasks. Its 7.85% is not comparable to an in-domain
  LIBERO-X training result.
- OpenVLA and ACT have **no single unified checkpoint** — each of the 4
  LIBERO suites (`spatial`/`object`/`goal`/`10`) is served by its own
  independently fine-tuned (OpenVLA) or from-scratch-trained (ACT)
  checkpoint. Their LIBERO results are same-distribution; their
  LIBERO-Plus/LIBERO-Pro results are zero-shot transfer, exactly like
  π0.5's own LIBERO-Plus/LIBERO-Pro/LIBERO-X results. This is not an
  architecture-fair comparison across policies — see "Cross-policy
  comparison" above for the specific confounds (checkpoint count, action
  horizon, model scale, language conditioning).
- OpenVLA's 0/2,000 (100%) failure on LIBERO-Pro's `swap(position)`
  perturbation is a verified genuine policy failure (no infra errors, every
  episode ran to `max_steps`), not a data or harness bug.
- ACT has no language conditioning by design (a deliberate choice in this
  campaign's harness, not an oversight) — its `Language Instructions`
  (LIBERO-Plus) and `lan(semantic)` (LIBERO-Pro) perturbation scores reflect
  its baseline visual performance on those scenes, not language robustness.
- The recorded π0.5 runs used a pinned upstream `openpi` commit plus local
  evaluation patches; the recorded OpenVLA/ACT runs used an internal harness
  fork with no git history (never committed). The reports preserve the exact
  provenance caveat for each.

See [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md),
[`analysis-report/report_zh.md`](analysis-report/report_zh.md), and the
benchmark-specific `EVAL_REPORT.md` files for details.

## License

Original harness and analysis code is released under MIT. Benchmark code,
assets, model weights, and generated videos remain subject to their original
owners' licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
