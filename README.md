# pi0.5 on LIBERO: auditable evaluation archive

[Rollout videos](https://huggingface.co/datasets/ctegdf/pi05-libero-rollouts) |
[Analysis report](analysis-report/report_zh.md) |
[Result protocol](docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md) |
[Release notes](docs/RELEASE_NOTES.md)

Reproducible evaluation harnesses, aggregate results, and sanitized episode
metadata for the official `pi05_libero` checkpoint on four LIBERO-family
benchmarks, plus a comparison campaign evaluating **OpenVLA-7B** and **ACT**
(both trained/fine-tuned per-suite, no single unified checkpoint) on LIBERO,
LIBERO-Plus, and LIBERO-Pro under the same protocol. Evaluation code and
metadata live in this repository; rollout videos live in the companion
Hugging Face dataset(s).

## Evaluation results

### pi0.5 (`pi05_libero`)

| Benchmark | Successes | Evaluated episodes | Success rate | Scope |
| --- | ---: | ---: | ---: | --- |
| LIBERO | 1,942 | 2,000 | 97.10% | Official four-suite protocol |
| LIBERO-Plus | 8,390 | 10,030 | 83.65% | Full generated matrix |
| LIBERO-Pro | 4,703 | 8,000 | 58.79% | 16/20 cells available; 2,000 episodes N/A |
| LIBERO-X | 2,711 | 34,520 | 7.85% | Zero-shot transfer, not paper reproduction |

### OpenVLA-7B vs ACT (per-suite checkpoints, protocol v1)

| Benchmark | π0.5 | OpenVLA | ACT | Scope |
| --- | ---: | ---: | ---: | --- |
| LIBERO | 97.10% | 69.15% | 24.90% | Same-distribution (2,000 episodes) |
| LIBERO-Plus | 83.65% | 25.26% | 13.06% | Zero-shot (10,030 episodes) |
| LIBERO-Pro | 58.79% | 33.84% | 11.11% | Zero-shot; 16/20 cells available (8,000 episodes) |

Full breakdown, per-suite/per-perturbation tables, and the key findings
(OpenVLA's 0/2,000 catastrophic failure on the `swap(position)` perturbation;
ACT's language-blind architecture) are in
[`analysis-report/cross_policy_report_zh.md`](analysis-report/cross_policy_report_zh.md)
(Chinese) and the per-model `docs/*-EVAL_REPORT.md` files linked below. This
is not an architecture-fair comparison — see the report's caveats section
before citing these numbers as an architecture verdict.

## Video inventory

The complete MP4 archive is published in
[Hugging Face](https://huggingface.co/datasets/ctegdf/pi05-libero-rollouts).

| Benchmark | Full | Smoke | MP4s |
| --- | ---: | ---: | ---: |
| LIBERO official | 2,000 | 8 | 2,008 |
| LIBERO Base assets | 2,000 | 8 | 2,008 |
| LIBERO-Plus | 10,030 | 28 | 10,058 |
| LIBERO-Pro | 8,000 | 16 | 8,016 |
| LIBERO-X | 34,520 | 29 | 34,549 |
| **Total** | **56,550** | **89** | **56,639** |

| Inventory item | Count | Reconciliation |
| --- | ---: | --- |
| Episode metadata records | 56,647 | All retained episode records |
| Available MP4 files | 56,639 | Every file is indexed by size and SHA-256 |
| Metadata-only episodes | 8 | LIBERO-Plus smoke, `video_status=not_recorded` |

The GitHub repository intentionally contains no checkpoints, videos, runtime
directories, remote logs, or simulator assets.

## Contents

| Path | Purpose |
| --- | --- |
| [`harness/`](harness) | Evaluation launchers, inventory checks, integrity checks, and unit tests |
| [`results/`](results) | Aggregate summaries and sanitized episode metadata with relative video paths |
| [`analysis-report/`](analysis-report) | Derived tables, figures, Chinese report, and rendered HTML report |
| [`docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md`](docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md) | Stable result contract for ACT and OpenVLA evaluations |
| [`scripts/export_to_protocol.py`](scripts/export_to_protocol.py) | Export a new model run into the shared result layout |
| [`scripts/validate_run.py`](scripts/validate_run.py) | Validate a published run and report every protocol violation |
| [`scripts/stage_videos.py`](scripts/stage_videos.py) | Build an isolated video tree without modifying the source archive |
| [`scripts/upload_videos_to_hf.py`](scripts/upload_videos_to_hf.py) | Validate and upload the video inventory in resumable batches |

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
python3 analysis-report/generate_report.py
python3 -m unittest discover -s harness/plus-pro -p 'test_*.py' -v
python3 -m py_compile harness/libero-x/*.py
```

The checked-in report is already generated. Regenerating it requires the
original evaluation archive mounted beside this checkout, because the
generator validates raw episode/video paths and the final audit; the GitHub
checkout intentionally does not redistribute those large source directories.

Running a simulator evaluation requires the pinned upstream repositories,
benchmark assets, a compatible CUDA/MuJoCo environment, and the checkpoint.
The benchmark projects and checkpoint are not redistributed here.

## Video archive

The upload helper requires a local source archive and an authenticated Hugging
Face account:

```bash
python3 -m pip install huggingface_hub
hf auth login
python3 scripts/upload_videos_to_hf.py \
  --source-root /path/to/evaluation-archive \
  --repo-id YOUR_ACCOUNT/pi05-libero-rollouts \
  --commit-message 'Upload complete evaluation rollouts'
```

The script validates the expected 56,639 MP4 files and uploads them under
`videos/{official,base-libero-assets,plus,pro,libero-x}/`. It also uploads a
manifest and the dataset card. Use a dataset repository, not a model
repository. To keep the GitHub project and large files separate, stage the
videos into a second directory first:

```bash
python3 scripts/stage_videos.py \
  --source-root /path/to/evaluation-archive \
  --destination-root /path/to/pi05-libero-video-release
```

The current source archive contains 6,000 LIBERO-X LEVEL1 MP4s and all 6,000
LEVEL1 episode records resolve to non-empty files. Eight LIBERO-Plus smoke
records have `video_status=not_recorded` and are retained as unavailable
metadata. The uploadable inventory therefore contains 56,639 real MP4s. The
preparation script records any missing episode as `video_available=false`
instead of fabricating a placeholder.

## Interpretation limits

- LIBERO-Pro has four unavailable official `env` cells. They are N/A, not
  failures, and are excluded from the 58.79% denominator.
- LIBERO-X evaluates a standard-LIBERO-finetuned checkpoint zero-shot on new
  tasks. Its 7.85% is not comparable to an in-domain LIBERO-X training result.
- The recorded runs used a pinned upstream commit plus local evaluation
  patches. The reports preserve the exact provenance caveat.

See [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md),
[`analysis-report/report_zh.md`](analysis-report/report_zh.md), and the
benchmark-specific `EVAL_REPORT.md` files for details.

## License

Original harness and analysis code is released under MIT. Benchmark code,
assets, model weights, and generated videos remain subject to their original
owners' licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
