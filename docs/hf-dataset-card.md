---
language:
- en
license: other
task_categories:
- robotics
tags:
- LIBERO
- robot-learning
- pi05
- openvla
- act
- rollout-videos
pretty_name: pi0.5 / OpenVLA / ACT LIBERO evaluation rollouts
---

# pi0.5 / OpenVLA / ACT LIBERO evaluation rollouts

**Evaluation code, results, and reports:** [GitHub repository](https://github.com/ctegdf/pi05-libero-eval)

Rendered rollout videos for three policies evaluated under the same LIBERO /
LIBERO-Plus / LIBERO-Pro protocol: **π0.5** (`pi05_libero`, single unified
checkpoint), **OpenVLA-7B**, and **ACT** (both OpenVLA and ACT use 4
independently fine-tuned/trained per-suite checkpoints, not one unified
weight). The [companion GitHub repository](https://github.com/ctegdf/pi05-libero-eval)
contains the evaluation harness, summaries, and episode metadata; OpenVLA/ACT
results additionally follow the versioned
[`MODEL_EVAL_RESULT_PROTOCOL_V1`](https://github.com/ctegdf/pi05-libero-eval/blob/main/docs/MODEL_EVAL_RESULT_PROTOCOL_V1.md)
layout under `results/runs/<model_id>/<benchmark_id>/<run_id>/`.

## Coverage

**π0.5** (`videos/{official,base-libero-assets,plus,pro,libero-x}/`):
- LIBERO official and smoke controls
- LIBERO-Plus full and smoke runs
- LIBERO-Pro full and smoke runs (the available 8,000-episode matrix)
- LIBERO-X LEVEL1-5 full and smoke runs

**OpenVLA / ACT** (`videos/{openvla,act}/<benchmark>/<run_id>/`, protocol v1 layout):
- LIBERO (`libero`), LIBERO-Plus (`libero-plus`), LIBERO-Pro (`libero-pro`) — `full` phase only, 4 per-suite checkpoints each

| Source | Benchmark | Full | Smoke | MP4s |
| --- | --- | ---: | ---: | ---: |
| π0.5 | LIBERO official | 2,000 | 8 | 2,008 |
| π0.5 | LIBERO Base assets | 2,000 | 8 | 2,008 |
| π0.5 | LIBERO-Plus | 10,030 | 28 | 10,058 |
| π0.5 | LIBERO-Pro | 8,000 | 16 | 8,016 |
| π0.5 | LIBERO-X | 34,520 | 29 | 34,549 |
| OpenVLA | LIBERO | 2,000 | — | 2,000 |
| OpenVLA | LIBERO-Plus | 10,030 | — | 10,030 |
| OpenVLA | LIBERO-Pro | 8,000 | — | 8,000 |
| ACT | LIBERO | 2,000 | — | 2,000 |
| ACT | LIBERO-Plus | 10,030 | — | 10,030 |
| ACT | LIBERO-Pro | 8,000 | — | 8,000 |
| **Total** |  |  |  | **96,699** |

`manifests/video-manifest.jsonl` (π0.5) and `manifests/video-manifest.jsonl`
(shared, appended by the OpenVLA/ACT uploader) record the byte size and
SHA-256 for every file.

All 6,000 LIBERO-X LEVEL1 episode records resolve to non-empty source MP4s in
the current archive. Eight LIBERO-Plus smoke records (π0.5 only) are
explicitly marked `video_available=false` because their run status is
`not_recorded`. Every OpenVLA/ACT `full`-phase episode has a matched video
(`videos_available == total` for all 6 runs, per each run's `run.json`).
Missing records are retained in companion metadata rather than represented by
fabricated files.

## Interpretation

LIBERO-Pro has 2,000 unavailable `env` episodes per policy; they are N/A, not
failures. LIBERO-X (π0.5 only) is a zero-shot transfer evaluation of a
checkpoint fine-tuned on standard LIBERO, not an in-domain LIBERO-X training
result. OpenVLA/ACT's LIBERO-Plus and LIBERO-Pro runs are also zero-shot
transfer (their checkpoints were fine-tuned/trained only on standard LIBERO);
their standard-LIBERO runs are same-distribution. See the GitHub repository's
`docs/*-EVAL_REPORT.md` and `analysis-report/cross_policy_report_zh.md` for
the full per-model analysis, including OpenVLA's 0/2,000 catastrophic failure
on the LIBERO-Pro `swap(position)` perturbation and ACT's language-blind
architecture (documented so its `Language Instructions`/`lan(semantic)`
perturbation scores aren't misread as language robustness).

## Licensing

These are generated rollouts, not a redistribution of model weights or
simulator assets. Users must follow the upstream licenses and citation terms
for OpenPI, OpenVLA, `lerobot`/ACT, LIBERO, LIBERO-Plus, LIBERO-Pro, and
LIBERO-X before using or redistributing the videos.
