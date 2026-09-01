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
- rollout-videos
pretty_name: pi0.5 LIBERO evaluation rollouts
---

# pi0.5 LIBERO evaluation rollouts

**Evaluation code, results, and reports:** [GitHub repository](https://github.com/ctegdf/pi05-libero-eval)

Rendered rollout videos for the public `pi05_libero` evaluation archive. The
[companion GitHub repository](https://github.com/ctegdf/pi05-libero-eval)
contains the evaluation harness, summaries, and episode metadata.

## Coverage

- LIBERO official and smoke controls
- LIBERO-Plus full and smoke runs
- LIBERO-Pro full and smoke runs (the available 8,000-episode matrix)
- LIBERO-X LEVEL1-5 full and smoke runs

The archive contains 56,639 available MP4 files. Files are grouped below `videos/` by
benchmark and phase. `manifests/video-manifest.jsonl` records the byte size and
SHA-256 for every file.

| Benchmark | Full | Smoke | MP4s |
| --- | ---: | ---: | ---: |
| LIBERO official | 2,000 | 8 | 2,008 |
| LIBERO Base assets | 2,000 | 8 | 2,008 |
| LIBERO-Plus | 10,030 | 28 | 10,058 |
| LIBERO-Pro | 8,000 | 16 | 8,016 |
| LIBERO-X | 34,520 | 29 | 34,549 |
| **Total** |  |  | **56,639** |

All 6,000 LIBERO-X LEVEL1 episode records resolve to non-empty source MP4s in
the current archive. Eight LIBERO-Plus smoke records are explicitly marked
`video_available=false` because their run status is `not_recorded`. Missing
records are retained in companion metadata rather than represented by
fabricated files.

## Interpretation

LIBERO-Pro has 2,000 unavailable `env` episodes; they are N/A, not failures.
LIBERO-X is a zero-shot transfer evaluation of a checkpoint fine-tuned on
standard LIBERO, not an in-domain LIBERO-X training result.

## Licensing

These are generated rollouts, not a redistribution of model weights or
simulator assets. Users must follow the upstream licenses and citation terms
for OpenPI, LIBERO, LIBERO-Plus, LIBERO-Pro, and LIBERO-X before using or
redistributing the videos.
