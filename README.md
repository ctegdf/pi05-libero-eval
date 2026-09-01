# pi0.5 on LIBERO: auditable evaluation archive

**Rollout videos:** [Hugging Face dataset](https://huggingface.co/datasets/ctegdf/pi05-libero-rollouts)

Reproducible evaluation harnesses, aggregate results, and sanitized episode
metadata for the official `pi05_libero` checkpoint on four LIBERO-family
benchmarks:

| Benchmark | Result | Scope |
| --- | ---: | --- |
| LIBERO | 1,942 / 2,000 (97.10%) | official four-suite protocol |
| LIBERO-Plus | 8,390 / 10,030 (83.65%) | full generated matrix |
| LIBERO-Pro | 4,703 / 8,000 (58.79%) | 16/20 cells available; 2,000 episodes N/A |
| LIBERO-X | 2,711 / 34,520 (7.85%) | zero-shot transfer, not paper reproduction |

The 56,639 uploadable MP4s are grouped as follows:

| Benchmark | Full | Smoke | MP4s |
| --- | ---: | ---: | ---: |
| LIBERO official | 2,000 | 8 | 2,008 |
| LIBERO Base assets | 2,000 | 8 | 2,008 |
| LIBERO-Plus | 10,030 | 28 | 10,058 |
| LIBERO-Pro | 8,000 | 16 | 8,016 |
| LIBERO-X | 34,520 | 29 | 34,549 |
| **Total** |  |  | **56,639** |

The episode metadata contains 56,647 records. Eight LIBERO-Plus smoke
records have `video_status=not_recorded`, so they are retained as metadata but
do not have an MP4 upload.

The complete rendered rollout videos are published separately in the
[Hugging Face dataset](https://huggingface.co/datasets/ctegdf/pi05-libero-rollouts).
The GitHub repository intentionally contains no checkpoints, videos, runtime
directories, remote logs, or simulator assets.

## Contents

- `harness/`: standard-library inventory, launch, integrity, and unit-test code
- `results/`: summaries and sanitized episode metadata; video paths are relative
- `analysis-report/`: derived tables, figures, and the Chinese/HTML reports
- `scripts/upload_videos_to_hf.py`: upload an isolated local video tree to a
  Hugging Face dataset in resumable batches
- `scripts/stage_videos.py`: create that isolated video tree from the source
  archive without touching the original archive

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
