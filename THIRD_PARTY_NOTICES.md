# Third-party notices

This repository contains evaluation code and derived results. It does not
redistribute the upstream model checkpoint or simulator assets.

- OpenPI / `pi05_libero`: use the license and terms published by
  Physical Intelligence at the pinned upstream commit recorded in the reports.
- OpenVLA / `openvla-7b-finetuned-libero-*`: use the license and terms
  published by the OpenVLA project (Stanford/UC Berkeley et al.) for the
  model weights and code; the 4 per-suite checkpoints used here are the
  official Hugging Face releases, not custom fine-tunes.
- ACT / `lerobot`: use the license and terms published by Hugging Face's
  `lerobot` project for the `ACTConfig`/`modeling_act` implementation used to
  train the per-suite ACT policies from scratch (this repository does not
  redistribute those trained weights).
- LIBERO, LIBERO-Plus, LIBERO-Pro, and LIBERO-X: obtain each project from its
  official repository and follow its license, dataset terms, and citation
  requirements.
- The Hugging Face video dataset contains generated rollouts from those
  benchmarks. Publishing the videos does not grant rights to redistribute the
  underlying benchmark assets or model weights.

Before making the Hugging Face dataset public, verify that every benchmark
owner permits redistribution of rendered videos and that any required paper
citation is included in the dataset card.
