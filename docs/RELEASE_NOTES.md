# Release notes

This is a sanitized public artifact assembled from the completed evaluation
archive. Absolute paths, hostnames, runtime snapshots, credentials, checkpoint
files, and private cleanup backups were intentionally omitted.

The result metadata keeps episode-level outcomes and provenance fields needed
for analysis. Video fields are rewritten to relative paths matching the Hugging
Face dataset layout; they are not local filesystem paths.

The full videos are uploaded separately because GitHub is not an appropriate
large-file distribution service for this archive. The uploader streams from the
original archive, verifies the expected file inventory, and records SHA-256
hashes in the dataset manifest.

The current source archive contains 6,000 LIBERO-X LEVEL1 episode records and
6,000 corresponding non-empty MP4 files. Eight LIBERO-Plus smoke records have
no recording (`video_status=not_recorded`) and remain metadata-only. The
uploadable π0.5 inventory contains 56,639 real MP4 files. The preparation
script records missing episodes with `video_available=false`; it never
generates a placeholder video.

OpenVLA and ACT results follow the same sanitization principles via a
separate tool (`scripts/export_to_protocol.py`), producing
`results/runs/<model_id>/<benchmark_id>/<run_id>/` under
[`MODEL_EVAL_RESULT_PROTOCOL_V1`](MODEL_EVAL_RESULT_PROTOCOL_V1.md). Six
`full`-phase runs are published this way (2,000/10,030/8,000 episodes each
for LIBERO/LIBERO-Plus/LIBERO-Pro, per policy), each with a matched,
non-placeholder video for every episode (0 missing).
