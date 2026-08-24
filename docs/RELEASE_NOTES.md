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
uploadable inventory contains 56,639 real MP4 files. The preparation script
records missing episodes with `video_available=false`; it never generates a
placeholder video.
