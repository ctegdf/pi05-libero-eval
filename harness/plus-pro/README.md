# π0.5 LIBERO-Plus / LIBERO-Pro evaluation harness

This directory is a standalone evaluation harness. It does not patch the
OpenPI checkout, either benchmark checkout, or an existing LIBERO home. Every
run creates its own `runtime/<run-id><private> binds the
policy server to `127.0.0.1`, and only terminates the exact server child it
started.

The only supported policy is the official `pi05_libero` fine-tuned
checkpoint (`pi05_libero` config and
`physical-intelligence/libero` normalization statistics). A checkpoint or
preflight load failure produces diagnostics and no success-rate summary; it
is never converted to 0%.

## Fixed protocol

Both benchmarks use seed 7, 224×224 policy images, five executed actions per
replan, ten initial no-op steps, 7-D finite action validation, and EGL. The
suite horizons are 220 (`libero_spatial`), 280 (`libero_object`), 300
(`libero_goal`), and 520 (`libero_10`). Prompts are parsed from the concrete
BDDL `language_instruction` field (or the legacy in-file `language` field),
never reconstructed from a filename or `task.language`.

LIBERO-Plus uses the official
`libero/libero/benchmark/task_classification.json` as its task inventory. The
four required counts are 2402/2518/2591/2519, totaling exactly 10030. A full
run evaluates each classified task once. Smoke chooses the first available
task in each `(suite, category)` pair, up to four suites × seven categories.
The client also checks that classification IDs and registered-suite BDDL
order agree before rollout. `_language`, `_view`, `_table`, `_tb`, and
`_light` variants fall back to the original init-state file; `_add` and
`_level` variants resolve through `init_files/libero_newobj/<suite>`.

LIBERO-Pro reads only pre-generated BDDL and init-state files. It never calls
`perturbation.create_env`. The complete protocol is five perturbations
(`object`, `swap(position)`, `lan(semantic)`, `task`, `env`) × four suites ×
ten tasks × 50 trials = 10000 episodes. Smoke runs the first task/trial in
each available cell. The launcher builds a per-run symlink merge view; the
snapshot and code checkout remain unchanged.

If an entire Pro cell is absent, the run records it in
`manifests/compatibility-*.json` as `N/A` with its 500 unavailable episodes
and continues all complete cells. For example, a snapshot containing the 16
`lan/object/swap/task` cells but no four `*_env` cells evaluates 8000
episodes and reports 2000 as unavailable—not failures and not 0%. Once all
20 cells are present, strict 10000-episode expansion is automatic. A partial
result has result status `partial_incompatible` and
`protocol_applicable=false`.

## Environments and data

Create separate server and Python 3.8 client virtual environments outside
the old LIBERO environment. Install each official repository's locked
requirements in those environments; this harness introduces no dependency
versions of its own. Expected inputs are:

- an independent LIBERO-Plus or LIBERO-Pro code checkout;
- Plus data in that checkout, or the official Pro pre-generated snapshot;
- common LIBERO assets staged outside the checkouts (default:
  `data/libero-plus/assets`);
- the Plus ImageMagick runtime at
  `<plus-repo>/runtime/imagemagick/{prefix,root}`;
- a read-only OpenPI checkout and the official checkpoint directory.

For Plus the launcher exports `MAGICK_HOME=<runtime>/prefix`, prepends
`<runtime>/prefix/lib` to `LD_LIBRARY_PATH`, and sets
`MAGICK_CONFIGURE_PATH=<runtime>/root/etc/ImageMagick-6`. For both clients it
sets `MUJOCO_GL=egl`; the server is restricted with
`CUDA_VISIBLE_DEVICES=<gpu>` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`.

## Commands

Run an actual WebSocket inference before any matrix:

```bash
python run_pi05_eval.py \
  --benchmark plus --phase preflight \
  --benchmark-repo /srv/pi05-eval/repos/LIBERO-Plus \
  --openpi-repo /srv/openpi \
  --checkpoint-dir /srv/checkpoints \
  --client-python /srv/pi05-eval/venvs/plus/bin/python \
  --server-python /srv/openpi/.venv/bin/python \
  --gpu-id 4 --port 8130 \
  --output-dir /srv/pi05-eval/results/plus-preflight
```

Smoke and full use the same interface:

```bash
python run_pi05_eval.py \
  --benchmark plus --phase full \
  --benchmark-repo /srv/pi05-eval/repos/LIBERO-Plus \
  --openpi-repo /srv/openpi --checkpoint-dir /srv/checkpoints \
  --client-python /srv/pi05-eval/venvs/plus/bin/python \
  --server-python /srv/openpi/.venv/bin/python \
  --gpu-id 4 --port 8130 --output-dir /srv/pi05-eval/results/plus-full \
  --resume

python run_pi05_eval.py \
  --benchmark pro --phase full \
  --benchmark-repo /srv/pi05-eval/repos/LIBERO-Pro \
  --benchmark-data-dir /srv/pi05-eval/datasets/LIBERO-Pro \
  --assets-dir /srv/pi05-eval/data/libero-plus/assets \
  --openpi-repo /srv/openpi --checkpoint-dir /srv/checkpoints \
  --client-python /srv/pi05-eval/venvs/pro/bin/python \
  --server-python /srv/openpi/.venv/bin/python \
  --gpu-id 4 --port 8131 --output-dir /srv/pi05-eval/results/pro-full \
  --resume
```

`--checkpoint-dir` may name the `pi05_libero` directory directly or its
parent. Resume skips only episodes with a persisted `success` or `failure`;
environment, connection, policy-runtime, and checkpoint errors remain
retryable. Attempt IDs remain unique across reruns.

## Artifacts and verification

Each evaluated attempt is appended and fsynced to `episodes.jsonl`. Success
and ordinary policy-failure episodes require an MP4. `summary.json` and
`summary.csv` aggregate suite/category/difficulty/perturbation while
excluding infrastructure errors from success denominators. Pro N/A cells
also appear explicitly in both formats.

Per-run files under `manifests/` include environment/package/GPU inventory,
three Git states, source paths and SHA-256s, checkpoint/assets provenance,
preflight evidence, compatibility, server diagnostics, and a terminal result
manifest. Logs are unique per run and are never overwritten.

Static verification requires no simulator:

```bash
python -m unittest -v test_pi05_eval.py
python -m py_compile pi05_eval_support.py pi05_eval_server.py \
  pi05_eval_client.py run_pi05_eval.py test_pi05_eval.py
```

Full multi-GPU completion uses `finalize_plus_shards.sh` followed by
`finalize_all_results.sh`; `recover_finalizers.sh` safely retries either
stage if its original waiter exits without a passed artifact. The waiters
gate on completed summaries and manifests rather than treating process exit
as proof of completion.

The final audit independently reconstructs both benchmark matrices and
requires exact episode/attempt identities, exactly one policy outcome per
episode, and an exact set match between record video paths and the non-empty
`videos/*.mp4` inventory. It also binds all per-run manifests to one run ID,
recomputes source inventories against the current benchmark files, verifies
the fixed OpenPI and benchmark commits, and checks the official checkpoint's
15-file parameter tree (`12,439,083,567` bytes, tree SHA-256
`b5d2c61bb555413cba73b66b6876c5e895e9f6ea69e6eeb9827ea9ea7339fa45`).
That tree was byte-hashed independently in both the staging directory and the
official OpenPI cache, with an exact match. `final_audit_time_harness_sources_sha256`
is deliberately labeled as the source fingerprint at audit time; because the
harness was uncommitted during rollout, it is not claimed as a retrospective
runtime-source hash.

The same report re-audits the earlier LIBERO controls from their raw records,
not just their summaries. Each control must contain the exact four-suite ×
ten-task × fifty-trial grid with seed 7, 224-pixel inputs, five-step replans,
the suite-specific horizons, and one non-empty video per episode. The official
control is bound to the 15-file `pi05_libero` parameter tree above. The Base
control is separately bound to the 20-file `pi05_base` tree (12,441,721,931
bytes; SHA-256
`7ed18c089c75ccd1b2aa1506045a575177a4b81691a38d4687da0715fb7ba0cb`)
and to a symlink-only runtime view whose assets and normalization statistics
resolve to the official LIBERO checkpoint. Both old runs also require their
matching Git, environment, preflight, startup, checkpoint, and terminal result
manifests.

After the remote audit passes, `sync_final_archive.sh` performs the final
incremental download without embedding credentials. Set `SSHPASS` in the
invoking process and run the script; it refuses an unpassed remote audit,
avoids the checkpoint staging tree, preserves existing local files, verifies
all regular-file SHA-256 values, and checks that merged video links resolve.
