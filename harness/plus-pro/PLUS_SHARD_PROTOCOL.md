# Plus multi-GPU full-evaluation protocol

The official 10,030-episode Plus matrix is partitioned by suite after full
inventory expansion and validation:

- GPU 0 / port 8130: `libero_10`, then `libero_spatial` (4,921 planned)
- GPU 2 / port 8132: `libero_goal` (2,591 planned)
- GPU 3 / port 8133: `libero_object` (2,518 planned)

The partitions have pairwise-disjoint episode IDs and their union is the full
10,030-episode matrix. Each shard writes its own JSONL, summaries, manifests,
logs, runtime directory, and videos. Final merge must reject duplicate episode
or attempt IDs, require the exact four suite counts, and verify every video.

Environment randomness remains episode-local: every environment is seeded with
7 when it is created. Policy sampling is process-local: OpenPI initializes each
policy server at JAX key 0 and advances it per inference. Therefore multi-server
execution uses an independent reproducible policy RNG stream per shard and is
not bitwise equivalent to one continuous single-server RNG stream. Final
provenance must record `policy_rng_initial_key=0` and
`policy_rng_scope=server_process/per-suite`.

The first 2,141 completed `libero_10` outcomes were preserved from the original
GPU 0 single-server run. GPU 0 then resumed the authoritative
`libero_10`/`libero_spatial` shard; GPU 2 and GPU 3 started fresh suite shards.
The separately running PRO evaluation on GPU 1 is unchanged.
