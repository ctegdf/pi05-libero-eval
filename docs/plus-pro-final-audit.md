# OpenPI LIBERO evaluation audit

Status: **passed**

| Benchmark | Protocol | Status | Evaluated / Protocol | Successes | Failures | Success rate | Infra errors | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LIBERO | pi05_libero | complete | 2000 / 2000 | 1942 | 58 | 97.10% | 0 | official threshold passed |
| LIBERO | pi05_base/LIBERO-assets | complete | 2000 / 2000 | 0 | 2000 | 0.00% | 0 | no success-rate threshold |
| LIBERO | pi05_base/native-assets | not_applicable | 0 / 2000 | N/A | N/A | N/A | 0 | required physical-intelligence/libero norm_stats absent; not a zero-success result |
| LIBERO-Plus | pi05_libero/multi-server-suite-sharded | complete | 10030 / 10030 | 8390 | 1640 | 83.65% | 0 | environment seed 7; policy RNG key 0 scoped independently per suite server |
| LIBERO-Pro | pi05_libero | partial_incompatible | 8000 / 10000 | 4703 | 3297 | 58.79% | 0 | 8000 available episodes complete; 2000 env episodes N/A because official cells are absent |

Official LIBERO macro-suite acceptance: 97.10% >= 93.85% (**passed**).
LIBERO-Pro's unavailable 2,000 environment episodes are N/A and are not converted into failures.
LIBERO-Plus uses environment seed 7; policy sampling uses an independent JAX key-0 stream per suite server.
