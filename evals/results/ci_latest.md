_Run 2026-11-24T060033Z · 110 queries · 0 queries skipped (target expired)_

### synthetic (80 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.850 | 0.690 | 0.729 | 7 | 18 |
| vector [bge-small] | 0.912 | 0.843 | 0.860 | 13 | 21 |
| hybrid [bge-small] | 0.950 | 0.818 | 0.849 | 7 | 19 |
| hybrid_rerank [bge-small] | 0.950 | 0.909 | 0.919 | 43 | 56 |

### judged (30 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.444 | 0.878 | 0.630 | 3 | 10 |
| vector [bge-small] | 0.601 | 0.978 | 0.800 | 12 | 18 |
| hybrid [bge-small] | 0.540 | 0.967 | 0.767 | 5 | 12 |
| hybrid_rerank [bge-small] | 0.575 | 0.983 | 0.814 | 42 | 51 |

### synthetic MRR@10 by query style

| Config | keyword | natural | skills |
|---|---|---|---|
| keyword | 0.607 | 0.759 | 0.704 |
| vector [bge-small] | 0.833 | 0.830 | 0.865 |
| hybrid [bge-small] | 0.754 | 0.839 | 0.864 |
| hybrid_rerank [bge-small] | 0.892 | 0.926 | 0.908 |
