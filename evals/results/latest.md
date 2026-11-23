_Run 2026-11-23T160746Z · 110 queries · 0 queries skipped (target expired)_

### synthetic (80 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.738 | 0.483 | 0.545 | 54 | 150 |
| vector [bge-small] | 0.787 | 0.702 | 0.723 | 13 | 18 |
| hybrid [bge-small] | 0.838 | 0.670 | 0.711 | 57 | 166 |
| hybrid_rerank [bge-small] | 0.912 | 0.841 | 0.858 | 95 | 249 |

### judged (30 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.416 | 0.796 | 0.597 | 15 | 99 |
| vector [bge-small] | 0.604 | 0.978 | 0.803 | 11 | 13 |
| hybrid [bge-small] | 0.555 | 0.967 | 0.784 | 14 | 98 |
| hybrid_rerank [bge-small] | 0.590 | 0.983 | 0.820 | 49 | 133 |

### synthetic MRR@10 by query style

| Config | keyword | natural | skills |
|---|---|---|---|
| keyword | 0.481 | 0.542 | 0.422 |
| vector [bge-small] | 0.617 | 0.665 | 0.827 |
| hybrid [bge-small] | 0.600 | 0.704 | 0.706 |
| hybrid_rerank [bge-small] | 0.769 | 0.880 | 0.875 |
