_Run 2026-11-25T071207Z · 110 queries · 0 queries skipped (target expired)_

### synthetic (80 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.738 | 0.487 | 0.549 | 68 | 310 |
| vector [bge-small] | 0.787 | 0.702 | 0.723 | 16 | 53 |
| vector [minilm] | 0.650 | 0.454 | 0.501 | 14 | 19 |
| hybrid [bge-small] | 0.838 | 0.663 | 0.706 | 59 | 222 |
| hybrid [minilm] | 0.787 | 0.576 | 0.627 | 57 | 163 |
| hybrid_rerank [bge-small] | 0.912 | 0.841 | 0.858 | 106 | 258 |
| hybrid_rerank [minilm] | 0.850 | 0.786 | 0.802 | 111 | 212 |

### judged (30 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.415 | 0.796 | 0.595 | 19 | 110 |
| vector [bge-small] | 0.604 | 0.978 | 0.803 | 14 | 19 |
| vector [minilm] | 0.326 | 0.859 | 0.464 | 10 | 12 |
| hybrid [bge-small] | 0.557 | 0.967 | 0.785 | 17 | 107 |
| hybrid [minilm] | 0.441 | 0.933 | 0.637 | 17 | 107 |
| hybrid_rerank [bge-small] | 0.589 | 0.983 | 0.819 | 61 | 146 |
| hybrid_rerank [minilm] | 0.486 | 0.983 | 0.722 | 66 | 156 |

### synthetic MRR@10 by query style

| Config | keyword | natural | skills |
|---|---|---|---|
| keyword | 0.483 | 0.555 | 0.422 |
| vector [bge-small] | 0.617 | 0.665 | 0.827 |
| vector [minilm] | 0.309 | 0.505 | 0.551 |
| hybrid [bge-small] | 0.600 | 0.685 | 0.705 |
| hybrid [minilm] | 0.460 | 0.617 | 0.654 |
| hybrid_rerank [bge-small] | 0.769 | 0.880 | 0.875 |
| hybrid_rerank [minilm] | 0.716 | 0.846 | 0.796 |
