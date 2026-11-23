_Run 2026-11-23T152707Z · 80 queries · 0 queries skipped (target expired)_

### synthetic (80 queries)

| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| keyword | 0.300 | 0.129 | 0.169 | 198 | 427 |
| vector [bge-small] | 0.787 | 0.702 | 0.723 | 13 | 42 |
| hybrid [bge-small] | 0.662 | 0.399 | 0.461 | 199 | 378 |
| hybrid_rerank [bge-small] | 0.875 | 0.814 | 0.829 | 249 | 466 |

### synthetic MRR@10 by query style

| Config | keyword | natural | skills |
|---|---|---|---|
| keyword | 0.149 | 0.102 | 0.138 |
| vector [bge-small] | 0.617 | 0.665 | 0.827 |
| hybrid [bge-small] | 0.412 | 0.381 | 0.404 |
| hybrid_rerank [bge-small] | 0.759 | 0.846 | 0.837 |
