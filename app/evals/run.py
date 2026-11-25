"""Run the retrieval eval: every query x every mode (x every embedding model).

  python -m app.evals.run                                  # bge-small, all modes, all sets
  python -m app.evals.run --models bge-small minilm gemini
  python -m app.evals.run --modes keyword hybrid --limit 10   # quick smoke run
  python -m app.evals.run --ci             # CI corpus: writes ci_latest.md, fails under evals/ci_thresholds.json

Writes evals/results/<timestamp>.json (per-query detail, for error analysis) and
evals/results/latest.md (the summary table for the README).
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from app.db import make_pool, run
from app.evals.dataset import HANDLABELED_PATH, JUDGED_PATH, RESULTS_DIR, SYNTHETIC_PATH, EvalQuery, key_to_id_map, load
from app.evals.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from app.search.models import SearchMode, SearchRequest
from app.search.service import search

K = 10
THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "evals" / "ci_thresholds.json"


def check_thresholds(summary: dict, thresholds: dict[str, dict[str, float]]) -> list[str]:
    """thresholds: {"hybrid_rerank [bge-small]|synthetic": {"ndcg@10": 0.8}, ...} -> list of failures."""
    failures = []
    for group, mins in thresholds.items():
        got = summary.get(group)
        if got is None:
            failures.append(f"{group}: missing from results")
            continue
        for metric, floor in mins.items():
            if got[metric] < floor:
                failures.append(f"{group} {metric} = {got[metric]:.3f} < {floor:.3f}")
    return failures


async def _coverage(pool, model: str) -> tuple[int, int]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT (SELECT count(*) FROM jobs WHERE is_active) AS active,
                      (SELECT count(*) FROM job_embeddings e JOIN jobs j ON j.id = e.job_id
                       WHERE j.is_active AND e.model = %s AND e.content_hash = j.content_hash) AS embedded""",
            (model,),
        )
        r = await cur.fetchone()
        return r["embedded"], r["active"]


async def evaluate(models: list[str], modes: list[SearchMode], limit: int | None) -> dict:
    queries: list[EvalQuery] = load(SYNTHETIC_PATH) + load(JUDGED_PATH) + [q for q in load(HANDLABELED_PATH) if q.qrels]
    if limit:
        queries = queries[:limit]
    pool = make_pool()
    await pool.open()
    try:
        for m in models:
            done, active = await _coverage(pool, m)
            if done < active:
                print(f"WARNING: {m} has embeddings for {done}/{active} active jobs; vector results will be partial")

        id_by_key = await key_to_id_map(pool, {k for q in queries for k in q.qrels})
        key_by_id = {v: k for k, v in id_by_key.items()}
        # A synthetic query whose target job expired has no answer anymore: skip it.
        usable = [q for q in queries if any(g > 0 and k in id_by_key for k, g in q.qrels.items())]
        skipped = len(queries) - len(usable)

        configs = []
        for mode in modes:
            if mode is SearchMode.keyword:
                configs.append((mode, None))
            else:
                configs += [(mode, m) for m in models]

        per_query = []
        for mode, model in configs:
            label = mode.value if model is None else f"{mode.value} [{model}]"
            print(f"running {label} on {len(usable)} queries")
            for q in usable:
                resp = await search(pool, SearchRequest(query=q.query, mode=mode, k=K, model=model, filters=q.filters))
                # Unknown ids (e.g. jobs inserted after labelling) map to themselves: non-relevant.
                ranked = [key_by_id.get(h.job_id, f"id:{h.job_id}") for h in resp.results]
                per_query.append(
                    {
                        "config": label, "qid": q.qid, "set": q.set, "style": q.style, "query": q.query,
                        "recall": recall_at_k(ranked, q.qrels, K), "mrr": mrr_at_k(ranked, q.qrels, K),
                        "ndcg": ndcg_at_k(ranked, q.qrels, K), "latency_ms": resp.timings_ms["total"],
                        "top": ranked[:K],
                        "target_rank": next(
                            (i for i, d in enumerate(ranked, 1) if q.qrels.get(d, 0) > 0), None
                        ),
                    }
                )  # fmt: skip
    finally:
        await pool.close()
    return {"per_query": per_query, "skipped_expired": skipped, "n_queries": len(usable)}


def summarise(per_query: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in per_query:
        groups[(r["config"], r["set"])].append(r)
        groups[(r["config"], "all")].append(r)
        if r["style"]:
            groups[(r["config"], f"style:{r['style']}")].append(r)
    return {
        f"{cfg}|{s}": {
            "n": len(rs),
            "recall@10": mean(r["recall"] for r in rs),
            "mrr@10": mean(r["mrr"] for r in rs),
            "ndcg@10": mean(r["ndcg"] for r in rs),
            "p50_ms": percentile([r["latency_ms"] for r in rs], 50),
            "p95_ms": percentile([r["latency_ms"] for r in rs], 95),
        }
        for (cfg, s), rs in groups.items()
    }


def to_markdown(summary: dict, meta: dict) -> str:
    configs = list(dict.fromkeys(k.split("|")[0] for k in summary))
    sets = [s for s in ("synthetic", "judged", "handlabeled") if any(k.endswith(f"|{s}") for k in summary)]
    out = [f"_Run {meta['when']} · {meta['n_queries']} queries · {meta['active_note']}_\n"]
    for s in sets:
        n = next(v["n"] for k, v in summary.items() if k.endswith(f"|{s}"))
        out += [f"### {s} ({n} queries)\n", "| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
                "|---|---|---|---|---|---|"]  # fmt: skip
        for c in configs:
            v = summary.get(f"{c}|{s}")
            if v:
                out.append(
                    f"| {c} | {v['recall@10']:.3f} | {v['mrr@10']:.3f} | {v['ndcg@10']:.3f} "
                    f"| {v['p50_ms']:.0f} | {v['p95_ms']:.0f} |"
                )
        out.append("")
    styles = sorted({k.split("|")[1] for k in summary if "|style:" in k})
    if styles:
        out += ["### synthetic MRR@10 by query style\n", "| Config | " + " | ".join(s[6:] for s in styles) + " |",
                "|---|" + "---|" * len(styles)]  # fmt: skip
        for c in configs:
            cells = [f"{summary[f'{c}|{s}']['mrr@10']:.3f}" if f"{c}|{s}" in summary else "-" for s in styles]
            out.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["bge-small"])
    p.add_argument("--modes", nargs="+", default=[m.value for m in SearchMode], choices=[m.value for m in SearchMode])
    p.add_argument("--limit", type=int)
    p.add_argument("--ci", action="store_true", help="CI corpus run: ci_latest.md + threshold gate")
    a = p.parse_args()

    res = run(evaluate(a.models, [SearchMode(m) for m in a.modes], a.limit))
    if not res["per_query"]:
        raise SystemExit("no usable queries; run `python -m app.evals.generate` first")
    summary = summarise(res["per_query"])
    when = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    meta = {
        "when": when, "n_queries": res["n_queries"], "models": a.models, "modes": a.modes,
        "active_note": f"{res['skipped_expired']} queries skipped (target expired)",
    }  # fmt: skip
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{when}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "per_query": res["per_query"]}, indent=1), encoding="utf-8"
    )
    md = to_markdown(summary, meta)
    if not a.limit:  # don't let a smoke run overwrite the headline table
        (RESULTS_DIR / ("ci_latest.md" if a.ci else "latest.md")).write_text(md, encoding="utf-8")
    print("\n" + md)
    if a.ci:
        failures = check_thresholds(summary, json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))["retrieval"])
        for f in failures:
            print(f"REGRESSION: {f}")
        if failures:
            sys.exit(1)
        print("all retrieval thresholds met")


if __name__ == "__main__":
    main()
