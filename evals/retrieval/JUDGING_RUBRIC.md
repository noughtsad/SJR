# Relevance judging rubric (v1)

Used for `judged.jsonl`: 30 realistic queries, 607 pooled jobs (top 10 from each of keyword, vector, hybrid and hybrid+rerank, de-duplicated).
The judgements were made by Claude (Opus 5.5) as three parallel judges of 10 queries each. Each judge saw the title, company, location, remote flag, extracted seniority and skills, and the first ~600 characters of the description.
Judges did not see which retrieval mode returned a job.

The question for each job: *would a person who typed this query be glad to see this job?*

| Grade | Meaning |
|---|---|
| **2** | Matches everything that matters: role/field, the main skill named, and any explicit constraint (remote, seniority, location, part-time). One seniority level above is fine (lead/staff for "senior"). |
| **1** | Right area, but one thing is off or can't be confirmed: onsite when the query says remote, a level off, an adjacent role, or the key skill/credential isn't visible in the text. |
| **0** | Wrong job: different field or role, or not a real posting (e.g. a candidate advertising themselves). Keyword overlap alone doesn't count. |

## Known limitations
- **Judge strictness varies.** Judge A (queries 1–10) applied "unconfirmable → 1" strictly, e.g. Staff Accountant without a visible CPA mention. Grade split: A 58/118/39 (2/1/0), B 98/52/36, C 71/81/54. Every retrieval mode is scored against the same labels, so comparisons *between modes* are unaffected. Absolute nDCG across query groups is less comparable.
- **Pooling.** Jobs no mode returned in its top 10 are unjudged and count as non-relevant.
- **AI judgement, not human.** A human-labelled set (`handlabeled.jsonl`, via `python -m app.evals.label`) is still the gold standard and is kept separate.
