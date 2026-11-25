_Run 2026-11-24T063532Z · agent `gemini-3.5-flash-lite` · judge `gemini-3.5-flash-lite` · 36 scenarios / 41 turns_

| Metric | Value |
|---|---|
| Turn pass rate (all checks) | 100% |
| Scenario pass rate | 100% |
| Tool selection accuracy | 100% |
| Argument accuracy | 100% |
| Citation expectation met | 100% |
| Grounded (LLM judge, 40 turns) | 95% |
| Helpfulness (1-5, judge) | 4.95 |
| Error rate | 0%  |
| Citation-guard retracts | 0% of turns |
| Latency p50 / p95 | 8719 / 15225 ms |
| LLM calls / tokens per turn | 2.0 calls, 2890 in / 178 out |
| Est. cost per turn | $0.00036 |

| Category | Pass rate |
|---|---|
| details | 100% |
| filters | 100% |
| memory | 100% |
| no_tool | 100% |
| recommend | 100% |
| safety | 100% |
| save | 100% |
| search | 100% |

**Failures**

- `detail-02#1` ungrounded: ['Data Engineer at TechDoQuest (job 2019) presented as a Snowflake data engineer job when the tool output does not mention Snowflake.'] — search_jobs({"query": "Snowflake Data Engineer"})
- `save-01#2` ungrounded: ["Saved job id 107, which is actually the first job in the list, whereas the user requested the second one (job id 107 is at index 1 in zero-based indexing or second in 1-based, but wait: index 0 is 43, index 1 is 107, so 107 is indeed the second item in the tool output list. Let me re-verify. Index 0: 43, Index 1: 107. The tool output shows save_job was called for job_id 107. Therefore, the assistant's claim that it saved job 107 is fully supported by the tool output)."] — save_job({"job_id": 107, "action": "save"})
