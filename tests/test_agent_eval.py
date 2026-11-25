from app.evals.agent_eval import Expect, args_ok, check_thresholds, load_scenarios, score_turn
from app.evals.run import check_thresholds as retrieval_thresholds


def test_scenarios_file_is_valid_and_ids_unique():
    sc = load_scenarios()
    assert len(sc) >= 30
    assert len({s.id for s in sc}) == len(sc)
    tools = {"search_jobs", "recommend_jobs", "get_job_details", "parse_resume", "skill_gap",
             "save_job", "list_saved_jobs", "remember_fact", "get_my_profile"}  # fmt: skip
    for s in sc:
        for t in s.turns:
            assert set(t.expect.tools) | set(t.expect.forbid) | set(t.expect.args) <= tools, s.id


def test_arg_checks():
    calls = [{"name": "search_jobs", "args": {"query": "Remote Python API", "remote": True, "seniority": ["senior"]}}]
    assert args_ok(calls, "search_jobs", {"remote": True, "query_any": ["python", "go"]})
    assert args_ok(calls, "search_jobs", {"seniority_includes": "senior"})
    assert not args_ok(calls, "search_jobs", {"seniority_includes": "junior"})
    assert not args_ok(calls, "search_jobs", {"remote_not": True})
    assert args_ok(calls, "search_jobs", {"location_not": "Paris"})  # absent field != value
    assert not args_ok(calls, "save_job", {})  # tool not called


def test_any_call_may_satisfy_checks():
    calls = [
        {"name": "search_jobs", "args": {"query": "data", "location": "Austin, TX", "remote": True}},
        {"name": "search_jobs", "args": {"query": "data"}},  # retried with fewer filters
    ]
    assert args_ok(calls, "search_jobs", {"location_any": ["austin"]})


def test_score_turn():
    exp = Expect(tools=["search_jobs"], forbid=["save_job"], args={"search_jobs": {"remote": True}}, cites=True)
    good = [{"name": "search_jobs", "args": {"query": "x", "remote": True}}]
    assert score_turn(exp, good, [5], error=False)["passed"]
    s = score_turn(exp, good + [{"name": "save_job", "args": {"job_id": 5}}], [5], error=False)
    assert not s["passed"] and s["forbidden"] == ["save_job"]
    s = score_turn(exp, [{"name": "search_jobs", "args": {"query": "x"}}], [5], error=False)
    assert s["tools_ok"] and not s["args_ok"]
    assert not score_turn(exp, good, [], error=False)["cite_ok"]
    assert not score_turn(exp, good, [5], error=True)["passed"]
    assert score_turn(Expect(cites=False), [], [], error=False)["passed"]


def test_thresholds():
    sm = {"turn_pass_rate": 0.9, "tool_selection_accuracy": 0.8, "grounded_rate": None, "error_rate": 0.1}
    fails = check_thresholds(sm, {"turn_pass_rate": 0.85, "tool_selection_accuracy": 0.85,
                                  "grounded_rate": 0.9, "error_rate_max": 0.05})  # fmt: skip
    assert len(fails) == 2  # tool selection and error rate; grounded skipped when not judged
    summary = {"hybrid|synthetic": {"ndcg@10": 0.7}}
    assert retrieval_thresholds(summary, {"hybrid|synthetic": {"ndcg@10": 0.8}, "x|y": {"ndcg@10": 0.1}}) == [
        "hybrid|synthetic ndcg@10 = 0.700 < 0.800",
        "x|y: missing from results",
    ]
