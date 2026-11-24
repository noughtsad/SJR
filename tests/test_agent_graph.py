"""The real agent graph, run offline: a scripted chat model, stub tools,
in-memory checkpointer and store. Covers the tool loop, the citation guard
and its retries, tool-error recovery and thread memory."""

import asyncio
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent.graph import allowed_job_ids, build_graph, cited_job_ids
from app.agent.runtime import chat_stream


class ScriptedModel(BaseChatModel):
    """Returns the scripted replies in order and records what it was shown."""

    replies: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])


@tool
async def search_jobs(query: str) -> str:
    """stub"""
    return json.dumps([{"job_id": 5, "title": "Data Engineer"}, {"job_id": 7, "title": "ML Engineer"}])


@tool
async def get_job_details(job_id: int) -> str:
    """stub"""
    raise ValueError(f"no job with id {job_id}")


def _call(name, args, id_="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


async def _no_context(_user_id):
    return "- test profile", "- none yet"


def _graph(replies):
    model = ScriptedModel(replies=replies, seen=[])
    g = build_graph(
        None, InMemorySaver(), InMemoryStore(), llm=model, tools=[search_jobs, get_job_details], context_fn=_no_context
    )
    return g, model


def _turn(g, text, thread="t"):
    async def go():
        return [ev async for ev in chat_stream(g, "u1", thread, text)]

    return asyncio.run(go())


def test_citation_helpers():
    msgs = [ToolMessage('[{"job_id": 12}, {"job_id": 34}]', tool_call_id="x")]
    assert allowed_job_ids(msgs) == {12, 34}
    assert cited_job_ids("see [job 12] and [Job #34], not job 56") == [12, 34]


def test_tool_loop_and_valid_citation():
    g, _ = _graph([_call("search_jobs", {"query": "data"}), AIMessage("Try [job 5].")])
    events = _turn(g, "find data jobs")
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds and "tool_result" in kinds and "retract" not in kinds
    assert events[-1]["type"] == "final" and events[-1]["cited_job_ids"] == [5]


def test_invented_job_is_retracted_and_regenerated():
    g, model = _graph([AIMessage("You'd love [job 999]."), AIMessage("I need to search first; nothing cited.")])
    events = _turn(g, "anything for me?")
    assert any(e["type"] == "retract" for e in events)
    final = events[-1]
    assert final["type"] == "final" and final["cited_job_ids"] == []
    # the model's second call was told which id was invalid
    assert "[job 999]" in model.seen[1][-1].content
    # the bad draft and the validator note are not left in the thread
    msgs = g.get_state({"configurable": {"thread_id": "u1:t"}}).values["messages"]
    assert not any("999" in str(m.content) for m in msgs)


def test_persistent_hallucination_is_struck_out():
    g, _ = _graph([AIMessage("[job 999]"), AIMessage("[job 998]"), AIMessage("Definitely [job 997].")])
    final = _turn(g, "jobs?")[-1]
    assert final["cited_job_ids"] == [] and "[unverified job removed]" in final["text"]


def test_tool_error_goes_back_to_model():
    g, model = _graph([_call("get_job_details", {"job_id": 42}), AIMessage("That job doesn't exist.")])
    events = _turn(g, "details on 42")
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["preview"].startswith("Error:") and "no job with id 42" in result["preview"]
    assert events[-1]["type"] == "final"


def test_thread_memory_and_cross_turn_citations():
    g, model = _graph(
        [_call("search_jobs", {"query": "data"}), AIMessage("[job 5]"), AIMessage("Tell me more about [job 7].")]
    )
    _turn(g, "find data jobs")
    final = _turn(g, "what about the second one?")[-1]
    assert final["cited_job_ids"] == [7]  # returned by a tool in the earlier turn of the same thread
    assert any(isinstance(m, HumanMessage) and m.content == "find data jobs" for m in model.seen[-1])


def test_threads_are_isolated():
    g, _ = _graph(
        [
            _call("search_jobs", {"query": "data"}),
            AIMessage("[job 5]"),
            AIMessage("[job 5]"),
            AIMessage("Let me search."),
        ]
    )
    _turn(g, "find data jobs", thread="a")
    events = _turn(g, "and that one?", thread="b")  # job 5 was never returned in thread b
    assert any(e["type"] == "retract" for e in events)
    assert events[-1]["type"] == "final" and events[-1]["cited_job_ids"] == []


def test_tool_budget_forces_an_answer():
    calls = [_call("search_jobs", {"query": f"q{i}"}, id_=f"c{i}") for i in range(4)]
    g, model = _graph([*calls, AIMessage("Best I found: [job 5].")])
    events = _turn(g, "keep looking")
    assert sum(e["type"] == "tool_call" for e in events) == 4
    assert events[-1]["cited_job_ids"] == [5]
    assert "Tool budget for this turn is used up" in model.seen[-1][0].content
