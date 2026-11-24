"""Long-term memory: durable facts about a user, kept across conversations.

Short-term memory (the conversation itself) lives in the LangGraph Postgres
checkpointer, one thread per conversation. Facts live in the LangGraph
Postgres store under ("user_facts", user_id) and are injected into the system
prompt of every new thread. Keys are a fixed vocabulary so a new statement
overwrites the old one ("now I want hybrid" replaces "remote only") instead of
piling up contradictions.
"""

from typing import Literal

from langgraph.store.base import BaseStore

FACT_KEYS = Literal[
    "target_role",
    "location",
    "work_mode",  # remote / hybrid / onsite
    "salary_expectation",
    "seniority",
    "industries",
    "avoid",  # companies, industries or roles they don't want
    "visa_or_relocation",
    "availability",
    "other",
]


def _ns(user_id: str) -> tuple[str, ...]:
    return ("user_facts", user_id)


async def put_fact(store: BaseStore, user_id: str, key: str, value: str) -> None:
    await store.aput(_ns(user_id), key, {"value": value[:500]})


async def get_facts(store: BaseStore, user_id: str) -> dict[str, str]:
    items = await store.asearch(_ns(user_id), limit=50)
    return {it.key: it.value["value"] for it in items}


async def delete_fact(store: BaseStore, user_id: str, key: str) -> None:
    await store.adelete(_ns(user_id), key)
