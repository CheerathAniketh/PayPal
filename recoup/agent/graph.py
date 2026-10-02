"""Graph assembly and the module-level compiled ``graph``.

The graph carries ONE record through ONE attempt and never cycles.  Anything
cross-record -- dispatch order, budgets, contact caps, time advancement -- lives
in the scheduler, not here, so every attempt is a single clean Studio trace and
one audit row.

``graph`` is a module-level object so ``langgraph dev`` (Studio) and a FastAPI
app import the identical instance.  It is compiled without a checkpointer -- the
dev server injects persistence.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from recoup.agent.nodes import (
    decide_node,
    diagnose,
    execute,
    guardrail_check,
    ingest,
    log,
    narrate,
    route_after_decide,
    route_after_guardrail,
)
from recoup.agent.state import AgentState


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("ingest", ingest)
    g.add_node("diagnose", diagnose)
    g.add_node("decide", decide_node)
    g.add_node("guardrail_check", guardrail_check)
    g.add_node("execute", execute)
    g.add_node("narrate", narrate)
    g.add_node("log", log)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "diagnose")
    g.add_edge("diagnose", "decide")

    # decide -> abandon (terminal set in decide) | guardrail_check
    g.add_conditional_edges(
        "decide",
        route_after_decide,
        {"abandon": "narrate", "guardrail_check": "guardrail_check"},
    )
    # guardrail -> execute | abandon (terminal = escalated, set in guardrail)
    g.add_conditional_edges(
        "guardrail_check",
        route_after_guardrail,
        {"execute": "execute", "abandon": "narrate"},
    )

    g.add_edge("execute", "narrate")
    g.add_edge("narrate", "log")
    g.add_edge("log", END)
    return g


# Compiled once, at import, so Studio and FastAPI share this instance.
graph = build_graph().compile()
