"""Graph state and the enums the nodes share.

The state is a plain ``TypedDict`` -- LangGraph merges each node's returned dict
into it.  ``record`` is the Phase-1 failed-payment shape (amount in integer
paise); the controller sets ``run_id``, ``record``, ``attempt``, ``now_iso``
before invoke, and the nodes fill the rest.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

# Re-export the taxonomy types so the agent package is self-contained to import.
from config.taxonomy import FailureClass, Intervention  # noqa: F401


class TerminalStatus(str, Enum):
    """Where a single attempt through the graph ends up."""

    RECOVERED = "recovered"
    IN_PROGRESS = "in_progress"     # fired a contact; awaiting customer action
    ESCALATED = "escalated"         # a human now owns it
    ABANDONED = "abandoned"         # judged not worth another ask
    SCHEDULED = "scheduled"         # queued for a later window


class AgentState(TypedDict, total=False):
    # ---- set by the controller before invoke -------------------------
    run_id: str
    record: Dict[str, Any]
    attempt: int
    now_iso: str

    # ---- filled by the nodes -----------------------------------------
    diagnosis: Dict[str, Any]
    decision: Dict[str, Any]
    guardrail: Dict[str, Any]
    execution: Dict[str, Any]
    narration: str
    terminal_status: str
    audit_row: Dict[str, Any]
