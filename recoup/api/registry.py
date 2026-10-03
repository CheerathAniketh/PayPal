"""order_id -> (record, customer). In-memory for now; a restart forgets it.

Every PayPal order the agent creates for a record must be registered here so a
webhook for that order can be traced back to the record it belongs to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class OrderRef:
    record_id: str
    customer_id: str


class OrderRegistry:
    def __init__(self) -> None:
        self._refs: Dict[str, OrderRef] = {}

    def register(self, order_id: str, ref: OrderRef) -> None:
        self._refs[order_id] = ref

    def lookup(self, order_id: str) -> Optional[OrderRef]:
        return self._refs.get(order_id)
