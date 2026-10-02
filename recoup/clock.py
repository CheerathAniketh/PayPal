"""Simulated time.

The agent never reads the system clock.  Three reasons:

1. **Cooldowns would be no-ops.**  A batch finishes in seconds; "wait at least
   24h between retries" only means something against simulated time.
2. **Reproducibility.**  A real clock means the demo hits different salary
   windows depending on what day you record the video.
3. **Verifiable timestamps.**  "This retry happened 26 simulated hours after the
   failure" becomes checkable from the log rather than asserted.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta


class ClockRewound(RuntimeError):
    """Raised when time is asked to move backwards.

    An append-only audit trail whose timestamps could rewind would not be
    evidence of anything.
    """


class SimulatedClock:
    def __init__(self, start: datetime) -> None:
        self._now = start
        self._start = start

    # ---- reading -----------------------------------------------------
    def now(self) -> datetime:
        return self._now

    def iso(self) -> str:
        return self._now.isoformat()

    @property
    def start(self) -> datetime:
        return self._start

    def hours_since(self, moment: datetime) -> float:
        """Age of ``moment`` in simulated hours, floored at zero.

        Never negative: a record cannot be less than zero hours old, and a
        negative age would silently satisfy every cooldown check.
        """
        delta = (self._now - moment).total_seconds() / 3600.0
        return max(0.0, delta)

    # ---- moving ------------------------------------------------------
    def advance(self, hours: float) -> datetime:
        if hours < 0:
            raise ClockRewound(f"cannot advance by {hours} hours")
        self._now = self._now + timedelta(hours=hours)
        return self._now

    def advance_to(self, moment: datetime) -> datetime:
        if moment < self._now:
            raise ClockRewound(
                f"refusing to rewind from {self._now.isoformat()} "
                f"to {moment.isoformat()}"
            )
        self._now = moment
        return self._now

    # ---- scheduling --------------------------------------------------
    def next_high_liquidity_day(
        self, salary_day: int, min_delay_hours: float = 24.0
    ) -> datetime:
        """The next occurrence of the customer's salary day, at least
        ``min_delay_hours`` from now.

        The floor is the point: it means **scheduling and compliance cannot
        disagree**.  Without it the agent could plan a retry that the cooldown
        guardrail would then refuse, and the record would sit in limbo.

        Short months are handled by clamping to the month's real length --
        February resolves a salary day of 30 to the 27th (or 28th/29th), never
        a phantom date that would raise.
        """
        if min_delay_hours < 0:
            raise ValueError("min_delay_hours must be non-negative")
        earliest = self._now + timedelta(hours=min_delay_hours)
        year, month = earliest.year, earliest.month
        for _ in range(4):
            month_len = calendar.monthrange(year, month)[1]
            day = min(max(salary_day, 1), month_len)
            candidate = datetime(year, month, day, 9, 0, 0)
            if candidate >= earliest:
                return candidate
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise RuntimeError(  # pragma: no cover - unreachable for day <= 31
            f"could not resolve salary day {salary_day} after {earliest}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SimulatedClock({self._now.isoformat()})"
