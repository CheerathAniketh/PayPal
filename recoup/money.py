"""Money is integer paise. Never a float.

Binary floating point cannot represent most decimal currency values exactly:
``0.1 + 0.2 != 0.3``.  Across 180 records, several attempts each, with partial
debits taking 50% of an amount, that error compounds enough for a headline
figure to drift from the sum of its parts.  A reviewer who totals the audit log
by hand and gets a different number stops trusting every other number.

The tell was already in the code before this module existed:
``int(round(record.amount * 100))`` at the API boundary.  Razorpay's API works
in paise, so integers *remove* a conversion rather than adding one.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Union

Rupees = Union[int, float, str, Decimal]


def rupees_to_paise(rupees: Rupees) -> int:
    """Convert rupees to integer paise.  Use at the INGEST BOUNDARY only.

    Goes via :class:`~decimal.Decimal` so that ``"736.005"`` rounds the way a
    human expects rather than the way binary floating point happens to.
    """
    quantised = Decimal(str(rupees)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(quantised * 100)


def paise_to_rupees(paise: int) -> float:
    """Convert paise to rupees.  DISPLAY ONLY -- never feed this back into a
    calculation, or you have reintroduced the float you just removed."""
    _assert_paise(paise)
    return paise / 100.0


def format_inr(paise: int) -> str:
    """Render paise for humans: ``73600`` -> ``'Rs 736.00'``."""
    _assert_paise(paise)
    sign = "-" if paise < 0 else ""
    return f"{sign}Rs {abs(paise) / 100.0:,.2f}"


def split_paise(paise: int, fraction: float) -> int:
    """Take ``fraction`` of an amount, rounding DOWN.

    The rounding direction is deliberate.  When a direction must be chosen,
    choose the one that cannot inflate the headline recovery figure: a partial
    debit of Rs 736.005 is recorded as Rs 736.00, never Rs 736.01.
    """
    _assert_paise(paise)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")
    # Decimal, then floor -- float multiplication of 73600 * 0.5 is exact here
    # but is not exact in general, and "exact in this case" is not a policy.
    return int((Decimal(paise) * Decimal(str(fraction))).to_integral_value(rounding="ROUND_FLOOR"))


def total_paise(amounts: Iterable[int]) -> int:
    """Sum paise.  Exact by construction; exists so callers never reach for
    ``sum(float)`` out of habit."""
    running = 0
    for amount in amounts:
        _assert_paise(amount)
        running += amount
    return running


def _assert_paise(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"money must be integer paise, got {type(value).__name__}: {value!r}"
        )
