"""Time and execution honesty.

The claims these defend: simulated time is monotonic and cooldown-safe; the
executor never inflates the headline; and API success is not conflated with
revenue recovered.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.taxonomy import Intervention, spec
from recoup.clock import ClockRewound, SimulatedClock
from recoup.environment import RecoveryEnvironment
from recoup.executor import (
    TEST_MODE_SUPPORT,
    Executor,
    TestModeRequiresClient,
)
from recoup.models import ExecutionMode, Outcome
from tests.conftest import make_customer, make_latent, make_record

NOW = datetime(2026, 1, 16, 9, 0, 0)


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------
def test_clock_refuses_to_rewind():
    """Timestamps are monotonic.

    An append-only audit trail whose timestamps could rewind would not be
    evidence of anything.
    """
    clock = SimulatedClock(NOW)
    clock.advance(24)
    with pytest.raises(ClockRewound):
        clock.advance(-1)
    with pytest.raises(ClockRewound):
        clock.advance_to(NOW)


def test_hours_since_never_negative():
    """No negative record ages.

    A record cannot be less than zero hours old, and a negative age would
    silently satisfy every cooldown check.
    """
    clock = SimulatedClock(NOW)
    future = NOW + timedelta(hours=5)
    assert clock.hours_since(future) == 0.0
    clock.advance(10)
    assert clock.hours_since(NOW) == pytest.approx(10.0)


def test_salary_window_scheduling_respects_cooldown():
    """Scheduling and compliance agree.

    ``next_high_liquidity_day`` takes a floor, so the agent can never plan a
    retry the cooldown guardrail would then forbid.
    """
    clock = SimulatedClock(datetime(2026, 1, 27, 9))  # salary day itself
    nxt = clock.next_high_liquidity_day(salary_day=27, min_delay_hours=24.0)
    assert (nxt - clock.now()).total_seconds() / 3600.0 >= 24.0
    # It picks the NEXT occurrence, not today's already-passed one.
    assert nxt.month == 2


def test_salary_window_handles_short_months():
    """February is not treated as 31 days.

    A salary day of 30 in February resolves to the 27th/28th, never a phantom
    date that would raise.
    """
    clock = SimulatedClock(datetime(2026, 2, 10, 9))
    nxt = clock.next_high_liquidity_day(salary_day=30, min_delay_hours=24.0)
    assert nxt.month == 2
    assert nxt.day == 28  # 2026 is not a leap year


# --------------------------------------------------------------------------
# The executor: SIMULATED vs TEST_MODE
# --------------------------------------------------------------------------
def _executor(mode=ExecutionMode.SIMULATED, client=None):
    return Executor(RecoveryEnvironment(), mode=mode, client=client)


def test_simulated_mode_never_calls_an_api():
    """The batch runs offline, in CI, without keys."""
    ex = _executor()
    record = make_record()
    result = ex.execute(
        record, make_customer(), make_latent(),
        Intervention.RETRY_NOW, run_id="run_1", attempt_number=1, attempt_at=NOW,
    )
    assert result.api_called is False
    assert result.execution_mode is ExecutionMode.SIMULATED


def test_test_mode_requires_a_client():
    """No silent downgrade to simulated.

    Downgrading would mean the audit trail claims real calls that never
    happened.
    """
    with pytest.raises(TestModeRequiresClient):
        Executor(RecoveryEnvironment(), mode=ExecutionMode.TEST_MODE, client=None)


def test_every_intervention_declares_test_mode_support():
    """No intervention runs without an explicit real-or-mock decision."""
    for intervention in Intervention:
        assert intervention in TEST_MODE_SUPPORT, intervention
        support = TEST_MODE_SUPPORT[intervention]
        # An unsupported intervention must say WHY it is mocked.
        if not support.supported:
            assert support.mock_reason


def test_mocked_interventions_are_flagged_not_hidden():
    """Limitations reach the audit trail.

    ``re_auth_mandate`` has no honest test-mode equivalent, so in SIMULATED (and
    when called under TEST_MODE) it is flagged mocked, with the reason attached.
    """
    ex = _executor()
    record = make_record(reason="payer_action_required")
    result = ex.execute(
        record, make_customer(), make_latent(),
        Intervention.RE_AUTH_MANDATE, run_id="run_1", attempt_number=1,
        attempt_at=NOW,
    )
    assert result.was_mocked is True
    assert "authentication" in result.mock_reason


def test_partial_debit_recovers_only_part_of_the_amount():
    """The headline is not inflated.

    A ``retry_smaller_amount`` that succeeds recovers half, not all. Counting a
    partial as a full recovery would inflate the headline. $736 -> $368.
    """
    ex = _executor()
    record = make_record(amount_cents=73_600)
    # A latent set up to make the smaller-amount retry succeed.
    latent = make_latent(base_logodds=6.0)
    result = ex.execute(
        record, make_customer(), latent,
        Intervention.RETRY_SMALLER_AMOUNT, run_id="run_1", attempt_number=1,
        attempt_at=NOW,
    )
    assert result.outcome is Outcome.RECOVERED
    assert result.amount_attempted_cents == 36_800
    assert result.amount_recovered_cents == 36_800


def test_terminal_interventions_recover_nothing_and_touch_no_api():
    """Bookkeeping states stay bookkeeping."""
    ex = _executor()
    record = make_record()
    for intervention, outcome in (
        (Intervention.ESCALATE, Outcome.ESCALATED),
        (Intervention.GIVE_UP, Outcome.GAVE_UP),
    ):
        result = ex.execute(
            record, make_customer(), make_latent(), intervention,
            run_id="run_1", attempt_number=1, attempt_at=NOW,
        )
        assert result.outcome is outcome
        assert result.amount_recovered_cents == 0
        assert result.api_called is False
        assert spec(intervention).terminal


def test_timing_changes_outcomes():
    """The core thesis holds at the executor level too.

    The same intervention on the same record can recover at a good time and fail
    at a bad one, because the environment reads ``attempt_at``.
    """
    ex = _executor()
    customer = make_customer(salary_day=1, income_regularity=0.95)
    record = make_record(customer=customer, reason="insufficient_funds",
                         failed_at=datetime(2026, 1, 5, 9))
    latent = make_latent(income_regularity=0.95)

    clock = SimulatedClock(record.failed_at)
    good = clock.next_high_liquidity_day(1, 24.0)      # a salary day
    bad = good + timedelta(days=12)                     # mid-month drought

    p_good = ex.environment.true_prob(
        record, customer, latent, Intervention.RETRY_NOW, good
    )
    p_bad = ex.environment.true_prob(
        record, customer, latent, Intervention.RETRY_NOW, bad
    )
    assert p_good - p_bad > 0.15


def test_api_failure_is_not_recovery_failure():
    """A flaky network cannot quietly deflate the metrics.

    When the API raises, the error is recorded, but the environment still
    decides whether the money came back.
    """
    class BoomClient:
        def create_order(self, **_kwargs):
            raise RuntimeError("network timeout")

    ex = Executor(RecoveryEnvironment(), mode=ExecutionMode.TEST_MODE,
                  client=BoomClient())
    record = make_record()
    latent = make_latent(base_logodds=6.0)  # would recover in the world
    result = ex.execute(
        record, make_customer(), latent, Intervention.RETRY_NOW,
        run_id="run_1", attempt_number=1, attempt_at=NOW,
    )
    assert "network timeout" in result.api_error
    assert result.outcome is Outcome.RECOVERED  # the world decided, not the API
