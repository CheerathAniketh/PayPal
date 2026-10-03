import time

from recoup.agent.gemini_narrator import GeminiNarrator
from recoup.agent.runtime import TemplateNarrator


class _Resp:
    def __init__(self, text):
        self.output_text = text


class _Interactions:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        return self.fn(kw)


class _Client:
    def __init__(self, fn):
        self.interactions = _Interactions(fn)


def _state(cust="cust_0066", cents=134560, status="escalated"):
    return {
        "record": {"customer_id": cust, "amount_cents": cents, "error_reason": "instrument_declined"},
        "decision": {"chosen": "escalate_to_human", "rationale": "x"},
        "execution": {},
        "guardrail": {"reason": "attempt cap reached"},
        "terminal_status": status,
    }


def _template(state):
    return TemplateNarrator().narrate(state)


def test_valid_sentence_is_used():
    c = _Client(lambda kw: _Resp("Escalated cust_0066's $1,345.60 for human review after the card was declined."))
    out = GeminiNarrator(c).narrate(_state())
    assert out.startswith("Escalated cust_0066's $1,345.60 for human")
    assert c.interactions.calls == 1


def test_wrong_amount_is_rejected_and_template_used():
    c = _Client(lambda kw: _Resp("Escalated cust_0066's $1,000.00 for human review."))
    s = _state()
    assert GeminiNarrator(c).narrate(s) == _template(s)


def test_invented_number_is_rejected():
    c = _Client(lambda kw: _Resp("Escalated $1,345.60 for cust_0066 after 3 failed attempts."))
    s = _state()
    assert GeminiNarrator(c).narrate(s) == _template(s)


def test_api_error_falls_back():
    def boom(kw):
        raise RuntimeError("quota")

    s = _state()
    assert GeminiNarrator(_Client(boom)).narrate(s) == _template(s)


def test_timeout_falls_back():
    def slow(kw):
        time.sleep(0.5)
        return _Resp("Escalated cust_0066's $1,345.60.")

    s = _state()
    assert GeminiNarrator(_Client(slow), timeout_s=0.05).narrate(s) == _template(s)


def test_recovered_rows_never_call_the_model():
    c = _Client(lambda kw: _Resp("should not be used"))
    s = _state(status="recovered")
    s["execution"] = {"amount_recovered_cents": 134560}
    assert GeminiNarrator(c).narrate(s) == _template(s)
    assert c.interactions.calls == 0


def test_circuit_breaker_stops_calling_after_repeated_failures():
    def boom(kw):
        raise RuntimeError("down")

    c = _Client(boom)
    n = GeminiNarrator(c, max_consecutive_failures=3)
    for i in range(6):
        n.narrate(_state(cust=f"cust_{i:04d}", cents=10000 + i))
    assert c.interactions.calls == 3
    assert n.stats()["disabled_by_failures"] is True


def test_cache_avoids_repeat_calls():
    c = _Client(lambda kw: _Resp("Escalated cust_0066's $1,345.60 for human review."))
    n = GeminiNarrator(c)
    n.narrate(_state())
    n.narrate(_state())
    assert c.interactions.calls == 1


def test_outreach_rows_skipped_by_default():
    c = _Client(lambda kw: _Resp("Contacted cust_0066 about $1,345.60."))
    s = _state(status="in_progress")
    s["execution"] = {"settles_async": True}
    assert GeminiNarrator(c).narrate(s) == _template(s)
    assert c.interactions.calls == 0


def test_disk_cache_survives_a_new_instance(tmp_path):
    path = str(tmp_path / "cache.json")
    c1 = _Client(lambda kw: _Resp("Escalated cust_0066's $1,345.60 for human review."))
    GeminiNarrator(c1, cache_path=path).narrate(_state())
    c2 = _Client(lambda kw: _Resp("different"))
    out = GeminiNarrator(c2, cache_path=path).narrate(_state())
    assert out.startswith("Escalated cust_0066's $1,345.60")
    assert c2.interactions.calls == 0


def test_spelled_out_amount_is_rejected():
    c = _Client(lambda kw: _Resp("Escalated cust_0066 with five hundred dollars for human review."))
    s = _state()
    assert GeminiNarrator(c).narrate(s) == _template(s)


def test_reply_without_the_digit_amount_is_rejected():
    c = _Client(lambda kw: _Resp("Escalated cust_0066 for human review after a decline."))
    s = _state()
    assert GeminiNarrator(c).narrate(s) == _template(s)


def test_prompt_tells_the_model_the_guardrail_blocked_the_action():
    seen = {}

    def fn(kw):
        seen["p"] = kw["input"]
        return _Resp("Escalated cust_0066's $1,345.60 for human review.")

    GeminiNarrator(_Client(fn)).narrate(_state())
    assert "blocked it" in seen["p"] and "Nothing was sent" in seen["p"]


def test_payment_error_hidden_from_prompt_when_guardrail_blocked():
    seen = {}

    def fn(kw):
        seen["p"] = kw["input"]
        return _Resp("Escalated cust_0066's $1,345.60 for human review.")

    GeminiNarrator(_Client(fn)).narrate(_state())
    assert "instrument declined" not in seen["p"]


def test_payment_error_kept_in_prompt_when_no_guardrail_reason():
    seen = {}

    def fn(kw):
        seen["p"] = kw["input"]
        return _Resp("Stopped pursuing cust_0066's $1,345.60 after the instrument declined.")

    s = _state(status="abandoned")
    s["guardrail"] = {}
    GeminiNarrator(_Client(fn)).narrate(s)
    assert "instrument declined" in seen["p"]
