"""Gemini narrator: write-only prose for the audit row.

Safety contract (the pitch: the LLM never controls whether money moves):
  * It only reads a facts dict built by code and returns one sentence.
  * Every number in the reply must already appear in the facts, otherwise the
    reply is discarded and the deterministic TemplateNarrator text is used.
  * Any error, timeout or empty reply falls back to the template. After
    several failures in a row it stops calling the API for the process.
  * Nothing it returns feeds routing.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from recoup.agent.runtime import TemplateNarrator

DEFAULT_MODEL = "gemini-3.5-flash-lite"
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUM_WORDS = r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen|twice|thrice)\b"


def _usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _clean_num(n: str) -> str:
    return n.rstrip(".,")


class GeminiNarrator:
    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_MODEL,
        fallback: Optional[Any] = None,
        timeout_s: float = 8.0,
        max_calls: int = 400,
        max_consecutive_failures: int = 3,
        cache_path: Optional[str] = None,
        include_outreach: bool = False,
        progress_every: int = 10,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback or TemplateNarrator()
        self._timeout = timeout_s
        self._max_calls = max_calls
        self._max_fail = max_consecutive_failures
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._cache: Dict[str, str] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._outreach = include_outreach
        self._progress_every = progress_every
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except Exception:  # noqa: BLE001
                self._cache = {}
        self._fail_streak = 0
        self.api_calls = 0
        self.llm_used = 0
        self.fell_back = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Optional["GeminiNarrator"]:
        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("GEMINI_API_KEY"):
            return None
        from google import genai  # lazy: the Render API process never imports this

        cache = Path(__file__).resolve().parents[2] / "data" / "narration_cache.json"
        return cls(
            genai.Client(),
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            cache_path=str(cache),
            include_outreach=os.getenv("RECOUP_NARRATE_OUTREACH") == "1",
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "api_calls": self.api_calls,
            "llm_sentences_used": self.llm_used,
            "fell_back_to_template": self.fell_back,
            "disabled_by_failures": self._fail_streak >= self._max_fail,
        }

    # ------------------------------------------------------------------
    def narrate(self, state: Dict[str, Any]) -> str:
        base = ""
        try:
            base = self._fallback.narrate(state)
            facts = self._facts(state)
            if facts is None:
                return base
            if self._fail_streak >= self._max_fail or self.api_calls >= self._max_calls:
                self.fell_back += 1
                return base
            key = self._key(facts)
            if key in self._cache:
                return self._cache[key]
            text = self._ask(facts)
            if text and self._valid(text, facts):
                self._cache[key] = text
                self._save_cache()
                self.llm_used += 1
                return text
            self.fell_back += 1
            return base
        except Exception:  # noqa: BLE001 - narration must never break the graph
            self.fell_back += 1
            return base

    # ------------------------------------------------------------------
    def _key(self, facts: Dict[str, str]) -> str:
        return json.dumps({"m": self._model, "f": facts}, sort_keys=True)

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True))
        except Exception:  # noqa: BLE001
            pass

    def _facts(self, state: Dict[str, Any]) -> Optional[Dict[str, str]]:
        status = state.get("terminal_status")
        execution = state.get("execution") or {}
        if status == "escalated":
            kind = "escalation note for a human reviewer"
        elif status == "abandoned":
            kind = "close-out note explaining why recovery was stopped"
        elif self._outreach and status == "in_progress" and execution.get("settles_async"):
            kind = "note that the customer was contacted and the agent is waiting on them"
        else:
            return None
        record = state.get("record") or {}
        decision = state.get("decision") or {}
        guardrail = state.get("guardrail") or {}
        return {
            "note_type": kind,
            "outcome": str(status),
            "customer": str(record.get("customer_id", "?")),
            "amount": _usd(int(record.get("amount_cents", 0))),
            "action_proposed": str(decision.get("chosen", "?")).replace("_", " "),
            "failure_reason": "" if guardrail.get("reason") else str(record.get("error_reason", "") or "").replace("_", " "),
            "what_happened": GeminiNarrator._what_happened(
                str(status),
                str(decision.get("chosen", "?")).replace("_", " "),
                str(guardrail.get("reason") or ""),
            ),
        }

    @staticmethod
    def _what_happened(status: str, action: str, reason: str) -> str:
        """Code-written account of the outcome. The model only rephrases it."""
        if status == "escalated" and reason:
            return (
                f"The agent proposed '{action}', but a deterministic guardrail blocked it "
                f"({reason}). Nothing was sent to the payer and the case was handed to a human."
            )
        if status == "escalated":
            return "The agent handed the case to a human reviewer."
        if status == "abandoned":
            return "The agent decided to stop pursuing recovery for this payment."
        return f"The agent sent '{action}' and is waiting for the customer to act."

    @staticmethod
    def _prompt(facts: Dict[str, str]) -> str:
        lines = "\n".join(f"- {k}: {v}" for k, v in facts.items() if v)
        return (
            "You write one-line notes for an audit log of a payment-recovery system.\n"
            "Write exactly ONE plain sentence of at most 30 words: a "
            f"{facts['note_type']}.\n"
            "Use only the facts below. Write the amount exactly as given, with digits. "
            "Do not spell out numbers, and do not add any number, date, percentage or "
            "promise that is not in the facts. Do not say an action failed or was "
            "attempted unless the facts say so. Do not give a reason for the outcome other than the one in the facts. Plain English, no snake_case or code "
            "identifiers (except the customer id). No markdown, no quotes.\n"
            f"Facts:\n{lines}\n"
        )

    def _ask(self, facts: Dict[str, str]) -> str:
        self.api_calls += 1
        if self.api_calls == 1 or self.api_calls % self._progress_every == 0:
            print(f"[narrator] api call {self.api_calls} (used {self.llm_used}, fell back {self.fell_back})", file=sys.stderr, flush=True)
        try:
            fut = self._pool.submit(
                self._client.interactions.create,
                model=self._model,
                input=self._prompt(facts),
            )
            interaction = fut.result(timeout=self._timeout)
            text = (getattr(interaction, "output_text", "") or "").strip().strip("`\"' ")
        except Exception:  # noqa: BLE001
            self._fail_streak += 1
            return ""
        self._fail_streak = 0 if text else self._fail_streak + 1
        return text

    @staticmethod
    def _valid(text: str, facts: Dict[str, str]) -> bool:
        if "\n" in text or len(text) > 300 or len(text.split()) > 45:
            return False
        if facts["amount"] not in text:
            return False
        facts_text = " ".join(facts.values())
        if re.search(_NUM_WORDS, text, re.I) and not re.search(_NUM_WORDS, facts_text, re.I):
            return False
        allowed = {_clean_num(n) for v in facts.values() for n in _NUM.findall(v)}
        return all(_clean_num(n) in allowed for n in _NUM.findall(text))


def describe_choice() -> str:
    """For scripts: a one-line description of which narrator will be used."""
    return "gemini" if os.getenv("GEMINI_API_KEY") else "template (no GEMINI_API_KEY)"
