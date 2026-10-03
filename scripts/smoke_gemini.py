"""Check the Gemini key and model, then narrate one sample row.

Run:  python -m scripts.smoke_gemini
Set GEMINI_MODEL in .env to override the default model.
"""
import os

from dotenv import load_dotenv

load_dotenv()
from recoup.agent.gemini_narrator import DEFAULT_MODEL, GeminiNarrator  # noqa: E402


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set in .env")
        return
    from google import genai

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client()
    r = client.interactions.create(model=model, input="Reply with the single word: ok")
    print(f"raw call OK | model={model} | reply={r.output_text!r}")

    n = GeminiNarrator(client, model=model)
    state = {
        "record": {"customer_id": "cust_0066", "amount_cents": 134560, "error_reason": "instrument_declined"},
        "decision": {"chosen": "escalate_to_human"},
        "execution": {},
        "guardrail": {"reason": "attempt cap reached"},
        "terminal_status": "escalated",
    }
    print("narration:", n.narrate(state))
    print("stats:", n.stats())


if __name__ == "__main__":
    main()
