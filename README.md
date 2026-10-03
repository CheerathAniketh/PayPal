# Recoup

**Agentic failed-payment recovery for PayPal merchants. The AI proposes, the rules decide, every action is audited.**

When a payment fails, most merchants either retry blindly or give up. Recoup treats each failed payment as a decision: which recovery action has the best net expected value (recovered amount minus fees and contact cost), is that action allowed by hard rules, and what exactly happened? An LLM drafts the explanation. It never decides whether money moves.

Built for the [PayPal AI Hackathon](https://paypal-ai-hackathon.devpost.com/) (Devpost).

- Hosted demo: https://recoup-mrkf.onrender.com (free tier, so the first request after idle can be slow while the service wakes)
- Demo video: `TODO: YouTube URL`
- License: MIT

## What existed before, and what is new

The decision engine (agent graph, audit store, scheduler, economics and synthetic data generator) existed before the hackathon. Everything that touches PayPal, plus the dashboard and the LLM narrator, was built during it.

| Existed before the hackathon | New for this hackathon |
|---|---|
| LangGraph agent (ingest, diagnose, decide, guardrail check, execute, narrate, log) | PayPal sandbox client: OAuth2, create/get/capture order (`recoup/paypal/`) |
| SQLite append-only audit store | Executor rewritten for PayPal (`recoup/executor.py`) |
| Batch scheduler, net-EV economics, synthetic data generator | Failure taxonomy remapped to PayPal error vocabulary (`config/taxonomy.py`) |
| Guardrail framework and its tests | PayPal fee model, payment rails (card, PayPal balance, bank account) |
| | Webhook signature verification via PayPal's verify API, fails closed (`recoup/paypal/webhook_verify.py`) |
| | FastAPI service with webhook endpoint, per-record ledger, live event log (`recoup/api/`) |
| | AG Grid dashboard (`web/index.html`) |
| | Gemini narrator with a validation layer (`recoup/agent/gemini_narrator.py`) |
| | Render deployment config (`render.yaml`) |

## How it works

For each failed payment the agent:

1. **Diagnoses** the failure from the PayPal error reason (a lookup, never ML).
2. **Ranks** the available recovery actions (retry now, retry in a better window, ask the customer to update their payment method, hand to a human, give up) by net expected value.
3. **Checks six deterministic guardrails** before anything runs: attempt cap, cooldown between attempts, contact cap, amount gate (large amounts need human sign-off), subscription state (no silent retry against a cancelled, paused or halted subscription), and prior customer notice for recurring charges on a stored payment method. A failed check blocks the action and escalates the case to a human with the reason logged.
4. **Executes** through the PayPal sandbox.
5. **Narrates** the decision in one sentence and **logs** an append-only audit row: action, rationale, all guardrail results, outcome, idempotency key.

### Where the AI is, and where it is not

- The LLM (Gemini, via `google-genai`) writes the one-line narration on escalated and abandoned audit rows. It is **write-only**: it has no path into routing, so it cannot change whether money moves.
- Code, not the model, establishes what happened. The prompt contains only facts assembled by code, including a code-written account of the outcome ("the guardrail blocked X; nothing was sent"), and when a guardrail blocked an action the original payment error is withheld so the model cannot attribute the block to the wrong cause.
- Replies are validated before use: the exact dollar amount must appear, no spelled-out numbers, no digits that are not in the facts. A reply that fails validation, a timeout, or an API error falls back to a deterministic template. Several failures in a row switch the model off for the run.
- There is also an optional learned recovery-probability policy (LightGBM). It is not the default and is not what the headline numbers use. See limitations.

## Honesty rule: what is real and what is simulated

PayPal's sandbox cannot produce an organic declined payment on demand, so we keep the two questions separate:

> **PayPal answers "did the API call work, and with what error." The simulated environment answers "did the money come back."**

- Payment failures in the batch are **injected**, and retry outcomes are **sampled**. Every number in the dashboard's KPI header and both grids comes from this simulation and every row is tagged `source: "simulated"`.
- In test mode the executor creates real sandbox orders. When the simulation says a retry failed, it calls capture with PayPal's sandbox `PayPal-Mock-Response` header and records PayPal's error. These mock error bodies are canned (the same `debug_id` comes back every time); they are sandbox mock responses, not live declines.
- Two reasons used in the synthetic data, `insufficient_funds` and `card_expired`, are rejected by the sandbox's mock mechanism, so they exist only in simulated data.
- The **only real recovery path** is the webhook flow: an order is approved by a sandbox buyer, `CHECKOUT.ORDER.APPROVED` triggers a capture, and `PAYMENT.CAPTURE.COMPLETED` or `DENIED` is verified and fed to the same outcome ingestion the simulation uses. These events appear in a separate "Live PayPal sandbox events" panel and are never blended with simulated money.

## Results (simulated batch)

180 failed payments across 75 customers, $112,006.77 at risk, default rules policy:

| | |
|---|---|
| Recovered | 95 records, **$58,066.70 (51.8%)** |
| Escalated to a human | 62, each with a logged reason |
| Abandoned | 23 |
| Attempts | 250 |
| Attempts blocked by a failed guardrail | 11 (each escalated) |
| Compliance violations | 0 across 250 audit rows (no action executed with a failed subscription or customer-notice check) |
| Unrecoverable records | 15 of 15 "truly dead" records (in the generator's ground truth) were stopped cleanly |

These are simulated outcomes from one seeded batch, not claims about real merchants. The recovery rate depends on the generator's assumptions.

## Run it

Tested on Python 3.14, Linux.

```bash
git clone https://github.com/CheerathAniketh/PayPal.git && cd PayPal
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in the values you need, see below
python -m pytest -q           # unit tests, no network needed
uvicorn recoup.api.main:app   # dashboard at http://localhost:8000
```

The dashboard reads the committed `data/ledger.json`. To regenerate it:

```bash
python -m scripts.build_ledger
```

To regenerate it with Gemini narration (about 2.5 minutes the first time, then cached in `data/narration_cache.json`):

```bash
pip install -r requirements-narrator.txt
# set GEMINI_API_KEY in .env, optionally GEMINI_MODEL (default gemini-3.5-flash-lite)
python -m scripts.smoke_gemini                       # checks the key and model
RECOUP_NARRATOR=gemini python -m scripts.build_ledger
```

### Environment variables (`.env`)

| Variable | Needed for |
|---|---|
| `PAYPAL_ENV=sandbox` | PayPal calls. The client refuses any other value. |
| `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET` | PayPal sandbox calls (a Merchant sandbox app) |
| `PAYPAL_WEBHOOK_ID` | Verifying webhook signatures. Without it `/webhook` returns 503 so PayPal retries. |
| `DEMO_TOKEN` | Enabling `POST /api/demo/order`. Without it that endpoint returns 404. |
| `GEMINI_API_KEY` | Regenerating the ledger with Gemini narration only |

### Live webhook demo

1. `POST /api/demo/order` with header `X-Demo-Token: <DEMO_TOKEN>`. This creates a real sandbox order and returns an `approve_url`.
2. Open `approve_url` and approve as a sandbox buyer.
3. Watch `GET /api/webhook-events` or the live panel on the dashboard.

On Render's free tier the service spins down after 15 minutes idle, and the order registry and event log are in memory, so wake the service (`/healthz`) and run the whole flow within a few minutes.

### Other scripts

`python -m scripts.demo_paypal_execute` (creates about 10 real sandbox orders), `python -m scripts.probe_mock_errors`, `python -m scripts.test_paypal_connection`, `python -m scripts.run_batch`, `python -m scripts.generate_batch`.

## Tools used

- **PayPal**: Orders v2 (create, get, capture), OAuth2, Webhooks and the verify-webhook-signature API, sandbox negative testing via `PayPal-Mock-Response`
- **Google Gemini API** (`google-genai`, `gemini-3.5-flash-lite`): narration
- **AG Grid Community**: decision ledger, audit-trail and live-event grids
- **Render**: hosting (`render.yaml`)
- LangGraph, FastAPI, SQLite, LightGBM, scikit-learn, pytest

## Status and known limitations

Please read these before judging the project.

- **Live round trip: verified once on 3 Oct 2026 against the deployed Render service.** A real sandbox order was created through `POST /api/demo/order`, approved by a sandbox buyer, and PayPal's signature-verified webhooks arrived in order: `CHECKOUT.ORDER.APPROVED` (received twice, one second apart, under different event ids; the capture request carries an idempotency key, so the second one should not double-capture, though I did not test that separately), then `PAYMENT.CAPTURE.COMPLETED`, which the server matched to the registered order and recorded as `recovered`. The order id locations assumed for both event types turned out to be correct. Limits: it was run once, not as a repeated test; the order registry and event log are in memory (a restart on Render's free tier clears them); and a webhook recovery shows up in the live-events panel but is not yet applied to the batch ledger.
- **Fees**: the Checkout fee is modelled as 3.49% plus a fixed $0.49. The percentage is from PayPal's own page; the $0.49 comes from secondary sources and is unverified. Cross-border surcharges are not modelled.
- **Learned model**: the committed LightGBM propensity model was trained before the PayPal port, on a different set of payment rails, and has not been retrained. The headline numbers use the rules policy.
- **Coarse reason codes**: PayPal's error codes are coarse, so timing decisions (for example waiting for a payday window) come from customer signals and the model, not from the reason code.
- **Synthetic scale**: amounts are lognormal around $500, so the fixed fee barely registers in the demo.
- **State is in memory**: the live order registry and event log reset on restart. The API does not yet apply webhook recoveries to the batch store; recoveries land in the live-events panel only.
- **Legacy internal names**: some identifiers still use pre-port vocabulary (`pre_debit_notice`, `is_mandate_debit`, subscription statuses such as `halted`). The dashboard and the audit text use plain wording; the internal field names are unchanged to keep the schema and tests stable. The guardrail on prior customer notice is an internal policy we chose, not a claim about a PayPal rule.
- **Not built**: Disputes API, Payouts API, a separate frontend framework.

## Project layout

```
config/            settings, PayPal sandbox config, failure taxonomy
recoup/agent/      LangGraph nodes, guardrails (control.py), runtime, Gemini narrator
recoup/paypal/     OAuth, orders, webhook verification
recoup/api/        FastAPI app (webhook, ledger, live events, dashboard)
recoup/executor.py PayPal-backed executor
scripts/           batch, ledger, probes and demos
web/index.html     AG Grid dashboard
data/              frozen batch, ledger.json, narration cache
tests/             pytest suite
```

## License

MIT, see `LICENSE`.
