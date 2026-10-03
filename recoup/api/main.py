"""Uvicorn entrypoint:  uvicorn recoup.api.main:app"""
from pathlib import Path

from config.paypal_settings import load_settings
from recoup.agent.outcomes import OutcomeIngest
from recoup.api.app import create_app
from recoup.api.ledger import add_ledger_routes
from recoup.api.registry import OrderRegistry
from recoup.paypal.auth import PayPalAuth
from recoup.paypal.orders import PayPalOrders

LEDGER_PATH = Path(__file__).resolve().parents[2] / "data" / "ledger.json"

settings = load_settings()
auth = PayPalAuth(settings)
registry = OrderRegistry()
resolutions: list = []  # TODO: apply to the real store when the agent is wired in
ingest = OutcomeIngest(apply=resolutions.append)
app = create_app(
    settings=settings,
    auth=auth,
    orders=PayPalOrders(settings, auth),
    registry=registry,
    ingest=ingest,
)
app.state.resolutions = resolutions
add_ledger_routes(app, LEDGER_PATH)
