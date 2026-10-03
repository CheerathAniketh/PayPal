import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from recoup.api.ledger import add_ledger_routes


def test_missing_ledger_is_404(tmp_path):
    app = FastAPI()
    add_ledger_routes(app, tmp_path / "nope.json")
    assert TestClient(app).get("/api/ledger").status_code == 404


def test_ledger_is_served_verbatim(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"source": "simulated", "rows": [{"record_id": "r1"}]}))
    app = FastAPI()
    add_ledger_routes(app, path)
    r = TestClient(app).get("/api/ledger")
    assert r.status_code == 200
    assert r.json()["rows"][0]["record_id"] == "r1"
