"""Serve the precomputed simulated ledger."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException


def add_ledger_routes(app: FastAPI, path: Path) -> None:
    @app.get("/api/ledger")
    def ledger() -> Any:
        if not path.exists():
            raise HTTPException(
                404, "ledger not built: run python -m scripts.build_ledger"
            )
        return json.loads(path.read_text())
