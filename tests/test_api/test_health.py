from __future__ import annotations

from fastapi.testclient import TestClient

from doodle_doc.api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "colqwen_loaded" in data
    assert "siglip_loaded" not in data
