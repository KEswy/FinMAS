from fastapi.testclient import TestClient

from finmas.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_no_llm_path():
    response = client.post(
        "/predict",
        json={
            "event_date": "2024-07-22",
            "event_text": "LPR下调",
            "event_type": "monetary_policy",
            "industry_code": "801780",
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "prob_up" in payload
    assert "final_direction" in payload

