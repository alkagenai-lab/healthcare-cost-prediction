from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Healthcare Treatment Cost Prediction API"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_prediction_endpoint():
    payload = {
        "age": 45,
        "sex": "male",
        "bmi": 30.0,
        "children": 2,
        "smoker": "no",
        "region": "northwest",
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_cost" in data
    assert isinstance(data["predicted_cost"], (int, float))
    assert data["predicted_cost"] > 0