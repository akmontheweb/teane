import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_echo_with_json_body(client):
    resp = client.post("/echo", json={"a": 1})
    assert resp.status_code == 200
    assert resp.get_json() == {"echo": {"a": 1}}


def test_echo_missing_body_returns_400(client):
    resp = client.post("/echo")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "json body required"}


def test_echo_invalid_json_returns_400(client):
    resp = client.post("/echo", data="not-json", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "json body required"}
