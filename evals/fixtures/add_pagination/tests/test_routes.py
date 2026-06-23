import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_items_returns_default_page(client):
    """Default page is 10 items (limit=10, offset=0)."""
    resp = client.get("/items")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "items" in body
    assert len(body["items"]) == 10
    assert body["items"][0]["id"] == 1
    assert body["items"][-1]["id"] == 10


def test_items_respects_limit(client):
    resp = client.get("/items?limit=5")
    body = resp.get_json()
    assert len(body["items"]) == 5
    assert body["items"][0]["id"] == 1
    assert body["items"][-1]["id"] == 5


def test_items_respects_offset(client):
    resp = client.get("/items?limit=3&offset=5")
    body = resp.get_json()
    assert len(body["items"]) == 3
    assert body["items"][0]["id"] == 6
    assert body["items"][-1]["id"] == 8


def test_items_clamps_negative_inputs(client):
    # limit < 1 → clamped to 1; offset < 0 → clamped to 0.
    resp = client.get("/items?limit=0&offset=-10")
    body = resp.get_json()
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == 1
