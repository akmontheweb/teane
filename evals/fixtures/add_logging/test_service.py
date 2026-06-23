import logging

from service import handle_request


def test_handle_request_echoes():
    out = handle_request({"x": 1})
    assert out == {"ok": True, "echo": {"x": 1}}


def test_handle_request_logs_keys(caplog):
    with caplog.at_level(logging.INFO, logger="service"):
        handle_request({"foo": 1, "bar": 2})
    service_records = [r for r in caplog.records if r.name == "service" and r.levelno == logging.INFO]
    assert len(service_records) == 1, (
        f"expected exactly one INFO record on the 'service' logger; got {service_records!r}"
    )
