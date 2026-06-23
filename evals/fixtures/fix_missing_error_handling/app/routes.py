from flask import Blueprint, jsonify, request

bp = Blueprint("api", __name__)


@bp.post("/echo")
def echo():
    body = request.get_json()
    return jsonify({"echo": body})
