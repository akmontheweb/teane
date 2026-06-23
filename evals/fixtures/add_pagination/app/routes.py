from flask import Blueprint, jsonify, request

bp = Blueprint("api", __name__)

ITEMS = [{"id": i, "name": f"item-{i}"} for i in range(1, 26)]


@bp.get("/items")
def list_items():
    return jsonify({"items": ITEMS})
