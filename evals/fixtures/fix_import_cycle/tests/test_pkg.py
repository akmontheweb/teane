from pkg.a import A
from pkg.b import B


def test_a_greet():
    assert A().greet() == "A hello from B"


def test_b_partner():
    assert B().partner() == "A"
