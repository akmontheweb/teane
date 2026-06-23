from widgets import Widget


def test_widget_constructs():
    w = Widget(name="cog", weight=2.5)
    assert w.name == "cog"
    assert w.weight == 2.5


def test_heavier_than():
    a = Widget(name="a", weight=1.0)
    b = Widget(name="b", weight=2.0)
    assert b.heavier_than(a)
    assert not a.heavier_than(b)
