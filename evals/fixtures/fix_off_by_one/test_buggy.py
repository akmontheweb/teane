from buggy import sum_to


def test_sum_to_5():
    assert sum_to(5) == 15


def test_sum_to_0():
    assert sum_to(0) == 0


def test_sum_to_10():
    assert sum_to(10) == 55
