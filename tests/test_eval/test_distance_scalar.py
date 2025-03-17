from numpy import array

from caveat.evaluate.distance import scalar


def test_ape_binary():
    a = (array([0, 1]), array([0, 1]))
    assert scalar.mape(a, a) == 0
    b = (array([0, 1]), array([0, 100]))
    assert scalar.mape(a, b) == 0
    b = (array([0, 1]), array([1, 0]))
    assert scalar.mape(a, b) == 1
    b = (array([0, 1]), array([0.5, 0.5]))
    assert scalar.mape(a, b) == 1
    b = (array([0, 1]), array([5, 5]))
    assert scalar.mape(a, b) == 1
    # todo


def test_mse():
    a = (array([0, 1]), array([0, 1]))
    assert scalar.mse(a, a) == 0
    b = (array([0, 1]), array([1, 0]))
    assert scalar.mse(a, b) == 1


def test_abs_diff():
    a = (array([10]), array([10]))
    assert scalar.abs_av_diff(a, a) == 0
    b = (array([8, 10]), array([5, 5]))
    assert scalar.abs_av_diff(a, b) == 1
    c = (array([8, 12]), array([5, 5]))
    assert scalar.abs_av_diff(a, c) == 0
