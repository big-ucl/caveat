from numpy import array

from caveat.evaluate.features import utils


def test_equals():
    a = {"a": (array([1, 2, 3]), array([1, 2, 3]))}
    b = {"a": (array([1, 2, 3]), array([1, 2, 3]))}
    assert utils.equals(a, b)

    a = {"a": (array([1, 2, 3]), array([1, 2, 3]))}
    b = {"a": (array([1, 2, 3]), array([1, 2, 4]))}
    assert not utils.equals(a, b)

    a = {"a": (array([1, 2, 3]), array([1, 2, 3]))}
    b = {"a": (array([1, 2]), array([1, 2]))}
    assert not utils.equals(a, b)

    a = {"a": (array([1, 2, 3]), array([1, 2, 3]))}
    b = {"b": (array([1, 2, 3]), array([1, 2, 3]))}
    assert not utils.equals(a, b)
