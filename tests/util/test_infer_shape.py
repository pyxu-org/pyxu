import pytest

import pycsou.util as pycu


def test_infer_sum_shape():
    valid = (
        ((5, 3), (5, 3), (5, 3)),  # same shape
        ((5, 3), (1, 3), (5, 3)),  # codomain broadcast
        ((1, 3), (5, 3), (5, 3)),  # codomain broadcast (commutativity)
        ((5, 3), (5, None), (5, 3)),  # domain broadcast
        ((5, None), (5, 3), (5, 3)),  # domain broadcast (commutativity)
        ((5, None), (1, None), (5, None)),  # domain-agnostic
    )
    for A, B, C in valid:
        assert pycu.infer_sum_shape(A, B) == C

    invalid = (
        ((None, 3), (5, 3)),  # codomain-agnostic
        ((5, 3), (None, 1)),  # codomain-agnostic (commutativity)
        ((5, 1), (5, 3)),  # domain broadcast
        ((5, 2), (5, 3)),  # domain broadcast
        ((5, 3), (2, 3)),  # codomain broadcast
        ((5, None), (2, 3)),  # domain-agnostic broadcast
        ((2, 3), (5, None)),  # domain-agnostic broadcast (commutativity)
    )
    for A, B in invalid:
        with pytest.raises(ValueError):
            pycu.infer_sum_shape(A, B)
