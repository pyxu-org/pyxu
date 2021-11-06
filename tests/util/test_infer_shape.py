import pytest

import pycsou.util as pycu


class TestInferSumShape:
    @pytest.mark.parametrize(
        ["sh1", "sh2", "sh3"],
        [
            ((5, 3), (5, 3), (5, 3)),  # same shape
            ((5, 3), (1, 3), (5, 3)),  # codomain broadcast
            ((1, 3), (5, 3), (5, 3)),  # codomain broadcast (commutativity)
            ((5, 3), (5, None), (5, 3)),  # domain broadcast
            ((5, None), (5, 3), (5, 3)),  # domain broadcast (commutativity)
            ((5, None), (1, None), (5, None)),  # domain-agnostic
        ],
    )
    def test_valid(self, sh1, sh2, sh3):
        assert pycu.infer_sum_shape(sh1, sh2) == sh3

    @pytest.mark.parametrize(
        ["sh1", "sh2"],
        [
            ((None, 3), (5, 3)),  # codomain-agnostic
            ((5, 3), (None, 1)),  # codomain-agnostic (commutativity)
            ((5, 1), (5, 3)),  # domain broadcast
            ((5, 2), (5, 3)),  # domain broadcast
            ((5, 3), (2, 3)),  # codomain broadcast
            ((5, None), (2, 3)),  # domain-agnostic broadcast
            ((2, 3), (5, None)),  # domain-agnostic broadcast (commutativity)
        ],
    )
    def test_invalid(self, sh1, sh2):
        with pytest.raises(ValueError):
            pycu.infer_sum_shape(sh1, sh2)


class TestInferCompositionShape:
    @pytest.mark.parametrize(
        ["sh1", "sh2", "sh3"],
        [
            ((5, 3), (3, 4), (5, 4)),
            ((5, None), (3, 4), (5, 4)),
            ((5, 3), (3, None), (5, None)),
            ((5, None), (3, None), (5, None)),
        ],
    )
    def test_valid(self, sh1, sh2, sh3):
        assert pycu.infer_composition_shape(sh1, sh2) == sh3

    @pytest.mark.parametrize(
        ["sh1", "sh2"],
        [
            ((None, 3), (3, 4)),
            ((5, 3), (None, 4)),
            ((5, 3), (1, 4)),
            ((5, 3), (1, None)),
        ],
    )
    def test_invalid(self, sh1, sh2):
        with pytest.raises(ValueError):
            pycu.infer_composition_shape(sh1, sh2)
