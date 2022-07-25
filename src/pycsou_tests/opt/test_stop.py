import collections.abc as cabc

import numpy as np
import pytest

import pycsou.abc.solver as pycs
import pycsou.opt.stop as stop

# We do not test MaxIter(), ManualStop() and MaxDuration() since they are trivial.


class TestAbsError:
    @pytest.mark.parametrize(
        ["eps", "f", "state", "stop_val"],
        [
            [3, None, 4, False],
            [3, None, 3, True],
            [3, None, 2, True],
            [3, lambda _: _**2, 4, False],
            [3, lambda _: _**2, 3, False],
            [3, lambda _: _**2, np.sqrt(3), True],
            [3, lambda _: _**2, 1, True],
        ],
    )
    def test_scalar_in(self, eps, f, state, stop_val):
        sc = stop.AbsError(eps=eps, f=f)
        state = dict(x=state)
        assert sc.stop(state) == stop_val
        sc.info()  # just to make sure it doesn't crash

    @pytest.mark.parametrize(
        ["eps", "f", "satisfy_all", "state", "stop_val"],
        [
            # 1 input, function
            [np.sqrt(14), None, True, np.arange(1, 4), True],
            [np.sqrt(14), None, True, np.arange(1, 5), False],
            [6, lambda _: _.sum(axis=-1, keepdims=True), True, np.broadcast_to(np.arange(1, 4), (1, 3)), True],
            [6, lambda _: _.sum(axis=-1, keepdims=True), True, np.broadcast_to(np.arange(1, 5), (1, 4)), False],
            # N input, satisfy_[any/all]
            [6.5, None, True, np.array([np.linspace(1, 4, 5), np.linspace(1, 5, 5)]), False],
            [6.5, None, False, np.array([np.linspace(1, 4, 5), np.linspace(1, 5, 5)]), True],
        ],
    )
    def test_array_in(self, eps, f, satisfy_all, state, stop_val, xp):
        sc = stop.AbsError(eps=eps, f=f, satisfy_all=satisfy_all)
        state = dict(x=xp.array(state))  # test all possible array backends.
        assert sc.stop(state) == stop_val
        sc.info()  # just to make sure it doesn't crash


class TestRelError:
    @pytest.mark.parametrize(
        ["eps", "f", "state", "stop_val"],
        [
            [0.5, None, (1, 2), False],
            [0.5, None, (2, 3), True],
            [0.5, None, (3, 4), True],
            [0.3, lambda _: _**2, (1, 2), False],
            [0.3, lambda _: _**2, (2, 3), False],
            [0.9, lambda _: _**2, (3, 4), True],
        ],
    )
    def test_scalar_in(self, eps, f, state, stop_val):
        sc = stop.RelError(eps=eps, f=f)
        state0 = dict(x=state[0])
        assert sc.stop(state0) == False
        state1 = dict(x=state[1])
        assert sc.stop(state1) == stop_val
        sc.info()  # just to make sure it doesn't crash

    @pytest.mark.parametrize(
        ["eps", "f", "satisfy_all", "state", "stop_val"],
        [
            # 1 input, function
            [0.6, None, True, (np.r_[1, 1, 1], np.r_[1, 2, 1]), True],
            [0.23, None, True, (np.r_[2, 2, 2], np.r_[2, 3, 2]), False],
            [
                0.4,
                lambda _: _.sum(axis=-1, keepdims=True),
                True,
                (
                    np.broadcast_to(np.r_[1, 1, 1], (1, 3)),
                    np.broadcast_to(np.r_[1, 2, 1], (1, 3)),
                ),
                True,
            ],
            [
                0.16,
                lambda _: _.sum(axis=-1, keepdims=True),
                True,
                (
                    np.broadcast_to(np.r_[2, 2, 2], (1, 3)),
                    np.broadcast_to(np.r_[2, 3, 2], (1, 3)),
                ),
                False,
            ],
            # N input, satisfy_[any/all]
            [
                0.3,
                None,
                True,
                (
                    np.array([[1, 1, 1], [2, 2, 2]]),
                    np.array([[1, 2, 1], [2, 3, 2]]),
                ),
                False,
            ],
            [
                0.3,
                None,
                False,
                (
                    np.array([[1, 1, 1], [2, 2, 2]]),
                    np.array([[1, 2, 1], [2, 3, 2]]),
                ),
                True,
            ],
        ],
    )
    def test_array_in(self, eps, f, satisfy_all, state, stop_val, xp):
        sc = stop.RelError(eps=eps, f=f, satisfy_all=satisfy_all)
        state0 = dict(x=xp.array(state[0]))  # test all possible array backends.
        assert sc.stop(state0) == False
        state1 = dict(x=xp.array(state[1]))  # test all possible array backends.
        assert sc.stop(state1) == stop_val
        sc.info()  # just to make sure it doesn't crash


@pytest.mark.parametrize(
    ["sc", "state_stream"],
    [
        [stop.MaxIter(n=10), [{}] * 12],  # state meaningless
        [stop.ManualStop(), [{}] * 12],  # state meaningless
        # [stop.MaxDuration(), [{}] * 10],  # MaxDuration is never 100% reproducible
        [stop.AbsError(eps=3), [dict(x=_) for _ in np.arange(10, 0, -1)]],
        [stop.RelError(eps=1 / 6), [dict(x=_) for _ in np.arange(10)]],
    ],
)
def test_clear(
    sc: pycs.StoppingCriterion,
    state_stream: cabc.Sequence[cabc.Mapping],
):
    """
    Verify that `StoppingCriterion.clear()` makes `sc` reusable.

    Algorithm:
    * feed `sc` with `state_stream`;
    * memoize stop()/info() outputs produced;
    * call clear();
    * verify that `sc` fed with `state_stream` again produces exactly the same output.
    """

    def memoize(sc, state_stream):
        h_stop, h_info = [], []  # history
        sc.clear()
        for state in state_stream:
            h_stop.append(sc.stop(state))
            h_info.append(sc.info())
        return h_stop, h_info

    h1_stop, h1_info = memoize(sc, state_stream)
    h2_stop, h2_info = memoize(sc, state_stream)
    assert h1_stop == h2_stop
    assert h1_info == h2_info
