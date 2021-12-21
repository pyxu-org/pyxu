import collections.abc as cabc

import numpy as np
import pytest

import pycsou.abc.solver as pycs
import pycsou.opt.stop as stop


@pytest.mark.parametrize(
    ["sc", "state_stream"],
    [
        [stop.MaxIter(n=10), [{}] * 12],  # state meaningless
        [stop.ManualStop(), [{}] * 12],  # state meaningless
        # [stop.MaxDuration(), [{}] * 10],  # MaxDuration is never 100% reproducible
        [stop.AbsError(eps=3), [dict(primal=_) for _ in np.arange(10, 0, -1)]],
        [stop.RelError(eps=1 / 6), [dict(primal=_) for _ in np.arange(10)]],
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
