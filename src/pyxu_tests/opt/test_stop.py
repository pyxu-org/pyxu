import collections.abc as cabc

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.opt.stop as pxst
from pyxu_tests.operator.examples.test_linfunc import Sum

# We do not test MaxIter(), ManualStop() and MaxDuration() since they are trivial.


class TestAbsError:
    @pytest.mark.parametrize("var", ["x", "y"])
    @pytest.mark.parametrize(
        ["eps", "rank", "f", "norm", "satisfy_all", "state", "stop_val"],
        [
            # L0-norm, rank 1, 1 input ----------------------------------------
            [0.5, 1, None, 0, False, np.r_[1e-4], False],
            [1, 1, None, 0, False, np.r_[1e-4], True],
            [2.999, 1, None, 0, False, np.r_[:4], False],
            [4, 1, None, 0, False, np.r_[:4], True],
            # L0-norm, rank 1, 2 input ----------------------------------------
            [2, 1, None, 0, True, np.r_[:4].reshape(2, 2), True],
            [1, 1, None, 0, True, np.r_[:4].reshape(2, 2), False],  # playing with satisfy_all
            [1, 1, None, 0, False, np.r_[:4].reshape(2, 2), True],  # playing with satisfy_all
            # L0-norm, rank 2, 1 input ----------------------------------------
            [3, 2, None, 0, True, np.r_[:4].reshape(2, 2), True],
            [2.999, 2, None, 0, True, np.r_[:4].reshape(2, 2), False],
            # L0-norm, rank 2, 2 input ----------------------------------------
            [6, 2, None, 0, True, np.r_[:12].reshape(2, 2, 3), True],
            [4.9, 2, None, 0, False, np.r_[:12].reshape(2, 2, 3), False],
            [5, 2, None, 0, False, np.r_[:12].reshape(2, 2, 3), True],
            # L0.5-norm, f given, rank 1, 6 input -----------------------------
            [25, 1, Sum((5, 3, 4)), 0.5, False, np.r_[:360].reshape(3, 2, 5, 3, 4) / 360, True],
            [25, 1, Sum((5, 3, 4)), 0.5, True, np.r_[:360].reshape(3, 2, 5, 3, 4) / 360, False],
            [3016, 1, Sum((5, 3, 4)), 0.5, True, np.r_[:360].reshape(3, 2, 5, 3, 4) / 360, True],
        ],
    )
    def test_stop(
        self,
        var,
        # ---------
        eps,
        rank,
        f,
        norm,
        satisfy_all,
        state,
        stop_val,
        # ---------
        xp,
        width,
    ):
        sc = pxst.AbsError(
            eps=eps,
            var=var,
            rank=rank,
            f=f,
            norm=norm,
            satisfy_all=satisfy_all,
        )
        state = {var: xp.array(state, dtype=width.value)}

        assert sc.stop(state) == stop_val
        sc.info()  # just to make sure it doesn't crash


class TestRelError:
    # We disable RuntimeWarnings which may arise due to NaNs. (See comment below.)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("var", ["x", "y"])
    @pytest.mark.parametrize(
        ["eps", "rank", "f", "norm", "satisfy_all", "state0", "state1", "stop_val"],
        [
            # L1-norm, rank 1, 1 input ----------------------------------------
            [1, 1, None, 1, True, np.full((5,), 1), np.full((5,), 2), True],
            [0.9, 1, None, 1, True, np.full((5,), 1), np.full((5,), 2), False],
            [1e-6, 1, None, 1, True, np.full((5,), 0), np.full((5,), 0), True],  # 0/0 at 2nd iteration forces stop
            [1e-6, 1, None, 1, True, np.full((5,), 0), np.full((5,), 1), False],  # 1/0 at 2nd iteration forces continue
        ],
    )
    def test_stop(
        self,
        var,
        # ---------
        eps,
        rank,
        f,
        norm,
        satisfy_all,
        state0,
        state1,
        stop_val,
        # ---------
        xp,
        width,
    ):
        sc = pxst.RelError(
            eps=eps,
            var=var,
            rank=rank,
            f=f,
            norm=norm,
            satisfy_all=satisfy_all,
        )
        state0 = {var: xp.array(state0, dtype=width.value)}
        state1 = {var: xp.array(state1, dtype=width.value)}

        assert sc.stop(state0) is False
        assert sc.stop(state1) == stop_val
        sc.info()  # just to make sure it doesn't crash


@pytest.mark.parametrize(
    ["sc", "state_stream"],
    [
        [pxst.MaxIter(n=10), [{}] * 12],  # state meaningless
        [pxst.ManualStop(), [{}] * 12],  # state meaningless
        [pxst.AbsError(eps=3), [dict(x=np.r_[x]) for x in np.arange(10, 0, -1)]],
        [pxst.RelError(eps=1 / 6), [dict(x=np.r_[x]) for x in np.arange(10)]],
    ],
)
def test_clear(
    sc: pxa.StoppingCriterion,
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
