import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.math as pxm
import pyxu.operator as pxo
import pyxu_tests.conftest as ct


# Helper methods --------------------------------------------------------------
def reshape(
    op: pxt.OpT,
    dim: pxt.NDArrayShape = None,
    codim: pxt.NDArrayShape = None,
) -> pxt.OpT:
    # Reshape an operator to new dim/codim_shape.
    # (Useful for testing square operators where dim != codim.)
    opR = 1
    if dim is not None:
        opR = pxo.ReshapeAxes(
            dim_shape=dim,
            codim_shape=op.dim_shape,
        )

    opL = 1
    if codim is not None:
        opL = pxo.ReshapeAxes(
            dim_shape=op.codim_shape,
            codim_shape=codim,
        )

    op_reshape = opL * op * opR
    return op_reshape


def op_squareop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_squareop as tc

    return tc.CumSum(dim_shape=dim_shape)


class TestTrace:
    @pytest.fixture(
        params=[
            # dim_shape == codim_shape case -----------------------------------
            op_squareop((5,)),
            op_squareop((5, 3, 4)),
            # dim_shape != codim_shape case -----------------------------------
            reshape(op_squareop((5, 3, 4)), dim=None, codim=(2, 30)),
            reshape(op_squareop((5, 3, 4)), dim=(2, 30), codim=None),
        ]
    )
    def op(self, request) -> pxt.OpT:
        return request.param

    @pytest.fixture
    def _op_trace(self, op) -> float:
        # Ground truth trace
        A = op.asarray()
        B = A.reshape(op.codim_size, op.dim_size)
        tr = B.trace()
        return tr

    def test_value_explicit(self, op, _op_trace, xp, width):
        tr = pxm.trace(op, xp=xp, dtype=width.value)
        assert ct.allclose(tr, _op_trace, as_dtype=width.value)

    @pytest.mark.parametrize("seed", [0, 5, 135])
    def test_value_hutchpp(self, op, _op_trace, xp, width, seed):
        # Ensure computed trace (w/ default parameter values) satisfies statistical property stated
        # in hutchpp() docstring, i.e.: estimation error smaller than 1e-2 w/ probability 0.9
        N_trial = 100

        tr = np.array(
            [
                pxm.hutchpp(
                    op,
                    xp=xp,
                    dtype=width.value,
                    seed=seed,
                )
                for _ in range(N_trial)
            ]
        )

        N_pass = sum(np.abs(tr - _op_trace) <= 1e-2)
        assert N_pass >= 0.9 * N_trial
