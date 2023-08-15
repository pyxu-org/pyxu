import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.math.linalg as pxlg
import pyxu.runtime as pxrt
import pyxu_tests.conftest as ct


def op_squareop():
    import pyxu_tests.operator.examples.test_squareop as tc

    return tc.CumSum(N=7)


def op_normalop():
    import pyxu_tests.operator.examples.test_normalop as tc

    rng = np.random.default_rng(seed=2)
    h = rng.normal(size=(7,))
    return tc.CircularConvolution(h=h)


def op_unitop():
    import pyxu_tests.operator.examples.test_unitop as tc

    return tc.Permutation(N=7)


def op_selfadjointop():
    import pyxu_tests.operator.examples.test_selfadjointop as tc

    return tc.SelfAdjointConvolution(N=7)


def op_posdefop():
    import pyxu_tests.operator.examples.test_posdefop as tc

    return tc.PSDConvolution(N=7)


def op_projop():
    import pyxu_tests.operator.examples.test_projop as tc

    return tc.Oblique(N=7, alpha=np.pi / 4)


def op_orthprojop():
    import pyxu_tests.operator.examples.test_orthprojop as tc

    return tc.ScaleDown(N=7)


class TestTrace:
    @pytest.fixture(
        params=[
            op_squareop(),
            op_normalop(),
            op_unitop(),
            op_selfadjointop(),
            op_posdefop(),
            op_projop(),
            op_orthprojop(),
        ]
    )
    def op(self, request) -> pxt.OpT:
        return request.param

    @pytest.fixture
    def _op_trace(self, op) -> float:
        # Ground truth trace
        tr = op.asarray().trace()
        return tr

    def test_value_explicit(self, op, _op_trace):
        tr = pxlg.trace(op)
        assert ct.allclose(tr, _op_trace, as_dtype=pxrt.getPrecision().value)

    def test_value_hutchpp(self, op, _op_trace):
        # Ensure computed trace (w/ default parameter values) satisfies statistical property stated
        # in hutchpp() docstring, i.e.: estimation error smaller than 1e-2 w/ probability 0.9
        N_trial = 100
        tr = np.array([pxlg.hutchpp(op) for _ in range(N_trial)])
        N_pass = sum(np.abs(tr - _op_trace) <= 1e-2)
        assert N_pass >= 0.9 * N_trial
