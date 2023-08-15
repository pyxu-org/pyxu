import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.math.linalg as pxlg
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class L1Norm(pxa.ProxFunc):
    # f: \bR^{M} -> \bR
    #      x     -> \norm{x}{1}
    def __init__(self, M: int):
        super().__init__(shape=(1, M))
        self.lipschitz = np.sqrt(M)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        y = pxlg.norm(arr, ord=1, axis=-1, keepdims=True)
        return y

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr, tau):
        xp = pxu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y

    def asloss(self, data: pxt.NDArray = None) -> pxt.OpT:
        from pyxu.operator.func.loss import shift_loss

        op = shift_loss(op=self, data=data)
        return op


class TestL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, L1Norm(M=5)),),  # dim, op
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([7]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((5,)),
                    tau=1,
                ),
                out=np.zeros((5,)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2),
                    tau=1,
                ),
                out=np.array([-2, -1, 0, 0, 0]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 5, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestL1NormMoreau(conftest.DiffFuncT):
    disable_test = frozenset(
        conftest.DiffFuncT.disable_test
        | {
            # .asloss().moreau_envelope() makes sense, not the converse.
            "test_interface_asloss",
        }
    )

    @pytest.fixture(
        params=itertools.product(
            ((4, 2, L1Norm(M=4).moreau_envelope(mu=2)),),  # dim, mu, op
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][2], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def mu(self, _spec) -> int:
        return _spec[0][1]

    @pytest.fixture
    def op_orig(self, dim):
        return L1Norm(M=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_apply(self, op_orig, mu):
        dim = self._sanitize(op_orig.dim, 3)
        x = self._random_array((dim,), seed=7)
        y = op_orig.prox(x, mu)
        z = op_orig.apply(y) + (0.5 / mu) * (np.linalg.norm(y - x) ** 2)
        return dict(
            in_=dict(arr=x),
            out=z,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 6, self._sanitize(dim, 3)
        return self._random_array((N_test, dim), seed=5)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test, dim = 6, self._sanitize(dim, 3)
        return self._random_array((N_test, dim), seed=6)

    @pytest.fixture
    def data_grad(self, op_orig, mu):
        dim = self._sanitize(op_orig.dim, 3)
        x = self._random_array((dim,), seed=7)
        y = (x - op_orig.prox(x, mu)) / mu
        return dict(
            in_=dict(arr=x),
            out=y,
        )
