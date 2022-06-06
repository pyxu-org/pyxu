import warnings

import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


class L1Norm(pyco.ProxFunc):
    # f: \bR^{M} -> \bR
    #      x     -> \norm{x}{1}
    def __init__(self, M: int = None):
        super().__init__(shape=(1, M))
        self._lipschitz = self.lipschitz(M=M, warn=False)

    def lipschitz(
        self,
        M: pyct.Real = None,
        warn: bool = True,
    ):
        if self.dim is not None:
            return np.sqrt(self.dim)
        elif M is not None:
            return np.sqrt(M)
        else:
            if warn:
                msg = " ".join(
                    [
                        "Cannot infer tight Lipschitz constant for domain-agnostic function.",
                        "Recommendation: instantiate L1Norm() with the known dimension,",
                        "or provide a dimension hint to lipschitz() via Parameter[M].",
                    ]
                )
                warnings.warn(msg)
            return np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.linalg.norm(arr, ord=1, axis=-1, keepdims=True).astype(arr.dtype)
        return y

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr, tau):
        xp = pycu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau) * xp.sign(arr)
        return y

    def asloss(self, data=None):
        if data is None:
            return self
        else:
            return self.argshift(-data)


class TestL1Norm(conftest.ProxFuncT):
    @pytest.fixture(params=[None, 5])
    def dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def op(self, dim):
        return L1Norm(M=dim)

    @pytest.fixture
    def data_shape(self, op, dim):
        return (1, dim)

    @pytest.fixture
    def data_lipschitz(self, dim):
        return dict(
            in_=dict(
                M=dim,
                warn=False,
            ),
            out=np.inf if (dim is None) else np.sqrt(dim),
        )

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
