import itertools

import numpy as np
import numpy.linalg as npl
import pytest
import scipy.ndimage as scimage

import pycsou.operator as pycob
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest

if pycd.CUPY_ENABLED:
    import cupy.linalg as cpl

import collections.abc as cabc


# We disable PrecisionWarnings since Stencil() is not precision-agnostic, but the outputs
# computed must still be valid. We also disable NumbaPerformanceWarnings, as these appear
# due to using the overkill use of GPU for very small test input arrays.
@pytest.mark.filterwarnings(
    "ignore::pycsou.util.warning.PrecisionWarning", "ignore::numba.core.errors.NumbaPerformanceWarning"
)
class TestStencil(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            [2], [0]  # (1) 1D and (3) 3D kernels  # [0, 2]  # right or left centered kernels (wrt origin=1)
        )
    )
    def stencil_params(self, request):
        ndim, pos = request.param
        stencil_coefs = np.expand_dims(self._random_array((2,), seed=20), axis=list(np.arange(ndim - 1)))
        center = np.zeros(ndim, dtype=int)
        center[-1] = pos
        return stencil_coefs, center

    @pytest.fixture
    def stencil_coefs(self, stencil_params):
        return stencil_params[0]

    @pytest.fixture
    def center(self, stencil_params):
        return stencil_params[1]

    @pytest.fixture
    def arg_shape(self, stencil_params):
        return (8,) * stencil_params[0].ndim

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture
    def spec(
        self,
        stencil_coefs,
        center,
        arg_shape,
        ndi,
        width,
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        op = pycob.linop.base._StencilOp(
            stencil_coefs=ndi.module().asarray(stencil_coefs, dtype=width.value),
            center=center,
            arg_shape=arg_shape,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, op, arg_shape) -> conftest.DataLike:
        arr = self._random_array(((arg_shape[0] - 4) ** len(arg_shape),), seed=20)  # random seed for reproducibility
        arr = np.pad(arr.reshape(*[s - 4 for s in arg_shape]), ((2, 2),) * len(arg_shape), constant_values=0.0).ravel()
        kernel, center = op.stencil_coefs, op.center
        xp = pycu.get_array_module(kernel)
        kernel = kernel.get() if xp.__name__ == "cupy" else kernel

        origin = [
            0,
        ] + list(center)
        origin[-1] -= 1

        out = scimage.correlate(
            arr.reshape(-1, *op.arg_shape),
            kernel.reshape(1, *kernel.shape),
            origin=origin,
            mode="constant",
            cval=np.nan,
        )
        out[np.isnan(out)] = 0

        return dict(
            in_=dict(arr=arr),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_adjoint(self, op, arg_shape) -> conftest.DataLike:
        arr = self._random_array(((arg_shape[0] - 4) ** len(arg_shape),), seed=20)  # random seed for reproducibility
        arr = np.pad(arr.reshape(*[s - 4 for s in arg_shape]), ((2, 2),) * len(arg_shape), constant_values=0.0).ravel()

        out = (op.asarray().T @ arr.T).T

        return dict(
            in_=dict(arr=arr),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.OpShape:
        size = np.prod(arg_shape).item()
        sh = (size, size)
        return sh

    @pytest.fixture
    def data_pinv(self, op, _damp, data_apply):

        arr = data_apply["out"]
        xp = pycu.get_array_module(arr)
        xpl = cpl if xp.__name__ == "cupy" else npl

        B = op.asarray(xp=xp, dtype=pycrt.Width.DOUBLE.value)
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim):
            A[i, i] += _damp

        out, *_ = xpl.lstsq(A, B.T @ arr)
        data = dict(
            in_=dict(
                arr=arr,
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out,
        )
        return data

    @pytest.fixture
    def data_pinvT(self, op, _damp, data_apply):
        arr = data_apply["in_"]["arr"]
        xp = pycu.get_array_module(arr)
        xpl = cpl if xp.__name__ == "cupy" else npl
        B = op.asarray(xp=xp, dtype=pycrt.Width.DOUBLE.value)
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim):
            A[i, i] += _damp

        out, *_ = xpl.lstsq(A, arr)
        out = B @ out
        data = dict(
            in_=dict(
                arr=arr,
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out,
        )
        return data

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        shape = (N_test,) + ((op.arg_shape[0] - 4) ** len(op.arg_shape),)
        x = self._random_array(shape)
        x = np.pad(
            x.reshape(N_test, *[s - 4 for s in op.arg_shape]),
            ((0, 0),) + ((2, 2),) * len(op.arg_shape),
            constant_values=0.0,
        ).reshape(N_test, -1)
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        shape = (N_test,) + ((op.arg_shape[0] - 4) ** len(op.arg_shape),)
        x = self._random_array(shape)
        x = np.pad(
            x.reshape(N_test, *[s - 4 for s in op.arg_shape]),
            ((0, 0),) + ((2, 2),) * len(op.arg_shape),
            constant_values=0.0,
        ).reshape(N_test, -1)
        return x

    def test_math_adjoint(self, op, xp, width):
        # Added zero-padding (wrt conftest)
        self._skip_if_disabled()
        N = 20
        shape = (N,) + ((op.arg_shape[0] - 4) ** len(op.arg_shape),)
        x = self._random_array(shape, xp=xp, width=width)
        x = np.pad(
            x.reshape(N, *[s - 4 for s in op.arg_shape]), ((0, 0),) + ((2, 2),) * len(op.arg_shape), constant_values=0.0
        ).ravel()

        y = self._random_array(shape, xp=xp, width=width)
        y = np.pad(
            y.reshape(N, *[s - 4 for s in op.arg_shape]), ((0, 0),) + ((2, 2),) * len(op.arg_shape), constant_values=0.0
        ).ravel()

        ip = lambda a, b: (a * b).sum(axis=-1)  # (N, Q) * (N, Q) -> (N,)
        lhs = ip(op.adjoint(x), y)
        rhs = ip(x, op.apply(y))

        assert conftest.allclose(lhs, rhs, as_dtype=width.value)
