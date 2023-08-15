# How from_jax() tests work:
#
# The idea is to run the same tests as pyxu_tests/operator/examples, but where the operator is
# created using from_jax().
#
# To test a JAX-backed operator, inherit from JaxMixin and the suitable (pre-created) conftest.MapT
# subclass from examples/.
#
# Only pyxu.abc.operator.QuadraticFunc() is untested. (Reason: cannot import the `xp` fixture in
# Fixture[kwargs], hence test logic complexity should increase a lot just to test this operator.)
# [2023.04.05] Manual tests of QuadraticFunc() show it works via from_jax().

import itertools

import jax
import jax.numpy as jnp
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop as pxio
import pyxu.runtime as pxrt
import pyxu_tests.operator.examples.test_difffunc as test_difffunc
import pyxu_tests.operator.examples.test_diffmap as test_diffmap
import pyxu_tests.operator.examples.test_func as test_func
import pyxu_tests.operator.examples.test_linfunc as test_linfunc
import pyxu_tests.operator.examples.test_linop as test_linop
import pyxu_tests.operator.examples.test_map as test_map
import pyxu_tests.operator.examples.test_normalop as test_normalop
import pyxu_tests.operator.examples.test_orthprojop as test_orthprojop
import pyxu_tests.operator.examples.test_posdefop as test_posdefop
import pyxu_tests.operator.examples.test_projop as test_projop
import pyxu_tests.operator.examples.test_proxdifffunc as test_proxdifffunc
import pyxu_tests.operator.examples.test_proxfunc as test_proxfunc
import pyxu_tests.operator.examples.test_selfadjointop as test_selfadjointop
import pyxu_tests.operator.examples.test_squareop as test_squareop
import pyxu_tests.operator.examples.test_unitop as test_unitop


class JaxMixin:
    disable_test = {
        # from_jax() does not always respect input precision in absence of context manager.
        # (See from_jax() notes.)
        "test_prec_apply",
        "test_prec_call",
        "test_prec_grad",
        "test_prec_prox",
        "test_prec_fenchel_prox",
        "test_prec_adjoint",
        "test_prec_pinv",
        "test_prec_call_dagger",
        "test_prec_apply_dagger",
        "test_prec_to_sciop",
        "test_interface_asloss",  # not worth modifying test architecture specifically for this test.
    }

    # Internal helpers --------------------------------------------------------
    @classmethod
    def _metric(
        cls,
        a: pxt.NDArray,
        b: pxt.NDArray,
        as_dtype: pxt.DType,
    ) -> bool:
        # JAX precision cannot be changed at runtime.
        # Values are thus compared based on the actual precision.
        using_f64 = jax.config.jax_enable_x64
        width = {True: pxrt.Width.DOUBLE, False: pxrt.Width.SINGLE}[using_f64]
        return super()._metric(a, b, as_dtype=width.value)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[True, False])
    def jit(self, request) -> bool:
        # Should JAX methods be JIT-compiled?
        return request.param

    @pytest.fixture(params=[True, False])
    def vectorize(self, request) -> bool:
        # Should JAX methods be vectorized?
        return request.param

    @pytest.fixture
    def kwargs(self) -> dict:
        # JAX functions which define the operator.
        # Arithmetic methods are assumed to work w/ stacking dimensions.
        #
        # This fixture should be overriden by sub-classes.
        #
        # If arithmetic methods can be omitted from `kwargs` and be auto-inferred by JAX, it is
        # recommended to test this by parameterizing the fixture.
        raise NotImplementedError

    @pytest.fixture
    def _op(self, data_shape, vectorize, jit, kwargs) -> pxt.OpT:
        # Use from_jax() to create a Pyxu operator.
        if vectorize:
            from pyxu.operator.interop.jax import _FromJax

            # only keep arithmetic methods
            vec = frozenset(kwargs.keys()) & _FromJax._meth
        else:
            vec = frozenset()

        op = pxio.from_jax(
            cls=self.base,
            shape=data_shape,
            vectorize=vec,
            jit=jit,
            enable_warnings=False,  # Warnings are only emitted in _to_jax(), _from_jax():
            **kwargs,
        )
        return op

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
                # DASK inputs are not supported.
            ],
            pxrt.Width,
        )
    )
    def spec(self, _op, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        return _op, ndi, width


class TestJaxReLU(JaxMixin, test_map.TestReLU):
    disable_test = test_map.TestReLU.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            out = arr.clip(min=0)
            return out

        return dict(apply=j_apply)


class TestJaxMedian(JaxMixin, test_func.TestMedian):
    disable_test = test_func.TestMedian.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            out = jnp.median(arr, axis=-1, keepdims=True)
            return out

        return dict(apply=j_apply)


class TestJaxSin(JaxMixin, test_diffmap.TestSin):
    disable_test = test_diffmap.TestSin.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            out = jnp.sin(arr)
            return out

        return dict(apply=j_apply)


class TestJaxSquaredL2Norm(JaxMixin, test_difffunc.TestSquaredL2Norm):
    disable_test = test_difffunc.TestSquaredL2Norm.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = (arr**2).sum(axis=-1, keepdims=True)
            return out

        def j_grad(arr: jax.Array) -> jax.Array:
            out = 2 * arr
            return out

        data = dict(apply=j_apply)
        if define_secondary:
            data.update(
                grad=j_grad,
            )
        return data


class TestJaxL1Norm(JaxMixin, test_proxfunc.TestL1Norm):
    disable_test = test_proxfunc.TestL1Norm.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self, dim) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            out = jnp.abs(arr).sum(axis=-1, keepdims=True)
            return out

        def j_prox(arr: jax.Array, tau: pxt.Real) -> jax.Array:
            out = jnp.fmax(0, jnp.fabs(arr) - tau) * jnp.sign(arr)
            return out

        data = dict(
            apply=j_apply,
            prox=j_prox,
        )
        return data


class TestJaxSquaredL2Norm2(JaxMixin, test_proxdifffunc.TestSquaredL2Norm):
    disable_test = test_proxdifffunc.TestSquaredL2Norm.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = (arr**2).sum(axis=-1, keepdims=True)
            return out

        def j_grad(arr: jax.Array) -> jax.Array:
            out = 2 * arr
            return out

        def j_prox(arr: jax.Array, tau: pxt.Real) -> jax.Array:
            out = arr / (2 * tau + 1)
            return out

        data = dict(
            apply=j_apply,
            prox=j_prox,
        )
        if define_secondary:
            data.update(
                grad=j_grad,
            )
        return data


class TestJaxTile(JaxMixin, test_linop.TestTile):
    disable_test = test_linop.TestTile.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, codim, dim, request) -> dict:
        define_secondary = request.param
        M = codim // dim

        def j_apply(arr: jax.Array) -> jax.Array:
            out = jnp.concatenate([arr] * M, axis=-1)
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            sh = (*arr.shape[:-1], dim)
            out = arr.reshape((-1, M, dim)).sum(axis=-2).reshape(sh)
            return out

        data = dict(apply=j_apply)
        if define_secondary:
            data.update(
                adjoint=j_adjoint,
            )
        return data


class TestJaxScaledSum(JaxMixin, test_linfunc.TestScaledSum):
    disable_test = test_linfunc.TestScaledSum.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = arr.cumsum(axis=-1).sum(axis=-1, keepdims=True)
            return out

        def j_grad(arr: jax.Array) -> jax.Array:
            g = jnp.arange(dim, 0, -1, dtype=arr.dtype)
            out = jnp.broadcast_to(g, arr.shape)
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            scale = jnp.arange(dim, 0, -1, dtype=arr.dtype)
            out = scale * arr
            return out

        data = dict(apply=j_apply)
        if define_secondary:
            data.update(
                grad=j_grad,
                adjoint=j_adjoint,
            )
        return data


class TestJaxCumSum(JaxMixin, test_squareop.TestCumSum):
    disable_test = test_squareop.TestCumSum.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = arr.cumsum(axis=-1)
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            out = arr[..., ::-1].cumsum(axis=-1)[..., ::-1]
            return out

        data = dict(apply=j_apply)
        if define_secondary:
            data.update(
                adjoint=j_adjoint,
            )
        return data


class TestJaxCircularConvolution(JaxMixin, test_normalop.TestCircularConvolution):
    disable_test = test_normalop.TestCircularConvolution.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        h_FW = test_normalop.TestCircularConvolution.filter.reshape(-1)
        dim = h_FW.size
        h_BW = h_FW[jnp.array([0, *jnp.arange(1, dim)[::-1]])]

        def _circ_convolve(h: jax.Array, x: jax.Array) -> jax.Array:
            hF = jnp.fft.fft(h, axis=-1)
            xF = jnp.fft.fft(x, axis=-1)
            out = jnp.fft.ifft(hF * xF, axis=-1).real
            return out

        def j_apply(arr: jax.Array) -> jax.Array:
            out = _circ_convolve(h_FW, arr)
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            out = _circ_convolve(h_BW, arr)
            return out

        data = dict(apply=j_apply)
        if define_secondary:
            data.update(
                adjoint=j_adjoint,
            )
        return data


class TestJaxPermutation(JaxMixin, test_unitop.TestPermutation):
    disable_test = test_unitop.TestPermutation.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = arr[..., ::-1]
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            out = j_apply(arr)
            return out

        def j_pinv(arr: jax.Array, damp: pxt.Real) -> jax.Array:
            out = j_adjoint(arr) / (1 + damp)
            return out

        data = dict(
            apply=j_apply,
        )
        if define_secondary:
            data.update(
                adjoint=j_adjoint,
                pinv=j_pinv,
            )
        return data


class TestJaxSelfAdjointConvolution(JaxMixin, test_selfadjointop.TestSelfAdjointConvolution):
    disable_test = test_selfadjointop.TestSelfAdjointConvolution.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self, dim) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            hF = jnp.asarray(test_selfadjointop.filterF(dim))
            out = jnp.fft.ifft(jnp.fft.fft(arr, axis=-1) * hF, axis=-1).real
            return out

        return dict(apply=j_apply)


class TestJaxPSDConvolution(JaxMixin, test_posdefop.TestPSDConvolution):
    disable_test = test_posdefop.TestPSDConvolution.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self, dim) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            hF = jnp.asarray(test_posdefop.filterF(dim))
            out = jnp.fft.ifft(jnp.fft.fft(arr, axis=-1) * hF, axis=-1).real
            return out

        return dict(apply=j_apply)


class TestJaxOblique(JaxMixin, test_projop.TestOblique):
    disable_test = test_projop.TestOblique.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, alpha, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            x = jnp.zeros_like(arr)
            out = jax.lax.dynamic_update_index_in_dim(
                x,
                update=alpha * arr[..., 0] + arr[..., -1],
                index=-1,
                axis=-1,
            )
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            x = jnp.zeros_like(arr)
            y = jax.lax.dynamic_update_index_in_dim(
                x,
                update=alpha * arr[..., -1],
                index=0,
                axis=-1,
            )
            out = jax.lax.dynamic_update_index_in_dim(
                y,
                update=arr[..., -1],
                index=-1,
                axis=-1,
            )
            return out

        data = dict(
            apply=j_apply,
        )
        if define_secondary:
            data.update(
                adjoint=j_adjoint,
            )
        return data


class TestJaxScaleDown(JaxMixin, test_orthprojop.TestScaleDown):
    disable_test = test_orthprojop.TestScaleDown.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = jax.lax.dynamic_update_index_in_dim(
                arr,
                update=jnp.zeros_like(arr, shape=(*arr.shape[:-1], 1)),
                index=-1,
                axis=-1,
            )
            return out

        def j_pinv(arr: jax.Array, damp: pxt.Real) -> jax.Array:
            out = j_apply(arr) / (1 + damp)
            return out

        data = dict(
            apply=j_apply,
        )
        if define_secondary:
            data.update(
                pinv=j_pinv,
            )
        return data
