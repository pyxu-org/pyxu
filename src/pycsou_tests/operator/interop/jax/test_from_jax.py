# How from_jax() tests work:
#
# The idea is to run the same tests as pycsou_tests/operator/examples, but where the operator is
# created using from_jax().
#
# To test a JAX-backed operator, inherit from JaxMixin and the suitable (pre-created) conftest.MapT
# subclass from examples/.

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pycsou.operator.interop as pycio
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.examples.test_difffunc as test_difffunc
import pycsou_tests.operator.examples.test_diffmap as test_diffmap
import pycsou_tests.operator.examples.test_func as test_func
import pycsou_tests.operator.examples.test_linop as test_linop
import pycsou_tests.operator.examples.test_map as test_map
import pycsou_tests.operator.examples.test_proxdifffunc as test_proxdifffunc
import pycsou_tests.operator.examples.test_proxfunc as test_proxfunc


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
        # JAX functions + arithmetic attributes which define the operator.
        # Arithmetic methods are assumed to work w/ stacking dimensions.
        #
        # This fixture should be overriden by sub-classes.
        #
        # If arithmetic methods can be omitted from `kwargs` and be auto-inferred by JAX, it is
        # recommended to test this by parameterizing the fixture.
        raise NotImplementedError

    @pytest.fixture
    def _op(self, data_shape, vectorize, jit, kwargs) -> pyct.OpT:
        # Use from_jax() to create a Pycsou operator.
        if vectorize:
            from pycsou.operator.interop.jax import _FromJax

            # only keep arithmetic methods
            vec = frozenset(kwargs.keys()) & _FromJax._meth
        else:
            vec = frozenset()

        op = pycio.from_jax(
            cls=self.base,
            shape=data_shape,
            vectorize=vec,
            jit=jit,
            enable_warnings=False,
            # Warnings are only emitted in _to_jax(), _from_jax():
            # Purpose-built tests to validate those methods are located in ./test_conversion.py
            **kwargs,
        )
        return op

    @pytest.fixture(
        params=itertools.product(
            [
                pycd.NDArrayInfo.NUMPY,
                pycd.NDArrayInfo.CUPY,
                # DASK inputs are not supported.
            ],
            pycrt.Width,
        )
    )
    def spec(self, _op, request) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
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

        return dict(
            _lipschitz=1,
            _diff_lipschitz=1,
            apply=j_apply,
        )


class TestJaxSquaredL2Norm(JaxMixin, test_difffunc.TestSquaredL2Norm):
    disable_test = test_difffunc.TestSquaredL2Norm.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_grad = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = (arr**2).sum(axis=-1, keepdims=True)
            return out

        def j_grad(arr: jax.Array) -> jax.Array:
            out = 2 * arr
            return out

        data = dict(
            _diff_lipschitz=2,
            apply=j_apply,
        )
        if define_grad:
            data.update(grad=j_grad)
        return data


class TestJaxL1Norm(JaxMixin, test_proxfunc.TestL1Norm):
    disable_test = test_proxfunc.TestL1Norm.disable_test | JaxMixin.disable_test

    @pytest.fixture
    def kwargs(self, dim) -> dict:
        def j_apply(arr: jax.Array) -> jax.Array:
            out = jnp.abs(arr).sum(axis=-1, keepdims=True)
            return out

        def j_prox(arr: jax.Array, tau: pyct.Real) -> jax.Array:
            out = jnp.fmax(0, jnp.fabs(arr) - tau) * jnp.sign(arr)
            return out

        data = dict(
            apply=j_apply,
            prox=j_prox,
        )
        if dim is not None:
            data.update(_lipschitz=np.sqrt(dim))
        return data


class TestJaxSquaredL2Norm2(JaxMixin, test_proxdifffunc.TestSquaredL2Norm):
    disable_test = test_proxdifffunc.TestSquaredL2Norm.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_grad = request.param

        def j_apply(arr: jax.Array) -> jax.Array:
            out = (arr**2).sum(axis=-1, keepdims=True)
            return out

        def j_grad(arr: jax.Array) -> jax.Array:
            out = 2 * arr
            return out

        def j_prox(arr: jax.Array, tau: pyct.Real) -> jax.Array:
            out = arr / (2 * tau + 1)
            return out

        data = dict(
            _diff_lipschitz=2,
            apply=j_apply,
            prox=j_prox,
        )
        if define_grad:
            data.update(grad=j_grad)
        return data


class TestJaxTile(JaxMixin, test_linop.TestTile):
    disable_test = test_linop.TestTile.disable_test | JaxMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, codim, dim, request) -> dict:
        define_adjoint = request.param
        M = codim // dim

        def j_apply(arr: jax.Array) -> jax.Array:
            out = jnp.concatenate([arr] * M, axis=-1)
            return out

        def j_adjoint(arr: jax.Array) -> jax.Array:
            sh = (*arr.shape[:-1], dim)
            out = arr.reshape((-1, M, dim)).sum(axis=-2).reshape(sh)
            return out

        data = dict(
            _lipschitz=np.sqrt(M),
            apply=j_apply,
        )
        if define_adjoint:
            data.update(adjoint=j_adjoint)
        return data
