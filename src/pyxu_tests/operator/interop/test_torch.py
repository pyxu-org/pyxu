# How from_torch() tests work:
#
# The idea is to run the same tests as pyxu_tests/operator/examples, but where the operator is
# created using from_torch().
#
# To test a torch-backed operator, inherit from TorchMixin and the suitable (pre-created) conftest.MapT
# subclass from examples/.
#
# Only pyxu.abc.operator.QuadraticFunc() is untested. (Reason: cannot import the `xp` fixture in
# Fixture[kwargs], hence test logic complexity should increase a lot just to test this operator.)

import itertools

import numpy as np
import pytest
import torch

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop as pxio
import pyxu.runtime as pxrt
import pyxu_tests.operator.examples.test_difffunc as test_difffunc
import pyxu_tests.operator.examples.test_diffmap as test_diffmap
import pyxu_tests.operator.examples.test_linfunc as test_linfunc
import pyxu_tests.operator.examples.test_linop as test_linop
import pyxu_tests.operator.examples.test_map as test_map
import pyxu_tests.operator.examples.test_normalop as test_normalop
import pyxu_tests.operator.examples.test_orthprojop as test_orthprojop
import pyxu_tests.operator.examples.test_projop as test_projop
import pyxu_tests.operator.examples.test_proxdifffunc as test_proxdifffunc
import pyxu_tests.operator.examples.test_proxfunc as test_proxfunc
import pyxu_tests.operator.examples.test_squareop as test_squareop
import pyxu_tests.operator.examples.test_unitop as test_unitop


class TorchMixin:
    disable_test = {
        # from_torch() does not always respect input precision in absence of context manager.
        # (See from_torch() notes.)
        "test_prec_apply",
        "test_prec_call",
        "test_prec_grad",
        "test_prec_prox",
        "test_prec_fenchel_prox",
        "test_prec_adjoint",
        "test_prec_pinv",
        "test_prec_call_dagger",
        "test_prec_apply_dagger",
    }

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[False])
    def jit(self, request) -> bool:
        # Should Torch methods be JIT-compiled?
        return request.param

    @pytest.fixture(params=[True, False])
    def vectorize(self, request) -> bool:
        # Should Torch methods be vectorized?
        return request.param

    @pytest.fixture
    def kwargs(self) -> dict:
        # Torch functions which define the operator.
        # Arithmetic methods are assumed to work w/ stacking dimensions.
        #
        # This fixture should be overriden by sub-classes.
        #
        # If arithmetic methods can be omitted from `kwargs` and be auto-inferred by Torch, it is
        # recommended to test this by parameterizing the fixture.
        raise NotImplementedError

    @pytest.fixture
    def _op(self, dim_shape, codim_shape, vectorize, jit, kwargs) -> pxt.OpT:
        # Use from_torch() to create a Pyxu operator.
        if vectorize:
            from pyxu.operator.interop.torch import _FromTorch

            # only keep arithmetic methods
            vec = frozenset(kwargs.keys()) & _FromTorch._meth
        else:
            vec = frozenset()

        op = pxio.from_torch(
            cls=self.base,
            dim_shape=dim_shape,
            codim_shape=codim_shape,
            vectorize=vec,
            jit=jit,
            enable_warnings=False,  # Warnings are only emitted in _to_torch(), _from_torch():
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


class TestTorchReLU(TorchMixin, test_map.TestReLU):
    disable_test = test_map.TestReLU.disable_test | TorchMixin.disable_test

    @pytest.fixture
    def kwargs(self) -> dict:
        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            out = arr.clip(min=0)
            return out

        return dict(apply=t_apply)


class TestTorchSin(TorchMixin, test_diffmap.TestSin):
    disable_test = test_diffmap.TestSin.disable_test | TorchMixin.disable_test

    @pytest.fixture
    def kwargs(self) -> dict:
        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            out = torch.sin(arr)
            return out

        return dict(apply=t_apply)


class TestTorchSquaredL2Norm(TorchMixin, test_difffunc.TestSquaredL2Norm):
    disable_test = test_difffunc.TestSquaredL2Norm.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim_shape, request) -> dict:
        define_secondary = request.param
        dim_rank = len(dim_shape)

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            axis = tuple(range(-dim_rank, 0))
            out = (arr**2).sum(dim=axis).unsqueeze(-1)
            return out

        def t_grad(arr: torch.Tensor) -> torch.Tensor:
            out = 2 * arr
            return out

        data = dict(apply=t_apply)
        if define_secondary:
            data.update(
                grad=t_grad,
            )
        return data


class TestTorchL1Norm(TorchMixin, test_proxfunc.TestL1Norm):
    disable_test = test_proxfunc.TestL1Norm.disable_test | TorchMixin.disable_test

    @pytest.fixture
    def kwargs(self, dim_shape) -> dict:
        dim_rank = len(dim_shape)

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            axis = tuple(range(-dim_rank, 0))
            out = torch.abs(arr).sum(dim=axis).unsqueeze(-1)
            return out

        def t_prox(arr: torch.Tensor, tau: pxt.Real) -> torch.Tensor:
            out = torch.max(torch.tensor([0.0], dtype=arr.dtype), torch.abs(arr) - tau) * torch.sign(arr)
            return out

        data = dict(
            apply=t_apply,
            prox=t_prox,
        )
        return data


class TestTorchSquaredL2Norm2(TorchMixin, test_proxdifffunc.TestSquaredL2Norm):
    disable_test = test_proxdifffunc.TestSquaredL2Norm.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim_shape, request) -> dict:
        define_secondary = request.param
        dim_rank = len(dim_shape)

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            axis = tuple(range(-dim_rank, 0))
            out = (arr**2).sum(dim=axis)
            return out

        def t_grad(arr: torch.Tensor) -> torch.Tensor:
            out = 2 * arr
            return out

        def t_prox(arr: torch.Tensor, tau: pxt.Real) -> torch.Tensor:
            out = arr / (2 * tau + 1)
            return out

        data = dict(
            apply=t_apply,
            prox=t_prox,
        )
        if define_secondary:
            data.update(
                grad=t_grad,
            )
        return data


class TestTorchSum(TorchMixin, test_linop.TestSum):
    disable_test = test_linop.TestSum.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim_shape, codim_shape, request) -> dict:
        define_secondary = request.param
        dim_rank = len(dim_shape)
        codim_rank = len(codim_shape)

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            if dim_rank == 1:
                out = arr.sum(dim=-1, keepdims=True)
            else:
                out = arr.sum(dim=-1)
            return out

        def t_adjoint(arr: torch.Tensor) -> torch.Tensor:
            sh = arr.shape[:-codim_rank]

            if dim_rank == 1:
                out = torch.broadcast_to(arr, (*sh, *dim_shape))
            else:
                out = torch.broadcast_to(arr.unsqueeze(-1), (*sh, *dim_shape))
            return out

        data = dict(apply=t_apply)
        if define_secondary:
            data.update(
                adjoint=t_adjoint,
            )
        return data


class TestTorchSum2(TorchMixin, test_linfunc.TestSum):
    disable_test = test_linfunc.TestSum.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim_shape, codim_shape, request) -> dict:
        define_secondary = request.param
        dim_rank = len(dim_shape)
        codim_rank = len(codim_shape)

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            axis = tuple(range(-dim_rank, 0))
            out = arr.sum(dim=axis).unsqueeze(-1)
            return out

        def t_grad(arr: torch.Tensor) -> torch.Tensor:
            sh = arr.shape[:-dim_rank]
            out = torch.ones((*sh, *dim_shape), dtype=arr.dtype)
            return out

        def t_adjoint(arr: torch.Tensor) -> torch.Tensor:
            sh = arr.shape[:-codim_rank]
            extend = (np.newaxis,) * (dim_rank - 1)

            out = torch.broadcast_to(
                arr[..., *extend],
                (*sh, *dim_shape),
            )
            return out

        data = dict(apply=t_apply)
        if define_secondary:
            data.update(
                grad=t_grad,
                adjoint=t_adjoint,
            )
        return data


class TestTorchCumSum(TorchMixin, test_squareop.TestCumSum):
    disable_test = test_squareop.TestCumSum.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            out = arr.cumsum(dim=-1)
            return out

        def t_adjoint(arr: torch.Tensor) -> torch.Tensor:
            out = arr.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
            return out

        data = dict(apply=t_apply)
        if define_secondary:
            data.update(
                adjoint=t_adjoint,
            )
        return data


class TestTorchCircularConvolution(TorchMixin, test_normalop.TestCircularConvolution):
    disable_test = test_normalop.TestCircularConvolution.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, conv_filter, request) -> dict:
        define_secondary = request.param

        h_FW = conv_filter
        dim = h_FW.size
        h_BW = h_FW[np.array([0, *np.arange(1, dim)[::-1]])]

        h_FW = torch.tensor(h_FW)
        h_BW = torch.tensor(h_BW)

        def _circ_convolve(h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            hF = torch.fft.fft(h, dim=-1)
            xF = torch.fft.fft(x, dim=-1)
            out = torch.fft.ifft(hF * xF, dim=-1).real
            return out

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            out = _circ_convolve(h_FW, arr)
            return out

        def t_adjoint(arr: torch.Tensor) -> torch.Tensor:
            out = _circ_convolve(h_BW, arr)
            return out

        data = dict(apply=t_apply)
        if define_secondary:
            data.update(
                adjoint=t_adjoint,
            )
        return data


class TestTorchPermutation(TorchMixin, test_unitop.TestPermutation):
    disable_test = test_unitop.TestPermutation.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, dim_shape, codim_shape, request) -> dict:
        define_secondary = request.param
        dim_rank = len(dim_shape)
        codim_rank = len(codim_shape)

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            axis = list(range(-dim_rank, 0))
            out = torch.flip(arr, dims=axis)
            return out

        def t_adjoint(arr: torch.Tensor) -> torch.Tensor:
            axis = list(range(-codim_rank, 0))
            out = torch.flip(arr, dims=axis)
            return out

        def t_pinv(arr: torch.Tensor, damp: pxt.Real) -> torch.Tensor:
            out = t_adjoint(arr) / (1 + damp)
            return out

        data = dict(
            apply=t_apply,
        )
        if define_secondary:
            data.update(
                adjoint=t_adjoint,
                pinv=t_pinv,
            )
        return data


class TestTorchOblique(TorchMixin, test_projop.TestOblique):
    disable_test = test_projop.TestOblique.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, alpha, request) -> dict:
        define_secondary = request.param

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(arr)
            out[..., -1] = alpha * arr[..., 0] + arr[..., -1]
            return out

        def t_adjoint(arr: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(arr)
            out[..., 0] = alpha * arr[..., -1]  # Update the first element along the last dimension
            out[..., -1] = arr[..., -1]  # Update the last element along the last dimension
            return out

        data = dict(
            apply=t_apply,
        )
        if define_secondary:
            data.update(
                adjoint=t_adjoint,
            )
        return data


class TestTorchScaleDown(TorchMixin, test_orthprojop.TestScaleDown):
    disable_test = test_orthprojop.TestScaleDown.disable_test | TorchMixin.disable_test

    @pytest.fixture(params=[True, False])
    def kwargs(self, request) -> dict:
        define_secondary = request.param

        def t_apply(arr: torch.Tensor) -> torch.Tensor:
            out = arr.clone()
            out[..., -1] = torch.zeros_like(out[..., -1])
            return out

        def t_pinv(arr: torch.Tensor, damp: pxt.Real) -> torch.Tensor:
            out = t_apply(arr) / (1 + damp)
            return out

        data = dict(
            apply=t_apply,
        )
        if define_secondary:
            data.update(
                pinv=t_pinv,
            )
        return data
