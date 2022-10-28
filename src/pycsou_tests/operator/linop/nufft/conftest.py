import numpy as np
import pytest

import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


class NUFFT_Mixin:
    disable_test = frozenset(
        conftest.LinOpT.disable_test
        | {
            # NUFFT does not support evaluating inputs at different precisions.
            "test_precCM_adjoint",
            "test_precCM_adjoint_dagger",
            "test_precCM_adjoint_T",
            "test_precCM_apply",
            "test_precCM_apply_dagger",
            "test_precCM_apply_T",
            "test_precCM_call",
            "test_precCM_call_dagger",
            "test_precCM_call_T",
            "test_precCM_eigvals",
            "test_precCM_eigvals",
            "test_precCM_pinv",
            "test_precCM_svdvals",
            # from_sciop() tests try round trip NUFFT<>to_sciop()<>from_sciop().
            # Compounded effect of approximations make most tests fail.
            # There is no reason to use from_sciop() in NUFFT -> safe to disable.
            "test_value_from_sciop",
            "test_prec_from_sciop",
            "test_backend_from_sciop",
        }
    )

    # Internal helpers --------------------------------------------------------
    @classmethod
    def _metric(
        cls,
        a: pyct.NDArray,
        b: pyct.NDArray,
        as_dtype: pyct.DType,
    ) -> bool:
        # NUFFT is an approximate transform.
        # Based on [FINUFFT], results hold up to a small relative error.
        #
        # We choose a conservative threshold, irrespective of the `eps` parameter chosen by the
        # user. Additional tests below test explicitly if computed values correctly obey `eps`.
        eps_default = 1e-2

        cast = lambda x: pycu.compute(x)
        lhs = np.linalg.norm(pycu.to_NUMPY(cast(a) - cast(b)), axis=-1)
        rhs = np.linalg.norm(pycu.to_NUMPY(cast(b)), axis=-1)
        return conftest.less_equal(lhs, eps_default * rhs, as_dtype=as_dtype).all()

    # Fixtures which parametrize operator -------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def transform_dimension(self, request) -> int:
        return request.param

    @pytest.fixture(params=[-1, 1])
    def transform_sign(self, request) -> int:
        return request.param

    @pytest.fixture(params=[0, 1e-4, 1e-6])
    def transform_eps(self, request) -> float:
        return request.param

    @pytest.fixture(params=[1, 2])
    def transform_ntrans(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1, 2])
    def transform_nthreads(self, request) -> int:
        return request.param

    @pytest.fixture(params=[False, True])
    def transform_real(self, request) -> bool:
        return request.param

    # Overridden Tests --------------------------------------------------------
    def test_value_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Input is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_value_to_sciop(_op_sciop, _data_to_sciop)

    def test_backend_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Input is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_backend_to_sciop(_op_sciop, _data_to_sciop)

    def test_prec_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Input is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_prec_to_sciop(_op_sciop, _data_to_sciop)
