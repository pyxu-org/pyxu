import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class NUFFT_Mixin:
    # Internal helpers --------------------------------------------------------
    @classmethod
    def _metric(
        cls,
        a: pxt.NDArray,
        b: pxt.NDArray,
        as_dtype: pxt.DType,
    ) -> bool:
        # NUFFT is an approximate transform.
        # Based on [FINUFFT], results hold up to a small relative error.
        #
        # We choose a conservative threshold, irrespective of the `eps` parameter chosen by the
        # user. Additional tests below test explicitly if computed values correctly obey `eps`.
        eps_default = 1e-2

        cast = lambda x: pxu.compute(x)
        lhs = np.linalg.norm(pxu.to_NUMPY(cast(a) - cast(b)), axis=-1)
        rhs = np.linalg.norm(pxu.to_NUMPY(cast(b)), axis=-1)
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

    @pytest.fixture(
        params=[
            0,  # LINEAR order
            1,  # FFT order
        ]
    )
    def transform_modeord(self, request) -> int:
        return request.param

    # Overridden Tests --------------------------------------------------------
    def test_value_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Last axis is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_value_to_sciop(_op_sciop, _data_to_sciop)

    def test_backend_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Last axis is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_backend_to_sciop(_op_sciop, _data_to_sciop)

    def test_prec_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Last axis is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_prec_to_sciop(_op_sciop, _data_to_sciop)
