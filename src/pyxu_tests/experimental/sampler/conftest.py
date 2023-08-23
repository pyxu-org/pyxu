import collections.abc as cabc
import typing as typ

import pytest

import pyxu.experimental.sampler as pxs
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.conftest as ct


class SamplerT(ct.DisableTestMixin):
    # Internal helpers --------------------------------------------------------
    @classmethod
    def _check_shape(
        cls,
        samples_list,
    ):
        assert all([s.shape == samples_list[0].shape for s in samples_list[1:]])

    @staticmethod
    def _check_backend(samples_list):
        assert all([isinstance(s, type(samples_list[0])) for s in samples_list[1:]])

    @staticmethod
    def _check_prec(samples_list):
        assert all([s.dtype == samples_list[0].dtype for s in samples_list[1:]])

    @staticmethod
    def _check_precCM(
        gen,
        widths: cabc.Collection[pxrt.Width] = pxrt.Width,
    ):
        stats = dict()
        for w in widths:
            with pxrt.Precision(w):
                out = next(gen)
            stats[w] = out.dtype == w.value
        assert all(stats.values())

    @staticmethod
    def _check_reproducibility(samples_1, samples_2, num_samples, xp):
        with pxrt.EnforcePrecision(False):
            for _ in range(num_samples):
                sample_1 = next(samples_1)
                sample_2 = next(samples_2)
                assert xp.allclose(sample_1, sample_2)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def spec(self) -> tuple[typ.Generator, pxd.NDArrayInfo, pxrt.Width]:
        # override in subclass to return:
        # * the sampler.samples() generator, where sampler is the sampler to test;
        # * the backend of accepted input arrays;
        # * the precision of accepted input arrays.
        #
        # The triplet (sampler, backend, precision) must be provided since some samplers may not be
        # backend/precision-agnostic.
        raise NotImplementedError

    @pytest.fixture
    def samples(self, spec) -> typ.Generator:
        return spec[0]

    @pytest.fixture
    def samples_copy(self) -> typ.Generator:
        # override in subclass to return a copy of the sampler.samples() generator (with the same rng)
        raise NotImplementedError

    @pytest.fixture
    def sampler(self) -> pxs._Sampler:
        # override in subclass to return the sampler object being tested.
        raise NotImplementedError

    @pytest.fixture
    def seed(self) -> pxrt.Width:
        return 1234

    @pytest.fixture
    def num_samples(self) -> int:
        return 5

    @pytest.fixture
    def samples_list(self, num_samples, samples) -> list[pxt.NDArray]:
        with pxrt.EnforcePrecision(False):
            s_list = [next(samples) for _ in range(num_samples)]
        return s_list

    @pytest.fixture
    def ndi(self, spec) -> pxd.NDArrayInfo:
        ndi_ = spec[1]
        if ndi_.module() is None:
            pytest.skip(f"{ndi_} unsupported on this machine.")
        return ndi_

    @pytest.fixture
    def xp(self, ndi) -> pxt.ArrayModule:
        return ndi.module()

    @pytest.fixture
    def width(self, spec) -> pxrt.Width:
        return spec[2]

    # Tests -------------------------------------------------------------------
    def test_shape_samples(self, samples_list):
        self._skip_if_disabled()
        self._check_shape(samples_list)

    def test_backend_samples(self, samples_list):
        self._skip_if_disabled()
        self._check_backend(samples_list)

    def test_prec_samples(self, samples_list):
        self._skip_if_disabled()
        self._check_prec(samples_list)

    def test_precCM_samples(self, samples):
        self._skip_if_disabled()
        self._check_precCM(samples)

    def test_reproducibility_samples(self, samples, samples_copy, num_samples, xp):
        self._skip_if_disabled()
        self._check_reproducibility(samples, samples_copy, num_samples, xp)
