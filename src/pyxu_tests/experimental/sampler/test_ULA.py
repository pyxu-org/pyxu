import itertools
import warnings

import numpy as np
import pytest

import pyxu.experimental.sampler as pxs
import pyxu.info.deps as pxd
import pyxu.info.warning as pxw
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.experimental.sampler.conftest as ct


class TestULA(ct.SamplerT):
    @staticmethod
    def _check_no_side_effect(sampler, seed, x0):
        # idea:
        # * eval next(gen_1) once [out_1]
        # * in-place update out_1, ex: scale by constant factor [scale]
        # * eval next(gen_2) with same seed and same x0 [out_2]
        # * assert out_1 == out_2, i.e. input x0 was not modified

        xp = pxu.get_array_module(x0)
        rng_1 = xp.random.default_rng(seed=seed)
        gen_1 = sampler.samples(rng=rng_1, x0=x0)

        scale = 10

        with pxrt.EnforcePrecision(False):
            out_1 = next(gen_1)

            if pxu.copy_if_unsafe(out_1) is not out_1:
                # out_1 is read_only -> safe
                return
            else:
                # out_1 is writeable -> test correctness
                out_gt = out_1.copy()
                out_1 *= scale  # Scale output -> does it change x0 ?
                rng_2 = xp.random.default_rng(seed=seed)
                gen_2 = sampler.samples(rng=rng_2, x0=x0)
                out_2 = next(gen_2)

            try:
                assert xp.allclose(out_gt, out_2)
            except AssertionError:
                # Function is non-transparent, but which backend caused it?
                N = pxd.NDArrayInfo
                ndi = N.from_obj(out_1)
                if ndi == N.CUPY:
                    # warn about CuPy-only non-transparency.
                    msg = "\n".join(
                        [
                            f"{sampler} is not transparent when applied to CuPy starting points.",
                            f"If the same test fails for non-CuPy inputs, then {sampler}'s implementation is at fault -> user fix required.",
                            "If the same test passes for non-CuPy inputs, then this warning can be safely ignored.",
                        ]
                    )
                    warnings.warn(msg, pxw.NonTransparentWarning)
                else:
                    raise

    @pytest.fixture
    def dim(self):
        return 4

    @pytest.fixture
    def x0_np(self, dim):
        stack_dim = 2, 3
        return np.arange(np.prod(stack_dim) * dim).reshape((*stack_dim, dim))

    @pytest.fixture
    def x0(self, x0_np, xp, width):
        return xp.array(x0_np, dtype=width.value)

    @pytest.fixture
    def func(self, dim):
        return 1 / 2 * pxo.SquaredL2Norm(dim=dim)  # Sampling of standard normal distribution

    @pytest.fixture
    def sampler(self, func):
        return pxs.ULA(f=func)

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, sampler, seed, x0_np, request):
        ndi = request.param[0]
        if (xp := ndi.module()) is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        width = request.param[1]

        gen = sampler.samples(rng=xp.random.default_rng(seed=seed), x0=xp.array(x0_np, dtype=width.value))
        return gen, *request.param

    @pytest.fixture
    def samples_copy(self, func, sampler, seed, x0_np, xp, width):
        rng_1 = xp.random.default_rng(seed=seed)
        rng_2 = xp.random.default_rng(seed=seed)
        sampler._rng = rng_1  # Reset rng object of sampler
        sampler_copy = pxs.ULA(f=func)  # Create copy of sampler
        samples_copy = sampler_copy.samples(rng=rng_2, x0=xp.array(x0_np, dtype=width.value))
        return samples_copy

    def test_transparent_samples(self, sampler, seed, x0):
        self._skip_if_disabled()
        self._check_no_side_effect(sampler, seed, x0)


class TestMYULA(TestULA):
    @pytest.fixture
    def prox_func(self, dim):
        return pxo.L1Norm(dim=dim)

    @pytest.fixture
    def sampler(self, func, prox_func):
        return pxs.MYULA(f=func, g=prox_func)

    @pytest.fixture
    def samples_copy(self, func, prox_func, sampler, seed, x0_np, xp, width):
        rng_1 = xp.random.default_rng(seed=seed)
        rng_2 = xp.random.default_rng(seed=seed)
        sampler._rng = rng_1  # Reset rng object of sampler
        sampler_copy = pxs.MYULA(f=func, g=prox_func)  # Create copy of sampler
        return sampler_copy.samples(rng=rng_2, x0=xp.array(x0_np, dtype=width.value))
