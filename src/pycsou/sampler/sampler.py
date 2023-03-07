import collections.abc as cabc
import math

import pycsou.abc as pyca
import pycsou.operator.func as pycof
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.random as pycr


class _Sampler:
    """Abstract base class for samplers."""

    def samples(self, seed: pyct.Integer = None, **kwargs) -> cabc.Generator:
        """Returns a generator; samples are drawn by calling next(generator)."""
        self._sample_init(seed, **kwargs)
        while True:
            yield self._sample()

    def _sample_init(self, seed, **kwargs):
        """Optional method to set initial state of the sampler (e.g., a starting point)."""
        pass

    def _sample(self) -> pyct.NDArray:
        """Method to be implemented by subclasses that returns the next sample."""
        raise NotImplementedError


class ULA(_Sampler):
    r"""
    Samples from distribution with PDF $p(x) \propto \exp(-f(x))$ with f differentiable.
    """

    def __init__(self, f: pyca.DiffFunc, gamma: pyct.Real = None):
        self._f = f
        self._beta = f.diff_lipschitz()
        self._gamma = self._set_gamma(gamma)
        self._rng = None
        self.x = None

    def _sample_init(self, seed: pyct.Integer, x0: pyct.NDArray):
        self.x = x0
        ndi = pycd.NDArrayInfo.from_obj(x0)
        self._rng = pycr.random_generator(ndi, seed)

    def _sample(self) -> pyct.NDArray:
        self.x += -self._gamma * self._f.grad(self.x)
        self.x += math.sqrt(2 * self._gamma) * self._rng.standard_normal(size=self.x.shape, dtype=self.x.dtype)
        return self.x

    def objective_func(self) -> pyct.Real:
        return self._f.apply(self.x)

    def _set_gamma(self, gamma: pyct.Real = None) -> pyct.Real:
        if gamma is None:
            if math.isfinite(self._beta):
                return pycrt.coerce(0.98 / self._beta)
            else:
                msg = "If f has unbounded Lipschitz gradient, the gamma parameter must be provided."
            raise ValueError(msg)
        else:
            try:
                assert gamma > 0
            except:
                raise ValueError(f"gamma must be positive, got {gamma}.")
            return pycrt.coerce(gamma)


class MYULA(ULA):
    r"""
    Samples from distribution with PDF $p(x) \propto \exp(-(f(x) + g(x)))$ with f differentiable and g proximable.
    """

    def __init__(
        self, f: pyca.DiffFunc = None, g: pyca.ProxFunc = None, gamma: pyct.Real = None, lamb: pyct.Real = None
    ):

        dim = None
        if f is not None:
            dim = f.dim
        if g is not None:
            if dim is None:
                dim = g.dim
            else:
                assert g.dim == dim
        if dim is None:
            raise ValueError("One of f or g must be nonzero.")

        self._f_diff = pycof.NullFunc(dim=dim) if (f is None) else f
        self._g = pycof.NullFunc(dim=dim) if (g is None) else g

        self._lambda = self._set_lambda(lamb)
        f = self._f_diff + self._g.moreau_envelope(self._lambda)
        f.diff_lipschitz()
        super().__init__(f, gamma)

    def _set_lambda(self, lamb: pyct.Real = None) -> pyct.Real:
        if lamb is None:
            if math.isfinite(dl := self._f_diff._diff_lipschitz):
                return pycrt.coerce(2) if dl == 0 else pycrt.coerce(min(2, 1 / dl))
            else:
                msg = "If f has unbounded Lipschitz gradient, the lambda parameter must be provided."
            raise ValueError(msg)
        else:
            return pycrt.coerce(lamb)
