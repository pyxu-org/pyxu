import functools

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util.operator as pycuo
import pycsou.util.ptype as pyct


class _ProxPnPFunc(pyco.ProxFunc):
    def __init__(self, dim, denoiser: callable, vectorized: bool = True, **kwargs):
        super().__init__(shape=(1, dim))
        self._denoiser = lambda x: functools.partial(denoiser, **kwargs)(x)
        self._vectorized = vectorized

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplemented

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real = None) -> pyct.NDArray:
        if self._vectorized:
            return self._denoiser(arr)
        else:
            return pycuo.vectorize(i="x")(self._denoiser)(arr)
