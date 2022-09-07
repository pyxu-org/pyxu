"""
This file implements a few common operators, useful in the development of PFW for the LASSO problem.
This file is only temporary, and these methods should later on be imported from the tested implementations provided by
Pycsou (when they will be available).
"""

import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class SquaredL2Norm(pyco.ProxDiffFunc):
    # f: \bR^{M} -> \bR
    #      x     -> \norm{x}{2}^{2}
    def __init__(self, M: int = None):
        super().__init__(shape=(1, M))
        self._lipschitz = np.inf
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.linalg.norm(arr, axis=-1, keepdims=True)
        y2 = xp.power(y, 2, dtype=arr.dtype)
        return y2

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return 2 * arr

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        y = arr / (2 * tau + 1)
        return y

    @pycrt.enforce_precision(i="data", allow_None=True, o=False)
    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.ProxFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)


class L1Norm(pyco.ProxFunc):
    def __init__(self, M: int = None):
        super().__init__(shape=(1, M))
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.linalg.norm(arr, ord=1, axis=-1, keepdims=True).astype(arr.dtype)
        return y

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau) * xp.sign(arr)
        return y

    @pycrt.enforce_precision(i="data", allow_None=True)
    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.ProxFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)


class SubSampling(pyco.LinOp):
    r"""
    SubSampling operator that extracts a sub array of smaller size from another array.

    Output module is the module of the sampling indices. Maybe this behavior needs to be changed.
    """

    # todo needs to be tested (in normal and adjoint mode) with stacked array inputs.

    def __init__(self, size: int, sampling_indices: typ.Union[pyct.NDArray, list]):
        r"""

        Parameters
        ----------
        size: int
            Dimension of the ambient space from which the samples are taken.
        sampling_indices: NDArray | list
            Indices of the samples that should be extracted.
        """
        if isinstance(sampling_indices, list):
            import numpy as xp
        else:
            xp = pycu.get_array_module(sampling_indices)

        self.sampling_indices = xp.asarray(sampling_indices).reshape(-1)
        self.input_size = size
        self.nb_of_samples = self.sampling_indices.size
        super(SubSampling, self).__init__(shape=(self.nb_of_samples, self.input_size))
        self._lipschitz = 1.0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""

        Parameters
        ----------
        arr: NDArray
            Array the samples are extracted from.

        Returns
        -------
            A sub-sampled array.
        """
        return arr[..., self.sampling_indices]

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        # arr = arr.reshape(1, -1) if (arr.ndim == 1) else arr
        # y = xp.zeros((*arr.shape[:-1], self.input_size), dtype=arr.dtype)
        y = (
            xp.zeros(self.input_size, dtype=arr.dtype)
            if (arr.ndim == 1)
            else xp.zeros((*arr.shape[:-1], self.input_size), dtype=arr.dtype)
        )
        y[..., self.sampling_indices] = arr
        return y


class NonNegativeOrthant(pyco.ProxFunc):
    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if arr.ndim <= 1:
            return 0 if xp.all(arr >= 0) else xp.inf
        else:
            res = xp.zeros(arr.shape[:-1])
            res[xp.any(arr < 0, axis=-1)] = xp.infty
            return res

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        res = xp.copy(arr)
        res[res < 0] = 0
        return res

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        pass


if __name__ == "__main__":
    N = 10
    posIndicator = NonNegativeOrthant(shape=(1, None))

    a = np.random.normal(size=N)
    b = posIndicator.prox(a, tau=1)
    print(posIndicator(a))
    print(b)
    print(posIndicator(b))

    print("0-d input: {}".format(posIndicator(np.r_[-1])))
    print("3-d input: {}".format(posIndicator(np.arange(24).reshape((2, 3, 4)) - 3)))
