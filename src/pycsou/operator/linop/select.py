import typing as typ

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "SubSampling",
]


class SubSampling(pyca.LinOp):
    r"""
    Given a set of indices :math: `\mathcal{I} \subset \{1, \dots, M\}` of size :math: `L`, the *subsampling
    linear operator* extracts a sub array from an input array :math:`\mathbf{x} \in \mathbb{R}^M` corresponding
    to the associated indices. Its Lipschitz constant is 1.

    The adjoint operator embeds an array of size :math: `L` within a larger array of size :math: `M` by filling
    the provided indices with the inut array values.

    The dimension of the input space :math: `M` needs to be specified.

    Notes
    -----
    :py:func:`~pycsou.operator.linop.select.SubSampling` instances are **not arraymodule-agnostic**: they will only
    work with NDArrays belonging to the same module as the ``sampling_indices`` array.
    """

    def __init__(self, M: int, sampling_indices: typ.Union[pyct.NDArray, list]):
        r"""

        Parameters
        ----------
        M: int
            Dimension of the ambient space from which the samples are taken.
        sampling_indices: NDArray | list
            Indices of the samples that should be extracted.
        """
        if isinstance(sampling_indices, list):
            import numpy as xp
        else:
            xp = pycu.get_array_module(sampling_indices)

        self.sampling_indices = xp.asarray(sampling_indices).reshape(-1)
        self.input_size = M
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
        y = xp.zeros((*arr.shape[:-1], self.input_size), dtype=arr.dtype)
        y[..., self.sampling_indices] = arr
        return y
