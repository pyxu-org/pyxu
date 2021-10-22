import pycsou.abc.operator as pycop
import pycsou.abc
import numpy as np

NDArray = pycsou.abc.NDArray


class ExplicitLinFunc(pycop.LinFunc):
    def __init__(self, vec: NDArray):
        self.vec = vec.copy().reshape(-1)
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))

    def apply(self, arr: NDArray) -> NDArray:
        return (self.vec.conj()[None, :] * arr).sum(axis=-1)

    def adjoint(self, arr: NDArray) -> NDArray:
        return arr * self.vec[None, :]

    def gradient(self, arr: NDArray) -> NDArray:
        return np.broadcast_to(self.vec, arr.shape)
