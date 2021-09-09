import abc
import numpy.typing as npt
import numpy as np
from numbers import Number

NDArray = npt.ArrayLike


class Property:
    _supported_properties = ['apply', 'jacobianT', 'gradient', 'adjoint', 'prox']


class ApplyProp(abc.ABC, Property):

    def __call__(self, arr: NDArray) -> NDArray:
        return self.apply(arr)

    @abc.abstractmethod
    def apply(self, arr: NDArray) -> NDArray:
        pass

    def apply_along_axis(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.apply, axis=axis, arr=arr)

    def lipschitz(self) -> float:
        return np.infty

    @property
    def has_apply(self) -> bool:
        return True

    @property
    def is_lipschitzian(self) -> bool:
        return True if self.lipschitz() < np.infty else False


class _DiffProp(Property):

    def diff_lipschitz(self) -> float:
        return np.infty

    @property
    def is_diff_lipschitzian(self) -> bool:
        return True if self.diff_lipschitz() < np.infty else False


class JacProp(abc.ABC, _DiffProp):

    @abc.abstractmethod
    def jacobianT(self, arr: NDArray) -> Property:
        pass

    @property
    def has_jacobianT(self) -> bool:
        return True


class GradProp(abc.ABC, _DiffProp):

    @abc.abstractmethod
    def gradient(self, arr: NDArray) -> NDArray:
        pass

    @property
    def has_gradient(self) -> bool:
        return True

    def gradient_along_axis(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.gradient, axis=axis, arr=arr)


class AdjProp(abc.ABC, Property):

    @abc.abstractmethod
    def adjoint(self, arr: NDArray) -> NDArray:
        pass

    @property
    def has_adjoint(self) -> bool:
        return True

    def adjoint_along_axis(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.adjoint, axis=axis, arr=arr)


class ProxProp(abc.ABC, Property):

    @abc.abstractmethod
    def prox(self, arr: NDArray, tau: Number) -> NDArray:
        pass

    @property
    def has_prox(self) -> bool:
        return True

    def prox_along_axis(self, arr: NDArray, tau: Number, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.prox, axis=axis, arr=arr, tau=tau)

    def fenchel_prox(self, arr: NDArray, sigma: Number) -> NDArray:
        return arr - sigma * self.prox(arr=arr / sigma, tau=1 / sigma)

    def fenchel_prox_along_axis(self, arr: NDArray, sigma: Number, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.fenchel_prox, axis=axis, arr=arr, sigma=sigma)
