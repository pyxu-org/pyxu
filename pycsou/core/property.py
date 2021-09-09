import abc
import numpy.typing as npt
import numpy as np
from numbers import Number

NDArray = npt.ArrayLike


class Property:
    _supported_properties = ['apply', 'jacobianT']


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
    def is_evaluable(self) -> bool:
        return True

    @property
    def is_lipschitzian(self) -> bool:
        return True if self.lipschitz() < np.infty else False


class _DiffProp(Property):

    def diff_lipschitz(self) -> float:
        return np.infty

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def is_diff_lipschitzian(self) -> bool:
        return True if self.diff_lipschitz() < np.infty else False


class JacProp(abc.ABC, _DiffProp):

    @abc.abstractmethod
    def jacobianT(self, arr: NDArray) -> Property:
        pass


class GradProp(abc.ABC, _DiffProp):

    @abc.abstractmethod
    def gradient(self, arr: NDArray) -> NDArray:
        pass

    def gradient_along_axis(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.gradient, axis=axis, arr=arr)


class AdjProp(abc.ABC, Property):

    @abc.abstractmethod
    def adjoint(self, arr: NDArray) -> NDArray:
        pass

    def adjoint_along_axis(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.adjoint, axis=axis, arr=arr)


class ProxProp(abc.ABC, Property):

    @abc.abstractmethod
    def prox(self, arr: NDArray, tau: Number) -> NDArray:
        pass

    def prox_along_axis(self, arr: NDArray, tau: Number, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.prox, axis=axis, arr=arr, tau=tau)