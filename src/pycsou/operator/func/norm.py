import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = ["L1Norm", "L2Norm", "SquaredL2Norm", "L21Norm"]


class ShiftLossMixin:
    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        from pycsou.operator.func.loss import shift_loss

        op = shift_loss(op=self, data=data)
        return op


class L1Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_1`-norm, :math:`\Vert\mathbf{x}\Vert_1:=\sum_{i=1}^N |x_i|`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)

        Notes
        -----
        The operator's Lipschitz constant is set to :math:`\infty` if domain-agnostic.
        It is recommended to set `dim` explicitly to compute a tight closed-form.
        """
        super().__init__(shape=(1, dim))
        if dim is None:
            self._lipschitz = np.inf
        else:
            self._lipschitz = np.sqrt(dim)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y


class L2Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_2`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\sqrt{\sum_{i=1}^N |x_i|^2}`.
    """

    def __init__(self, dim: pyct.Integer = None):
        super().__init__(shape=(1, dim))
        self._lipschitz = 1
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=2, axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        scale = 1 - tau / xp.fmax(self.apply(arr), tau)

        y = arr.copy()
        y *= scale.astype(dtype=arr.dtype)
        return y


class SquaredL2Norm(pyca._QuadraticFunc):
    r"""
    :math:`\ell^2_2`-norm, :math:`\Vert\mathbf{x}\Vert^2_2:=\sum_{i=1}^N |x_i|^2`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, axis=-1, keepdims=True)
        y **= 2
        return y

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return 2 * arr

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        y = arr.copy()
        y /= 2 * tau + 1
        return y

    def _hessian(self) -> pyct.OpT:
        from pycsou.operator.linop import IdentityOp

        if self.dim is None:
            msg = "\n".join(
                [
                    "hessian: domain-agnostic functionals unsupported.",
                    f"Explicitly set `dim` in {self.__class__}.__init__().",
                ]
            )
            raise ValueError(msg)
        return IdentityOp(dim=self.dim).squeeze()


class L21Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Mixed :math:`\ell_2-\ell_1` norm :math:`\Vert\mathbf{x}\Vert_{2, 1}:=\sum_{i=1}^N \sqrt{ \sum_{j=1}^M x_{i, j}^2}`,
    for arrays of dimension :math:`\geq 2`.
    Notes
    _____
    The input array need not be 2-dimensional: the :math:`\ell_2` norm is applied along a predefined subset of the
    dimensions, and the :math:`\ell_1` norm on the remaining ones.
    """

    def __init__(self, arg_shape: tuple[int, int], l2_axis: typ.Union[int, tuple[int, ...]] = (0,)):
        r"""
        Parameters
        ----------
        arg_shape: tuple[int, ...]
            Shape of the multidimensional input array.
        l2_axis: int or tuple[int, ...], optional
            Dimension(s) along which the :math:`\ell_2` norm is applied.
        """
        super().__init__(shape=(1, np.prod(arg_shape)))
        self.arg_shape = arg_shape
        if isinstance(l2_axis, int):
            l2_axis = (l2_axis,)
        self.l2_axis = l2_axis
        ax_l2 = [a if a < 0 else (a - len(arg_shape)) for a in l2_axis]
        self._l1_axis = np.setdiff1d(np.arange(-len(arg_shape), 0), ax_l2)  # Axes where l1 norm is applied
        self._lipschitz = np.inf

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        x = arr.copy().reshape(arr.shape[:-1] + self.arg_shape)
        x = xp.moveaxis(x, self._l1_axis, np.arange(-len(self._l1_axis), 0))  # Move l1 axis to trailing dimensions
        # Reshape so that l1 and l2 are a single dimension
        x = x.reshape(arr.shape[:-1] + (np.prod([self.arg_shape[a] for a in self.l2_axis]),) + (-1,))
        return pylinalg.norm(pylinalg.norm(x, ord=2, axis=-2), ord=1, axis=-1, keepdims=True)

    @pycrt.enforce_precision(["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real):
        xp = pycu.get_array_module(arr)
        x = arr.copy().reshape(arr.shape[:-1] + self.arg_shape)
        x = xp.moveaxis(x, self._l1_axis, range(-len(self._l1_axis), 0))  # Move l1 axis to trailing dimensions
        # Reshape so that l1 and l2 are a single dimension
        x = x.reshape(arr.shape[:-1] + (np.prod([self.arg_shape[a] for a in self.l2_axis]),) + (-1,))
        x = (1 - tau / xp.fmax(pylinalg.norm(x, ord=2, axis=-2, keepdims=True), tau)) * x
        x = xp.moveaxis(x, range(-len(self._l1_axis), 0), self._l1_axis)  # Move back dimensions to their original place
        return x.reshape(arr.shape[:-1] + (-1,))
