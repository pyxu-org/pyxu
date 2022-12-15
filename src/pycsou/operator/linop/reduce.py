import collections.abc as cabc
import types

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "Sum",
]


def Sum(
    arg_shape: pyct.NDArrayShape,
    axis: pyct.NDArrayAxis = None,
) -> pyct.OpT:
    r"""
    Multi-dimensional sum reduction.

    This operator re-arranges the input array to a multidimensional array of shape ``arg_shape``,
    then reduces it via summation across one or more ``axis``.

    For example, assuming the input array :math:`\mathbf{x} \in \mathbb{R}^{N_1 \times N_2 \times
    N_3}` and ``axis=-1``, then

    .. math::

       \mathbf{y}_{i,j} = \sum_{k}{\mathbf{x}_{i,j,k}}.

    Parameters
    ----------
    arg_shape: pyct.NDArrayShape
        Shape of the data to be reduced.
    axis: pyct.NDArrayAxis
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all the elements of the input array.
        If axis is negative it counts from the last to the first axis.

    Notes
    -----
    The matrix operator of a 1D reduction applied to :math:`\mathbf{x} \in \mathbb{R}^{N}` is given
    by

    .. math::

       \mathbf{A}(x) = \mathbf{1}^{T} \mathbf{x},

    where :math:`\sigma_{\max}(\mathbf{A}) = \sqrt{N}`.
    An ND reduction is a chain of 1D reductions in orthogonal dimensions.
    Hence the Lipschitz constant of an ND reduction is the product of Lipschitz constants of all
    1D reductions involved, i.e.:

    .. math::

       L = \sqrt{\prod_{i_{k}} N_{i_{k}}},

    where :math:`\{i_{k}\}_{k}` denotes the axes being summed over.
    """

    def as_array(obj) -> np.ndarray:
        if isinstance(obj, cabc.Sequence):
            pass
        else:
            obj = [obj]
        return np.array(obj, dtype=int)

    arg_shape = as_array(arg_shape)
    assert np.all(arg_shape > 0)
    N_dim = len(arg_shape)

    if axis is None:
        axis = np.arange(N_dim)
    axis = np.unique(as_array(axis))  # drop potential duplicates
    assert np.all((-N_dim <= axis) & (axis < N_dim))  # all axes in valid range
    axis = (axis + N_dim) % N_dim  # get rid of negative axes

    sum_shape = arg_shape.copy()  # array shape after reduction
    sum_shape[axis] = 1

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(sh + _._arg_shape)

        axis = tuple(ax + len(sh) for ax in _._axis)
        out = arr.sum(axis=axis).reshape(*sh, -1)

        return out

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(sh + _._sum_shape)

        xp = pycu.get_array_module(arr)
        out = xp.broadcast_to(arr, sh + _._arg_shape)

        out = out.reshape(*sh, -1)
        return out

    def op_lipschitz(_, **kwargs) -> pyct.Real:
        N = np.prod(_._arg_shape) / np.prod(_._sum_shape)
        _._lipschitz = np.sqrt(N)
        return _._lipschitz

    dim = arg_shape.prod()
    codim = dim // arg_shape[axis].prod()

    klass = pyca.LinOp if codim > 1 else pyca.LinFunc
    op = klass(shape=(codim, dim))
    op._axis = tuple(axis)
    op._arg_shape = tuple(arg_shape)
    op._sum_shape = tuple(sum_shape)

    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.lipschitz = types.MethodType(op_lipschitz, op)
    op._name = "Sum"
    return op
