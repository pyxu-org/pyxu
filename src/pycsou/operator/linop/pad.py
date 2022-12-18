import collections.abc as cabc
import functools
import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct

__all__ = [
    "Pad",
]

WidthSpec = typ.Union[
    pyct.Integer,
    cabc.Sequence[pyct.Integer],
    cabc.Sequence[tuple[pyct.Integer, pyct.Integer]],
]
ModeSpec = typ.Union[str, cabc.Sequence[str]]


class _Pad1D(pyca.LinOp):
    def __init__(
        self,
        arg_shape,
        axis,
        pad_width,
        mode,
    ):
        self.codom_shape = tuple(s + np.sum(pad_width) * (i == axis) for i, s in enumerate(arg_shape))
        self.arg_shape = tuple(arg_shape)
        self.axis = axis
        self.mode = mode

        # Create multidimensional padding tuple, with padding defined only for axis
        self.pad_width = [(0, 0)] * (len(arg_shape) + 1)
        self.pad_width[axis + 1] = pad_width

        # Check that extended boundaries do not overlap
        if np.sum(pad_width) > arg_shape[axis]:
            warnings.warn(
                f"The default Lipschitz constant is estimated assuming that the number of padded elements "
                f"({np.sum(pad_width)}) is smaller than the size of the input array ({arg_shape[axis]}). "
                f"For a better estimate call the method `op.lipschitz(recompute=True)`."
            )
            if np.any(np.array(pad_width) > arg_shape[axis]):
                raise ValueError(
                    f"The number of padded elements in each side {pad_width} must not be larger than the "
                    f"size of the input array ({arg_shape[axis]}) at axis {axis}."
                )

        super().__init__(
            shape=(
                np.prod(self.codom_shape).item(),
                np.prod(arg_shape).item(),
            )
        )

        # Define Lipschitz constant (see `Notes` in PadOp)
        if self.mode == "constant":
            self._lipschitz = 1.0
        elif self.mode == "edge":
            self._lipschitz = np.sqrt(1 + np.sum(pad_width) ** 2)
        else:
            np.sqrt(2)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        """
        Pad input array in the one dimension defined by `axis`.
        """
        return np.pad(
            array=arr.reshape(-1, *self.arg_shape),
            pad_width=self.pad_width,
            mode=self.mode,
        ).reshape(*arr.shape[:-1], self.codim)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        """
        Adjoint of padding in one dimension.

        The adjoint method is performed in two steps, trimming and cumulative summation (see `Notes`
        in PadOp).
        """
        # Trim
        arr_shape = (-1,) + self.codom_shape
        out_shape = (-1,) + self.arg_shape
        slices = []
        for i, (start, end) in enumerate(self.pad_width):
            end = out_shape[i] + self.pad_width[i][0] if arr_shape[i] != -1 else None
            slices.append(slice(start, end))
        out = arr.reshape(*arr_shape)[tuple(slices)].copy()

        # Cumulative sum
        if self.mode == "constant":
            return out.reshape(*arr.shape[:-1], self.dim)

        # Slices of output onto which the input (padded) elements are summed to.
        slices_out = [np.copy(slices), np.copy(slices)]
        if self.mode == "wrap":
            slices_out[0][self.axis + 1] = slice(-self.pad_width[self.axis + 1][0], None)
            slices_out[1][self.axis + 1] = slice(0, self.pad_width[self.axis + 1][1])
        elif self.mode in ["reflect", "symmetric"]:
            # reflect and symmetric only differ by a 1 element displacement, captured by the
            # following `aux` variable.
            aux = self.mode == "reflect"
            slices_out[0][self.axis + 1] = slice(self.pad_width[self.axis + 1][0] + aux - 1, (0 if aux else None), -1)
            slices_out[1][self.axis + 1] = slice(-(1 + aux), -(self.pad_width[self.axis + 1][1] + 1 + aux), -1)
        elif self.mode == "edge":
            slices_out[0][self.axis + 1] = slice(0, 1)
            slices_out[1][self.axis + 1] = slice(-1, None)
        else:
            raise NotImplementedError(f"mode {self.mode} is not supported.")

        # Slices of input array to be summed to output
        slices_arr = [np.copy(slices), np.copy(slices)]
        slices_arr[0][self.axis + 1] = slice(0, self.pad_width[self.axis + 1][0])
        slices_arr[1][self.axis + 1] = slice(self.codom_shape[self.axis] - (self.pad_width[self.axis + 1][1]), None)

        # Perform cumulative summation
        for i, slice_ in enumerate(slices_arr):
            if arr.reshape(*arr_shape)[tuple(slice_)].size:
                if self.mode == "edge":
                    out[tuple(slices_out[i])] += arr.reshape(*arr_shape)[tuple(slice_)].sum(
                        axis=self.axis + 1,
                        keepdims=True,
                    )
                else:
                    out[tuple(slices_out[i])] += arr.reshape(*arr_shape)[tuple(slice_)]

        return out.reshape(*arr.shape[:-1], -1)


def Pad(
    arg_shape: pyct.NDArrayShape,
    pad_width: WidthSpec,
    mode: ModeSpec = "constant",
) -> pyct.OpT:
    r"""
    Multi-dimensional padding operator.

    This operator pads the input array in each dimension according to specified widths.

    Parameters
    ----------
    arg_shape: pyct.NDArrayShape
        Shape of the input array.
    pad_width: WidthSpec
        Number of values padded to the edges of each axis.
        Multiple forms are accepted:

        * int: pad each dimension's head/tail by `pad_width`.
        * tuple[int, ...]: pad dimension[k]'s head/tail by `pad_width[k]`.
        * tuple[tuple[int, int], ...]: pad dimension[k]'s head/tail by `pad_width[k][0]` /
          `pad_width[k][1]` respectively.

    mode: str | list(str)
        Padding mode.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (zero-padding)
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: pad dimension[k] using `mode[k]`.

        (See :py:func:`numpy.pad` for details.)

    Returns
    -------
    op: pyct.OpT

    Notes
    -----
    * If inputs are D-dimensional, then some of the padding of later axes are calculated from
      padding of previous axes.
    * The *adjoint* of the padding operator performs a cumulative summation over the original
      positions used to pad.
      Its effect is clear from its matrix form.
      For example the matrix-form of ``Pad(arg_shape=(3,), mode="wrap", pad_width=(1, 1))`` is:

      .. math::

         \mathbf{A}
         =
         \left[
            \begin{array}{ccc}
                0 & 0 & 1 \\
                1 & 0 & 0 \\
                0 & 1 & 0 \\
                0 & 0 & 1 \\
                1 & 0 & 0
            \end{array}
         \right].

      The adjoint of :math:`\mathbf{A}` corresponds to its matrix transpose:

      .. math::

         \mathbf{A}^{\ast}
         =
         \left[
             \begin{array}{ccccc}
                 0 & 1 & 0 & 0 & 1 \\
                 0 & 0 & 1 & 0 & 0 \\
                 1 & 0 & 0 & 1 & 0
             \end{array}
         \right].

      This operation can be seen as a trimming (:math:`\mathbf{T}`) plus a cumulative summation
      (:math:`\mathbf{S}`):

      .. math::

         \mathbf{A}^{\ast}
         =
         \mathbf{T} + \mathbf{S}
         =
         \left[
            \begin{array}{ccccc}
                0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0
            \end{array}
         \right]
         +
         \left[
            \begin{array}{ccccc}
                0 & 0 & 0 & 0 & 1 \\
                0 & 0 & 0 & 0 & 0 \\
                1 & 0 & 0 & 0 & 0
            \end{array}
         \right],

      where both :math:`\mathbf{T}` and :math:`\mathbf{S}` are efficiently implemented in
      matrix-free form.


    * The Lipschitz constant of the multi-dimensional padding operator is the product of Lipschitz
      constants of the uni-dimensional paddings applied per dimension, i.e.:

      .. math::

         L = \prod_{i} L_{i}, \qquad i \in \{0, \dots, N-1\},

      where an upper-bound to :math:`L_{i}` depends on the boundary condition at the :math:`i`-th
      axis:

      - mode='constant' corresponds to an up-sampling operator, hence :math:`L_{i} = 1`.
      - In mode='wrap'/'reflect'/'symmetric', the padding operator is adding :math:`p` elements from
        the input signal:

          .. math::

             \| P_{i}(\mathbf{x}) - P_{i}(\mathbf{y}) \|^{2}_{2}
             \leq
             \| \mathbf{x} - \mathbf{y} \|^{2}_{2} + \| \mathbf{x}_{p} - \mathbf{y}_{p} \|^{2}_{2},

          where :math:`\mathbf{x}_{p}` contains the :math:`p` elements of :math:`\mathbf{x}` used
          for padding.
          The right-hand-side of the inequality is itself bounded by the distance between the
          original vectors:

          .. math::

             \| \mathbf{x} - \mathbf{y} \|^{2}_{2} + \| \mathbf{x}_{p} - \mathbf{y}_{p} \|^{2}_{2}
             \leq
             2 \| \mathbf{x} - \mathbf{y} \|^{2}_{2}
             \Longrightarrow
             L_{i}
             =
             \sqrt{2}

      - In mode='edge', the padding operator is adding :math:`p = \sum{\text{pad_width}}` elements
        at the extremes.
        In this case, the upper bound is:

          .. math::

             \begin{align*}
                 \| P_{i}(\mathbf{x}) - P_{i}(\mathbf{y}) \|^{2}_{2}
                 & \leq
                 \| \mathbf{x} - \mathbf{y} \|^{2}_{2} + p^{2}  \| \mathbf{x} - \mathbf{y} \|^{2}_{\infty} \\
                 & \leq
                 (1 + p^{2}) \| \mathbf{x} - \mathbf{y} \|^{2}_{2}
                 \Longrightarrow
                 L_{i}
                 =
                 \sqrt{1 + p^{2}}
             \end{align*}
    """
    arg_shape = tuple(arg_shape)
    N_dim = len(arg_shape)

    # transform `pad_width` to canonical form tuple[tuple[int, int]]
    is_seq = lambda _: isinstance(_, cabc.Sequence)
    if not is_seq(pad_width):  # int-form
        pad_width = ((pad_width, pad_width),) * N_dim
    assert len(pad_width) == N_dim, f"arg_shape/pad_width are length-mismatched."
    if not is_seq(pad_width[0]):  # tuple[int, ...] form
        pad_width = tuple((w, w) for w in pad_width)
    else:  # tuple[tulpe[int, int], ...] form
        pass

    if isinstance(mode, str):  # shared mode
        mode = (mode,) * N_dim
    elif isinstance(mode, cabc.Sequence):  # tuple[str, ...]: different modes
        assert len(mode) == N_dim, "arg_shape/mode are length-mismatched."
        mode = tuple(mode)
    else:
        raise ValueError(f"Unkwown mode encountered: {mode}.")
    mode = tuple(map(lambda _: _.strip().lower(), mode))
    assert set(mode) <= {
        "constant",
        "wrap",
        "reflect",
        "symmetric",
        "edge",
    }, "Unknown mode(s) encountered."

    # 1d padding operators in each dimension.
    arg_shape_ = list(arg_shape)
    op_list = []
    for d in range(ndim):
        op_list.append(
            _Pad1D(
                arg_shape=arg_shape_.copy(),
                axis=d,
                pad_width=pad_width[d],
                mode=mode[d],
            )
        )
        arg_shape_[d] += np.sum(pad_width[d])

    # Compose 1d padding operators into multi-dimensional padding operator.
    op = functools.reduce(lambda x, y: x * y, op_list[::-1])
    op._name = "Pad"
    return op
