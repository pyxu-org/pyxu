import collections.abc as cabc
import types
import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "Pad",
]


class Pad(pyca.LinOp):
    r"""
    Multi-dimensional padding operator.

    This operator pads the input array in each dimension according to specified widths.

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
    WidthSpec = typ.Union[
        pyct.Integer,
        cabc.Sequence[pyct.Integer],
        cabc.Sequence[tuple[pyct.Integer, pyct.Integer]],
    ]
    ModeSpec = typ.Union[str, cabc.Sequence[str]]

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        pad_width: WidthSpec,
        mode: ModeSpec = "constant",
    ):
        r"""
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
        """
        self._arg_shape = tuple(arg_shape)
        assert all(_ > 0 for _ in self._arg_shape)
        N_dim = len(self._arg_shape)

        # transform `pad_width` to canonical form tuple[tuple[int, int], ...]
        is_seq = lambda _: isinstance(_, cabc.Sequence)
        if not is_seq(pad_width):  # int-form
            pad_width = ((pad_width, pad_width),) * N_dim
        assert len(pad_width) == N_dim, f"arg_shape/pad_width are length-mismatched."
        if not is_seq(pad_width[0]):  # tuple[int, ...] form
            pad_width = tuple((w, w) for w in pad_width)
        else:  # tuple[tulpe[int, int], ...] form
            pass
        assert all(0 <= min(lhs, rhs) for (lhs, rhs) in pad_width)
        self._pad_width = pad_width

        # transform `mode` to canonical form tuple[str, ...]
        if isinstance(mode, str):  # shared mode
            mode = (mode,) * N_dim
        elif isinstance(mode, cabc.Sequence):  # tuple[str, ...]: different modes
            assert len(mode) == N_dim, "arg_shape/mode are length-mismatched."
            mode = tuple(mode)
        else:
            raise ValueError(f"Unkwown mode encountered: {mode}.")
        self._mode = tuple(map(lambda _: _.strip().lower(), mode))
        assert set(self._mode) <= {
            "constant",
            "wrap",
            "reflect",
            "symmetric",
            "edge",
        }, "Unknown mode(s) encountered."

        # Useful constant: shape of padded outputs
        self._pad_shape = list(self._arg_shape)
        for i, (lhs, rhs) in enumerate(self._pad_width):
            self._pad_shape[i] += lhs + rhs
        self._pad_shape = tuple(self._pad_shape)

        codim = np.prod(self._pad_shape)
        dim = np.prod(self._arg_shape)
        super().__init__(shape=(codim, dim))

        # Some modes have awkward interpretations when pad-widths cross certain thresholds.
        # Supported pad-widths are thus limited to sensible regions.
        for i in range(N_dim):
            N = self._arg_shape[i]
            w_max = dict(
                constant=np.inf,
                wrap=N,
                reflect=N - 1,
                symmetric=N,
                edge=N,  # Lipschitz constant known analytically up to this limit
            )[self._mode[i]]
            lhs, rhs = self._pad_width[i]
            assert max(lhs, rhs) <= w_max, f"pad_width along dim-{i} is limited to {w_max}."

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self._arg_shape)
        N_dim = len(self._arg_shape)

        xp = pycu.get_array_module(arr)
        pad_width_sh = ((0, 0),) * len(sh)  # don't pad stack-dims

        if len(set(self._mode)) == 1:  # mono-mode: one-shot padding
            out = xp.pad(
                array=arr,
                pad_width=pad_width_sh + self._pad_width,
                mode=self._mode[0],
            )
        else:  # multi-mode: pad iteratively
            out = arr
            for i in range(N_dim):
                pad_width = [(0, 0)] * N_dim
                pad_width[i] = self._pad_width[i]
                pad_width = tuple(pad_width)

                out = xp.pad(
                    array=out,
                    pad_width=pad_width_sh + pad_width,
                    mode=self._mode[i],
                )

        out = out.reshape(*sh, self.codim)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        # todo: implement
        pass

    def lipschitz(self, **kwargs) -> pyct.Real:
        if kwargs.get("recompute", False):
            self._lipschitz = super().lipschitz(**kwargs)
        else:
            L = []  # 1D pad-op Lipschitz constants
            for N, m, (lhs, rhs) in zip(self._arg_shape, self._mode, self._pad_width):
                # Numbers obtained by evaluating L numerically over the entire range
                # of supported (lhs, rhs) pad-widths.
                p = lhs + rhs
                if m == "constant":
                    _L = 1
                elif m in {"wrap", "symmetric"}:
                    if p == 0:
                        _L = 1
                    elif 1 <= p <= N:
                        _L = np.sqrt(2)
                    else:  # N + 1 <= p <= 2 N
                        _L = np.sqrt(3)
                elif m == "reflect":
                    if p == 0:
                        _L = 1
                    elif 1 <= p <= N - 2:
                        _L = np.sqrt(2)
                    else:  # N - 1 <= p <= 2 N - 2
                        _L = np.sqrt(3)
                elif m == "edge":
                    _L = np.sqrt(1 + min(p, N))
                L.append(_L)
            self._lipschitz = np.prod(L)
        return self._lipschitz

    def gram(self) -> pyct.OpT:
        if all(m == "constant" for m in self._mode):
            from pycsou.operator.linop.base import IdentityOp

            op = IdentityOp(dim=self.dim)
        else:
            op = super().gram()
        return op

    def cogram(self) -> pyct.OpT:
        if all(m == "constant" for m in self._mode):
            # orthogonal projection
            op = pyca.OrthProjOp(shape=(self.codim, self.codim))
            op._op = self

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
                sh = arr.shape[:-1]
                pad_shape = _._op._pad_shape
                arr = arr.reshape(*sh, *pad_shape)

                pad_width = _._op._pad_width
                selector = [slice(None)] * len(sh)
                for N, (lhs, rhs) in zip(pad_shape, pad_width):
                    s = slice(lhs, N - rhs)
                    selector.append(s)
                selector = tuple(selector)

                pad_width_sh = ((0, 0),) * len(sh)  # don't pad stack-dims
                xp = pycu.get_array_module(arr)
                out = xp.pad(
                    array=arr[selector],
                    pad_width=pad_width_sh + pad_width,
                    mode="constant",
                )

                out = out.reshape(*sh, _.codim)
                return out

            op.apply = types.MethodType(op_apply, op)
        else:
            op = super().cogram()
        return op
