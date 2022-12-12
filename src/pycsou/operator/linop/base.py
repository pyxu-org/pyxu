import types
import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "IdentityOp",
    "NullOp",
    "NullFunc",
    "HomothetyOp",
    "DiagonalOp",
]


class IdentityOp(pyca.OrthProjOp):
    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return pycu.read_only(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return pycu.read_only(arr)

    def svdvals(self, **kwargs) -> pyct.NDArray:
        return pyca.UnitOp.svdvals(self, **kwargs)

    def eigvals(self, **kwargs) -> pyct.NDArray:
        return pyca.UnitOp.svdvals(self, **kwargs)

    def asarray(self, **kwargs) -> pyct.NDArray:
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())
        A = xp.eye(N=self.dim, dtype=dtype)
        return A

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = arr.copy()
        out /= 1 + kwargs.pop("damp", 0)
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        cst = 1 / (1 + kwargs.pop("damp", 0))
        op = HomothetyOp(cst=cst, dim=self.dim)
        return op

    def trace(self, **kwargs) -> pyct.Real:
        return float(self.dim)


class NullOp(pyca.LinOp):
    """
    Null operator.

    This operator maps any input vector on the null vector.
    """

    def __init__(self, shape: pyct.OpShape):
        super().__init__(shape=shape)
        self._lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.codim),
        )

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.dim),
        )

    def svdvals(self, **kwargs) -> pyct.NDArray:
        N = pycd.NDArrayInfo
        xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
        D = xp.zeros(kwargs.pop("k"), dtype=pycrt.getPrecision().value)
        return D

    def gram(self) -> pyct.OpT:
        op = NullOp(shape=(self.dim, self.dim))
        return op.asop(pyca.SelfAdjointOp).squeeze()

    def cogram(self) -> pyct.OpT:
        op = NullOp(shape=(self.codim, self.codim))
        return op.asop(pyca.SelfAdjointOp).squeeze()

    def asarray(self, **kwargs) -> pyct.NDArray:
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())
        A = xp.zeros(self.shape, dtype=dtype)
        return A

    def trace(self, **kwargs) -> pyct.Real:
        return float(0)


def NullFunc(dim: pyct.Integer) -> pyct.OpT:
    """
    Null functional.

    This functional maps any input vector on the null scalar.
    """
    op = NullOp(shape=(1, dim)).squeeze()
    op._name = "NullFunc"
    return op


def HomothetyOp(cst: pyct.Real, dim: pyct.Integer) -> pyct.OpT:
    """
    Scaling operator.

    Parameters
    ----------
    cst: pyct.Real
        Scaling factor.
    dim: pyct.Integer
        Dimension of the domain.

    Returns
    -------
    op: pyct.OpT
        (dim, dim) scaling operator.

    Notes
    -----
    This operator is not defined in terms of :py:func:`~pycsou.operator.linop.DiagonalOp` since it
    is array-backend-agnostic.
    """
    assert isinstance(cst, pyct.Real), f"cst: expected real, got {cst}."

    if np.isclose(cst, 0):
        op = NullOp(shape=(dim, dim))
    elif np.isclose(cst, 1):
        op = IdentityOp(dim=dim)
    else:  # build PosDef or SelfAdjointOp

        @pycrt.enforce_precision(i="arr")
        def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
            out = arr.copy()
            out *= _._cst
            return out

        def op_svdvals(_, **kwargs) -> pyct.NDArray:
            N = pycd.NDArrayInfo
            xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
            D = xp.full(
                shape=kwargs.pop("k"),
                fill_value=abs(_._cst),
                dtype=pycrt.getPrecision().value,
            )
            return D

        def op_eigvals(_, **kwargs) -> pyct.NDArray:
            D = _.svdvals(**kwargs)
            D *= np.sign(_._cst)
            return D

        @pycrt.enforce_precision(i="arr")
        def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
            out = arr.copy()
            scale = _._cst / (_._cst**2 + kwargs.pop("damp", 0))
            out *= scale
            return out

        def op_dagger(_, **kwargs) -> pyct.OpT:
            scale = _._cst / (_._cst**2 + kwargs.pop("damp", 0))
            op = HomothetyOp(cst=scale, dim=_.dim)
            return op

        def op_gram(_):
            return HomothetyOp(cst=_._cst**2, dim=_.dim)

        def op_trace(_, **kwargs):
            out = _._cst * _.codim
            return float(out)

        klass = pyca.PosDefOp if (cst > 0) else pyca.SelfAdjointOp
        op = klass(shape=(dim, dim))
        op._cst = cst
        op._lipschitz = abs(cst)
        op.apply = types.MethodType(op_apply, op)
        op.svdvals = types.MethodType(op_svdvals, op)
        op.eigvals = types.MethodType(op_eigvals, op)
        op.pinv = types.MethodType(op_pinv, op)
        op.dagger = types.MethodType(op_dagger, op)
        op.gram = types.MethodType(op_gram, op)
        op.cogram = op.gram
        op.trace = types.MethodType(op_trace, op)
        op._name = "HomothetyOp"

    return op.squeeze()


def DiagonalOp(
    vec: pyct.NDArray,
    enable_warnings: bool = True,
) -> pyct.OpT:
    r"""
    Diagonal linear operator :math:`L: \mathbf{x} \to \text{diag}(\mathbf{v}) \mathbf{x}`.

    Notes
    -----
    :py:func:`~pycsou.operator.linop.base.DiagonalOp` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``vec``.
    Moreover, inner computations may cast input arrays when the precision of ``vec`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.

    Parameters
    ----------
    vec: pyct.NDArray
        (N,) diagonal scale factors.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    """
    assert len(vec) == np.prod(vec.shape), f"vec: {vec.shape} is not a DiagonalOp generator."
    if (dim := vec.size) == 1:  # Module-agnostic
        return HomothetyOp(cst=float(vec), dim=1)
    else:
        xp = pycu.get_array_module(vec)
        if pycu.compute(xp.allclose(vec, 0)):
            op = NullOp(shape=(dim, dim))
        elif pycu.compute(xp.allclose(vec, 1)):
            op = IdentityOp(dim=dim)
        else:  # build PosDef or SelfAdjointOp

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr):
                if (_._vec.dtype != arr.dtype) and _._enable_warnings:
                    msg = "Computation may not be performed at the requested precision."
                    warnings.warn(msg, pycuw.PrecisionWarning)
                out = arr.copy()
                out *= _._vec
                return out

            def op_asarray(_, **kwargs) -> pyct.NDArray:
                N = pycd.NDArrayInfo
                dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
                xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())

                v = pycu.compute(_._vec.astype(dtype=dtype, copy=False))
                v = pycu.to_NUMPY(v)
                A = xp.diag(v)
                return A

            def op_gram(_):
                return DiagonalOp(
                    vec=_._vec**2,
                    enable_warnings=_._enable_warnings,
                )

            def op_svdvals(_, **kwargs):
                k = kwargs.pop("k")
                which = kwargs.pop("which", "LM")
                N = pycd.NDArrayInfo
                xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
                D = xp.abs(pycu.compute(_._vec))
                D = D[xp.argsort(D)]
                D = D.astype(pycrt.getPrecision().value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            def op_eigvals(_, **kwargs):
                k = kwargs.pop("k")
                which = kwargs.pop("which", "LM")
                N = pycd.NDArrayInfo
                xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
                D = pycu.compute(_._vec)
                D = D[xp.argsort(xp.abs(D))]
                D = D.astype(pycrt.getPrecision().value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            @pycrt.enforce_precision(i="arr")
            def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
                damp = kwargs.pop("damp", 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                out = arr.copy()
                out *= scale
                return out

            def op_dagger(_, **kwargs) -> pyct.OpT:
                damp = kwargs.pop("damp", 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                return DiagonalOp(
                    vec=scale,
                    enable_warnings=_._enable_warnings,
                )

            def op_trace(_, **kwargs):
                return float(_._vec.sum())

            klass = pyca.PosDefOp if pycu.compute(xp.all(vec > 0)) else pyca.SelfAdjointOp
            op = klass(shape=(dim, dim))
            op._vec = vec
            op._enable_warnings = bool(enable_warnings)
            op._lipschitz = pycu.compute(xp.abs(vec).max())
            op.apply = types.MethodType(op_apply, op)
            op.asarray = types.MethodType(op_asarray, op)
            op.gram = types.MethodType(op_gram, op)
            op.cogram = op.gram
            op.svdvals = types.MethodType(op_svdvals, op)
            op.eigvals = types.MethodType(op_eigvals, op)
            op.pinv = types.MethodType(op_pinv, op)
            op.dagger = types.MethodType(op_dagger, op)
            op.trace = types.MethodType(op_trace, op)
            op._name = "DiagonalOp"

        return op.squeeze()


def _ExplicitLinOp(
    cls: pyct.OpC,
    mat: typ.Union[pyct.NDArray, pyct.SparseArray],
    enable_warnings: bool = True,
) -> pyct.OpT:
    r"""
    Build a linear operator from its matrix representation.

    Given a matrix :math:`\mathbf{A}\in\mathbb{R}^{M\times N}`, the *explicit linear operator*
    associated to :math:`\mathbf{A}` is defined as

    .. math::

       f_\mathbf{A}(\mathbf{x})
       =
       \mathbf{A}\mathbf{x},
       \qquad
       \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

       f^\ast_\mathbf{A}(\mathbf{z})
       =
       \mathbf{A}^T\mathbf{z},
       \qquad
       \forall \mathbf{z}\in\mathbb{R}^M.

    Parameters
    ----------
    cls: pyct.OpC
        LinOp sub-class to instantiate.
    mat: pyct.NDArray | pyct.SparseArray
        (M, N) matrix generator.
        The input array can be *dense* or *sparse*.
        Accepted sparse arrays are:

        * CPU: COO/CSC/CSR/BSR/GCXS
        * GPU: COO/CSC/CSR
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.

    Notes
    -----
    * :py:class:`~pycsou.operator.linop.base._ExplicitLinOp` instances are **not
      arraymodule-agnostic**: they will only work with NDArrays belonging to the same (dense) array
      module as ``mat``.
      Moreover, inner computations may cast input arrays when the precision of ``mat`` does not
      match the user-requested precision.
      If such a situation occurs, a warning is raised.

    * The matrix provided in ``__init__()`` is used as-is and can be accessed via ``.mat``.
    """

    def _standard_form(A):
        fail_dense = False
        try:
            pycd.NDArrayInfo.from_obj(A)
        except:
            fail_dense = True

        fail_sparse = False
        try:
            pycd.SparseArrayInfo.from_obj(A)
        except:
            fail_sparse = True

        if fail_dense and fail_sparse:
            raise ValueError("mat: format could not be inferred.")
        else:
            return A

    def _matmat(A, b, warn: bool = True) -> pyct.NDArray:
        # A: (M, N) dense/sparse
        # b: (..., N) dense
        # out: (..., M) dense
        if (A.dtype != b.dtype) and warn:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pycuw.PrecisionWarning)

        M, N = A.shape
        sh_out = (*b.shape[:-1], M)
        b = b.reshape((-1, N)).T  # (N, (...).prod)
        out = A.dot(b)  # (M, (...).prod)
        return out.T.reshape(sh_out)

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        return _matmat(_.mat, arr, warn=_._enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        return _matmat(_.mat.T, arr, warn=_._enable_warnings)

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        N = pycd.NDArrayInfo
        S = pycd.SparseArrayInfo
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())

        try:  # Sparse arrays
            info = S.from_obj(_.mat)
            if info in (S.SCIPY_SPARSE, S.CUPY_SPARSE):
                f = lambda _: _.toarray()
            elif info == S.PYDATA_SPARSE:
                f = lambda _: _.todense()
            A = f(_.mat.astype(dtype))  # `copy` field not ubiquitous
        except:  # Dense arrays
            info = N.from_obj(_.mat)
            A = pycu.compute(_.mat.astype(dtype, copy=False))

            if info == N.DASK:
                # Chunks may have been mapped to sparse arrays.
                # Call .asarray() again for a 2nd pass.
                _op = _ExplicitLinOp(_.__class__, mat=A)
                A = _op.asarray(xp=N.NUMPY.module(), dtype=A.dtype)
        finally:
            A = pycu.to_NUMPY(A)

        return xp.array(A, dtype=dtype)

    def op_trace(_, **kwargs) -> pyct.Real:
        if _.dim != _.codim:
            raise NotImplementedError
        else:
            try:
                tr = _.mat.trace()
            except:
                # .trace() missing for [PYDATA,CUPY]_SPARSE API.
                S = pycd.SparseArrayInfo
                info = S.from_obj(_.mat)
                if info == S.PYDATA_SPARSE:
                    # use `sparse.diagonal().sum()`, but array must be COO.
                    try:
                        A = _.mat.tocoo()  # GCXS inputs
                    except:
                        A = _.mat  # COO inputs
                    finally:
                        tr = info.module().diagonal(A).sum()
                elif info == S.CUPY_SPARSE:
                    tr = _.mat.diagonal().sum()
                else:
                    raise ValueError(f"Unknown sparse format {_.mat}.")
            return float(tr)

    def op_lipschitz(_, **kwargs) -> pyct.Real:
        # We want to piggy-back onto Lin[Op,Func].lipschitz() to compute the Lipschitz constant L.
        # Problem: LinOp.lipschitz() relies on svdvals() or hutchpp() to compute L, and they take
        # different parameters to do computations on the GPU.
        # Solution:
        # * we add the relevant kwargs before calling the LinOp.lipschitz() + drop all unrecognized
        #   kwargs there as needed.
        # * similarly for LinFunc.lipschitz().
        N = pycd.NDArrayInfo
        S = pycd.SparseArrayInfo

        try:  # Dense arrays
            info = N.from_obj(_.mat)
            kwargs.update(
                xp=info.module(),
                gpu=info == N.CUPY,
            )
        except:  # Sparse arrays
            info = S.from_obj(_.mat)
            gpu = info == S.CUPY_SPARSE
            kwargs.update(
                xp=N.CUPY.module() if gpu else N.NUMPY.module(),
                gpu=gpu,
            )

        if _.codim == 1:
            L = pyca.LinFunc.lipschitz(_, **kwargs)
        else:
            L = _.__class__.lipschitz(_, **kwargs)
        return L

    op = cls(shape=mat.shape)
    op.mat = _standard_form(mat)
    op._enable_warnings = bool(enable_warnings)
    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.asarray = types.MethodType(op_asarray, op)
    op.lipschitz = types.MethodType(op_lipschitz, op)
    op.trace = types.MethodType(op_trace, op)
    op._name = "_ExplicitLinOp"
    return op


class _Pad1D(pyca.LinOp):
    def __init__(self, arg_shape, axis, pad_width, mode):

        self.codom_shape = tuple([s + np.sum(pad_width) * (i == axis) for i, s in enumerate(arg_shape)])
        self.arg_shape = tuple(arg_shape)
        self.axis = axis
        self.mode = mode

        # Create multidimensional padding tuple, with padding defined only for axis
        self.pad_width = [
            (0, 0),
        ] * (len(arg_shape) + 1)
        self.pad_width[axis + 1] = pad_width

        # Check that extended boundaries do not overlap
        if np.sum(pad_width) > arg_shape[axis]:
            warnings.warn(
                f"The default Lipschitz constant is estimated assuming that the number of padded elements "
                f"({np.sum(pad_width)})is smaller than the size of the input array ({arg_shape[axis]}). "
                f"For a better estimate call the method `op.lipschitz(recompute=True)`."
            )
            if (np.array(pad_width) > arg_shape[axis]).any():
                raise ValueError(
                    f"The number of padded elements in each side {pad_width} must not be larger than the "
                    f"size of the input array ({arg_shape[axis]}) at axis {axis}."
                )

        super().__init__(shape=(np.prod(self.codom_shape).item(), np.prod(arg_shape).item()))
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
        return np.pad(arr.reshape(-1, *self.arg_shape), self.pad_width, self.mode).reshape(*arr.shape[:-1], self.codim)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        """
        Adjoint of padding in one dimension.

        The adjoint method is performed in two steps, trimming and cumulative summation (see `Notes` in PadOp).
        """

        # Trim
        arr_shape = (-1,) + self.codom_shape
        out_shape = (-1,) + self.arg_shape
        slices = []
        for i, (start, end) in enumerate(self.pad_width):
            end = out_shape[i] + self.pad_width[i][0] if arr_shape[i] != -1 else None
            slices.append(slice(start, end))
        out = arr.reshape(*arr_shape)[tuple(slices)].copy()

        # cumulative sum
        if self.mode == "constant":
            return out.reshape(*arr.shape[:-1], self.dim)

        # Slices of output onto which the input (padded) elements are summed to.
        slices_out = [np.copy(slices), np.copy(slices)]
        if self.mode == "wrap":
            slices_out[0][self.axis + 1] = slice(-self.pad_width[self.axis + 1][0], None)
            slices_out[1][self.axis + 1] = slice(0, self.pad_width[self.axis + 1][1])
        elif self.mode in ["reflect", "symmetric"]:
            # reflect and symmetric only differ by a 1 element displacement, captured by the following `aux` variable.
            aux = self.mode == "reflect"
            slices_out[0][self.axis + 1] = slice(self.pad_width[self.axis + 1][0] + aux - 1, (0 if aux else None), -1)
            slices_out[1][self.axis + 1] = slice(-(1 + aux), -(self.pad_width[self.axis + 1][1] + 1 + aux), -1)
        elif self.mode == "edge":
            slices_out[0][self.axis + 1] = slice(0, 1)
            slices_out[1][self.axis + 1] = slice(-1, None)
        else:
            raise NotImplementedError(f"mode {self.mode} is not supported. ")

        # Slices of input array to be summed to output
        slices_arr = [np.copy(slices), np.copy(slices)]
        slices_arr[0][self.axis + 1] = slice(0, self.pad_width[self.axis + 1][0])
        slices_arr[1][self.axis + 1] = slice(self.codom_shape[self.axis] - (self.pad_width[self.axis + 1][1]), None)

        # Perform cumulative summation
        for i, slice_ in enumerate(slices_arr):
            if arr.reshape(*arr_shape)[tuple(slice_)].size:
                if self.mode == "edge":
                    out[tuple(slices_out[i])] += arr.reshape(*arr_shape)[tuple(slice_)].sum(
                        axis=self.axis + 1, keepdims=True
                    )
                else:
                    out[tuple(slices_out[i])] += arr.reshape(*arr_shape)[tuple(slice_)]

        return out.reshape(*arr.shape[:-1], -1)
