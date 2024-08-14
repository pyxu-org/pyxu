import types
import typing as typ
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "IdentityOp",
    "NullOp",
    "NullFunc",
    "HomothetyOp",
    "DiagonalOp",
]


class IdentityOp(pxa.OrthProjOp):
    """
    Identity operator.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        return pxu.read_only(arr)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        return pxu.read_only(arr)

    def svdvals(self, **kwargs) -> pxt.NDArray:
        return pxa.UnitOp.svdvals(self, **kwargs)

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        A = xp.eye(N=self.dim_size, dtype=dtype)
        B = A.reshape(*self.codim_shape, *self.dim_shape)
        return B

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = arr.copy()
        out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = HomothetyOp(
            cst=1 / (1 + damp),
            dim_shape=self.dim_shape,
        )
        return op

    def trace(self, **kwargs) -> pxt.Real:
        return self.dim_size


class NullOp(pxa.LinOp):
    """
    Null operator.

    This operator maps any input vector on the null vector.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        self.lipschitz = 0

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        ndi = pxd.NDArrayInfo.from_obj(arr)
        kwargs = dict()
        if ndi == pxd.NDArrayInfo.DASK:
            stack_chunks = arr.chunks[: -self.dim_rank]
            core_chunks = ("auto",) * self.codim_rank
            kwargs.update(chunks=stack_chunks + core_chunks)

        xp = ndi.module()
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[: -self.dim_rank], *self.codim_shape),
            **kwargs,
        )

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        ndi = pxd.NDArrayInfo.from_obj(arr)
        kwargs = dict()
        if ndi == pxd.NDArrayInfo.DASK:
            stack_chunks = arr.chunks[: -self.codim_rank]
            core_chunks = ("auto",) * self.dim_rank
            kwargs.update(chunks=stack_chunks + core_chunks)

        xp = ndi.module()
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[: -self.codim_rank], *self.dim_shape),
            **kwargs,
        )

    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)

        D = xp.zeros(kwargs["k"], dtype=dtype)
        return D

    def gram(self) -> pxt.OpT:
        op = NullOp(
            dim_shape=self.dim_shape,
            codim_shape=self.dim_shape,
        )
        return op.asop(pxa.SelfAdjointOp)

    def cogram(self) -> pxt.OpT:
        op = NullOp(
            dim_shape=self.codim_shape,
            codim_shape=self.codim_shape,
        )
        return op.asop(pxa.SelfAdjointOp)

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        A = xp.broadcast_to(
            xp.array(0, dtype=dtype),
            (*self.codim_shape, *self.dim_shape),
        )
        return A

    def trace(self, **kwargs) -> pxt.Real:
        return 0


def NullFunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    """
    Null functional.

    This functional maps any input vector on the null scalar.
    """
    op = NullOp(
        dim_shape=dim_shape,
        codim_shape=1,
    ).asop(pxa.LinFunc)
    op._name = "NullFunc"
    return op


def HomothetyOp(dim_shape: pxt.NDArrayShape, cst: pxt.Real) -> pxt.OpT:
    """
    Constant scaling operator.

    Parameters
    ----------
    cst: Real
        Scaling factor.

    Returns
    -------
    op: OpT
        Scaling operator.

    Note
    ----
    This operator is not defined in terms of :py:func:`~pyxu.operator.DiagonalOp` since it is array-backend-agnostic.
    """
    assert isinstance(cst, pxt.Real), f"cst: expected real, got {cst}."

    if np.isclose(cst, 0):
        op = NullOp(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
    elif np.isclose(cst, 1):
        op = IdentityOp(dim_shape=dim_shape)
    else:  # build PosDef or SelfAdjointOp

        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            out = arr.copy()
            out *= _._cst
            return out

        def op_svdvals(_, **kwargs) -> pxt.NDArray:
            gpu = kwargs.get("gpu", False)
            xp = pxd.NDArrayInfo.from_flag(gpu).module()
            dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)

            D = xp.full(
                shape=kwargs["k"],
                fill_value=abs(_._cst),
                dtype=dtype,
            )
            return D

        def op_pinv(_, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
            out = arr.copy()
            out *= _._cst / (_._cst**2 + damp)
            return out

        def op_dagger(_, damp: pxt.Real, **kwargs) -> pxt.OpT:
            op = HomothetyOp(
                cst=_._cst / (_._cst**2 + damp),
                dim_shape=_.dim_shape,
            )
            return op

        def op_gram(_):
            op = HomothetyOp(
                cst=_._cst**2,
                dim_shape=_.dim_shape,
            )
            return op

        def op_estimate_lipschitz(_, **kwargs) -> pxt.Real:
            L = abs(_._cst)
            return L

        def op_trace(_, **kwargs):
            out = _._cst * _.dim_size
            return out

        op = px_src.from_source(
            cls=pxa.PosDefOp if (cst > 0) else pxa.SelfAdjointOp,
            dim_shape=dim_shape,
            codim_shape=dim_shape,
            embed=dict(
                _name="HomothetyOp",
                _cst=cst,
                _lipschitz=float(abs(cst)),
            ),
            apply=op_apply,
            svdvals=op_svdvals,
            pinv=op_pinv,
            gram=op_gram,
            cogram=op_gram,
            trace=op_trace,
            estimate_lipschitz=op_estimate_lipschitz,
        )
        op.dagger = types.MethodType(op_dagger, op)
    return op


def DiagonalOp(
    vec: pxt.NDArray,
    dim_shape: pxt.NDArrayShape = None,
    enable_warnings: bool = True,
) -> pxt.OpT:
    r"""
    Element-wise scaling operator.

    Note
    ----
    * :py:func:`~pyxu.operator.DiagonalOp` instances are **not arraymodule-agnostic**: they will only work with NDArrays
      belonging to the same array module as `vec`.  Moreover, inner computations may cast input arrays when the
      precision of `vec` does not match the user-requested precision.  If such a situation occurs, a warning is raised.
    * If `vec` is a DASK array, the operator will be a :py:class:`~pyxu.abc.SelfAdjointOp`.  If `vec` is a NUMPY/CUPY
      array, the created operator specializes to :py:class:`~pyxu.abc.PosDefOp` when possible.  Specialization is not
      automatic for DASK inputs because operators should be quick to build under all circumstances, and this is not
      guaranteed if we have to check that all entries are positive for out-of-core arrays.  Users who know that all
      `vec` entries are positive can manually cast to :py:class:`~pyxu.abc.PosDefOp` afterwards if required.

    Parameters
    ----------
    dim_shape: NDArrayShape
        (M1,...,MD) shape of operator's domain.
        Defaults to the shape of `vec` when omitted.
    vec: NDArray
        Scale factors. If `dim_shape` is provided, then `vec` must be broadcastable with arrays of size `dim_shape`.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    """
    if dim_shape is None:
        dim_shape = vec.shape
    else:
        dim_shape = pxu.as_canonical_shape(dim_shape)
        sh = np.broadcast_shapes(vec.shape, dim_shape)

        # Getting here means `vec` and `dim_shape` are broadcastable, but we don't know yet
        # which one defines the upper bound.
        assert all(s <= d for (s, d) in zip(sh, dim_shape)), "vec and dim_shape are incompatible."

    def op_apply(_, arr):
        if (_._vec.dtype != arr.dtype) and _._enable_warnings:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pxw.PrecisionWarning)
        out = arr.copy()
        out *= _._vec
        return out

    def op_asarray(_, **kwargs) -> pxt.NDArray:
        xp = pxu.get_array_module(_._vec)
        vec = xp.broadcast_to(_._vec, _.dim_shape)
        A = xp.diag(vec.reshape(-1)).reshape((*_.codim_shape, *_.dim_shape))

        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        B = xp.array(pxu.to_NUMPY(A), dtype=dtype)
        return B

    def op_gram(_):
        return DiagonalOp(
            vec=_._vec**2,
            dim_shape=_.dim_shape,
            enable_warnings=_._enable_warnings,
        )

    def op_svdvals(_, **kwargs):
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        k = kwargs["k"]
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)

        vec = xp.broadcast_to(
            xp.abs(_._vec),
            _.dim_shape,
        ).reshape(-1)
        if ndi == pxd.NDArrayInfo.DASK:
            D = xp.topk(vec, k)
        else:
            vec = vec[vec.argsort()]
            D = vec[-k:]
        return D.astype(dtype)

    def op_pinv(_, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale = _._vec / (_._vec**2 + damp)
            scale[xp.isnan(scale)] = 0
        out = arr.copy()
        out *= scale
        return out

    def op_dagger(_, damp: pxt.Real, **kwargs) -> pxt.OpT:
        xp = pxu.get_array_module(_._vec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale = _._vec / (_._vec**2 + damp)
            scale[xp.isnan(scale)] = 0
        return DiagonalOp(
            vec=scale,
            dim_shape=_.dim_shape,
            enable_warnings=_._enable_warnings,
        )

    def op_trace(_, **kwargs):
        xp = pxu.get_array_module(_._vec)
        vec = xp.broadcast_to(_._vec, _.dim_shape)
        return float(vec.sum())

    def op_estimate_lipschitz(_, **kwargs):
        xp = pxu.get_array_module(_._vec)
        _.lipschitz = float(xp.fabs(vec).max())
        return _.lipschitz

    ndi = pxd.NDArrayInfo.from_obj(vec)
    if ndi == pxd.NDArrayInfo.DASK:
        klass = pxa.SelfAdjointOp
    else:
        positive = (vec > 0).all()
        klass = pxa.PosDefOp if positive else pxa.SelfAdjointOp
    op = px_src.from_source(
        cls=klass,
        dim_shape=dim_shape,
        codim_shape=dim_shape,
        embed=dict(
            _name="DiagonalOp",
            _vec=vec,
            _enable_warnings=bool(enable_warnings),
        ),
        apply=op_apply,
        estimate_lipschitz=op_estimate_lipschitz,
        asarray=op_asarray,
        gram=op_gram,
        cogram=op_gram,
        svdvals=op_svdvals,
        pinv=op_pinv,
        trace=op_trace,
    )
    op.dagger = types.MethodType(op_dagger, op)
    return op


def _ExplicitLinOp(
    cls: pxt.OpC,
    mat: typ.Union[pxt.NDArray, pxt.SparseArray],
    dim_rank: pxt.Integer = None,
    enable_warnings: bool = True,
) -> pxt.OpT:
    r"""
    Build a linear operator from its matrix representation.

    Given an array :math:`\mathbf{A} \in \mathbb{R}^{N_{1} \times\cdots\times N_{K} \times M_{1} \times\cdots\times
    M_{D}}`, the *explicit linear operator* associated to :math:`\mathbf{A}` is defined as

    .. math::

       [\mathbf{A}\mathbf{x}]_{n_{1},\ldots,n_{K}}
       =
       \langle \mathbf{A}[n_{1},\ldots,n_{K},\ldots], \mathbf{x} \rangle
       \qquad
       \forall \mathbf{x} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}.

    Parameters
    ----------
    cls: OpC
        LinOp sub-class to instantiate.
    mat: NDArray, SparseArray
        (N1,...,NK, M1,...,MD) matrix generator.
        The input array can be *dense* or *sparse*.
        Accepted 2D sparse arrays are:

        * CPU: COO/CSC/CSR/BSR
        * GPU: COO/CSC/CSR
    dim_rank: Integer
        Rank of operator's domain. (D)
        It can be omitted if `mat` is 2D since auto-inferred to 1.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.

    Notes
    -----
    * :py:class:`~pyxu.operator.linop.base._ExplicitLinOp` instances are **not arraymodule-agnostic**:
      they will only work with NDArrays belonging to the same (dense) array module as `mat`.  Moreover, inner
      computations may cast input arrays when the precision of `mat` does not match the user-requested precision.  If
      such a situation occurs, a warning is raised.

    * The matrix provided to :py:func:`~pyxu.operator.linop.base._ExplicitLinOp` is used as-is and can be accessed via
      ``.mat``.
    """

    def is_dense(A) -> bool:
        # Ensure `A` is a supported array format, then return
        #    `True` if  `A` is dense
        #    `False` if `A` is sparse
        fail_dense = False
        try:
            pxd.NDArrayInfo.from_obj(A)
            return True
        except Exception:
            fail_dense = True

        fail_sparse = False
        try:
            pxd.SparseArrayInfo.from_obj(A)
            return False
        except Exception:
            fail_sparse = True

        if fail_dense and fail_sparse:
            raise ValueError("mat: format could not be inferred.")

    def tensordot(A, b, dim_rank, warn: bool):
        # Parameters
        # ----------
        # A: (N1,...,NK, M1,...,MD) dense or sparse (2D)
        # b: (S1,...,SL, M1,...,MD) dense
        # dim_rank: D
        # warn: bool
        #
        # Returns
        # -------
        # out: (S1,...,SL, N1,...,NK) dense
        if (A.dtype != b.dtype) and warn:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pxw.PrecisionWarning)

        dim_shape = A.shape[-dim_rank:]
        dim_size = np.prod(dim_shape)
        codim_shape = A.shape[:-dim_rank]
        codim_size = np.prod(codim_shape)

        sh = b.shape[:-dim_rank]
        if not is_dense(A):  # sparse matrix -> necessarily 2D
            b = b.reshape(-1, dim_size)
            out = A.dot(b.T).T  # (prod(sh), codim_size)
            out = out.reshape(*sh, codim_size)
        else:  # ND dense array
            N = pxd.NDArrayInfo  # short-hand
            ndi = N.from_obj(A)
            xp = ndi.module()

            if ndi != N.DASK:
                # NUMPY/CUPY.tensordot() works -> use it.
                out = xp.tensordot(  # (S1,...,SL, N1,...,NK)
                    b,  # (S1,...,SL, M1,...,MD)
                    A,  # (N1,...,NK, M1,...,MD)
                    axes=[
                        list(range(-dim_rank, 0)),
                        list(range(-dim_rank, 0)),
                    ],
                )
            else:  # DASK-backed `mat`
                # DASK.tensordot() broken -> use 2D-ops instead
                msg = "[2023.12] DASK's tensordot() is broken. -> fallback onto 2D-shaped ops."
                pxw.warn_dask_perf(msg)

                A_2D = A.reshape(codim_size, dim_size)
                b = b.reshape(-1, dim_size)
                out = A_2D.dot(b.T).T  # (prod(sh), codim_size)
                out = out.reshape(*sh, *codim_shape)
        return out

    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
        out = tensordot(
            A=_.mat,
            b=arr,
            dim_rank=_.dim_rank,
            warn=_._enable_warnings,
        )
        return out

    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
        if is_dense(_.mat):
            axes = (
                *tuple(range(-_.dim_rank, 0)),
                *tuple(range(_.codim_rank)),
            )
        else:
            axes = None  # transposes all axes for 2D sparse arrays
        out = tensordot(
            A=_.mat.transpose(axes),
            b=arr,
            dim_rank=_.codim_rank,
            warn=_._enable_warnings,
        )
        return out

    def op_estimate_lipscthitz(_, **kwargs) -> pxt.Real:
        N = pxd.NDArrayInfo
        S = pxd.SparseArrayInfo

        if is_dense(_.mat):
            ndi = N.from_obj(_.mat)
        else:
            sdi = S.from_obj(_.mat)
            if sdi == S.SCIPY_SPARSE:
                ndi = N.NUMPY
            elif sdi == S.CUPY_SPARSE:
                ndi = N.CUPY
            else:
                raise NotImplementedError

        kwargs.update(
            xp=ndi.module(),
            gpu=ndi == N.CUPY,
            dtype=_.mat.dtype,
        )
        klass = _.__class__
        return klass.estimate_lipschitz(_, **kwargs)

    def op_asarray(_, **kwargs) -> pxt.NDArray:
        N = pxd.NDArrayInfo
        xp = kwargs.get("xp", N.default().module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)

        if is_dense(_.mat):
            A = _.mat.astype(dtype, copy=False)
        else:
            A = _.mat.astype(dtype).toarray()  # `copy field not ubiquitous`
        B = xp.array(pxu.to_NUMPY(A), dtype=dtype)
        return B

    def op_trace(_, **kwargs) -> pxt.Real:
        if _.dim_size != _.codim_size:
            raise NotImplementedError
        elif len(_.mat.shape) == 2:  # dense or sparse
            try:
                tr = _.mat.trace()
            except AttributeError:
                # Not all sparse types have a .trace() method ...
                tr = _.mat.diagonal().sum()
        else:  # ND dense arrays only
            # We don't want to reshape `mat` if DASK-backed for performance reasons, so the trace is built by indexing
            # the "diagonal" manually.
            tr = 0
            for idx in range(_.dim_size):
                dim_idx = np.unravel_index(idx, _.dim_shape)
                codim_idx = np.unravel_index(idx, _.codim_shape)
                tr += _.mat[*codim_idx, *dim_idx]
        return float(tr)

    is_dense(mat)  # We were given a dense/sparse array ...
    # ... but is dim_rank correctly specified?
    assert len(mat.shape) >= 2, "Only 2D+ arrays are supported."
    if len(mat.shape) == 2:
        dim_rank = 1  # doesn't matter what the user specified.
    else:  # rank > 2
        # if ND -> mandatory supplied & (1 <= dim_rank < mat.ndim)
        assert dim_rank is not None, "Dimension rank must be specified for ND operators."
        assert 1 <= dim_rank < len(mat.shape)

    op = px_src.from_source(
        cls=cls,
        dim_shape=mat.shape[-dim_rank:],
        codim_shape=mat.shape[:-dim_rank],
        embed=dict(
            _name="_ExplicitLinOp",
            mat=mat,
            _enable_warnings=bool(enable_warnings),
        ),
        apply=op_apply,
        adjoint=op_adjoint,
        asarray=op_asarray,
        trace=op_trace,
        estimate_lipschitz=op_estimate_lipscthitz,
    )
    return op
