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

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        return pxu.read_only(arr)

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        return pxu.read_only(arr)

    def svdvals(self, **kwargs) -> pxt.NDArray:
        return pxa.UnitOp.svdvals(self, **kwargs)

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        A = xp.eye(N=self.dim, dtype=dtype)
        return A

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = arr.copy()
        out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        cst = 1 / (1 + damp)
        op = HomothetyOp(cst=cst, dim=self.dim)
        return op

    @pxrt.enforce_precision()
    def trace(self, **kwargs) -> pxt.Real:
        return self.dim


class NullOp(pxa.LinOp):
    """
    Null operator.

    This operator maps any input vector on the null vector.
    """

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        self.lipschitz = 0

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.codim),
        )

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.dim),
        )

    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        width = pxrt.getPrecision()

        D = xp.zeros(kwargs["k"], dtype=width.value)
        return D

    def gram(self) -> pxt.OpT:
        op = NullOp(shape=(self.dim, self.dim))
        return op.asop(pxa.SelfAdjointOp).squeeze()

    def cogram(self) -> pxt.OpT:
        op = NullOp(shape=(self.codim, self.codim))
        return op.asop(pxa.SelfAdjointOp).squeeze()

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        A = xp.zeros(self.shape, dtype=dtype)
        return A

    @pxrt.enforce_precision()
    def trace(self, **kwargs) -> pxt.Real:
        return 0


def NullFunc(dim: pxt.Integer) -> pxt.OpT:
    """
    Null functional.

    This functional maps any input vector on the null scalar.
    """
    op = NullOp(shape=(1, dim)).squeeze()
    op._name = "NullFunc"
    return op


def HomothetyOp(dim: pxt.Integer, cst: pxt.Real) -> pxt.OpT:
    """
    Constant scaling operator.

    Parameters
    ----------
    dim: Integer
        Dimension of the domain.
    cst: Real
        Scaling factor.

    Returns
    -------
    op: OpT
        (dim, dim) scaling operator.

    Note
    ----
    This operator is not defined in terms of :py:func:`~pyxu.operator.DiagonalOp` since it is array-backend-agnostic.
    """
    assert isinstance(cst, pxt.Real), f"cst: expected real, got {cst}."

    if np.isclose(cst, 0):
        op = NullOp(shape=(dim, dim))
    elif np.isclose(cst, 1):
        op = IdentityOp(dim=dim)
    else:  # build PosDef or SelfAdjointOp

        @pxrt.enforce_precision(i="arr")
        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            out = arr.copy()
            out *= _._cst
            return out

        def op_svdvals(_, **kwargs) -> pxt.NDArray:
            gpu = kwargs.get("gpu", False)
            xp = pxd.NDArrayInfo.from_flag(gpu).module()
            width = pxrt.getPrecision()

            D = xp.full(
                shape=kwargs["k"],
                fill_value=abs(_._cst),
                dtype=width.value,
            )
            return D

        @pxrt.enforce_precision(i=("arr", "damp"))
        def op_pinv(_, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
            out = arr.copy()
            scale = _._cst / (_._cst**2 + damp)
            out *= scale
            return out

        def op_dagger(_, damp: pxt.Real, **kwargs) -> pxt.OpT:
            scale = _._cst / (_._cst**2 + damp)
            op = HomothetyOp(cst=scale, dim=_.dim)
            return op

        def op_gram(_):
            return HomothetyOp(cst=_._cst**2, dim=_.dim)

        @pxrt.enforce_precision()
        def op_trace(_, **kwargs):
            out = _._cst * _.codim
            return out

        op = px_src.from_source(
            cls=pxa.PosDefOp if (cst > 0) else pxa.SelfAdjointOp,
            shape=(dim, dim),
            embed=dict(
                _name="HomothetyOp",
                _cst=cst,
            ),
            apply=op_apply,
            svdvals=op_svdvals,
            pinv=op_pinv,
            gram=op_gram,
            cogram=op_gram,
            trace=op_trace,
        )
        op.dagger = types.MethodType(op_dagger, op)
        op.lipschitz = abs(cst)
    return op.squeeze()


def DiagonalOp(
    vec: pxt.NDArray,
    enable_warnings: bool = True,
) -> pxt.OpT:
    r"""
    Element-wise scaling operator.

    Note
    ----
    :py:func:`~pyxu.operator.DiagonalOp` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as `vec`.  Moreover, inner computations may
    cast input arrays when the precision of `vec` does not match the user-requested precision.  If such a situation
    occurs, a warning is raised.

    Parameters
    ----------
    vec: NDArray
        (N,) diagonal scale factors.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    """
    assert len(vec) == np.prod(vec.shape), f"vec: {vec.shape} is not a DiagonalOp generator."
    if (dim := vec.size) == 1:  # Module-agnostic
        return HomothetyOp(cst=float(vec), dim=1)
    else:
        xp = pxu.get_array_module(vec)
        if pxu.compute(xp.allclose(vec, 0)):
            op = NullOp(shape=(dim, dim))
        elif pxu.compute(xp.allclose(vec, 1)):
            op = IdentityOp(dim=dim)
        else:  # build PosDef or SelfAdjointOp

            @pxrt.enforce_precision(i="arr")
            def op_apply(_, arr):
                if (_._vec.dtype != arr.dtype) and _._enable_warnings:
                    msg = "Computation may not be performed at the requested precision."
                    warnings.warn(msg, pxw.PrecisionWarning)
                out = arr.copy()
                out *= _._vec
                return out

            def op_asarray(_, **kwargs) -> pxt.NDArray:
                xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
                dtype = kwargs.get("dtype", pxrt.getPrecision().value)

                v = pxu.compute(_._vec.astype(dtype=dtype, copy=False))
                v = pxu.to_NUMPY(v)
                A = xp.diag(v)
                return A

            def op_gram(_):
                return DiagonalOp(
                    vec=_._vec**2,
                    enable_warnings=_._enable_warnings,
                )

            def op_svdvals(_, **kwargs):
                gpu = kwargs.get("gpu", False)
                xp = pxd.NDArrayInfo.from_flag(gpu).module()
                width = pxrt.getPrecision()

                k = kwargs["k"]
                which = kwargs.get("which", "LM")

                D = xp.abs(pxu.compute(_._vec))
                D = D[xp.argsort(D)]
                D = D.astype(width.value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            @pxrt.enforce_precision(i=("arr", "damp"))
            def op_pinv(_, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                out = arr.copy()
                out *= scale
                return out

            def op_dagger(_, damp: pxt.Real, **kwargs) -> pxt.OpT:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                return DiagonalOp(
                    vec=scale,
                    enable_warnings=_._enable_warnings,
                )

            @pxrt.enforce_precision()
            def op_trace(_, **kwargs):
                return float(_._vec.sum())

            def op_estimate_lipschitz(_, **kwargs):
                # Calling LinOp's generic method=svd solver may fail since it relies on LinearOperator.
                # We insead use the fact that _lipschitz is computed exactly at construction time.
                return _._lipschitz

            op = px_src.from_source(
                cls=pxa.PosDefOp if pxu.compute(xp.all(vec > 0)) else pxa.SelfAdjointOp,
                shape=(dim, dim),
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
            op.lipschitz = float(abs(vec).max())
        return op.squeeze()


def _ExplicitLinOp(
    cls: pxt.OpC,
    mat: typ.Union[pxt.NDArray, pxt.SparseArray],
    enable_warnings: bool = True,
) -> pxt.OpT:
    r"""
    Build a linear operator from its matrix representation.

    Given a matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times M}`, the *explicit linear operator* associated to
    :math:`\mathbf{A}` is defined as

    .. math::

       f_\mathbf{A}(\mathbf{x})
       =
       \mathbf{A}\mathbf{x},
       \qquad
       \forall \mathbf{x} \in \mathbb{R}^{M},

    with adjoint given by:

    .. math::

       f^{\ast}_{\mathbf{A}}(\mathbf{z})
       =
       \mathbf{A}^{T}\mathbf{z},
       \qquad
       \forall \mathbf{z} \in \mathbb{R}^{N}.

    Parameters
    ----------
    cls: OpC
        LinOp sub-class to instantiate.
    mat: NDArray, SparseArray
        (M, N) matrix generator.
        The input array can be *dense* or *sparse*.
        Accepted sparse arrays are:

        * CPU: COO/CSC/CSR/BSR/GCXS
        * GPU: COO/CSC/CSR
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

    def _standard_form(A):
        fail_dense = False
        try:
            pxd.NDArrayInfo.from_obj(A)
        except Exception:
            fail_dense = True

        fail_sparse = False
        try:
            pxd.SparseArrayInfo.from_obj(A)
        except Exception:
            fail_sparse = True

        if fail_dense and fail_sparse:
            raise ValueError("mat: format could not be inferred.")
        else:
            return A

    def _matmat(A, b, warn: bool) -> pxt.NDArray:
        # A: (M, N) dense/sparse
        # b: (..., N) dense
        # out: (..., M) dense
        if (A.dtype != b.dtype) and warn:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pxw.PrecisionWarning)

        M, N = A.shape
        sh_out = (*b.shape[:-1], M)
        b = b.reshape((-1, N)).T  # (N, (...).prod)
        out = A.dot(b)  # (M, (...).prod)
        return out.T.reshape(sh_out)

    @pxrt.enforce_precision(i="arr")
    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
        return _matmat(_.mat, arr, warn=_._enable_warnings)

    @pxrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
        return _matmat(_.mat.T, arr, warn=_._enable_warnings)

    def op_estimate_lipscthitz(_, **kwargs) -> pxt.Real:
        N = pxd.NDArrayInfo
        S = pxd.SparseArrayInfo

        try:  # Sparse arrays
            sdi = S.from_obj(_.mat)
            if sdi in (S.SCIPY_SPARSE, S.PYDATA_SPARSE):
                ndi = N.NUMPY
            else:  # S.CUPY_SPARSE
                ndi = N.CUPY
        except Exception:  # Dense arrays
            ndi = N.from_obj(_.mat)

        kwargs.update(xp=ndi.module())
        klass = _.__class__
        return klass.estimate_lipschitz(_, **kwargs)

    def op_asarray(_, **kwargs) -> pxt.NDArray:
        N = pxd.NDArrayInfo
        S = pxd.SparseArrayInfo
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)

        try:  # Sparse arrays
            info = S.from_obj(_.mat)
            if info in (S.SCIPY_SPARSE, S.CUPY_SPARSE):
                f = lambda _: _.toarray()
            elif info == S.PYDATA_SPARSE:
                f = lambda _: _.todense()
            A = f(_.mat.astype(dtype))  # `copy` field not ubiquitous
        except Exception:  # Dense arrays
            info = N.from_obj(_.mat)
            A = pxu.compute(_.mat.astype(dtype, copy=False))

            if info == N.DASK:
                # Chunks may have been mapped to sparse arrays.
                # Call .asarray() again for a 2nd pass.
                _op = _ExplicitLinOp(_.__class__, mat=A)
                A = _op.asarray(xp=N.NUMPY.module(), dtype=A.dtype)
        finally:
            A = pxu.to_NUMPY(A)

        return xp.array(A, dtype=dtype)

    @pxrt.enforce_precision()
    def op_trace(_, **kwargs) -> pxt.Real:
        if _.dim != _.codim:
            raise NotImplementedError
        else:
            try:
                tr = _.mat.trace()
            except Exception:
                # .trace() missing for [PYDATA,CUPY]_SPARSE API.
                S = pxd.SparseArrayInfo
                info = S.from_obj(_.mat)
                if info == S.PYDATA_SPARSE:
                    # use `sparse.diagonal().sum()`, but array must be COO.
                    try:
                        A = _.mat.tocoo()  # GCXS inputs
                    except Exception:
                        A = _.mat  # COO inputs
                    finally:
                        tr = info.module().diagonal(A).sum()
                elif info == S.CUPY_SPARSE:
                    tr = _.mat.diagonal().sum()
                else:
                    raise ValueError(f"Unknown sparse format {_.mat}.")
            return float(tr)

    op = px_src.from_source(
        cls=cls,
        shape=mat.shape,
        embed=dict(
            _name="_ExplicitLinOp",
            mat=_standard_form(mat),
            _enable_warnings=bool(enable_warnings),
        ),
        apply=op_apply,
        adjoint=op_adjoint,
        asarray=op_asarray,
        trace=op_trace,
        estimate_lipschitz=op_estimate_lipscthitz,
    )
    return op
