import types

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "kron",
    "khatri_rao",
]


def kron(A: pyct.OpT, B: pyct.OpT) -> pyct.OpT:
    r"""
    Kronecker product :math:`A \otimes B` between two linear operators.

    The Kronecker product :math:`A \otimes B` is defined as

    .. math::

       A \otimes B
       =
       \left[
           \begin{array}{ccc}
               A_{11} B     & \cdots & A_{1N_{A}} B \\
               \vdots       & \ddots & \vdots   \\
               A_{M_{A}1} B & \cdots & A_{M_{A}N_{A}} B \\
           \end{array}
       \right],

    where :math:`A : \mathbb{R}^{N_{A}} \to \mathbb{R}^{M_{A}}`,
    and :math:`B : \mathbb{R}^{N_{B}} \to \mathbb{R}^{M_{B}}`.

    Parameters
    ----------
    A: pyct.OpT
        (mA, nA) linear operator
    B: pyct.OpT
        (mB, nB) linear operator

    Returns
    -------
    op: pyct.OpT
        (mA*mB, nA*nB) linear operator.

    Notes
    -----
    This implementation is **matrix-free** by leveraging properties of the Kronecker product, i.e.
    :math:`A` and :math:`B` need not be known explicitly.
    In particular :math:`(A \otimes B) x` and :math:`(A \otimes B)^{*} x` are computed implicitly
    via the relation:

    .. math::

       \text{vec}(\mathbf{A}\mathbf{B}\mathbf{C})
       =
       (\mathbf{C}^{T} \otimes \mathbf{A}) \text{vec}(\mathbf{B}),

    where :math:`\mathbf{A}`, :math:`\mathbf{B}`, and :math:`\mathbf{C}` are matrices.
    """

    def _infer_op_shape(shA: pyct.Shape, shB: pyct.Shape) -> pyct.Shape:
        sh = (shA[0] * shB[0], shA[1] * shB[1])
        return sh

    def _infer_op_klass(A: pyct.OpT, B: pyct.OpT) -> pyct.OpC:
        # linear \kron linear -> linear
        # square (if output square)
        # normal \kron normal -> normal
        # unit \kron unit -> unit
        # self-adj \kron self-adj -> self-adj
        # pos-def \kron pos-def -> pos-def
        # idemp \kron idemp -> idemp
        # func \kron func -> func
        properties = set(A.properties() & B.properties())
        sh = _infer_op_shape(A.shape, B.shape)
        if sh[0] == sh[1]:
            properties.add(pyco.Property.LINEAR_SQUARE)
        if pyco.Property.FUNCTIONAL in properties:
            klass = pyco.LinFunc
        else:
            klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        # If `x` is a vector, then:
        #     (A \kron B)(x) = vec(B * mat(x) * A.T)
        sh_prefix = arr.shape[:-1]
        sh_dim = len(sh_prefix)

        x = arr.reshape((*sh_prefix, _._A.dim, _._B.dim))  # (..., A.dim, B.dim)
        y = _._B.apply(x)  # (..., A.dim, B.codim)
        z = y.transpose((*range(sh_dim), -1, -2))  # (..., B.codim, A.dim)
        t = _._A.apply(z)  # (..., B.codim, A.codim)
        u = t.transpose((*range(sh_dim), -1, -2))  # (..., A.codim, B.codim)

        out = u.reshape((*sh_prefix, -1))  # (..., A.codim * B.codim)
        return out

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        # If `x` is a vector, then:
        #     (A \kron B).H(x) = vec(B.H * mat(x) * A.conj)
        sh_prefix = arr.shape[:-1]
        sh_dim = len(sh_prefix)

        x = arr.reshape((*sh_prefix, _._A.codim, _._B.codim))  # (..., A.codim, B.codim)
        y = _._B.adjoint(x)  # (..., A.codim, B.dim)
        z = y.transpose((*range(sh_dim), -1, -2))  # (..., B.dim, A.codim)
        t = _._A.adjoint(z)  # (..., B.dim, A.dim)
        u = t.transpose((*range(sh_dim), -1, -2))  # (..., A.dim, B.dim)

        out = u.reshape((*sh_prefix, -1))  # (..., A.dim * B.dim)
        return out

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        # (A \kron B).asarray() = A.asarray() \kron B.asarray()
        A = _._A.asarray(**kwargs)
        B = _._B.asarray(**kwargs)
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        C = xp.tensordot(A, B, axes=0).transpose((0, 2, 1, 3)).reshape(_.shape)
        return C

    def op_gram(_) -> pyct.OpT:
        # (A \kron B).gram() = A.gram() \kron B.gram()
        A = _._A.gram()
        B = _._B.gram()
        op = kron(A, B)
        return op

    def op_cogram(_) -> pyct.OpT:
        # (A \kron B).cogram() = A.cogram() \kron B.cogram()
        A = _._A.cogram()
        B = _._B.cogram()
        op = kron(A, B)
        return op

    def op_svdvals(_, **kwargs) -> pyct.NDArray:
        # (A \kron B).svdvals(k, which)
        # = outer(
        #     A.svdvals(k, which),
        #     B.svdvals(k, which)
        #   ).[top|bottom](k)
        k = kwargs.get("k", 1)
        which = kwargs.get("which", "LM")
        if which.upper() == "SM":
            # `scipy.sparse.linalg.svds()` and `scipy.linalg.svd()` will only return up to
            # min(shape) singular values.
            # As such (A.svdvals(), B.svdvals()) output is insufficient to infer SM-singular values
            # of (A \kron B).
            D_C = _.__class__.svdvals(_, **kwargs)
        else:
            D_A = _._A.svdvals(**kwargs)
            D_B = _._B.svdvals(**kwargs)
            xp = pycu.get_array_module(D_A)
            pad_length = np.fmax(k - len(D_A) * len(D_B), 0)
            D_C = xp.concatenate(
                [
                    xp.zeros(pad_length, dtype=D_A.dtype),
                    xp.sort(xp.outer(D_A, D_B), axis=None),
                ]
            )[-k:]
        return D_C

    def op_eigvals(_, **kwargs) -> pyct.NDArray:
        # (A \kron B).eigvals(k, which)
        # = outer(
        #     A.eigvals(k, which),
        #     B.eigvals(k, which)
        #   ).abs().[top|bottom](k)
        if not _.has(pyco.Property.LINEAR_NORMAL):
            raise NotImplementedError

        D_A = _._A.eigvals(**kwargs)
        D_B = _._B.eigvals(**kwargs)

        xp = pycu.get_array_module(D_A)
        D_C = xp.outer(D_A, D_B).reshape(-1)
        D_C = D_C[xp.argsort(xp.abs(D_C))]

        k = kwargs.get("k", 1)
        which = kwargs.get("which", "LM")
        return D_C[:k] if (which.upper() == "SM") else D_C[-k:]

    @pycrt.enforce_precision(i="arr")
    def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        damp = kwargs.get("damp", 0)
        if np.isclose(damp, 0):
            # (A \kron B).dagger() = A.dagger() \kron B.dagger()
            op_d = kron(_._A.dagger(**kwargs), _._B.dagger(**kwargs))
            out = op_d.apply(arr)
        else:
            # default algorithm
            out = _.__class__.pinv(_, arr, **kwargs)
        return out

    def op_trace(_, **kwargs) -> pyct.Real:
        # tr(A \kron B) = tr(A) * tr(B)
        # [if both square, else default algorithm]
        P = pyco.Property.LINEAR_SQUARE
        if not _.has(P):
            raise NotImplementedError

        if _._A.has(P) and _._B.has(P):
            tr = _._A.trace(**kwargs) * _._B.trace(**kwargs)
        else:
            tr = _.__class__.trace(_, **kwargs)
        return float(tr)

    def op_expr(_) -> tuple:
        return ("kron", _._A, _._B)

    _A = A.squeeze()
    _B = B.squeeze()
    if not (klass := _infer_op_klass(_A, _B)).has(pyco.Property.LINEAR):
        raise ValueError("Kronecker product is defined for linear operators only.")
    op = klass(shape=_infer_op_shape(_A.shape, _B.shape))
    op._name = "kron"
    op._A = _A  # embed for introspection
    op._B = _B  # embed for introspection

    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.asarray = types.MethodType(op_asarray, op)
    op.gram = types.MethodType(op_gram, op)
    op.cogram = types.MethodType(op_cogram, op)
    op.svdvals = types.MethodType(op_svdvals, op)
    op.eigvals = types.MethodType(op_eigvals, op)
    op.pinv = types.MethodType(op_pinv, op)
    op.trace = types.MethodType(op_trace, op)
    op._expr = types.MethodType(op_expr, op)
    return op


def khatri_rao(A: pyct.OpT, B: pyct.OpT) -> pyct.OpT:
    r"""
    Column-wise Khatri-Rao product :math:`A \circ B` between two linear operators.

    The Khatri-Rao product :math:`A \circ B` is defined as

    .. math::

       A \circ B
       =
       \left[
           \begin{array}{ccc}
           \mathbf{a}_{1} \otimes \mathbf{b}_{1} & \cdots & \mathbf{a}_{N} \otimes \mathbf{b}_{N}
           \end{array}
       \right],

    where :math:`A : \mathbb{R}^{N} \to \mathbb{R}^{M_{A}}`,
    :math:`B : \mathbb{R}^{N} \to \mathbb{R}^{M_{B}}`,
    and :math:`\mathbf{a}_{k}` (repectively :math:`\mathbf{b}_{k}`) denotes the :math:`k`-th column of :math:`A`
    (respectively :math:`B`).

    Parameters
    ----------
    A: pyct.OpT
        (mA, n) linear operator
    B: pyct.OpT
        (mB, n) linear operator

    Returns
    -------
    op: pyct.OpT
        (mA*mB, n) linear operator.

    Notes
    -----
    This implementation is **matrix-free** by leveraging properties of the Khatri-Rao product, i.e.
    :math:`A` and :math:`B` need not be known explicitly.
    In particular :math:`(A \circ B) x` and :math:`(A \circ B)^{*} x` are computed implicitly via
    the relation:

    .. math::

       \text{vec}(\mathbf{A}\text{diag}(\mathbf{b})\mathbf{C})
       =
       (\mathbf{C}^{T} \circ \mathbf{A}) \mathbf{b},

    where :math:`\mathbf{A}`, :math:`\mathbf{C}` are matrices, and :math:`\mathbf{b}` is a vector.

    Note however that a matrix-free implementation of the Khatri-Rao product does not permit the
    same optimizations as a matrix-based implementation.
    Thus the Khatri-Rao product as implemented here is only marginally more efficient than applying
    :py:func:`~pycsou.operator.linop.kron.kron` and pruning its output.
    """

    def _infer_op_shape(shA: pyct.Shape, shB: pyct.Shape) -> pyct.Shape:
        if shA[1] != shB[1]:
            raise ValueError(f"Khatri-Rao product of {shA} and {shB} operators forbidden.")
        sh = (shA[0] * shB[0], shA[1])
        return sh

    def _infer_op_klass(A: pyct.OpT, B: pyct.OpT) -> pyct.OpC:
        # linear \kr linear -> linear
        # square (if output square)
        sh = _infer_op_shape(A.shape, B.shape)
        if sh[0] == 1:
            klass = pyco.LinFunc
        else:
            properties = set(pyco.LinOp.properties())
            if sh[0] == sh[1]:
                properties.add(pyco.Property.LINEAR_SQUARE)
            klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        # If `x` is a vector, then:
        #     (A \kr B)(x) = vec(B * diag(x) * A.T)
        sh_prefix = arr.shape[:-1]
        sh_dim = len(sh_prefix)
        xp = pycu.get_array_module(arr)
        I = xp.eye(N=_.dim, dtype=arr.dtype)  # (dim, dim)

        x = arr.reshape((*sh_prefix, 1, _.dim))  # (..., 1, dim)
        y = _._B.apply(x * I)  # (..., dim, B.codim)
        z = y.transpose((*range(sh_dim), -1, -2))  # (..., B.codim, dim)
        t = _._A.apply(z)  # (..., B.codim, A.codim)
        u = t.transpose((*range(sh_dim), -1, -2))  # (..., A.codim, B.codim)

        out = u.reshape((*sh_prefix, -1))  # (..., A.codim * B.codim)
        return out

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        # If `x` is a vector, then:
        #     (A \kr B).H(x) = diag(B.H * mat(x) * A.conj)
        sh_prefix = arr.shape[:-1]
        sh_dim = len(sh_prefix)
        xp = pycu.get_array_module(arr)
        I = xp.eye(N=_.dim, dtype=arr.dtype)

        x = arr.reshape((*sh_prefix, _._A.codim, _._B.codim))  # (..., A.codim, B.codim)
        y = _._B.adjoint(x)  # (..., A.codim, B.dim)
        z = y.transpose((*range(sh_dim), -1, -2))  # (..., dim, A.codim)
        t = pycu.copy_if_unsafe(_._A.adjoint(z))  # (..., dim, dim)
        t *= I

        out = t.sum(axis=-1)  # (..., dim)
        return out

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        # (A \kr B).asarray()[:,i] = A.asarray()[:,i] \kron B.asarray()[:,i]
        A = _._A.asarray(**kwargs).T.reshape((_.dim, _._A.codim, 1))
        B = _._B.asarray(**kwargs).T.reshape((_.dim, 1, _._B.codim))
        C = (A * B).reshape((_.dim, -1)).T
        return C

    def op_expr(_) -> tuple:
        return ("khatri_rao", _._A, _._B)

    _A = A.squeeze()
    _B = B.squeeze()
    if not (klass := _infer_op_klass(_A, _B)).has(pyco.Property.LINEAR):
        raise ValueError("Khatri-Rao product is defined for linear operators only.")
    op = klass(shape=_infer_op_shape(_A.shape, _B.shape))
    op._name = "khatri_rao"
    op._A = _A  # embed for introspection
    op._B = _B  # embed for introspection

    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.asarray = types.MethodType(op_asarray, op)
    op._expr = types.MethodType(op_expr, op)
    return op
