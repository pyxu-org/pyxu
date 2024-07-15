import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
import pyxu.util as pxu

__all__ = [
    "kron",
    "khatri_rao",
]


def kron(A: pxt.OpT, B: pxt.OpT) -> pxt.OpT:
    r"""
    `Kronecker product <https://en.wikipedia.org/wiki/Kronecker_product#Definition>`_ :math:`A \otimes B` between two
    linear operators.

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

    where :math:`A : \mathbb{R}^{N_{A}} \to \mathbb{R}^{M_{A}}`, and :math:`B : \mathbb{R}^{N_{B}} \to
    \mathbb{R}^{M_{B}}`.

    Parameters
    ----------
    A: OpT
        (mA, nA) linear operator
    B: OpT
        (mB, nB) linear operator

    Returns
    -------
    op: OpT
        (mA*mB, nA*nB) linear operator.

    Notes
    -----
    This implementation is **matrix-free** by leveraging properties of the Kronecker product, i.e.  :math:`A` and
    :math:`B` need not be known explicitly.  In particular :math:`(A \otimes B) x` and :math:`(A \otimes B)^{*} x` are
    computed implicitly via the relation:

    .. math::

       \text{vec}(\mathbf{A}\mathbf{B}\mathbf{C})
       =
       (\mathbf{C}^{T} \otimes \mathbf{A}) \text{vec}(\mathbf{B}),

    where :math:`\mathbf{A}`, :math:`\mathbf{B}`, and :math:`\mathbf{C}` are matrices.
    """

    def _infer_op_shape(shA: pxt.NDArrayShape, shB: pxt.NDArrayShape) -> pxt.NDArrayShape:
        sh = (shA[0] * shB[0], shA[1] * shB[1])
        return sh

    def _infer_op_klass(A: pxt.OpT, B: pxt.OpT) -> pxt.OpC:
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
            properties.add(pxa.Property.LINEAR_SQUARE)
        if pxa.Property.FUNCTIONAL in properties:
            klass = pxa.LinFunc
        else:
            klass = pxa.Operator._infer_operator_type(properties)
        return klass

    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
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

    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
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

    def op_estimate_lipschitz(_, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L_A = _._A.lipschitz
            L_B = _._B.lipschitz
            L = L_A * L_B
        else:
            L = _.__class__.estimate_lipschitz(_, **kwargs)
        return L

    def op_asarray(_, **kwargs) -> pxt.NDArray:
        # (A \kron B).asarray() = A.asarray() \kron B.asarray()
        A = _._A.asarray(**kwargs)
        B = _._B.asarray(**kwargs)
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        C = xp.tensordot(A, B, axes=0).transpose((0, 2, 1, 3)).reshape(_.shape)
        return C

    def op_gram(_) -> pxt.OpT:
        # (A \kron B).gram() = A.gram() \kron B.gram()
        A = _._A.gram()
        B = _._B.gram()
        op = kron(A, B)
        return op

    def op_cogram(_) -> pxt.OpT:
        # (A \kron B).cogram() = A.cogram() \kron B.cogram()
        A = _._A.cogram()
        B = _._B.cogram()
        op = kron(A, B)
        return op

    def op_svdvals(_, **kwargs) -> pxt.NDArray:
        # (A \kron B).svdvals(k)
        # = outer(
        #     A.svdvals(k),
        #     B.svdvals(k)
        #   ).top(k)
        k = kwargs.get("k", 1)

        D_A = _._A.svdvals(**kwargs)
        D_B = _._B.svdvals(**kwargs)
        xp = pxu.get_array_module(D_A)
        D_C = xp.concatenate([D_A, D_B])[-k:]
        return D_C

    def op_pinv(_, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        if np.isclose(damp, 0):
            # (A \kron B).dagger() = A.dagger() \kron B.dagger()
            op_d = kron(_._A.dagger(damp, **kwargs), _._B.dagger(damp, **kwargs))
            out = op_d.apply(arr)
        else:
            # default algorithm
            out = _.__class__.pinv(_, arr, damp, **kwargs)
        return out

    def op_trace(_, **kwargs) -> pxt.Real:
        # tr(A \kron B) = tr(A) * tr(B)
        # [if both square, else default algorithm]
        P = pxa.Property.LINEAR_SQUARE
        if not _.has(P):
            raise NotImplementedError

        if _._A.has(P) and _._B.has(P):
            tr = _._A.trace(**kwargs) * _._B.trace(**kwargs)
        else:
            tr = _.__class__.trace(_, **kwargs)
        return tr

    _A = A.squeeze()
    _B = B.squeeze()
    assert (klass := _infer_op_klass(_A, _B)).has(pxa.Property.LINEAR)
    is_scalar = lambda _: _.shape == (1, 1)
    if is_scalar(_A) and is_scalar(_B):
        from pyxu.operator.linop.base import HomothetyOp

        return HomothetyOp(cst=(_A.asarray() * _B.asarray()).item(), dim=1)
    elif is_scalar(_A) and (not is_scalar(_B)):
        return _A.asarray().item() * _B
    elif (not is_scalar(_A)) and is_scalar(B):
        return _A * _B.asarray().item()
    else:
        op = px_src.from_source(
            cls=klass,
            shape=_infer_op_shape(_A.shape, _B.shape),
            embed=dict(
                _name="kron",
                _A=_A,
                _B=_B,
            ),
            apply=op_apply,
            adjoint=op_adjoint,
            asarray=op_asarray,
            gram=op_gram,
            cogram=op_cogram,
            svdvals=op_svdvals,
            pinv=op_pinv,
            trace=op_trace,
            estimate_lipschitz=op_estimate_lipschitz,
            _expr=lambda _: (_._name, _._A, _._B),
        )
        op.lipschitz = op.estimate_lipschitz(__rule=True)
    return op


def khatri_rao(A: pxt.OpT, B: pxt.OpT) -> pxt.OpT:
    r"""
    `Column-wise Khatri-Rao product
    <https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Column-wise_Kronecker_product>`_ :math:`A \circ B` between
    two linear operators.

    The Khatri-Rao product :math:`A \circ B` is defined as

    .. math::

       A \circ B
       =
       \left[
           \begin{array}{ccc}
           \mathbf{a}_{1} \otimes \mathbf{b}_{1} & \cdots & \mathbf{a}_{N} \otimes \mathbf{b}_{N}
           \end{array}
       \right],

    where :math:`A : \mathbb{R}^{N} \to \mathbb{R}^{M_{A}}`, :math:`B : \mathbb{R}^{N} \to \mathbb{R}^{M_{B}}`, and
    :math:`\mathbf{a}_{k}` (repectively :math:`\mathbf{b}_{k}`) denotes the :math:`k`-th column of :math:`A`
    (respectively :math:`B`).

    Parameters
    ----------
    A: OpT
        (mA, n) linear operator
    B: OpT
        (mB, n) linear operator

    Returns
    -------
    op: OpT
        (mA*mB, n) linear operator.

    Notes
    -----
    This implementation is **matrix-free** by leveraging properties of the Khatri-Rao product, i.e.  :math:`A` and
    :math:`B` need not be known explicitly.  In particular :math:`(A \circ B) x` and :math:`(A \circ B)^{*} x` are
    computed implicitly via the relation:

    .. math::

       \text{vec}(\mathbf{A}\text{diag}(\mathbf{b})\mathbf{C})
       =
       (\mathbf{C}^{T} \circ \mathbf{A}) \mathbf{b},

    where :math:`\mathbf{A}`, :math:`\mathbf{C}` are matrices, and :math:`\mathbf{b}` is a vector.

    Note however that a matrix-free implementation of the Khatri-Rao product does not permit the same optimizations as a
    matrix-based implementation.  Thus the Khatri-Rao product as implemented here is only marginally more efficient than
    applying :py:func:`~pyxu.operator.kron` and pruning its output.
    """

    def _infer_op_shape(shA: pxt.NDArrayShape, shB: pxt.NDArrayShape) -> pxt.NDArrayShape:
        if shA[1] != shB[1]:
            raise ValueError(f"Khatri-Rao product of {shA} and {shB} operators forbidden.")
        sh = (shA[0] * shB[0], shA[1])
        return sh

    def _infer_op_klass(A: pxt.OpT, B: pxt.OpT) -> pxt.OpC:
        # linear \kr linear -> linear
        # square (if output square)
        sh = _infer_op_shape(A.shape, B.shape)
        if sh[0] == 1:
            klass = pxa.LinFunc
        else:
            properties = set(pxa.LinOp.properties())
            if sh[0] == sh[1]:
                properties.add(pxa.Property.LINEAR_SQUARE)
            klass = pxa.Operator._infer_operator_type(properties)
        return klass

    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
        # If `x` is a vector, then:
        #     (A \kr B)(x) = vec(B * diag(x) * A.T)
        sh_prefix = arr.shape[:-1]
        sh_dim = len(sh_prefix)
        xp = pxu.get_array_module(arr)
        I = xp.eye(N=_.dim, dtype=arr.dtype)  # noqa: E741

        x = arr.reshape((*sh_prefix, 1, _.dim))  # (..., 1, dim)
        y = _._B.apply(x * I)  # (..., dim, B.codim)
        z = y.transpose((*range(sh_dim), -1, -2))  # (..., B.codim, dim)
        t = _._A.apply(z)  # (..., B.codim, A.codim)
        u = t.transpose((*range(sh_dim), -1, -2))  # (..., A.codim, B.codim)

        out = u.reshape((*sh_prefix, -1))  # (..., A.codim * B.codim)
        return out

    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
        # If `x` is a vector, then:
        #     (A \kr B).H(x) = diag(B.H * mat(x) * A.conj)
        sh_prefix = arr.shape[:-1]
        sh_dim = len(sh_prefix)
        xp = pxu.get_array_module(arr)
        I = xp.eye(N=_.dim, dtype=arr.dtype)  # noqa: E741

        x = arr.reshape((*sh_prefix, _._A.codim, _._B.codim))  # (..., A.codim, B.codim)
        y = _._B.adjoint(x)  # (..., A.codim, B.dim)
        z = y.transpose((*range(sh_dim), -1, -2))  # (..., dim, A.codim)
        t = pxu.copy_if_unsafe(_._A.adjoint(z))  # (..., dim, dim)
        t *= I

        out = t.sum(axis=-1)  # (..., dim)
        return out

    def op_asarray(_, **kwargs) -> pxt.NDArray:
        # (A \kr B).asarray()[:,i] = A.asarray()[:,i] \kron B.asarray()[:,i]
        A = _._A.asarray(**kwargs).T.reshape((_.dim, _._A.codim, 1))
        B = _._B.asarray(**kwargs).T.reshape((_.dim, 1, _._B.codim))
        C = (A * B).reshape((_.dim, -1)).T
        return C

    def op_lipschitz(_, **kwargs) -> pxt.Real:
        if kwargs.get("tight", False):
            _._lipschitz = _.__class__.lipschitz(_, **kwargs)
        else:
            op = kron(_._A, _._B)
            _._lipschitz = op.lipschitz(**kwargs)
        return _._lipschitz

    _A = A.squeeze()
    _B = B.squeeze()
    assert (klass := _infer_op_klass(_A, _B)).has(pxa.Property.LINEAR)

    op = px_src.from_source(
        cls=klass,
        shape=_infer_op_shape(_A.shape, _B.shape),
        embed=dict(
            _name="khatri_rao",
            _A=_A,
            _B=_B,
        ),
        apply=op_apply,
        adjoint=op_adjoint,
        asarray=op_asarray,
        _expr=lambda _: (_._name, _._A, _._B),
    )

    # kr(A,B) = kron(A,B) + sub-sampling -> upper-bound provided by kron(A,B).lipschitz
    op.lipschitz = kron(_A, _B).lipschitz
    return op
