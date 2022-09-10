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
    """
    # Preserved properties:
    #     square (if output square)
    #     normal \kron normal -> normal
    #     unit \kron unit -> unit
    #     self-adj \kron self-adj -> self-adj
    #     pos-def \kron pos-def -> pos-def
    #     idemp \kron idemp -> idemp
    # Rules
    #     (A \kron B)(x) = vec(B * mat(x) * A.T)
    #     (A \kron B).H(x) = vec(B.H * mat(x) * A.conj)
    #     (A \kron B)._lipschitz = inf
    #     (A \kron B).asarray() = A.asarray() \kron B.asarray()
    #     (A \kron B).gram() = A.gram() \kron B.gram()
    #     (A \kron B).cogram() = A.cogram() \kron B.cogram()
    #     (A \kron B).svdvals(k, which) = outer(A.svdvals(k, which), B.svdvals(k, which)).[top|bottom](k)
    #     (A \kron B).eigvals(k, which) = outer(A.eigvals(k, which), B.eigvals(k, which)).abs().[top|bottom](k)
    #     (A \kron B).pinv(x, tau) = (A \kron B).dagger(tau).apply(x)
    #     (A \kron B).dagger(tau) = A.dagger() \kron B.dagger() [if tau=0, else default algorithm]
    #     tr(A \kron B) = tr(A) * tr(B)  [if both square, else default algorithm]
    #     (A \kron B)._expr() = ("kron", A, B)
    # squeeze before return
    pass


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
    """
    # Preserved properties:
    #     square (if output square)
    # Rules
    #     (A \kr B)(x) = vec(B * diag(x) * A.T)
    #     (A \kr B).H(x) = diag(B.H * mat(x) * A.conj)
    #     (A \kr B)._lipschitz = inf
    #     (A \kr B).asarray() = A.asarray() \kr B.asarray()
    #     (A \kr B).gram() =   default algorithm
    #     (A \kr B).cogram() = default algorithm
    #     (A \kr B).svdvals(k, which) = default algorithm
    #     (A \kr B).eigvals(k, which) = default algorithm
    #     (A \kr B).pinv(x, tau) = default algorithm
    #     (A \kr B).dagger(tau) = default algorithm
    #     tr(A \kr B) = default algorithm
    #     (A \kr B)._expr() = ("kr", A, B)
    # squeeze before return
    pass
