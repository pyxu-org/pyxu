import jax
import jax.flatten_util
import jaxtyping as jt
import lineax as lx
import opt_einsum as oe
import optax.tree_utils as otu

from ..abc import LinearOperator
from ..typing import Array, Arrays


def pinv(op: LinearOperator, y: Arrays, tau: jt.Scalar = 0, **kwargs) -> Arrays:
    r"""
    Find the least-squares solution to the system :math:`\bbA \bbx = \bby`, where :math:`\bbA: \cI \to \cO` is a linear operator.

    The solution is estimated via the :math:`\tau`-dampened normal equation:

    .. math::

       (\bbA^{\ast} \bbA + \tau I) \bbx = \bbA^{\ast} \bby

    Parameters
    ----------
    y: Arrays
        Co-domain vector :math:`\bby \in \cO`.
    tau: Scalar
        Positive dampening factor regularizing the pseudo-inverse.
    kwargs: dict
        Extra parameters passed on to :py:class:`~lineax.CG`.
        You should specify `rtol` and `atol` at minimum.

    Returns
    -------
    x: Arrays
        Pseudo-inverse solution :math:`\hat{\bbx} \in \cI`.
    diagnostics: lineax.Solution
        CG diagnostic information.
        Can be queried to learn more about the solution.
    """
    assert op.dim_shape is not None
    dtype = otu.tree_dtype(y, mixed_dtype_handler="promote")
    dim_info = jax.tree.map(
        lambda _: jax.ShapeDtypeStruct(shape=_.shape, dtype=dtype),
        op.dim_shape,
    )

    lx_op = lx.FunctionLinearOperator(
        fn=lambda x: otu.tree_add_scalar_mul(
            op.adjoint(op.apply(x)),
            tau,
            x,
        ),
        input_structure=dim_info,
        tags=lx.positive_semidefinite_tag,
    )
    solver = lx.CG(**kwargs)
    solution = lx.linear_solve(lx_op, op.adjoint(y), solver)

    x = solution.value
    return x, solution


def hadamard_outer(x: Array, *args: list[Array]) -> Array:
    r"""
    Compute Hadamard product of `x` with outer product of `args`:

    .. math::

       y = x \odot (A_{1} \otimes\cdots\otimes A_{D})

    Parameters
    ----------
    x: Array
        (N1,...,ND)
    args[k]: Array
        (Nk,)

    Returns
    -------
    y: Array
        (N1,...,ND)

    Note
    ----
    All inputs must share the same dtype precision.
    """
    D = len(args)
    assert all(A.ndim == 1 for A in args)
    sh = tuple(A.size for A in args)

    assert x.shape == sh

    x_ind = (*range(D),)
    o_ind = (*range(D),)
    outer_args = [None] * (2 * D)
    for d in range(D):
        outer_args[2 * d] = args[d]
        outer_args[2 * d + 1] = (d,)

    y = oe.contract(  # (N1,...,ND)
        *(x, x_ind),
        *outer_args,
        o_ind,
        use_blas=True,
        optimize="auto",
        backend="jax",
    )
    return y
