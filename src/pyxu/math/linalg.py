import functools

import jax
import jax.experimental.sparse.linalg as jesl
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jnd
import jaxtyping as jt
import lineax as lx
import optax
import optax.tree_utils as otu

from ..abc import LinearOperator
from ..typing import Array, Arrays


def svdvals(op: LinearOperator, k: int, **kwargs) -> jt.Scalar:
    """
    Compute leading singular values of a linear operator.

    The singular values of small operators are computed by materializing the operator and computing its SVD.
    The singular values of large operators are computed using the LOBPCG method.

    Parameters
    ----------
    op: LinearOperator
        Operator for which singular values are desired.
    k: int
        Number of singular values to compute.
        This value should be relatively small.
    kwargs: dict
        Extra parameters passed on to :py:class:`~jax.experimental.sparse.linalg.lobpcg_standard`.

    Returns
    -------
    D: Array
        (k,) leading singular values of ``op``.
    """
    assert op.dim_info is not None
    assert op.codim_info is not None

    N = min(op.dim_size, op.codim_size)
    if not (5 * k < N):
        # small operator case: materialize + SVD
        lx_op = lx.FunctionLinearOperator(
            fn=op.apply,
            input_structure=op.dim_info,
        )
        A = lx_op.as_matrix()

        D = jnp.linalg.svdvals(A)
        D = jnp.sort(D, descending=True)[:k]
        return D

    # large operator case: LOBPCG
    elif op.codim_size <= op.dim_size:
        x = jax.tree.map(lambda _: jnp.zeros(_.shape, _.dtype), op.codim_info)
        x_flat, f_unravel = jax.flatten_util.ravel_pytree(x)
        x_dtype = x_flat.dtype

        @functools.partial(jax.vmap, in_axes=1, out_axes=1)
        def A(y: Array) -> Array:
            _y = f_unravel(y)
            _x = op.apply(op.adjoint(_y))
            x, _ = jax.flatten_util.ravel_pytree(_x)
            return x

        X = jnd.normal(jnd.key(0), (op.codim_size, k), x_dtype)
    else:
        y = jax.tree.map(lambda _: jnp.zeros(_.shape, _.dtype), op.dim_info)
        y_flat, f_unravel = jax.flatten_util.ravel_pytree(y)
        y_dtype = y_flat.dtype

        @functools.partial(jax.vmap, in_axes=1, out_axes=1)
        def A(x: Array) -> Array:
            _x = f_unravel(x)
            _y = op.adjoint(op.apply(_x))
            y, _ = jax.flatten_util.ravel_pytree(_y)
            return y

        X = jnd.normal(jnd.key(0), (op.dim_size, k), y_dtype)

    D_eig, V_eig, N_iter = jesl.lobpcg_standard(A, X, **kwargs)
    D = jnp.sqrt(D_eig)
    return D


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
        t Pseudo-inverse solution :math:`\hat{\bbx} \in \cI`.
    diagnostics: lineax.Solution
        CG diagnostic information.
        Can be queried to learn more about the solution.
    """
    assert op.dim_info is not None

    lx_op = lx.FunctionLinearOperator(
        fn=lambda x: otu.tree_add_scalar_mul(
            op.adjoint(op.apply(x)),
            tau,
            x,
        ),
        input_structure=op.dim_info,
        tags=lx.positive_semidefinite_tag,
    )
    solver = lx.CG(**kwargs)
    solution = lx.linear_solve(lx_op, op.adjoint(y), solver)

    x = solution.value
    return x, solution
