from ..abc import LinearOperator
from ..util.typing import Arrays

import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax


def svdvals(op: LinearOperator, k: int, **kwargs) -> jt.Scalar:
    """
    Compute leading singular values of a linear operator.

    The computation is based on the power method.

    Parameters
    ----------
    op: LinearOperator
        Operator for which singular values are desired.
    k: int
        Number of singular values to compute.

    Returns
    -------
    D: Array
        (k,) leading singular values of ``op``.
    """
    # TODO: upgrade to LOBPCG method; converges faster & computes (k >= 1) singular values.
    # Needs flattening/etc to be performed since jax.experimental.sparse.linalg.lobpcg() accepts a callable with a single array in/out.

    assert k == 1, "Only leading singular value can be computed (for now)."

    assert (op.dim_shape is not None) and (op.codim_shape is not None), (
        "in/out shapes must be known."
    )

    if op.codim_size <= op.dim_size:

        def f(y: Arrays) -> Arrays:
            return op.apply(op.adjoint(y))

        v0 = jax.tree.map(lambda sh: jnp.ones(sh.shape), op.codim_shape)
    else:

        def f(x: Arrays) -> Arrays:
            return op.adjoint(op.apply(x))

        v0 = jax.tree.map(lambda sh: jnp.ones(sh.shape), op.dim_shape)

    if "v0" in kwargs:
        pass
    else:
        kwargs.update(v0=v0)

    D_eig, _ = optax.power_iteration(f, **kwargs)

    D = jnp.sqrt(D_eig)
    return D
