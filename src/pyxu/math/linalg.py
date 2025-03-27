import opt_einsum as oe

from ..typing import Array


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
