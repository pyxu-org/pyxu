import pyxu.info.ptype as pxt

__all__ = [
    "shift_loss",
]


def shift_loss(
    op: pxt.OpT,
    data: pxt.NDArray = None,
) -> pxt.OpT:
    r"""
    Shift a functional :math:`f(x)` to a loss functional :math:`g(x) = f(x - c)`.

    Parameters
    ----------
    data: pxt.NDArray
        (M,) input data.

    Returns
    -------
    op: pxt.OpT
        (1, M) Loss functionial.
        If `data` is omitted, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)
