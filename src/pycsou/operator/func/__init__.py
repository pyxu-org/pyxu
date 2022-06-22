import pycsou.util.ptype as pyct


def shift_loss(
    op: pyct.MapT,
    data: pyct.NDArray = None,
) -> pyct.MapT:
    """
    Shift a functional into a loss functional.

    Parameters
    ----------
    data: pyct.NDArray
        (N,) input data.

    Returns
    -------
    op: pyct.MapT
        Loss functionial.
        If `data = None`, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)
