"""
Custom warnings used inside Pycsou.
"""


class PrecisionWarning(UserWarning):
    """
    Use for precision-related warnings.
    """

    pass


class DenseWarning(UserWarning):
    """
    Use for sparse-based algos which revert to dense arrays.
    """
