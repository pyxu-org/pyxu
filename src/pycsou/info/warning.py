"""
Custom warnings used inside Pycsou.
"""


class PycsouWarning(UserWarning):
    """
    Parent class of all warnings raised in Pycsou.
    """

    pass


class AutoInferenceWarning(PycsouWarning):
    """
    Use when a quantity was auto-inferenced with possible caveats.
    """

    pass


class PerformanceWarning(PycsouWarning):
    """
    Use for performance-related warnings.
    """


class PrecisionWarning(PycsouWarning):
    """
    Use for precision-related warnings.
    """


class DenseWarning(PycsouWarning):
    """
    Use for sparse-based algos which revert to dense arrays.
    """


class NonTransparentWarning(PycsouWarning):
    """
    [Dev-team only] Inform test suite runner of (safe) non-transparent function call.
    """


class BackendWarning(PycsouWarning):
    """
    Inform user of a backend-specific problem to be aware of.
    """
