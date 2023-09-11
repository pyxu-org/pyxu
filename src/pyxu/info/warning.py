# Custom warnings used inside Pyxu.


class PyxuWarning(UserWarning):
    """
    Parent class of all warnings raised in Pyxu.
    """

    pass


class AutoInferenceWarning(PyxuWarning):
    """
    Use when a quantity was auto-inferenced with possible caveats.
    """

    pass


class PerformanceWarning(PyxuWarning):
    """
    Use for performance-related warnings.
    """


class PrecisionWarning(PyxuWarning):
    """
    Use for precision-related warnings.
    """


class DenseWarning(PyxuWarning):
    """
    Use for sparse-based algos which revert to dense arrays.
    """


class NonTransparentWarning(PyxuWarning):
    """
    Inform test suite runner of (safe) non-transparent function call.
    """


class BackendWarning(PyxuWarning):
    """
    Inform user of a backend-specific problem to be aware of.
    """


class ContributionWarning(PyxuWarning):
    """
    Use for warnings related to Pyxu plugins.
    """
