.. _developer-notes:

Developer Notes
===============

API Rules
---------

To implement novel operators or algorithms, the user should subclass one of the abstract classes in
Pycsou's class hierarchy. Doing so requires defining the fundamental methods attached to the chosen
subclass (e.g. :py:meth:`~pycsou.abc.operator.Apply.apply`,
:py:meth:`~pycsou.abc.operator.Gradient.gradient`, :py:meth:`~pycsou.abc.operator.Proximal.prox` or
:py:meth:`~pycsou.abc.operator.Adjoint.adjoint`...). When marked as such in the documentation, such
a user-defined method should abide by the following set of rules:

- It must handle properly the case where the input array is a 1-D or N-D array. In the latter case,
  the  first N-1 dimensions of the input array should be considered as stacking dimensions, i.e.
  the method is applied along the last axis.

- In the case of N-D inputs, the output should have the same number of dimensions as the input.

- It should be decorated with  :py:func:`~pycsou.runtime.enforce_precision`, which (together with
  the context manager :py:class:`~pycsou.runtime.Precision`) controls the numerical precision (e.g.
  *half*, *single*, *double*, *quad*) of the inputs/outputs. If possible, the computation performed
  by the method itself should also be carried out at the user-specified precision, accessible via
  the function :py:func:`~pycsou.runtime.getPrecision`. If preserving the precision is not
  possible, a warning should be raised.

- Whenever possible, it should be compatible with the array modules supported by Pycsou (use
  :py:func:`~pycsou.util.deps.supported_array_modules` for an up-to-date list). This can be
  achieved via the function :py:func:`~pycsou.util.get_array_module` which allows to write
  `module-agnostic
  <https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code>`_ code
  easily.


Example of a rule-abiding operator
----------------------------------

As an example, consider the following code snippet, defining the median operator:

.. code-block:: python3

    import pycsou.abc as pyca
    import pycsou.runtime as pycrt
    import pycsou.util as pycu

    class Median(pyca.Map):
        def __init__(self):
            super(Median, self).__init__(shape=(1, None))  # The Median operator is domain-agnostic.

        @pycrt.enforce_precision(i='arr', o=True)  # Enforce input/output precision.
        def apply(self, arr: NDArray) -> NDArray:
            xp = pycu.get_array_module(arr)  # Find the array module for the provided input.
            return xp.median(arr, axis=-1, keepdims=True)  # The median function is applied to the last axis.
                                                           # The returned array has the same dimensions as the input thanks to the keyword keepdims=True.

This operator can then be fed various arrays as inputs:

>>> import pycsou.util.deps as pycd
>>> m = Median()
>>> for xp in pycd.supported_array_modules():
...     out = m.apply(xp.arange(26).reshape(2, 13))  # Applies the operator in turn on a various array types.

If called from within the context manager :py:class:`~pycsou.runtime.Precision`, the decorated
apply method will automatically coerce the input/output to the user-specified precision:

>>> with pycu.Precision(pycu.Width.SINGLE):
...     out = m.apply(np.arange(26).reshape(2, 13))  # Single precision is used for the computation


Common pitfalls and performance issues
--------------------------------------

In the previous example, the :py:func:`~numpy.median` function works at the precision of the input
array. Therefore, the argument ``o=True`` in the decorator
:py:func:`~pycsou.runtime.enforce_precision` is not needed since the output is already guaranteed
to be at the user-specified precision. This is however, not always the case, as illustrated by the
following example:

.. code-block:: python3

    @pycrt.enforce_precision(i='arr', o=True)  # Enforce input/output precision.
    def f(self, arr: NDArray) -> NDArray:
        return np.random.poisson(arr)

Without the argument ``o=True`` in the decorator :py:func:`~pycsou.runtime.enforce_precision`, the
:py:func:`f` function above would output an array with dtype ``int32`` or ``int64`` (which are the
default dtypes for the function :py:func:`~numpy.random.poisson`). Thanks to the decorator however,
the output array is coerced as a float with prescribed precision.

In certain cases moreover, the inner computation may force a recast of the input array dtype. In
which case a warning should be raised. This is typically the case for the following example:

.. code-block:: python3

    import warnings

    class Matrix(pyca.LinOp):
        def __init__(self, mat: NDArray):
            assert mat.ndim == 2
            super(Matrix, self).__init__(shape=mat.shape)
            self.mat = mat

        @pycrt.enforce_precision(i='arr', o=True)  # Enforce input/output precision.
        def apply(self, arr: NDArray) -> NDArray:
            xp = pycu.get_array_module(arr)  # Find the array module for the provided input.
            assert xp is pycu.get_array_module(self.mat)  # Raise an error if self.mat and arr belong to different array modules.
            if self.mat.dtype != pycrt.getPrecision():
                warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
            return xp.matmul(self.mat, arr[...,None]).squeeze()  # matmul may cast arr to the dtype of self.mat

Observe that if the argument ``mat`` provided to the constructor of the ``Matrix`` class had a
dtype ``np.float64`` and the precision was set by the user to single precision (i.e.
``np.float32``), then the computation would not be performed at the correct precision. Indeed, the
:py:func:`~numpy.matmul` function invoked on the last line would automatically recast ``arr`` to
``np.float64`` before performing the matrix product. Thanks to the
:py:func:`~pycsou.runtime.enforce_precision` decorator, the output to the :py:meth:`apply` method
would still have single precision but the computation itself would not have been performed at such
precision (and would hence be slower than expected).

Note that a potential fix to ensure a computation at the requested precision in the example above
could have been to dynamically change the dtype of ``self.mat``, i.e. replacing the return
statement

.. code-block:: python3

    return xp.matmul(self.mat, arr[...,None]).squeeze()

by

.. code-block:: python3

    return xp.matmul(self.mat.astype(pycrt.getPrecision()), arr[...,None]).squeeze()

This recasting may however be memory-intensive to perform for very large arrays, and should not be
done without the explicit consent/knowledge of the user.
