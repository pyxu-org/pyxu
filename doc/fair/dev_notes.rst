Developer Notes
===============

API Rules
---------

To implement novel operators or algorithms, users must subclass an abstract base class from Pyxu's
:py:class:`~pyxu.abc.operator.Operator` hierarchy. Doing so requires defining the fundamental methods attached to the
subclass (e.g. :py:meth:`~pyxu.abc.operator.Map.apply`, :py:meth:`~pyxu.abc.operator.DiffFunc.grad`,
:py:meth:`~pyxu.abc.operator.ProxFunc.prox`, :py:meth:`~pyxu.abc.operator.LinOp.adjoint`, ...). When marked as such in
the documentation, such a user-defined method should abide by the following set of rules:

- It must handle properly the case where the input array is a 1-D or N-D array. In the latter case, the  first N-1
  dimensions of the input array should be considered as stacking dimensions, i.e. the method is applied along the last
  axis.

- In the case of N-D inputs, the output should have the same number of dimensions as the input.

- It should be decorated with :py:func:`~pyxu.runtime.enforce_precision`. Together with the
  :py:class:`~pyxu.runtime.Precision` context manager, the former controls the numerical precision (e.g. *single*,
  *double*) of the inputs/outputs. If possible, the computation performed by the method itself should also be carried
  out at the user-specified precision, accessible via :py:func:`~pyxu.runtime.getPrecision`.

- Whenever possible, it should be compatible with the array modules supported by Pyxu. (Use
  :py:func:`~pyxu.info.deps.supported_array_modules` for an up-to-date list).
  :py:func:`~pyxu.util.array_module.get_array_module` can be used to write `module-agnostic
  <https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code>`_ code easily.


Example of a rule-abiding operator
----------------------------------

As an example, consider the following code snippet, defining the median operator:

.. code-block:: python3

   import pyxu.abc as pxa
   import pyxu.runtime as pxrt
   import pyxu.util as pxu

   class Median(pxa.Map):
       def __init__(self, dim: int):
           super().__init__(shape=(1, dim))

       @pxrt.enforce_precision(i="arr")  # enforce input/output precision.
       def apply(self, arr):
           xp = pxu.get_array_module(arr)  # find array module of `arr`.
           return xp.median(arr, axis=-1, keepdims=True)  # median() is applied to the last axis.

This operator can then be fed various arrays as inputs:

.. code-block:: python3

   import pyxu.info.deps as pxd

   N = 5
   op = Median(N)
   for xp in pxd.supported_array_modules():
       out = op.apply(xp.arange(2*N).reshape(2, N))  # apply the operator to various array types.

If called from within the :py:class:`~pyxu.runtime.Precision` context manager, the decorated ``apply()`` method will
automatically *coerce* the input/output to the user-specified precision:

.. code-block:: python3

   with pxrt.Precision(pxrt.Width.SINGLE):
       out = op.apply(np.arange(N))  # float32 computation
