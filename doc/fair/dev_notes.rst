Developer Notes
===============

API Rules
---------

To implement novel operators or algorithms, users must subclass an abstract base class from Pyxu's
:py:class:`~pyxu.abc.Operator` hierarchy. Doing so requires defining the fundamental methods attached to the subclass
(e.g. :py:meth:`~pyxu.abc.Map.apply`, :py:meth:`~pyxu.abc.DiffFunc.grad`, :py:meth:`~pyxu.abc.ProxFunc.prox`,
:py:meth:`~pyxu.abc.LinOp.adjoint`, ...). When marked as such in the documentation, such a user-defined method should
abide by the following set of rules:

- It must handle properly the case where the input array is a 1-D or N-D array. In the latter case, the  first N-1
  dimensions of the input array should be considered as stacking dimensions, i.e. the method is applied along the last
  axis.

- In the case of N-D inputs, the output should have the same number of dimensions as the input.

- It should control the numerical precision (e.g. *single*,
  *double*) of the inputs/outputs. If possible, the computation performed by the method itself should also be carried
  out at the input array's precision.

- Whenever possible, it should be compatible with the array modules supported by Pyxu. (Use
  :py:func:`~pyxu.info.deps.supported_array_modules` for an up-to-date list).  :py:func:`~pyxu.util.get_array_module`
  can be used to write `module-agnostic
  <https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code>`_ code easily.


Example of a rule-abiding operator
----------------------------------

As an example, consider the following code snippet, defining the median operator:

.. code-block:: python3

   import pyxu.abc as pxa
   import pyxu.runtime as pxrt
   import pyxu.util as pxu

   class Median(pxa.Map):
       def __init__(self, dim_shape: tuple):
           super().__init__(dim_shape=dim_shape, codim_shape=1)

       def apply(self, arr):
           xp = pxu.get_array_module(arr)  # find array module of `arr`.
           axis = tuple(range(-len(self.dim_shape), 0)) # apply median to core dimensions
           return xp.median(arr, axis=axis, keepdims=False) # apply is vectorized to batch dimensions

This operator can then be fed various arrays as inputs:

.. code-block:: python3

   import pyxu.info.deps as pxd

   N = () # batch size
   dim_shape = (4, 3)
   op = Median(dim_shape)
   for xp in pxd.supported_array_modules():
       for width in pxrt.Width:
           arr = xp.random.normal(size=(N + dim_shape)).astype(width.value)
           out = op.apply(arr)  # apply the operator to various array types.
