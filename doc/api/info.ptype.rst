pyxu.info.ptype
===============

.. [Sepand, 2023.09.15]
.. `pyxu.info.ptype` contains alias types for Python type checkers.  When auto-documented via Sphinx, alias types
.. contain "alias of ..." as part of their docstring, which depends on the machine config.  (Was unable to remove them
.. via Sphinx's `autodoc-process-docstring` event.) To avoid machine-dependant outputs in HTML docs, we provide custom
.. machine-agnostic documentation for some aliases as required. Provided descriptions match those given as comments in
.. `pyxu.info.ptype`.


.. automodule:: pyxu.info.ptype
   :exclude-members: NDArray, ArrayModule, SparseArray, SparseModule, OpT, Property, SolverT, SolverM, Real, DType

   .. class:: NDArray

      Supported dense array types.

   .. class:: ArrayModule

      Supported dense array modules.

   .. class:: SparseArray

      Supported sparse array types.

   .. class:: SparseModule

      Supported sparse array modules.

   .. class:: OpT

      Top-level abstract :py:class:`~pyxu.abc.Operator` interface exposed to users.

   .. autodata:: Property
      :no-value:

   .. class:: SolverT

      Top-level abstract :py:class:`~pyxu.abc.Solver` interface exposed to users.

   .. class:: SolverM

      Solver run-mode(s).

   .. class:: Real

      Alias of :py:class:`numbers.Real`.

   .. data:: DType

      :py:attr:`~pyxu.info.ptype.NDArray` dtype specifier.
