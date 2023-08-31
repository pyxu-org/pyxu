pyxu.abc
========

.. The ABC module contains too many corner cases for autodocs' automodule construct.
.. We therefore choose explicitly what to show.

.. contents:: Table of Contents
   :local:
   :depth: 2

pyxu.abc.operator
-----------------

.. autoclass:: pyxu.abc.operator.Property

.. autoclass:: pyxu.abc.operator.Operator
   :special-members: __add__, __sub__, __neg__, __mul__, __pow__

.. autoclass:: pyxu.abc.operator.Map
   :special-members: __call__
   :members: apply, lipschitz, estimate_lipschitz

.. autoclass:: pyxu.abc.operator.Func
   :no-members:
   :members: asloss

.. autoclass:: pyxu.abc.operator.DiffMap
   :no-members:
   :members: jacobian, diff_lipschitz, estimate_diff_lipschitz

.. autoclass:: pyxu.abc.operator.DiffFunc
   :no-members:
   :members: grad

.. autoclass:: pyxu.abc.operator.ProxFunc
   :no-members:
   :members: prox, fenchel_prox, moreau_envelope

.. autoclass:: pyxu.abc.operator.ProxDiffFunc
   :no-members:

.. autoclass:: pyxu.abc.operator.QuadraticFunc
   :no-members:
   :special-members: __init__
   :private-members: _quad_spec

.. autoclass:: pyxu.abc.operator.LinOp
   :no-members:
   :members: adjoint, T, to_sciop, estimate_lipschitz, svdvals, asarray, gram, cogram, pinv, dagger, from_array

.. autoclass:: pyxu.abc.operator.SquareOp
   :no-members:
   :members: trace

.. autoclass:: pyxu.abc.operator.NormalOp
   :no-members:

.. autoclass:: pyxu.abc.operator.SelfAdjointOp
   :no-members:

.. autoclass:: pyxu.abc.operator.UnitOp
   :no-members:

.. autoclass:: pyxu.abc.operator.ProjOp
   :no-members:

.. autoclass:: pyxu.abc.operator.OrthProjOp
   :no-members:

.. autoclass:: pyxu.abc.operator.PosDefOp
   :no-members:

.. autoclass:: pyxu.abc.operator.LinFunc
   :no-members:

pyxu.abc.arithmetic
-------------------

.. autoclass:: pyxu.abc.arithmetic.Rule
   :members: op
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.ScaleRule
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.ArgScaleRule
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.ArgShiftRule
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.AddRule
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.ChainRule
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.PowerRule
   :no-members:

.. autoclass:: pyxu.abc.arithmetic.TransposeRule
   :no-members:

pyxu.abc.solver
---------------

.. automodule:: pyxu.abc.solver

   .. autoclass:: pyxu.abc.solver.Solver
      :noindex:

      .. autoattribute:: _mstate

      .. autoattribute:: _astate
