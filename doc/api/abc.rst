pyxu.abc
========

.. contents:: Table of Contents
   :local:
   :depth: 2

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

.. autoclass:: pyxu.abc.arithmetic.TransposeRule
   :no-members:

pyxu.abc.operator
-----------------

.. autoclass:: pyxu.abc.Property

.. autoclass:: pyxu.abc.Operator
   :special-members: __add__, __sub__, __neg__, __mul__, __pow__

.. autoclass:: pyxu.abc.Map
   :special-members: __call__
   :members: apply, lipschitz, estimate_lipschitz

.. autoclass:: pyxu.abc.Func
   :no-members:

.. autoclass:: pyxu.abc.DiffMap
   :no-members:
   :members: jacobian, diff_lipschitz, estimate_diff_lipschitz

.. autoclass:: pyxu.abc.DiffFunc
   :no-members:
   :members: grad

.. autoclass:: pyxu.abc.ProxFunc
   :no-members:
   :members: prox, fenchel_prox, moreau_envelope

.. autoclass:: pyxu.abc.ProxDiffFunc
   :no-members:

.. autoclass:: pyxu.abc.QuadraticFunc
   :no-members:
   :special-members: __init__
   :private-members: _quad_spec

.. autoclass:: pyxu.abc.LinOp
   :no-members:
   :members: adjoint, T, to_sciop, estimate_lipschitz, svdvals, asarray, gram, cogram, pinv, dagger, from_array

.. autoclass:: pyxu.abc.SquareOp
   :no-members:
   :members: trace

.. autoclass:: pyxu.abc.NormalOp
   :no-members:

.. autoclass:: pyxu.abc.SelfAdjointOp
   :no-members:

.. autoclass:: pyxu.abc.UnitOp
   :no-members:

.. autoclass:: pyxu.abc.ProjOp
   :no-members:

.. autoclass:: pyxu.abc.OrthProjOp
   :no-members:

.. autoclass:: pyxu.abc.PosDefOp
   :no-members:

.. autoclass:: pyxu.abc.LinFunc
   :no-members:

pyxu.abc.solver
---------------

.. autoclass:: pyxu.abc.Solver

   .. autoattribute:: _mstate

   .. autoattribute:: _astate

.. autoclass:: pyxu.abc.SolverMode

.. autoclass:: pyxu.abc.StoppingCriterion
