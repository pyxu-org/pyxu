API Reference
=============

pyxu.info.ptype
---------------

.. automodule:: pyxu.info.ptype

pyxu.info.deps
--------------

.. automodule:: pyxu.info.deps
   :members:
   :undoc-members:
   :show-inheritance:

pyxu.info.warning
-----------------

.. automodule:: pyxu.info.warning
   :no-special-members:
   :exclude-members: add_note, args, with_traceback

pyxu.abc
--------

.. The ABC module contains too many corner cases for autodocs' automodule construct.
.. We therefore choose explicitly what to show.

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

.. automodule:: pyxu.abc.solver

   .. autoclass:: pyxu.abc.solver.Solver
      :noindex:

      .. autoattribute:: _mstate

      .. autoattribute:: _astate

pyxu.math
---------

.. automodule:: pyxu.math.linalg

.. automodule:: pyxu.math.linesearch

pyxu.runtime
------------

.. automodule:: pyxu.runtime

pyxu.util
---------

.. automodule:: pyxu.util.array_module

.. automodule:: pyxu.util.complex

.. automodule:: pyxu.util.inspect

.. automodule:: pyxu.util.misc

   .. Force documentation of some non-public functions.

   .. autofunction:: peaks

   .. autofunction:: star_like_sample

.. automodule:: pyxu.util.operator

   .. Force documentation of some non-public functions.

   .. autofunction:: _dask_zip

   .. autofunction:: _array_ize

pyxu.opt
--------

.. autoclass:: pyxu.opt.stop.MaxIter
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.ManualStop
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.MaxDuration
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.MaxCarbon
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.Memorize
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.AbsError
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.RelError
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.solver.pgd.PGD
   :no-members:

.. autoclass:: pyxu.opt.solver.cg.CG
   :no-members:

.. autoclass:: pyxu.opt.solver.nlcg.NLCG
   :no-members:

.. autoclass:: pyxu.opt.solver.prox_adam.ProxAdam
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.CondatVu
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.CV
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.PD3O
   :no-members:

.. autofunction:: pyxu.opt.solver.pds.ChambollePock

.. autofunction:: pyxu.opt.solver.pds.CP

.. autoclass:: pyxu.opt.solver.pds.LorisVerhoeven
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.LV
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.DavisYin
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.DY
   :no-members:

.. autofunction:: pyxu.opt.solver.pds.DouglasRachford

.. autofunction:: pyxu.opt.solver.pds.DR

.. autoclass:: pyxu.opt.solver.pds.ForwardBackward
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.FB
   :no-members:

.. autofunction:: pyxu.opt.solver.pds.ProximalPoint

.. autofunction:: pyxu.opt.solver.pds.PP

.. autoclass:: pyxu.opt.solver.pds.ADMM
   :no-members:
