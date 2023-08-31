API Reference
=============

Welcome to the official API reference for Pyxu, your go-to library for cutting-edge scientific computing in Python. This API documentation is intended to serve as a comprehensive guide to the library's various modules, classes, functions, and interfaces, providing you with detailed descriptions of each component's role, relations, assumptions, and behavior. Please note that this API reference is not designed to be a tutorial; it's a technical resource aimed at users who are already familiar with the library's basics and wish to dive deeper into its functionalities. Whether you are troubleshooting, optimizing, or extending your Pyxu-based applications, this API reference is an invaluable tool for understanding the intricacies of the library.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. The goal of this page is to provide an alphabetical listing of all Pyxu objects exposed to users.  This is achieved
.. via the `autosummary` extension.  While `autosummary` understands `automodule`-documented packages, explicitly
.. listing a module's contents is  required.

Subpackages
-----------

pyxu.abc
^^^^^^^^
.. autosummary::

   ~pyxu.abc.operator.Property
   ~pyxu.abc.operator.Operator
   ~pyxu.abc.operator.Map
   ~pyxu.abc.operator.Func
   ~pyxu.abc.operator.DiffMap
   ~pyxu.abc.operator.DiffFunc
   ~pyxu.abc.operator.ProxFunc
   ~pyxu.abc.operator.ProxDiffFunc
   ~pyxu.abc.operator.QuadraticFunc
   ~pyxu.abc.operator.LinOp
   ~pyxu.abc.operator.SquareOp
   ~pyxu.abc.operator.NormalOp
   ~pyxu.abc.operator.SelfAdjointOp
   ~pyxu.abc.operator.UnitOp
   ~pyxu.abc.operator.ProjOp
   ~pyxu.abc.operator.OrthProjOp
   ~pyxu.abc.operator.PosDefOp
   ~pyxu.abc.operator.LinFunc
   ~pyxu.abc.arithmetic.Rule
   ~pyxu.abc.arithmetic.ScaleRule
   ~pyxu.abc.arithmetic.ArgScaleRule
   ~pyxu.abc.arithmetic.ArgShiftRule
   ~pyxu.abc.arithmetic.AddRule
   ~pyxu.abc.arithmetic.ChainRule
   ~pyxu.abc.arithmetic.PowerRule
   ~pyxu.abc.arithmetic.TransposeRule
   ~pyxu.abc.solver.Mode
   ~pyxu.abc.solver.StoppingCriterion
   ~pyxu.abc.solver.Solver

pyxu.info.ptype
^^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.info.ptype.NDArray
   ~pyxu.info.ptype.ArrayModule
   ~pyxu.info.ptype.SparseArray
   ~pyxu.info.ptype.SparseModule
   ~pyxu.info.ptype.OpT
   ~pyxu.info.ptype.OpC
   ~pyxu.info.ptype.Property
   ~pyxu.info.ptype.SolverT
   ~pyxu.info.ptype.SolverC
   ~pyxu.info.ptype.SolverM
   ~pyxu.info.ptype.Integer
   ~pyxu.info.ptype.Real
   ~pyxu.info.ptype.DType
   ~pyxu.info.ptype.OpShape
   ~pyxu.info.ptype.NDArrayAxis
   ~pyxu.info.ptype.NDArrayShape
   ~pyxu.info.ptype.Path
   ~pyxu.info.ptype.VarName

pyxu.info.deps
^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.info.deps.CUPY_ENABLED
   ~pyxu.info.deps.NDArrayInfo
   ~pyxu.info.deps.SparseArrayInfo
   ~pyxu.info.deps.supported_array_types
   ~pyxu.info.deps.supported_array_modules
   ~pyxu.info.deps.supported_sparse_types
   ~pyxu.info.deps.supported_sparse_modules

pyxu.info.warning
^^^^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.info.warning.PyxuWarning
   ~pyxu.info.warning.AutoInferenceWarning
   ~pyxu.info.warning.PerformanceWarning
   ~pyxu.info.warning.PrecisionWarning
   ~pyxu.info.warning.DenseWarning
   ~pyxu.info.warning.NonTransparentWarning
   ~pyxu.info.warning.BackendWarning

pyxu.experimental
^^^^^^^^^^^^^^^^^
.. rubric:: sampler

.. autosummary::

   ~pyxu.experimental.sampler.statistics.OnlineMoment
   ~pyxu.experimental.sampler.statistics.OnlineCenteredMoment
   ~pyxu.experimental.sampler.statistics.OnlineVariance
   ~pyxu.experimental.sampler.statistics.OnlineStd
   ~pyxu.experimental.sampler.statistics.OnlineSkewness
   ~pyxu.experimental.sampler.statistics.OnlineKurtosis
   ~pyxu.experimental.sampler._sampler.ULA
   ~pyxu.experimental.sampler._sampler.MYULA

pyxu.math
^^^^^^^^^
.. autosummary::

   ~pyxu.math.linalg.norm
   ~pyxu.math.linalg.trace
   ~pyxu.math.linalg.hutchpp
   ~pyxu.math.linesearch.backtracking_linesearch



.. toctree::
   :maxdepth: 2
   :hidden:

   abc
   opt
   operator/index
   runtime
   util
   math
   info
   experimental
   
