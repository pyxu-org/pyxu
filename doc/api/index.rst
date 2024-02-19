API Reference
=============

.. The goal of this page is to provide an alphabetical listing of all Pyxu objects exposed to users.  This is achieved
.. via the `autosummary` extension.  While `autosummary` understands `automodule`-documented packages, explicitly
.. listing a module's contents is required.

Welcome to the official API reference for Pyxu.  This API documentation is intended to serve as a comprehensive guide to
the library's various modules, classes, functions, and interfaces.  It provides detailed descriptions of each
component's role, relations, assumptions, and behavior.

Please note that this API reference is not designed to be a tutorial; it's a technical resource aimed at users who are
already familiar with the library's basics and wish to dive deeper into its functionalities.

Pyxu is broken down into the following top-level namespaces:

* ``pyxu.abc``: abstract base types and logic used throughout Pyxu.
* ``pyxu.info.deps``: array backend tools.
* ``pyxu.info.ptype``: type aliases for Python type checkers.
* ``pyxu.info.warning``: internal warnings.
* ``pyxu.operator``: operator collection.
* ``pyxu.operator.interop``: helpers to interface with external packages such as JAX, PyTorch, etc.
* ``pyxu.opt.solver``: solver collection.
* ``pyxu.opt.stop``: common stopping criteria.
* ``pyxu.runtime``: compute precision tools.
* ``pyxu.math``: math helpers.
* ``pyxu.util``: utility functions.
* ``pyxu.experimental``: experimental packages. These may change in the future without warning. Each sub-module under
  ``experimental`` must be imported individually:

  .. code-block:: python3

     from pyxu.experimental.sampler import ULA


Individual Pyxu components should be imported from these top-level modules. Some low-level routines are not exposed from
the former and must be imported explicitly:

.. code-block:: python3

   from pyxu.operator import Sum  # top-level import
   from pyxu.util.misc import peaks  # low-level import

The import path of each object can be inferred by looking at its canonical path in the alphabetical listings below.

.. contents:: Table of Contents
   :local:
   :depth: 2

pyxu.abc
--------

Operator-related
^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.abc.DiffFunc
   ~pyxu.abc.DiffMap
   ~pyxu.abc.Func
   ~pyxu.abc.LinFunc
   ~pyxu.abc.LinOp
   ~pyxu.abc.Map
   ~pyxu.abc.NormalOp
   ~pyxu.abc.Operator
   ~pyxu.abc.OrthProjOp
   ~pyxu.abc.PosDefOp
   ~pyxu.abc.ProjOp
   ~pyxu.abc.Property
   ~pyxu.abc.ProxDiffFunc
   ~pyxu.abc.ProxFunc
   ~pyxu.abc.QuadraticFunc
   ~pyxu.abc.SelfAdjointOp
   ~pyxu.abc.SquareOp
   ~pyxu.abc.UnitOp

Solver-related
^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.abc.Solver
   ~pyxu.abc.SolverMode
   ~pyxu.abc.StoppingCriterion

Arithmetic Rules (low-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   pyxu.abc.arithmetic.AddRule
   pyxu.abc.arithmetic.ArgScaleRule
   pyxu.abc.arithmetic.ArgShiftRule
   pyxu.abc.arithmetic.ChainRule
   pyxu.abc.arithmetic.Rule
   pyxu.abc.arithmetic.ScaleRule
   pyxu.abc.arithmetic.TransposeRule

pyxu.experimental
-----------------

Sampling Tools
^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.experimental.sampler.MYULA
   ~pyxu.experimental.sampler.OnlineCenteredMoment
   ~pyxu.experimental.sampler.OnlineKurtosis
   ~pyxu.experimental.sampler.OnlineMoment
   ~pyxu.experimental.sampler.OnlineSkewness
   ~pyxu.experimental.sampler.OnlineStd
   ~pyxu.experimental.sampler.OnlineVariance
   ~pyxu.experimental.sampler.ULA

Transforms
^^^^^^^^^^

.. autosummary::

   ~pyxu.experimental.xray.XRayTransform
   ~pyxu.experimental.xray._rt.RayXRT
   ~pyxu.experimental.xray._fourier.FourierXRT

pyxu.info.deps
--------------

.. autosummary::

   ~pyxu.info.deps.CUPY_ENABLED
   ~pyxu.info.deps.NDArrayInfo
   ~pyxu.info.deps.SparseArrayInfo
   ~pyxu.info.deps.supported_array_modules
   ~pyxu.info.deps.supported_array_types
   ~pyxu.info.deps.supported_sparse_modules
   ~pyxu.info.deps.supported_sparse_types

pyxu.info.ptype
---------------

.. autosummary::

   ~pyxu.info.ptype.ArrayModule
   ~pyxu.info.ptype.DType
   ~pyxu.info.ptype.Integer
   ~pyxu.info.ptype.NDArray
   ~pyxu.info.ptype.NDArrayAxis
   ~pyxu.info.ptype.NDArrayShape
   ~pyxu.info.ptype.OpC
   ~pyxu.info.ptype.OpT
   ~pyxu.info.ptype.Path
   ~pyxu.info.ptype.Property
   ~pyxu.info.ptype.Real
   ~pyxu.info.ptype.SolverC
   ~pyxu.info.ptype.SolverM
   ~pyxu.info.ptype.SolverT
   ~pyxu.info.ptype.SparseArray
   ~pyxu.info.ptype.SparseModule
   ~pyxu.info.ptype.VarName

pyxu.info.warning
-----------------

.. autosummary::

   ~pyxu.info.warning.AutoInferenceWarning
   ~pyxu.info.warning.BackendWarning
   ~pyxu.info.warning.ContributionWarning
   ~pyxu.info.warning.DenseWarning
   ~pyxu.info.warning.NonTransparentWarning
   ~pyxu.info.warning.PerformanceWarning
   ~pyxu.info.warning.PrecisionWarning
   ~pyxu.info.warning.PyxuWarning

pyxu.math
---------

.. autosummary::

   ~pyxu.math.backtracking_linesearch
   ~pyxu.math.hutchpp
   ~pyxu.math.trace

pyxu.operator.interop
---------------------

General
^^^^^^^

.. autosummary::

   ~pyxu.operator.interop.from_source

SciPy
^^^^^

.. autosummary::

   ~pyxu.operator.interop.from_sciop

JAX
^^^

.. autosummary::

   ~pyxu.operator.interop.jax._from_jax
   ~pyxu.operator.interop.jax._to_jax
   ~pyxu.operator.interop.from_jax

PyTorch
^^^^^^^

.. autosummary::

   ~pyxu.operator.interop.torch._from_torch
   ~pyxu.operator.interop.torch._to_torch
   ~pyxu.operator.interop.from_torch

pyxu.operator
-------------

Functionals
^^^^^^^^^^^

Norms & Loss Functions
++++++++++++++++++++++

.. autosummary::

   ~pyxu.operator.KLDivergence
   ~pyxu.operator.L1Norm
   ~pyxu.operator.L21Norm
   ~pyxu.operator.L2Norm
   ~pyxu.operator.LInfinityNorm
   ~pyxu.operator.PositiveL1Norm
   ~pyxu.operator.SquaredL1Norm
   ~pyxu.operator.SquaredL2Norm

Indicator Functions
+++++++++++++++++++

.. autosummary::

   ~pyxu.operator.HyperSlab
   ~pyxu.operator.L1Ball
   ~pyxu.operator.L2Ball
   ~pyxu.operator.LInfinityBall
   ~pyxu.operator.PositiveOrthant
   ~pyxu.operator.RangeSet

Linear Operators
^^^^^^^^^^^^^^^^

Basic Operators
+++++++++++++++

.. autosummary::

   ~pyxu.operator.DiagonalOp
   ~pyxu.operator.HomothetyOp
   ~pyxu.operator.IdentityOp
   ~pyxu.operator.NullFunc
   ~pyxu.operator.NullOp
   ~pyxu.operator.Pad
   ~pyxu.operator.SubSample
   ~pyxu.operator.Sum
   ~pyxu.operator.Trim

Transforms
++++++++++

.. autosummary::

   ~pyxu.operator.FFT
   ~pyxu.operator.CZT
   ~pyxu.operator.NUFFT1
   ~pyxu.operator.NUFFT2

Stencils & Convolutions
+++++++++++++++++++++++

.. autosummary::

   ~pyxu.operator.Convolve
   ~pyxu.operator.Correlate
   ~pyxu.operator.FFTConvolve
   ~pyxu.operator.FFTCorrelate
   ~pyxu.operator.Stencil
   ~pyxu.operator._Stencil
   ~pyxu.operator.UniformSpread

Filters
+++++++

.. autosummary::

   ~pyxu.operator.DifferenceOfGaussians
   ~pyxu.operator.DoG
   ~pyxu.operator.Gaussian
   ~pyxu.operator.Laplace
   ~pyxu.operator.MovingAverage
   ~pyxu.operator.Prewitt
   ~pyxu.operator.Scharr
   ~pyxu.operator.Sobel
   ~pyxu.operator.StructureTensor

Derivatives
+++++++++++

.. autosummary::

   ~pyxu.operator.DirectionalDerivative
   ~pyxu.operator.DirectionalGradient
   ~pyxu.operator.DirectionalHessian
   ~pyxu.operator.DirectionalLaplacian
   ~pyxu.operator.Divergence
   ~pyxu.operator.Gradient
   ~pyxu.operator.Hessian
   ~pyxu.operator.Jacobian
   ~pyxu.operator.Laplacian
   ~pyxu.operator.PartialDerivative

Tensor Products
+++++++++++++++

.. autosummary::

   ~pyxu.operator.khatri_rao
   ~pyxu.operator.kron

Misc
^^^^

.. autosummary::

   ~pyxu.operator.BroadcastAxes
   ~pyxu.operator.ConstantValued
   ~pyxu.operator.RechunkAxes
   ~pyxu.operator.ReshapeAxes
   ~pyxu.operator.SqueezeAxes
   ~pyxu.operator.TransposeAxes

Block-defined Operators
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.operator.block_diag
   ~pyxu.operator.stack

Element-wise Operators
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.operator.Abs
   ~pyxu.operator.ArcCos
   ~pyxu.operator.ArcCosh
   ~pyxu.operator.ArcSin
   ~pyxu.operator.ArcSinh
   ~pyxu.operator.ArcTan
   ~pyxu.operator.ArcTanh
   ~pyxu.operator.Cbrt
   ~pyxu.operator.Clip
   ~pyxu.operator.Cos
   ~pyxu.operator.Cosh
   ~pyxu.operator.Exp
   ~pyxu.operator.Gaussian
   ~pyxu.operator.LeakyReLU
   ~pyxu.operator.Log
   ~pyxu.operator.ReLU
   ~pyxu.operator.Sigmoid
   ~pyxu.operator.Sign
   ~pyxu.operator.SiLU
   ~pyxu.operator.Sin
   ~pyxu.operator.Sinh
   ~pyxu.operator.SoftPlus
   ~pyxu.operator.Sqrt
   ~pyxu.operator.Square
   ~pyxu.operator.Tan
   ~pyxu.operator.Tanh

Pulses
++++++

.. autosummary::

   ~pyxu.operator.Box
   ~pyxu.operator.Dirac
   ~pyxu.operator.FSSPulse
   ~pyxu.operator.KaiserBessel
   ~pyxu.operator.Triangle
   ~pyxu.operator.TruncatedGaussian

pyxu.opt.solver
---------------

.. autosummary::

   ~pyxu.opt.solver.Adam
   ~pyxu.opt.solver.ADMM
   ~pyxu.opt.solver.CG
   ~pyxu.opt.solver.ChambollePock
   ~pyxu.opt.solver.CondatVu
   ~pyxu.opt.solver.CP
   ~pyxu.opt.solver.CV
   ~pyxu.opt.solver.DavisYin
   ~pyxu.opt.solver.DouglasRachford
   ~pyxu.opt.solver.DR
   ~pyxu.opt.solver.DY
   ~pyxu.opt.solver.FB
   ~pyxu.opt.solver.ForwardBackward
   ~pyxu.opt.solver.LorisVerhoeven
   ~pyxu.opt.solver.LV
   ~pyxu.opt.solver.NLCG
   ~pyxu.opt.solver.PD3O
   ~pyxu.opt.solver.PGD
   ~pyxu.opt.solver.PP
   ~pyxu.opt.solver.ProximalPoint

pyxu.opt.stop
-------------

.. autosummary::

   ~pyxu.opt.stop.AbsError
   ~pyxu.opt.stop.ManualStop
   ~pyxu.opt.stop.MaxDuration
   ~pyxu.opt.stop.MaxIter
   ~pyxu.opt.stop.Memorize
   ~pyxu.opt.stop.RelError

pyxu.runtime
------------

.. autosummary::

   ~pyxu.runtime.coerce
   ~pyxu.runtime.CWidth
   ~pyxu.runtime.enforce_precision
   ~pyxu.runtime.EnforcePrecision
   ~pyxu.runtime.getCoerceState
   ~pyxu.runtime.getPrecision
   ~pyxu.runtime.Precision
   ~pyxu.runtime.Width

pyxu.util
---------

Array Backend-Related
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.util.compute
   ~pyxu.util.get_array_module
   ~pyxu.util.redirect
   ~pyxu.util.to_NUMPY

Complex Number Handling
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.util.as_real_op
   ~pyxu.util.require_viewable
   ~pyxu.util.view_as_complex
   ~pyxu.util.view_as_real

Operator-Related
^^^^^^^^^^^^^^^^

.. autosummary::

   ~pyxu.util.as_canonical_axes
   ~pyxu.util.as_canonical_shape
   ~pyxu.util.vectorize

Misc
^^^^

.. autosummary::

   ~pyxu.util.copy_if_unsafe
   ~pyxu.util.import_module
   ~pyxu.util.parse_params
   ~pyxu.util.read_only

.. rubric:: Low-lever Helpers

.. autosummary::

   pyxu.util.misc.peaks
   pyxu.util.misc.star_like_sample

.. toctree::
   :maxdepth: 2
   :hidden:

   abc
   info.deps
   info.ptype
   info.warning
   operator/index
   operator.interop
   opt.solver
   opt.stop
   runtime
   math
   util
   experimental/index
   ../references
