API Reference
=============

Welcome to the official API reference for Pyxu, your go-to library for cutting-edge scientific computing in Python. This
API documentation is intended to serve as a comprehensive guide to the library's various modules, classes, functions,
and interfaces, providing you with detailed descriptions of each component's role, relations, assumptions, and behavior.
Please note that this API reference is not designed to be a tutorial; it's a technical resource aimed at users who are
already familiar with the library's basics and wish to dive deeper into its functionalities. Whether you are
troubleshooting, optimizing, or extending your Pyxu-based applications, this API reference is an invaluable tool for
understanding the intricacies of the library.

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
.. rubric:: operator

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

.. rubric:: arithmetic

.. autosummary::

   ~pyxu.abc.arithmetic.Rule
   ~pyxu.abc.arithmetic.ScaleRule
   ~pyxu.abc.arithmetic.ArgScaleRule
   ~pyxu.abc.arithmetic.ArgShiftRule
   ~pyxu.abc.arithmetic.AddRule
   ~pyxu.abc.arithmetic.ChainRule
   ~pyxu.abc.arithmetic.PowerRule
   ~pyxu.abc.arithmetic.TransposeRule

.. rubric:: solver

.. autosummary::

   ~pyxu.abc.solver.Mode
   ~pyxu.abc.solver.StoppingCriterion
   ~pyxu.abc.solver.Solver

pyxu.info.ptype
^^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.info.ptype.ArrayModule
   ~pyxu.info.ptype.DType
   ~pyxu.info.ptype.Integer
   ~pyxu.info.ptype.NDArray
   ~pyxu.info.ptype.NDArrayAxis
   ~pyxu.info.ptype.NDArrayShape
   ~pyxu.info.ptype.OpC
   ~pyxu.info.ptype.OpShape
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

pyxu.info.deps
^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.info.deps.CUPY_ENABLED
   ~pyxu.info.deps.NDArrayInfo
   ~pyxu.info.deps.SparseArrayInfo
   ~pyxu.info.deps.supported_array_modules
   ~pyxu.info.deps.supported_array_types
   ~pyxu.info.deps.supported_sparse_modules
   ~pyxu.info.deps.supported_sparse_types

pyxu.info.warning
^^^^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.info.warning.PyxuWarning
   ~pyxu.info.warning.AutoInferenceWarning
   ~pyxu.info.warning.BackendWarning
   ~pyxu.info.warning.DenseWarning
   ~pyxu.info.warning.NonTransparentWarning
   ~pyxu.info.warning.PerformanceWarning
   ~pyxu.info.warning.PrecisionWarning

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

   ~pyxu.math.linalg.hutchpp
   ~pyxu.math.linalg.norm
   ~pyxu.math.linalg.trace
   ~pyxu.math.linesearch.backtracking_linesearch

pyxu.opt.stop
^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.opt.stop.AbsError
   ~pyxu.opt.stop.ManualStop
   ~pyxu.opt.stop.MaxCarbon
   ~pyxu.opt.stop.MaxDuration
   ~pyxu.opt.stop.MaxIter
   ~pyxu.opt.stop.Memorize
   ~pyxu.opt.stop.RelError

pyxu.opt.solver
^^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.opt.solver.pgd.PGD
   ~pyxu.opt.solver.cg.CG
   ~pyxu.opt.solver.nlcg.NLCG
   ~pyxu.opt.solver.prox_adam.ProxAdam
   ~pyxu.opt.solver.pds.CondatVu
   ~pyxu.opt.solver.pds.CV
   ~pyxu.opt.solver.pds.PD3O
   ~pyxu.opt.solver.pds.ChambollePock
   ~pyxu.opt.solver.pds.CP
   ~pyxu.opt.solver.pds.LorisVerhoeven
   ~pyxu.opt.solver.pds.LV
   ~pyxu.opt.solver.pds.DavisYin
   ~pyxu.opt.solver.pds.DY
   ~pyxu.opt.solver.pds.DouglasRachford
   ~pyxu.opt.solver.pds.DR
   ~pyxu.opt.solver.pds.ForwardBackward
   ~pyxu.opt.solver.pds.FB
   ~pyxu.opt.solver.pds.ProximalPoint
   ~pyxu.opt.solver.pds.PP
   ~pyxu.opt.solver.pds.ADMM

pyxu.runtime
^^^^^^^^^^^^
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
^^^^^^^^^
.. autosummary::

   ~pyxu.util.array_module.compute
   ~pyxu.util.array_module.copy_if_unsafe
   ~pyxu.util.array_module.get_array_module
   ~pyxu.util.array_module.read_only
   ~pyxu.util.array_module.redirect
   ~pyxu.util.array_module.to_NUMPY
   ~pyxu.util.complex.view_as_complex
   ~pyxu.util.complex.view_as_complex_mat
   ~pyxu.util.complex.view_as_real
   ~pyxu.util.complex.view_as_real_mat
   ~pyxu.util.inspect.import_module
   ~pyxu.util.inspect.parse_params
   ~pyxu.util.misc.as_canonical_shape
   ~pyxu.util.misc.next_fast_len
   ~pyxu.util.misc.peaks
   ~pyxu.util.misc.star_like_sample
   ~pyxu.util.operator._array_ize
   ~pyxu.util.operator._dask_zip
   ~pyxu.util.operator.infer_composition_shape
   ~pyxu.util.operator.infer_sum_shape
   ~pyxu.util.operator.vectorize

pyxu.operator
^^^^^^^^^^^^^
.. The operator module is huge, hence we break it down into categories to help users navigate.

.. rubric:: blocks

.. autosummary::

   ~pyxu.operator.blocks.stack
   ~pyxu.operator.blocks.vstack
   ~pyxu.operator.blocks.hstack
   ~pyxu.operator.blocks.block_diag
   ~pyxu.operator.blocks.block
   ~pyxu.operator.blocks.coo_block

.. rubric:: func

.. autosummary::

   ~pyxu.operator.func.norm.L1Norm
   ~pyxu.operator.func.norm.L2Norm
   ~pyxu.operator.func.norm.SquaredL2Norm
   ~pyxu.operator.func.norm.SquaredL1Norm
   ~pyxu.operator.func.norm.LInfinityNorm
   ~pyxu.operator.func.norm.L21Norm
   ~pyxu.operator.func.norm.PositiveL1Norm
   ~pyxu.operator.func.indicator.L1Ball
   ~pyxu.operator.func.indicator.L2Ball
   ~pyxu.operator.func.indicator.LInfinityBall
   ~pyxu.operator.func.indicator.PositiveOrthant
   ~pyxu.operator.func.indicator.HyperSlab
   ~pyxu.operator.func.indicator.RangeSet
   ~pyxu.operator.func.indicator.AffineSet
   ~pyxu.operator.func.indicator.ConvexSetIntersection

.. rubric:: linop: basic operators

.. autosummary::

   ~pyxu.operator.linop.select.SubSample
   ~pyxu.operator.linop.select.Trim
   ~pyxu.operator.linop.reduce.Sum
   ~pyxu.operator.linop.kron.kron
   ~pyxu.operator.linop.kron.khatri_rao
   ~pyxu.operator.linop.base.IdentityOp
   ~pyxu.operator.linop.base.NullOp
   ~pyxu.operator.linop.base.NullFunc
   ~pyxu.operator.linop.base.HomothetyOp
   ~pyxu.operator.linop.base.DiagonalOp
   ~pyxu.operator.linop.base._ExplicitLinOp
   ~pyxu.operator.linop.pad.Pad

.. rubric:: linop: stencils & convolutions

.. autosummary::

   ~pyxu.operator.linop.stencil._stencil._Stencil
   ~pyxu.operator.linop.stencil.stencil.Stencil
   ~pyxu.operator.linop.stencil.stencil.Correlate
   ~pyxu.operator.linop.stencil.stencil.Convolve

.. rubric:: linop: transforms

.. autosummary::

   ~pyxu.operator.linop.fft.fft.FFT
   ~pyxu.operator.linop.fft.nufft.NUFFT

.. rubric:: linop: derivatives

.. autosummary::

   ~pyxu.operator.linop.diff.PartialDerivative
   ~pyxu.operator.linop.diff.Gradient
   ~pyxu.operator.linop.diff.Jacobian
   ~pyxu.operator.linop.diff.Divergence
   ~pyxu.operator.linop.diff.Hessian
   ~pyxu.operator.linop.diff.Laplacian
   ~pyxu.operator.linop.diff.DirectionalDerivative
   ~pyxu.operator.linop.diff.DirectionalGradient
   ~pyxu.operator.linop.diff.DirectionalLaplacian
   ~pyxu.operator.linop.diff.DirectionalHessian

.. rubric:: linop: filters

.. autosummary::

   ~pyxu.operator.linop.filter.MovingAverage
   ~pyxu.operator.linop.filter.Gaussian
   ~pyxu.operator.linop.filter.DifferenceOfGaussians
   ~pyxu.operator.linop.filter.DoG
   ~pyxu.operator.linop.filter.Laplace
   ~pyxu.operator.linop.filter.Sobel
   ~pyxu.operator.linop.filter.Prewitt
   ~pyxu.operator.linop.filter.Scharr

.. rubric:: map

.. autosummary::

   ~pyxu.operator.map.base.ConstantValued
   ~pyxu.operator.map.ufunc.sin
   ~pyxu.operator.map.ufunc.cos
   ~pyxu.operator.map.ufunc.tan
   ~pyxu.operator.map.ufunc.arcsin
   ~pyxu.operator.map.ufunc.arccos
   ~pyxu.operator.map.ufunc.arctan
   ~pyxu.operator.map.ufunc.sinh
   ~pyxu.operator.map.ufunc.cosh
   ~pyxu.operator.map.ufunc.tanh
   ~pyxu.operator.map.ufunc.arcsinh
   ~pyxu.operator.map.ufunc.arccosh
   ~pyxu.operator.map.ufunc.arctanh
   ~pyxu.operator.map.ufunc.exp
   ~pyxu.operator.map.ufunc.log
   ~pyxu.operator.map.ufunc.clip
   ~pyxu.operator.map.ufunc.sqrt
   ~pyxu.operator.map.ufunc.cbrt
   ~pyxu.operator.map.ufunc.square
   ~pyxu.operator.map.ufunc.abs
   ~pyxu.operator.map.ufunc.sign
   ~pyxu.operator.map.ufunc.gaussian
   ~pyxu.operator.map.ufunc.sigmoid
   ~pyxu.operator.map.ufunc.softplus
   ~pyxu.operator.map.ufunc.leakyrelu
   ~pyxu.operator.map.ufunc.relu
   ~pyxu.operator.map.ufunc.silu
   ~pyxu.operator.map.ufunc.softmax

pyxu.operator.interop
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::

   ~pyxu.operator.interop.source.from_source
   ~pyxu.operator.interop.sciop.from_sciop
   ~pyxu.operator.interop.jax.from_jax
   ~pyxu.operator.interop.jax._from_jax
   ~pyxu.operator.interop.jax._to_jax
   ~pyxu.operator.interop.torch.from_torch
   ~pyxu.operator.interop.torch.astensor
   ~pyxu.operator.interop.torch.asarray


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
   ../references
