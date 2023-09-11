pyxu.operator
=============

.. pyxu.operator.map.base -----------------------------------------------------

.. autofunction:: pyxu.operator.ConstantValued

.. pyxu.operator.func.norm ----------------------------------------------------

.. autoclass:: pyxu.operator.L1Norm
   :no-members:

.. autoclass:: pyxu.operator.L2Norm
   :no-members:

.. autoclass:: pyxu.operator.SquaredL2Norm
   :no-members:

.. autoclass:: pyxu.operator.SquaredL1Norm
   :no-members:
   :members: prox
   :special-members: __init__

.. autofunction:: pyxu.operator.shift_loss

.. autoclass:: pyxu.operator.KLDivergence
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.LInfinityNorm
   :no-members:

.. autoclass:: pyxu.operator.L21Norm
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.PositiveL1Norm
   :no-members:

.. pyxu.operator.func.indicator -----------------------------------------------

.. autofunction:: pyxu.operator.L1Ball

.. autofunction:: pyxu.operator.L2Ball

.. autofunction:: pyxu.operator.LInfinityBall

.. autoclass:: pyxu.operator.PositiveOrthant
   :no-members:

.. autoclass:: pyxu.operator.HyperSlab
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.RangeSet
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.AffineSet
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.ConvexSetIntersection
   :no-members:
   :members: prox
   :special-members: __init__

.. pyxu.operator.blocks -------------------------------------------------------

.. autofunction:: pyxu.operator.stack

.. autofunction:: pyxu.operator.vstack

.. autofunction:: pyxu.operator.hstack

.. autofunction:: pyxu.operator.block_diag

.. autofunction:: pyxu.operator.block

.. autofunction:: pyxu.operator.coo_block

.. pyxu.operator.map.ufunc ----------------------------------------------------

.. autofunction:: pyxu.operator.sin

.. autofunction:: pyxu.operator.cos

.. autofunction:: pyxu.operator.tan

.. autofunction:: pyxu.operator.arcsin

.. autofunction:: pyxu.operator.arccos

.. autofunction:: pyxu.operator.arctan

.. autofunction:: pyxu.operator.sinh

.. autofunction:: pyxu.operator.cosh

.. autofunction:: pyxu.operator.tanh

.. autofunction:: pyxu.operator.arcsinh

.. autofunction:: pyxu.operator.arccosh

.. autofunction:: pyxu.operator.arctanh

.. autofunction:: pyxu.operator.exp

.. autofunction:: pyxu.operator.log

.. autofunction:: pyxu.operator.clip

.. autofunction:: pyxu.operator.sqrt

.. autofunction:: pyxu.operator.cbrt

.. autofunction:: pyxu.operator.square

.. autofunction:: pyxu.operator.abs

.. autofunction:: pyxu.operator.sign

.. autofunction:: pyxu.operator.gaussian

.. autofunction:: pyxu.operator.sigmoid

.. autofunction:: pyxu.operator.softplus

.. autofunction:: pyxu.operator.leakyrelu

.. autofunction:: pyxu.operator.relu

.. autofunction:: pyxu.operator.silu

.. autofunction:: pyxu.operator.softmax

.. pyxu.operator.linop.select -------------------------------------------------

.. autoclass:: pyxu.operator.SubSample
   :no-members:
   :members: apply, adjoint, TrimSpec, IndexSpec
   :special-members: __init__

.. autofunction:: pyxu.operator.Trim

.. pyxu.operator.linop.reduce -------------------------------------------------

.. autofunction:: pyxu.operator.Sum

.. pyxu.operator.linop.kron ---------------------------------------------------

.. autofunction:: pyxu.operator.kron

.. autofunction:: pyxu.operator.khatri_rao

.. pyxu.operator.linop.base ---------------------------------------------------

.. autoclass:: pyxu.operator.IdentityOp
   :no-members:

.. autoclass:: pyxu.operator.NullOp
   :no-members:

.. autofunction:: pyxu.operator.NullFunc

.. autofunction:: pyxu.operator.HomothetyOp

.. autofunction:: pyxu.operator.DiagonalOp

.. pyxu.operator.linop.pad ----------------------------------------------------

.. autoclass:: pyxu.operator.Pad
   :no-members:
   :members: WidthSpec, ModeSpec
   :special-members: __init__

.. pyxu.operator.linop.stencil ------------------------------------------------

.. autoclass:: pyxu.operator._Stencil
   :no-members:
   :members: init, apply, IndexSpec

.. autoclass:: pyxu.operator.Stencil
   :no-members:
   :members: KernelSpec, configure_dispatcher, kernel, center, relative_indices, visualize
   :special-members: __init__

.. autoclass:: pyxu.operator.Correlate

.. autofunction:: pyxu.operator.Convolve

.. pyxu.operator.linop.fft ----------------------------------------------------

.. autoclass:: pyxu.operator.FFT
   :no-members:
   :members: apply, adjoint
   :special-members: __init__

.. autoclass:: pyxu.operator.NUFFT
   :no-members:
   :members: type1, type2, type3, apply, adjoint, ascomplexarray, mesh, plot_kernel, params, auto_chunk, allocate, diagnostic_plot, stats

.. pyxu.operator.linop.diff ---------------------------------------------------

.. autoclass:: pyxu.operator.PartialDerivative
   :members: finite_difference, gaussian_derivative

.. autofunction:: pyxu.operator.Gradient

.. autofunction:: pyxu.operator.Jacobian

.. autofunction:: pyxu.operator.Divergence

.. autofunction:: pyxu.operator.Hessian

.. autofunction:: pyxu.operator.Laplacian

.. autofunction:: pyxu.operator.DirectionalDerivative

.. autofunction:: pyxu.operator.DirectionalGradient

.. autofunction:: pyxu.operator.DirectionalLaplacian

.. autofunction:: pyxu.operator.DirectionalHessian

.. pyxu.operator.linop.filter -------------------------------------------------

.. autofunction:: pyxu.operator.MovingAverage

.. autofunction:: pyxu.operator.Gaussian

.. autofunction:: pyxu.operator.DifferenceOfGaussians

.. autofunction:: pyxu.operator.DoG

.. autofunction:: pyxu.operator.Laplace

.. autofunction:: pyxu.operator.Sobel

.. autofunction:: pyxu.operator.Prewitt

.. autofunction:: pyxu.operator.Scharr

.. autoclass:: pyxu.operator.StructureTensor
