pyxu.operator
=============

.. autofunction:: pyxu.operator.map.base.ConstantValued

.. automodule:: pyxu.operator.map.ufunc

.. autoclass:: pyxu.operator.func.norm.L1Norm
   :no-members:

.. autoclass:: pyxu.operator.func.norm.L2Norm
   :no-members:

.. autoclass:: pyxu.operator.func.norm.SquaredL2Norm
   :no-members:

.. autoclass:: pyxu.operator.func.norm.SquaredL1Norm
   :no-members:
   :members: prox
   :special-members: __init__

.. autoclass:: pyxu.operator.func.norm.LInfinityNorm
   :no-members:

.. autoclass:: pyxu.operator.func.norm.L21Norm
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.func.norm.PositiveL1Norm
   :no-members:

.. autofunction:: pyxu.operator.func.indicator.L1Ball

.. autofunction:: pyxu.operator.func.indicator.L2Ball

.. autofunction:: pyxu.operator.func.indicator.LInfinityBall

.. autoclass:: pyxu.operator.func.indicator.PositiveOrthant
   :no-members:

.. autoclass:: pyxu.operator.func.indicator.HyperSlab
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.func.indicator.RangeSet
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.func.indicator.AffineSet
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.operator.func.indicator.ConvexSetIntersection
   :no-members:
   :members: prox
   :special-members: __init__

.. autofunction:: pyxu.operator.interop.source.from_source

.. autofunction:: pyxu.operator.interop.sciop.from_sciop

.. autofunction:: pyxu.operator.interop.jax.from_jax

.. autofunction:: pyxu.operator.interop.jax._from_jax

.. autofunction:: pyxu.operator.interop.jax._to_jax

.. autofunction:: pyxu.operator.interop.torch.astensor

.. autofunction:: pyxu.operator.interop.torch.asarray

.. autofunction:: pyxu.operator.interop.torch.from_torch

.. automodule:: pyxu.operator.blocks

.. autoclass:: pyxu.operator.linop.select.SubSample
   :no-members:
   :members: apply, adjoint, TrimSpec, IndexSpec
   :special-members: __init__

.. autofunction:: pyxu.operator.linop.select.Trim

.. autofunction:: pyxu.operator.linop.reduce.Sum

.. autofunction:: pyxu.operator.linop.kron.kron

.. autofunction:: pyxu.operator.linop.kron.khatri_rao

.. autoclass:: pyxu.operator.linop.base.IdentityOp
   :no-members:

.. autoclass:: pyxu.operator.linop.base.NullOp
   :no-members:

.. autofunction:: pyxu.operator.linop.base.NullFunc

.. autofunction:: pyxu.operator.linop.base.HomothetyOp

.. autofunction:: pyxu.operator.linop.base.DiagonalOp

.. autofunction:: pyxu.operator.linop.base._ExplicitLinOp

.. autoclass:: pyxu.operator.linop.pad.Pad
   :no-members:
   :members: WidthSpec, ModeSpec
   :special-members: __init__

.. autoclass:: pyxu.operator.linop.stencil._stencil._Stencil
   :no-members:
   :members: init, apply, IndexSpec

.. autoclass:: pyxu.operator.linop.stencil.stencil.Stencil
   :no-members:
   :members: KernelSpec, configure_dispatcher, kernel, center, relative_indices, visualize
   :special-members: __init__

.. autoclass:: pyxu.operator.linop.stencil.stencil.Correlate

.. autofunction:: pyxu.operator.linop.stencil.stencil.Convolve

.. autoclass:: pyxu.operator.linop.fft.fft.FFT
   :no-members:
   :members: apply, adjoint
   :special-members: __init__

.. autoclass:: pyxu.operator.linop.fft.nufft.NUFFT
   :no-members:
   :members: type1, type2, type3, apply, adjoint, ascomplexarray, mesh, plot_kernel, params, auto_chunk, allocate, diagnostic_plot, stats

.. autoclass:: pyxu.operator.linop.diff.PartialDerivative
   :members: finite_difference, gaussian_derivative

.. autofunction:: pyxu.operator.linop.diff.Gradient

.. autofunction:: pyxu.operator.linop.diff.Jacobian

.. autofunction:: pyxu.operator.linop.diff.Divergence

.. autofunction:: pyxu.operator.linop.diff.Hessian

.. autofunction:: pyxu.operator.linop.diff.Laplacian

.. autofunction:: pyxu.operator.linop.diff.DirectionalDerivative

.. autofunction:: pyxu.operator.linop.diff.DirectionalGradient

.. autofunction:: pyxu.operator.linop.diff.DirectionalLaplacian

.. autofunction:: pyxu.operator.linop.diff.DirectionalHessian

.. autofunction:: pyxu.operator.linop.filter.MovingAverage

.. autofunction:: pyxu.operator.linop.filter.Gaussian

.. autofunction:: pyxu.operator.linop.filter.DifferenceOfGaussians

.. autofunction:: pyxu.operator.linop.filter.DoG

.. autofunction:: pyxu.operator.linop.filter.Laplace

.. autofunction:: pyxu.operator.linop.filter.Sobel

.. autofunction:: pyxu.operator.linop.filter.Prewitt

.. autofunction:: pyxu.operator.linop.filter.Scharr
