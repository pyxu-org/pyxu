pyxu.operator.map
=================

.. contents:: Table of Contents
   :local:
   :depth: 2

Element-wise Operators
----------------------

.. autoclass:: pyxu.operator.Abs

.. autoclass:: pyxu.operator.ArcCos

.. autoclass:: pyxu.operator.ArcCosh

.. autoclass:: pyxu.operator.ArcSin

.. autoclass:: pyxu.operator.ArcSinh

.. autoclass:: pyxu.operator.ArcTan

.. autoclass:: pyxu.operator.ArcTanh

.. autoclass:: pyxu.operator.Cbrt

.. autoclass:: pyxu.operator.Clip

.. autoclass:: pyxu.operator.Cos

.. autoclass:: pyxu.operator.Cosh

.. autoclass:: pyxu.operator.Exp

.. autoclass:: pyxu.operator.Gaussian

.. autoclass:: pyxu.operator.LeakyReLU

.. autoclass:: pyxu.operator.Log

.. autoclass:: pyxu.operator.ReLU

.. autoclass:: pyxu.operator.Sigmoid

.. autoclass:: pyxu.operator.Sign

.. autoclass:: pyxu.operator.SiLU

.. autoclass:: pyxu.operator.Sin

.. autoclass:: pyxu.operator.Sinh

.. autoclass:: pyxu.operator.SoftPlus

.. autoclass:: pyxu.operator.Sqrt

.. autoclass:: pyxu.operator.Square

.. autoclass:: pyxu.operator.Tan

.. autoclass:: pyxu.operator.Tanh

Misc
----

.. autofunction:: pyxu.operator.ConstantValued

.. autoclass:: pyxu.operator.TransposeAxes
   :exclude-members: apply, adjoint

.. autoclass:: pyxu.operator.SqueezeAxes
   :exclude-members: apply, adjoint

.. autofunction:: pyxu.operator.RechunkAxes

.. autoclass:: pyxu.operator.ReshapeAxes
   :exclude-members: apply, adjoint

.. autoclass:: pyxu.operator.BroadcastAxes
   :exclude-members: apply, adjoint

Pulses
------

.. autoclass:: pyxu.operator.FSSPulse
   :exclude-members: argscale

.. autoclass:: pyxu.operator.Dirac
   :exclude-members: apply, applyF, support, supportF

.. autoclass:: pyxu.operator.Box
   :exclude-members: apply, applyF, support

.. autoclass:: pyxu.operator.Triangle
   :exclude-members: apply, applyF, support

.. autoclass:: pyxu.operator.TruncatedGaussian
   :exclude-members: apply, applyF, support

.. autoclass:: pyxu.operator.KaiserBessel
   :exclude-members: apply, applyF, support, supportF
