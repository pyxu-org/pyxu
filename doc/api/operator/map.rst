pyxu.operator.map
=================

.. contents:: Table of Contents
   :local:
   :depth: 2

Element-wise Operators
----------------------

.. autofunction:: pyxu.operator.abs

.. autofunction:: pyxu.operator.arccos

.. autofunction:: pyxu.operator.arccosh

.. autofunction:: pyxu.operator.arcsin

.. autofunction:: pyxu.operator.arcsinh

.. autofunction:: pyxu.operator.arctan

.. autofunction:: pyxu.operator.arctanh

.. autofunction:: pyxu.operator.cbrt

.. autofunction:: pyxu.operator.clip

.. autofunction:: pyxu.operator.cos

.. autofunction:: pyxu.operator.cosh

.. autofunction:: pyxu.operator.exp

.. autofunction:: pyxu.operator.gaussian

.. autofunction:: pyxu.operator.leakyrelu

.. autofunction:: pyxu.operator.log

.. autofunction:: pyxu.operator.relu

.. autofunction:: pyxu.operator.sigmoid

.. autofunction:: pyxu.operator.sign

.. autofunction:: pyxu.operator.silu

.. autofunction:: pyxu.operator.sin

.. autofunction:: pyxu.operator.sinh

.. autofunction:: pyxu.operator.sqrt

.. autofunction:: pyxu.operator.square

.. autofunction:: pyxu.operator.tan

.. autofunction:: pyxu.operator.tanh

Misc
----

.. autofunction:: pyxu.operator.ConstantValued

.. autoclass:: pyxu.operator.TransposeAxes
   :exclude-members: apply, adjoint, cogram

.. autofunction:: pyxu.operator.softmax

.. autofunction:: pyxu.operator.softplus

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
