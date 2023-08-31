pyxu.operator.func
==================

.. contents:: Table of Contents
   :local:
   :depth: 1
   
Norms 
-----

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

Indicator functions
-------------------

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