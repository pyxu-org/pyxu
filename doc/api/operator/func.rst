pyxu.operator.func
==================

.. contents:: Table of Contents
   :local:
   :depth: 2

Norms & Loss Functions
----------------------

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

Indicator Functions
-------------------

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
