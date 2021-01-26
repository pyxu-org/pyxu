Multidimensional Maps
=====================

Abstract classes for multidimensional (potentially nonlinear) maps.

.. currentmodule:: pycsou.core.map

.. autosummary:: 

   Map
   DifferentiableMap
   MapStack
   DiffMapStack

.. autoclass:: pycsou.core.map.Map
   :special-members: __init__, __call__, __add__, __mul__, __matmul__, __neg__, __sub__, __pow__, __truediv__

.. autoclass:: pycsou.core.map.DifferentiableMap
   :special-members: __init__

.. autoclass:: pycsou.core.map.MapStack
   :special-members: __init__

.. autoclass:: pycsou.core.map.MapVStack
   :special-members: __init__

.. autoclass:: pycsou.core.map.MapHStack
   :special-members: __init__

.. autoclass:: pycsou.core.map.DiffMapStack
   :special-members: __init__


.. autoclass:: pycsou.core.map.DiffMapVStack
   :special-members: __init__


.. autoclass:: pycsou.core.map.DiffMapHStack
   :special-members: __init__