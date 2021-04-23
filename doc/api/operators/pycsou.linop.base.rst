Constructors
============

Module: ``pycsou.linop.base``

.. automodule:: pycsou.linop.base
   :special-members: __init__

    .. rubric:: Interfaces with common scientific computing Python objects

    .. autosummary::

       PyLopLinearOperator
       ExplicitLinearOperator
       DenseLinearOperator
       SparseLinearOperator
       DaskLinearOperator

    .. rubric:: Basic operators

    .. autosummary::

       DiagonalOperator
       PolynomialLinearOperator
       IdentityOperator
       NullOperator

    .. rubric:: Special Structures

    .. autosummary::

       LinOpStack
       LinOpVStack
       LinOpHStack
       BlockOperator
       BlockDiagonalOperator
       KroneckerProduct
       KroneckerSum
       KhatriRaoProduct
