pyxu.opt
========

.. contents:: Table of Contents
   :local:
   :depth: 1

pyxu.opt.stop
-------------

.. autoclass:: pyxu.opt.stop.MaxIter
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.ManualStop
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.MaxDuration
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.MaxCarbon
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.Memorize
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.AbsError
   :no-members:
   :special-members: __init__

.. autoclass:: pyxu.opt.stop.RelError
   :no-members:
   :special-members: __init__

pyxu.opt.solver
---------------

.. autoclass:: pyxu.opt.solver.pgd.PGD
   :no-members:

.. autoclass:: pyxu.opt.solver.cg.CG
   :no-members:

.. autoclass:: pyxu.opt.solver.nlcg.NLCG
   :no-members:

.. autoclass:: pyxu.opt.solver.prox_adam.ProxAdam
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.CondatVu
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.CV
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.PD3O
   :no-members:

.. autofunction:: pyxu.opt.solver.pds.ChambollePock

.. autofunction:: pyxu.opt.solver.pds.CP

.. autoclass:: pyxu.opt.solver.pds.LorisVerhoeven
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.LV
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.DavisYin
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.DY
   :no-members:

.. autofunction:: pyxu.opt.solver.pds.DouglasRachford

.. autofunction:: pyxu.opt.solver.pds.DR

.. autoclass:: pyxu.opt.solver.pds.ForwardBackward
   :no-members:

.. autoclass:: pyxu.opt.solver.pds.FB
   :no-members:

.. autofunction:: pyxu.opt.solver.pds.ProximalPoint

.. autofunction:: pyxu.opt.solver.pds.PP

.. autoclass:: pyxu.opt.solver.pds.ADMM
   :no-members:
