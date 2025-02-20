.. image:: https://raw.githubusercontent.com/pyxu-org/pyxu/master/doc/_static/wide-logo.png
   :target: https://pyxu-org.github.io/
   :alt: Pyxu logo


Pyxu: Modular and Scalable Computational Imaging
================================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


Installation
------------

.. code-block:: bash

   # user install (CPU-only)
   pip install pyxu@git+https://github.com/pyxu-org/pyxu.git

   # user install (everything)
   pip install pyxu[all]@git+https://github.com/pyxu-org/pyxu.git

   # developer install
   git clone https://github.com/pyxu-org/pyxu.git
   cd pyxu/
   pip install -e ".[all,dev]"
   pre-commit install


Cite us
-------

::

   @software{pyxu-library,
     author       = {Matthieu Simeoni and
                     Sepand Kashani and
                     Joan Rué-Queralt and
                     Pyxu Developers},
     title        = {pyxu-org/pyxu: pyxu},
     publisher    = {Zenodo},
     doi          = {10.5281/zenodo.4486431},
     url          = {https://doi.org/10.5281/zenodo.4486431}
   }
