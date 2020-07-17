Installation
============

::

    $ git clone https://github.com/matthieumeo/pycsou
    $ cd <repository_dir>/
    $ # optional conda dependencies [See Remarks below]
    $ pip install --user -e . # option -e is for development mode only


Documentation
-------------

::

    $ conda install sphinx=='2.1.*'            \
                    sphinx_rtd_theme=='0.4.*'
    $ python3 setup.py build_sphinx  # Generate documentation (optional)


Remarks
-------

* pyoneer is developed and tested on x86_64 systems running MacOS and Linux.

* It is recommended to install dependencies using `Miniconda <https://conda.io/miniconda.html>`_ or
  `Anaconda <https://www.anaconda.com/download/#linux>`_::

    $ conda install --channel=conda-forge \
                    --file=requirements-conda.txt
