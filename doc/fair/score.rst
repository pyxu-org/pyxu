Pyxu Score Explained
====================

The **Pyxu Score** is a quantitative measure we use to rank and evaluate `Pyxu plugins <./plugins/index.html>`_.

.. warning::

    While the Pyxu Score offers insights into a plugin's quality and popularity, it's just one of many metrics.
    Users are encouraged to explore plugins individually and consider other factors before making a decision.

The score takes into account multiple criteria, ensuring you get a holistic understanding of a plugin's standing. Here's
a breakdown:

1. **Version Matching**: If a plugin's version matches the latest Pyxu version, and the latest Python version it gets a point.
2. **Pyxu Principles Adherence**: Plugins abiding by key `Pyxu principles <./dev_notes.html>`_, namely:

   * Input shape agnosticity (i.e., support for NDArrays),

   * Complete backend-agnosticity (i.e., support for Numpy, Cupy and Dask arrays),
   * Precision management (i.e., allow selection between computation at both single and double precision)

   *(each earn an additional point)*

3. **Development Stage Weight**:

   * **Early stages** (Planning, Pre-Alpha, Alpha) contribute 0.5 points.
   * **Advanced stages** (Beta, Production/Stable, Mature) contribute 1 point.
   * **Inactive** plugins do not recieive a score.
   See `the PyPI classifiers <https://pypi.org/classifiers/>`_ for more information on the development stages.

4. **Downloads**: Plugins get points based on the number of downloads in the last month, up to a maximum of 1 point
   for 1000 downloads.

The final score is then normalized by the number of criteria used (6 in this case) and presented as a percentage.


.. raw:: html

   <h3 style="margin-top: 0; font-weight: bold; text-align: left; ">Contribute to the Pyxu Score</h3>

Your insights matter! If you're a developer with ideas on how we can further refine the Pyxu Score, we'd love to hear
from you. 💡

We invite you to `open an issue <https://github.com/pyxu-org/pyxu/issues>`_ suggesting novel metrics or criteria
that, in your opinion, would better represent the value and quality of Pyxu plugins. By contributing your thoughts, you
play an integral role in refining and shaping the Pyxu community's standards.

Together, let's make the Pyxu Score the best reflection of plugin excellence! 🌟
