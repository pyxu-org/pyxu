.. _theory:

###################
Background Concepts
###################


Data Model
----------

Most real-life approximation problems can be formulated as inverse problems.

.. note::

   Consider an unknown *signal* :math:`f\in \mathcal{L}^2\left(\mathbb{R}^d\right)` and assume that the latter is *probed* by some sensing device, resulting in a data vector :math:`\mathbf{y}=[y_1,\ldots,y_L]\in\mathbb{R}^L`  of :math:`L` measurements. Recovering :math:`f` from the data vector :math:`\mathbf{y}` is called an inverse problem.

The following assumptions are standard:


* To account for sensing *inaccuracies*, the data vector  :math:`\mathbf{y}` is assumed to be the outcome of  a random vector :math:`\mathbf{Y}=[Y_1,\ldots,Y_L]:\Omega\rightarrow \mathbb{R}^L`, fluctuating according to some noise distribution. The entries of :math:`\mathbb{E}[\mathbf{Y}]=\tilde{\mathbf{y}}` are called the ideal measurements --these are the measurements that would be obtained in the absence of noise.
* The measurements are assumed unbiased and linear, i.e. :math:`\mathbb{E}[\mathbf{Y}]=\Phi^\ast f=\left[\langle f, \varphi_1\rangle, \ldots, \langle f,\varphi_L\rangle\right],` for some sampling functionals :math:`\{\varphi_1,\ldots,\varphi_L\}\subset \mathcal{L}^2\left(\mathbb{R}^d\right)`, modelling the acquisition system.

.. seealso::

	See :ref:`operators` for common sampling functionals provided by Pycsou (spatial sampling, subsampling, Fourier/Radon sampling, filtering, mean-pooling, etc...)

.. figure:: /images/deblurring.jpg
   :width: 90 %
   :align: center

   Image deblurring is a common example of inverse problem. 

Discretisation
--------------

Since the number of measurements is *finite*, it is reasonable to constrain the signal :math:`f` to be finite-dimensional:[#f1]_

.. math::

   f=\sum_{n=1}^N\alpha_n \psi_n=\Psi \mathbf{\alpha}, \qquad \mathbf{\alpha}=[\alpha_1,\ldots,\alpha_N]\in\mathbb{R}^N 


for some suitable basis functions :math:`\{\psi_n, \,n=1,\ldots,N\}\subset\mathcal{L}^2(\mathbb{R}^d)`. Typically, the basis functions are chosen as indicator functions of regular rectangular tiles of :math:`\mathbb{R}^d` called pixels. For example:

.. math::

   \psi_n(\mathbf{x})=\begin{cases}1 & \text{if} \,\mathbf{x}\in\left[c_1+(n-1)h_1, c_1+nh_1\right]\times\cdots\times\left[c_d+(n-1)h_d, c_d+ nh_d\right],\\
   0&\text{otherwise,}\end{cases}


where :math:`\mathbf{c}=[c_1,\ldots,c_d]` are the coordinates of the lower-left corner of the first pixel, and :math:`\{h_1,\ldots,h_d\}` are the sizes of the pixels across each dimension. The parametric signal :math:`f` is then a piecewise constant signal than can be **stored/manipulated/displayed** *efficiently* via multi-dimensional array (hence the popularity of pixel-based discretisation schemes).

.. figure:: /images/pixelisation.jpg
   :width: 90 %
   :align: center

   Example of a pixelated signal. 

.. note::

	Other popular choices of basis functions include: sines/cosines, radial basis functions, splines,  	polynomials... 

Pixelisation induces a *discrete inverse problem*:

Find :math:`\mathbf{\alpha}\in\mathbb{R}^N` from the noisy measurements :math:`\mathbf{y}\sim \mathbf{Y}` where :math:`\mathbb{E}[\mathbf{Y}]=\mathbf{G}\mathbf{\alpha}`.

The operator :math:`\mathbf{G}:\mathbb{R}^N\to \mathbb{R}^L` is a rectangular matrix given by:[#f2]_

.. math::
   
	\mathbb{R}^{L\times N} \ni\mathbf{G}
	&=
	\left[ \begin{array}{ccc}
	 \int_{\Omega_1} \varphi_1(\mathbf{x})d\mathbf{x} & \cdots& \int_{\Omega_N} \varphi_1(\mathbf{x})d\mathbf{x}\\
	 \vdots & \ddots & \vdots \\
	 \int_{\Omega_1} \varphi_L(\mathbf{x})d\mathbf{x} & \cdots& \int_{\Omega_N} \varphi_L(\mathbf{x})d\mathbf{x}
	 \end{array}\right]\\
	 &\simeq \eta
	 \left[ \begin{array}{ccc}
	\varphi_1(\mathbf{\xi}_1) & \cdots&\varphi_1(\mathbf{\xi}_N)\\
	 \vdots & \ddots & \vdots \\
	\varphi_L(\mathbf{\xi}_1) & \cdots&\varphi_L(\mathbf{\xi}_N)
	 \end{array}\right],


where :math:`\eta=\Pi_{k=1}^d h_k`, and :math:`\{\Omega_n\}_{n} \subset\mathcal{P}(\mathbb{R}^d)` and :math:`\{\mathbf{\xi}_n\}_n\subset\mathbb{R}^d` are the *supports* and *centroids* of each pixel, respectively. 

Inverse Problems are Ill-Posed
------------------------------

To solve the inverse problem one can approximate the mean :math:`\mathbb{E}[Y]` by its *one-sample empirical estimate* :math:`\mathbf{y}` and solve the linear problem: 

.. math::
   :label: discrete_pb
   
	\mathbf{y}=\mathbf{G}\mathbf{\alpha}.


Unfortunately, such problems are in general ill-posed:


* **There may exist no solutions.** If  :math:`\mathbf{G}` is not surjective, :math:`\mathcal{R}(\mathbf{G})\subsetneq \mathbb{R}^L`. Therefore the noisy data vector :math:`\mathbf{y}` is not guaranteed to belong to :math:`\mathcal{R}(\mathbf{G})`. 
* **There may exist more than one solution.** If :math:`L<N` indeed (or more generally if :math:`\mathbf{G}` is not injective), :math:`\mathcal{N}(\mathbf{G})\neq \{\mathbf{0}\}`. Therefore, if :math:`\mathbf{\alpha}^\star` is a solution to \eqref{inverse_pb_linear_system}, then :math:`\mathbf{\alpha}^\star + \mathbf{\beta}` is also a solution :math:`\forall\mathbf{\beta}\in \mathcal{N}(\mathbf{G})`:  

.. math::

   \mathbf{G}(\mathbf{\alpha}^\star + \mathbf{\beta})=\mathbf{G}\mathbf{\alpha}^\star + {\mathbf{G}\mathbf{\beta}}=\mathbf{G}\mathbf{\alpha}^\star=\mathbf{y}.

* **Solutions may be numerically unstable.** If :math:`\mathbf{G}` is surjective for example, then :math:`\mathbf{G}^\dagger=\mathbf{G}^T(\mathbf{G}\mathbf{G}^T)^{-1}` is a right-inverse of :math:`\mathbf{G}` and :math:`\mathbf{\alpha}^\star(\mathbf{y})=\mathbf{G}^T(\mathbf{G}\mathbf{G}^T)^{-1} \mathbf{y}` is a solution to \eqref{inverse_pb_linear_system}. We have then  

.. math::

   \|\mathbf{\alpha}^\star(\mathbf{y})\|_2\leq \|\mathbf{G}\|_2\|(\mathbf{G}^T\mathbf{G})^{-1}\|_2\|\mathbf{y}\|_2=\underbrace{\frac{\sqrt{\lambda_{max}(\mathbf{G}^T\mathbf{G})}}{\lambda_{min}(\mathbf{G}^T\mathbf{G})}}_{\text{Can be very large!}}\|\mathbf{y}\|_2, \qquad \forall \mathbf{y}\in \mathbb{R}^L.

The reconstruction linear map :math:`\mathbf{y}\mapsto \mathbf{\alpha}^\star(\mathbf{y})` can hence be virtually unbounded making it *unstable*.

.. figure:: /images/inverse_problem.png
   :width: 80 %
   :align: center

   Inverse problems are unstable. 

Regularising Inverse Problems
-----------------------------

The linear system :eq:`discrete_pb` is not only ill-posed but also non sensible: matching exactly the measurements is not desirable since the latter are in practice corrupted by instrumental noise.

A more sensible approach consists instead in solving the inverse problem by means of a penalised optimisation problem, confronting the physical evidence to the analyst’s a priori beliefs about the solution (e.g. smoothness, sparsity) via a data-fidelity and regularisation term, respectively: 

.. math::

	\min_{\mathbf{\alpha}\in\mathbb{R}^N} \,F(\mathbf{y}, \mathbf{G} \mathbf{\alpha})\quad+\quad \lambda\mathcal{R}(\mathbf{\alpha}).


The various quantities involved in the above equation can be interpreted as follows: 


* :math:`F:\mathbb{R}^L\times \mathbb{R}^L\rightarrow \mathbb{R}_+\cup\{+\infty\}` is a cost/data-fidelity/loss functional, measuring the discrepancy between the observed and predicted measurements :math:`\mathbf{y}` and :math:`\mathbf{G}\mathbf{\alpha}` respectively.
* :math:`\mathcal{R}:\mathbb{R}^N\to \mathbb{R}_+\cup\{+\infty\}` is a regularisation/penalty functional favouring simple and well-behaved solutions (typically with a finite number of degrees of freedom). 
* :math:`\lambda>0` is a regularisation/penalty parameter which controls the amount of regularisation by putting the regularisation functional and the cost functional on a similar scale. 


Choosing the Loss Functional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loss functional can be chosen as the negative log-likelihood of the data :math:`\mathbf{y}`:

.. math::

   F(\mathbf{y},\mathbf{G} \mathbf{\alpha})=-\ell(\mathbf{\alpha}\vert\mathbf{y})=-\log p_{Y_1,\ldots,Y_L}\left(y_1,\ldots,y_L | \mathbf{\alpha}\right).

When the noise distribution is not fully known or the likelihood too complex, one can also use general :math:`\ell_p` cost functionals 

.. math::

   F(\mathbf{y},\mathbf{G}\mathbf{\alpha})=\Vert\mathbf{y}-\mathbf{G}\mathbf{\alpha}\Vert_p^p=\sum_{i=1}^L\left\vert y_i-\sum_{n=1}^NG_{in} \alpha_n\right\vert^p,

where :math:`p\in [1,+\infty]` is typically chosen according to the tail behaviour of the noise distribution.

.. figure:: /images/lp_cost.png
   :width: 90 %
   :align: center

.. seealso::   
	
	See :ref:`losses` for a rich collection of commonly used loss functionals provided by Pycsou.  

Choosing the Penalty
~~~~~~~~~~~~~~~~~~~~

The penalty/regularisation functional is used to favour physically-admissible solutions with simple behaviours. It can be interpreted as implementing Occam’s razor principle:

Occam's razor principle is a philosophical principle also known as the *law of briefness* or in Latin *lex parsimoniae*. It was supposedly formulated by William of Ockham in the 14th century, who wrote in Latin *Entia non sunt multiplicanda praeter necessitatem*. In English, this translates to *More things should not be used than are necessary*.

In essence, this principle states that when two equally good explanations for a given phenomenon are available, one should always favour the simplest, i.e. the one that introduces the least explanatory variables.
What exactly is meant by "simple" solutions will depend on the specific application at hand. 

Common choices of regularisation strategies include: Tikhonov regularisation, TV regularisation, maximum entropy regularisation, etc...

.. seealso::   
	
	See :ref:`penalties` for a rich collection of commonly used penalty functionals provided by Pycsou.  

Proximal Algorithms
-------------------

Most optimisation problems used to solve inverse problems in practice
take the form:

.. math::
   :label: generic_form

   {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x})}

where: \* :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is
*convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous*
gradient. \*
:math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}`
and
:math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}`
are two *proper*, *lower semicontinuous* and *convex functions* with
*simple proximal operators*.

-  :math:`\mathbf{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *linear
   operator*.

Problems of the form :eq:`generic_form` can be solved by means of iterative **proximal
algorithms**:

-  **Primal-dual splitting:** solves for problems of the form
   :math:`{\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})+\mathcal{H}(\mathbf{K} \mathbf{x})}`
-  **Chambolle Pock splitting:** solves for problems of the form
   :math:`{\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})+\mathcal{H}(\mathbf{K} \mathbf{x})}`
-  **Douglas Rachford splitting/ADMM:** solves for problems of the form
   :math:`{\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})+\mathcal{H}(\mathbf{x})}`
-  **Forward-Backward splitting/APGD:** solves for problems of the form
   :math:`\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})`

These are all **first-order algorithms**: they rely only on the gradient
of :math:`\mathcal{F}`, and/or the proximal operators of
:math:`\mathcal{G}, \mathcal{H}`, and/or matrix/vector multiplications
with :math:`\mathbf{K}` and :math:`\mathbf{K}^T`.

.. seealso::   
	
	See :ref:`proxalgs` for implementations of the above mentionned algorithms in Pycsou.  


.. rubric:: Footnotes

.. [#f1] Infinite-dimensional signals may indeed have an infinite number of degrees of freedom, which cannot hope to estimate from a finite number of measurements only.
.. [#f2] The last approximate equality results from the midpoint rule.