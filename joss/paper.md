---
title: '**Pyxu**: A Python Framework for Computational Imaging with Distributed Computing and Hardware Acceleration'
tags:
  - Python
  - computational imaging
  - array computing
  - computer vision
  - image reconstruction
  - machine learning
  - bayesian methods 
authors:
  - name: Sepand Kashani
    orcid: 0000-0002-0735-371X
    equal-contrib: true
    corresponding: true 
    affiliation: 1
  - name: Joan Rué Queralt
    orcid: 0000-0002-9595-4557
    equal-contrib: true
    affiliation: 1
  - name: Matthieu Simeoni
    orcid: 0000-0002-4927-3697
    equal-contrib: true
    affiliation: 1

affiliations:
 
 - name: École Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland
   index: 1

date: 15 April 2024
bibliography: paper.bib

---

# Summary

The advent of computational imaging has brought about transformative changes in digital image restoration and enhancement, positioning these methodologies at the heart of numerous fields such as computer vision, biomedical imaging, and radio-interferometry. This evolution is particularly crucial in an era where the sheer volume of image data, combined with the complexity of restoration and enhancement tasks, poses significant computational challenges. The ability to effectively process and improve the quality of images captured under suboptimal conditions—enhancing resolution, reducing noise, and compensating for missing data—has become indispensable.

Amidst this backdrop, a critical challenge emerges: the processing of large-scale image datasets often exceeds the memory capacity of conventional computing systems, necessitating advanced solutions for out-of-memory data handling and distributed computing. Furthermore, the escalating demand for real-time processing and analysis calls for significant advancements in hardware acceleration. These challenges underscore the necessity for a new class of computational imaging tools designed to operate beyond the limitations of traditional systems.


# Statement of need
Digital image restoration and enhancement, central to the field of computer vision, aim to improve the quality of degraded or partially captured images, transforming them into versions of superior quality suitable for visualization or further analysis [@McCann:2019, @Mait:2018, @Haansen:2010]. The enhanced images typically feature improved resolution, reduced noise and blur, and even restored areas where data was missing.

The foundation of many image reconstruction methods is the resolution of mathematical inverse problems. This involves observing an object—whether a biological cell, a scene, or any other entity—through an acquisition system, like a microscope or camera, that captures data marred by noise and blur. The primary goal is to reconstruct the original object from these imperfect observations, effectively reversing the acquisition process to produce a clearer and more accurate representation. This process is mathematically represented as:

$$
\mathbf{y} =\mathcal{F}(\mathbf{x}) + \mathbf{\nu}
$$

where:
- $\mathbf{x}$ denotes the unknown image,
- $\mathcal{F}$ is an operator modelling the acquisition system (typically linear),
- $\mathbf{\nu}$ is some random noise, typically additive and zero-mean.
- $\mathbf{y}$ denote the measured observations,

Common examples of computational imaging tasks include for example:

<img src="../doc/_static/tutorial/recon_examples.jpg" alt="Computational imaging Examples." style="width: 80%;">


- **Image Denoising**: The process of eliminating noise artifacts to create a cleaner, crisper image.
- **Image Deblurring**: Restoration of a sharp image from a blurry input, enhancing focus and detail.
- **Image Inpainting**: Reconstructing missing or damaged regions within an image, often used for tasks like replacing lost blocks during coding and transmission or erasing watermark/logo overlays.
- **Image Super-Resolution**: Elevating the resolution of an image or an imaging system to provide finer detail.
- **Image Fusion**: The merging of two or more degraded images of the same object or scene into a single image that exceeds the quality of any individual input.
- **Image Filtering**: Modifying an image to promote particular features of interest, such as points, lines, or shapes.
- **Tomographic Reconstruction**: Rebuilding an image from its lower-dimensional projections, known as sinograms in the context of CT or PET scans.


Traditional software-based imaging pipelines have often relied on direct inversion techniques, namely estimating $\hat{x} = \mathcal{F}^{-1}(y)$. These methods aim to approximate the pseudoinverse of the sensing operator that models the image acquisition system. Although fast, intuitive, and relatively scalable, they are intrinsically limited in terms of accuracy, often resulting in images with poor resolution and significant reconstruction artifacts. It is evident that these methods struggle with noise, as the inverse operator amplifies the noise term in $\mathbf{y} =\mathcal{F}(\mathbf{x}) + \mathbf{\nu}$.

State-of-the-art image restoration methods employ powerful image priors that impose specific perceptual or structural characteristics typical of natural images. This effectively regularizes the ill-posed inverse problem and improves reconstruction accuracy. Such methods formulate Bayesian estimation problems to strike a balance between a likelihood term, which ensures fidelity to the observed data and its statistical attributes, and a prior term, which enforces the physical plausibility of the reconstruction. These problems are addressed using sophisticated iterative first-order proximal-splitting methods, which are apt for uncertainty quantification through Markov Chain Monte Carlo (MCMC) samplers.

On one hand, advancements in hardware, particularly the development of more powerful sensors, are significantly increasing the volume of data produced by measuring instruments [@Poger:2023, @Marone:2017]. These hardware innovations present a dual challenge; while they enhance image quality and resolution, benefiting scientists, they also impose substantial computational burdens with data volumes that frequently exceed memory capacities. This necessitates the development of out-of-memory algorithms and the integration of GPU acceleration to ensure reasonable image reconstruction times. In parallel,the integration of advanced algorithms [@Condat:2019], often relying on learned priors, has markedly transformed computational imaging [@McCann:2019, @Ongie:2020, @Suzuki:2017, @Ye:2023, @Arridge:2019]. This shift towards utilizing deep learning techniques has elevated image restoration, enhancement, and manipulation to unprecedented levels of performance, with some experts suggesting that we may have reached a plateau in terms of accuracy and performance [@Romano:2017]. Consequently, much of the current research in computational imaging is focused on incorporating robust deep learning models within reconstruction pipelines.

Pyxu distinguishes itself as a pioneering Python-based framework, meticulously designed to meet these advanced demands. It extends beyond the capabilities of specialized frameworks by offering a comprehensive, modular solution tailored to the broad and varied needs of the computational imaging community. Pyxu is characterized by its ability to facilitate the development and deployment of complex imaging pipelines, incorporating cutting-edge features such as operator algebra logic, seamless hardware acceleration, and deep learning interoperability. Fundamentally, Pyxu represents flexibility, high-performance computing, and a user-centric approach, making it an indispensable tool for managing the complexities of modern computational imaging, especially in handling large-scale datasets and leveraging distributed computing and hardware acceleration to address out-of-memory data processing challenges.


# Core Features of Pyxu

Pyxu stands as a pioneering open-source computational imaging software framework, meticulously crafted for Python. It is distinguished by its comprehensive support for hardware acceleration and distributed computing, encapsulating a suite of features designed to streamline the development and execution of computational imaging tasks across various modalities. Below, we delve into the key features that set Pyxu apart:

<img src="../doc/_static/microservice_hero.png" alt="Microservice architecture." style="width: 60%;">


### Operator Algebra Logic: 

Pyxu's operator algebra logic is a foundational element that enables the construction of intricate operators from simpler components. This logic serves as a cornerstone for solving customized inverse problems through advanced optimization algorithms. 

As an example, consider the composition of a differentiable functional (`DiffFunc`) $\mathcal{f}: \mathbb{R}^{N}\rightarrow\mathbb{R}$ with a differentiable map (`DiffMap`) $\mathcal{L}: \mathbb{R}^{M}\rightarrow\mathbb{R}^{N}$. Then their composition $\mathcal{h} = \mathcal{f}\circ \mathcal{L}$ (or `h = f * L` in Pyxu) is also a differentiable functional `DiffFunc`, with gradient given by:

`h.grad(x) = L.jacobian(x).adjoint(f.grad(L(x)))`

Another unique aspect of this setup is Pyxu's automated propagation of Lipschitz constants. This automation ensures that algorithms within Pyxu can automatically adjust their step-sizes for optimal convergence rates, eliminating the need for manual tuning. This feature significantly enhances the reliability and efficiency of optimization tasks, streamlining computational imaging processes.

### Microservice Architecture:

Pyxu's implementation of a microservice architecture, along with its intuitive plug-and-play API, fosters a dynamic and user-friendly environment for computational imaging. The architectural design is centered around offering a set of independent, application-agnostic, loosely coupled modules, such as mathematical operators, optimization algorithms, and Bayesian tools, all optimized for performance across different hardware backends and for out-of-core computing. This design enables users to effortlessly customize their computational imaging pipelines, streamlining interactions with these modular services for rapid integration, development, and deployment of new features in production environments with minimal disruption. Such modularity not only allows for quick adaptation to the latest advancements in computational imaging but also bolsters system reliability. This modular approach to building image reconstruction pipelines means that the services function in isolation, ensuring that any failure is localized to the specific operator or algorithm experiencing issues. Such isolation significantly diminishes the risk of system-wide outages, a critical advantage when constructing complex image reconstruction pipelines.

### Seamless Hardware Acceleration:
Pyxu's codebase is designed to be hardware-agnostic, which grants it the flexibility to process various array types without being confined to specific hardware configurations. This flexibility is crucial for ensuring Pyxu's compatibility with the evolving standards of array APIs and cutting-edge platforms like Intel’s oneAPI, thereby boosting its adaptability and efficiency for computationally demanding imaging tasks. Such a design allows Pyxu to smoothly switch between CPU-bound and GPU-bound array backends—employing NumPy [@Harris:2020] for CPU operations and CuPy [@Nishino:2017] for GPU tasks—utilizing standardized array APIs to achieve significant modularity and portability.

Importantly, Pyxu's architecture strategically reduces or completely avoids the need for data transfers between GPUs and CPUs, addressing a prevalent efficiency bottleneck in many computational imaging frameworks. By maintaining data exclusively on the GPU throughout the processing pipeline, Pyxu drastically cuts down on processing times and scales up its capability to handle extensive computations. This approach is especially beneficial in image processing, where the parallel processing power of GPUs markedly outperforms traditional CPU-based processing, delivering faster results and supporting larger datasets more effectively.


### Distributed Lazy Computing:

Pyxu's distributed computing capabilities are specifically engineered to address the unique challenges presented by large-scale datasets, which often exceed the memory capacities of conventional computing environments. By harnessing the power of lazy computing strategies alongside data chunking, Pyxu achieves efficient management and processing of datasets too large to fit into memory. Lazy computing allows Pyxu to construct a computation graph, deferring the execution of operations until absolutely necessary. Incorporating Dask's technology [@Rocklin:2015], Pyxu extends Python’s ecosystem to embrace out-of-core and distributed computing, mirroring Dask's ability to efficiently handle computations that surpass the limitations of a single machine. Dask's innovative approach, which includes building computation graphs for optimal evaluation and fragmenting large arrays into parallel-processed chunks, is integral to Pyxu's design. This enables Pyxu to execute imaging computations effectively across a diverse range of hardware setups, from multi-core CPUs to distributed GPU arrays.


### Other HPC features
Pyxu's operators are inherently vectorized, optimizing the processing of batches of data in parallel to exploit data-level parallelism. This, combined with Just-in-Time (JIT) compilation via Numba [@Lam:2015] for compute-intensive operations, allows Pyxu to bypass the Python interpreter, delivering performance on par with statically-typed languages. The utilization of Dask not only streamlines complex imaging tasks by judiciously managing computational resources but also underpins Pyxu's capability to efficiently navigate the complexities of large-scale data processing.

Through the integration of Dask's distributed and out-of-core computing functionalities, along with advanced strategies for minimizing unnecessary data transfers and leveraging vectorization and JIT compilation, Pyxu presents a robust framework for computational imaging. It is designed to simplify complex imaging tasks, ensuring scalability, adaptability, and unmatched efficiency in handling extensive imaging data.



### Interoperability with the PyData Stack and Deep Learning Frameworks
Pyxu distinguishes itself with exceptional interoperability within the PyData ecosystem and leading deep learning frameworks like JAX [@Bradbury:2018] and PyTorch [@Paszke:2019], significantly enhancing its utility in complex data science and machine learning workflows. Leveraging innovative Plug-and-Play (PnP) schemes, Pyxu utilizes denoising algorithms as implicit priors, replacing traditional proximity operators to broaden the integration of complex data models into imaging reconstruction, thereby capturing the inherent "truth" or structure of images more effectively. This approach is further enriched by the integration of neural networks, which have dramatically improved the quality and accuracy of reconstructed images by embodying sophisticated data structures and patterns. Pyxu's architecture is optimized for distributed computing environments, focusing on minimizing overhead and eliminating CPU-GPU data transfers to ensure peak efficiency. This streamlined, performance-oriented design makes Pyxu a versatile and powerful framework, setting new standards in computational imaging with its blend of flexibility, efficiency, and cutting-edge interoperability.

### Quality Assurance and Control:
Pyxu's development is underscored by a rigorous quality assurance process, with extensive testing of software components to ensure reliability. Operators undergo thourough testing across a broad spectrum of arguments to validate mathematical correctness, compatibility across different architectures, ND Array dimensional congruence, and behavior under multiple random seeds. This focus on quality extends to ensuring the accuracy of mathematical estimations, including Lipschitz constants, adjoints of linear operators, and other algorithmic approximations. Specialized testing protocols rigorously assess the mathematical and computational integrity of these elements, reinforcing the framework's dedication to providing precise and reliable computational imaging solutions. 

### Community-Driven Development:
Pyxu is characterized by its application-agnostic nature, versatility, and open-source foundation with minimal dependencies. Central to enhancing Pyxu's adaptability and usability for specific applications is the Pyxu FAIR platform, which streamlines the development, sharing, and integration of specialized plugins. By fostering community contributions and a collaborative environment, Pyxu FAIR significantly contributes to the expansion of the framework's capabilities, ensuring it responds effectively to the diverse needs of its user community.

Addressing the framework's broad applicability, Pyxu FAIR ensures Pyxu's adaptability to the unique requirements of various imaging communities. This initiative has facilitated the establishment of a user-friendly catalogue website, serving as a centralized hub for plugin discovery, exploration, and utilization, thereby enhancing Pyxu's adaptability and customization capabilities. Moreover, the introduction of a meta-programming framework alongside an interoperability protocol streamlines the plugin development process, ensuring their seamless integration into Pyxu's core framework. This methodical approach not only drives innovation within the Pyxu community but also broadens the framework's utility, enabling developers to seamlessly enhance or introduce new services to the ecosystem.

# Conclusion

Pyxu represents a significant advancement in the field of computational imaging, offering a robust and flexible solution to the challenges posed by processing large-scale image datasets. Its comprehensive suite of tools and features, combined with support for distributed computing and hardware acceleration, makes Pyxu an essential framework for researchers and practitioners in computational imaging.


# Acknowledgements

We extend our gratitude to all contributors, as detailed in the contributors list on GitHub. Special acknowledgment is given to the EPFL Center for Imaging and the EPFL LCAV research group for their invaluable support during the inception of this project. Additionally, we express our appreciation for the financial backing received from Meta, Zeiss, and various programs facilitated by ETH and SNF.

# References
