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

date: 4 June 2024
bibliography: references.bib

---

# Summary

Pyxu is a Python-based framework designed to democratize computational imaging by providing a user-friendly and highly efficient solution for both prototying and deploying imaging pipelines. Aimed at imaging scientists and engineers, Pyxu allows users to quickly, accurately, and robustly prototype imaging solutions without needing in-depth knowledge of mathematics, optimization, or computer science. Leveraging the PyData stack, Pyxu integrates seamlessly into existing workflows, making advanced imaging accessible to a broader audience.

# Statement of need

The advent of computational imaging (CI) has brought transformative changes in digital image reconstruction and restoration, positioning CI methods at the heart of computer vision, biomedical imaging, radio-interferometry, and other fields. This evolution is crucial in an era where the volume of image data and the complexity of optimization algorithms pose significant computational challenges. Efficient software implementations that can scale to handle large datasets and leverage GPU acceleration are essential to realize these technological advancements.

To address these needs, Pyxu offers a new class of modular, scalable, and high-performing CI tools. It provides a comprehensive framework that combines cutting-edge features, seamless hardware acceleration, and deep learning interoperability to streamline the development and deployment of advanced imaging pipelines.

## Computational Imaging Overview

Computational imaging aims to reconstruct or restore a scene from partial and degraded measurements, transforming them into an image or volume of sufficient quality for visualization or further analysis [@McCann:2019, @Mait:2018, @Hansen:2010]. Common computational imaging tasks include:

- **Image Denoising**: The process of eliminating noise artifacts to create a cleaner, crisper image.
- **Image Deblurring**: Restoration of a sharp image from a blurry input, enhancing focus and detail.
- **Image Inpainting**: Reconstructing missing or damaged regions within an image, often used for tasks like replacing lost blocks during coding and transmission or erasing watermark/logo overlays.
- **Image Super-Resolution**: Elevating the resolution of an image or an imaging system to provide finer detail.
- **Image Fusion**: The merging of several degraded images of the same object into a single image that exceeds the quality of any individual input.
- **Image Filtering**: Modifying an image to promote particular features of interest, such as points, lines, or shapes.
- **Tomographic Reconstruction**: Rebuilding an image from its lower-dimensional projections, known as sinograms in the context of CT or PET scans.

![Common examples of computational imaging tasks.\label{fig:CI_tasks}](../../doc/_static/tutorial/recon_examples.jpg){ width=80% }

Despite differences, most imaging problems can be categorized into forms such as Fourier sampling, spatial sampling, and tomographic sampling. The foundation of these computational imaging methods is the resolution of mathematical inverse problems. This involves modeling a scene of interest through an acquisition system, like a microscope or camera, that captures data degraded by noise and optical aberrations of the system. 

Traditionally, imaging pipelines have often relied on direct inversion techniques for reversing the acquisition process and reconstruct or restore an image. These methods aim to approximate the inverse of the sensing operator modeling the acquisition system. 

While direct methods, which are typically linear, are fast and deployed in most imaging devices in research and industry, such as MRI [@Pruessmann:2006], X-ray tomography and variants [@Withers:2021], Cryo-EM tomography [@Young:2023], and light microscopy [@Sage:2017] they have limitations in terms of accuracy, often resulting in images with poor resolution and significant reconstruction artifacts [@Fessler:2010].

## Physics-informed computational imaging
To overcome the limitations of direct methods, physics-based iterative algorithms have been developed in parallel in many imaging domains, allowing for example more efficient reconstruction in MRI and Radio Astronomy [@Greengard:2004, @Arras:2021, @Wang:2023], deblurring in microscopy [@Guo:2020], and tomographic reconstruction for medical imaging, materials science, and geophysical imaging [@O'Connell:2021, @Koneti:2019, @Kim:2006, @Rawlinson:2005].

Physics-based iterative algorithms add robustness to noise and to the partial measurements via the integration of prior knowledge of the scene of interest. When combined with sparsity prior, for example, they enable the reconstruction of high-quality images from fewer measurements, as demonstrated in compressed sensing for MRI [@Lustig:2007].


## The era of machine learning
More recently, machine-learning-based methods, which rely on trained neural networks to further improve reconstruction quality [@Ye:2023], have markedly transformed computational imaging [@McCann:2019, @Ongie:2020, @Suzuki:2017, @Ye:2023, @Arridge:2019]. They do so by exploiting signal dependencies across space [@Chandra:2021] and time [@Yoo:2021]. This shift towards deep learning has driven computational imaging to unprecedented levels of performance [@Jin:2017, @Monga:2021, @Kamilov:2023], with some experts suggesting that we may have reached a plateau in terms of accuracy and performance [@Romano:2017]. Consequently, much of the current research in computational imaging is focused on incorporating within reconstruction pipelines. However, their integration into practical workflows is often hindered by existing software limitations: 

- Lack of Modularity: Many existing solutions are end-to-end, monolithic pipelines that do not permit easy incorporation of novel algorithms or adaptation to new or custom imaging geometries.
- Inefficient Data Handling: Most state-of-the-art reconstruction software for GPU-based iterative algorithms implement frequent GPU/CPU data transfers that introduce unnecessary computational overhead and compromise performance.
- Scalability Issues: Most state-of-the-art open-source software for image reconstruction struggle with scalability, i.e., they are unable to efficiently process very large datasets

## The era of big data
In parallel, advancements in hardware, particularly the development of more powerful sensors with improved quality and resolution, benefiting science production.

Such advancements significantly increase the data volumes to process, often exceeding memory limits [@Poger:2023, @Marone:2017], hence impose substantial computational challenges. This calls for the development of out-of-memory algorithms and the integration of GPU acceleration to ensure reasonable image reconstruction times.

## Pressing need for computational-imaging software

The current landscape of imaging pipelines is fragmented, with each domain tackling increasingly complex problems, often under real-time constraints. This fragmentation results in significant duplicate efforts across different fields, leading to wasted developer time and hampering the flow of advances from one field to another due to domain-specific barriers.

To address these inefficiencies, it is essential to abstract core logic, CI methods, and algorithmic advancements into an application-agnostic package that can benefit the entire community. 

These challenges underscore a pressing need for a computational imaging software framework that is:

- Modular: Capable of integrating the latest operator and algorithmic advances, particularly in machine learning and AI.
- Scalable and High-Performing: Able to handle very large-scale data efficiently, meeting the growing demands of scientists.
- User-Friendly and Robust: Facilitates maintenance, is accessible to non-experts, and can be widely deployed within existing imaging infrastructures.

# Pyxu

[Pyxu](https://pyxu-org.github.io) is a Python-based framework designed from the ground-up to meet these advanced demands. It extends beyond the capabilities of specialized frameworks by offering a comprehensive, modular solution tailored to the broad and varied needs of the computational imaging community at large. It aims to speed up the development and deployment of complex imaging pipelines, incorporating cutting-edge features such as operator algebra logic, seamless hardware acceleration, deep learning interoperability, and out-of-core compute, while being maintainable and easy to extend [plugins].
It is distinguished by its comprehensive support for hardware acceleration and distributed computing, encapsulating a suite of features designed to streamline the development and execution of computational imaging tasks across various modalities. Below, we highlight its key features:

- User-Friendly Interface: Accessible to researchers and practitioners with varying expertise. Extensive documentation and tutorials ensure smooth onboarding and rapid prototyping.
- Quick Prototyping: Allows for the rapid development of imaging pipelines.
- Robust and Accurate: Ensures robustness and accuracy via advanced testing and automatic hyperparameter setting.
- Seamless Integration with PyData Stack: Utilizes familiar tools and libraries such as NumPy, CuPy, and Dask.


### Operator Algebra Logic:

Pyxu's operator algebra logic enables the construction of tailored operators from simpler components to solve customized inverse problems via advanced optimization algorithms. 

As an example, consider the composition of a differentiable functional (`DiffFunc`) $\mathcal{f}: \mathbb{R}^{N}\rightarrow\mathbb{R}$ with a differentiable map (`DiffMap`) $\mathcal{L}: \mathbb{R}^{M}\rightarrow\mathbb{R}^{N}$. Then their composition $\mathcal{h} = \mathcal{f}\circ \mathcal{L}$ (or `h = f * L` in Pyxu) is also a differentiable functional `DiffFunc`, with gradient given by:

`h.grad(x) = L.jacobian(x).adjoint(f.grad(L(x)))`

This setup allows Pyxu's automated propagation of Lipschitz constants, ensuring that algorithms can automatically adjust their step-sizes for optimal convergence rates, eliminating the tedious task of manual tuning.

### Microservice Architecture:

Pyxu's architectural design is centered around offering a set of independent, application-agnostic, loosely coupled modules, such as mathematical operators, optimization algorithms, and Bayesian tools, each optimized for performance across different hardware backends and for out-of-core computing.

This design enables users to easily craft computational imaging pipelines to their specific needs, irrespective of their application domain.

This architecture allows easy adoption of the latest CI advancements across fields, while decreasing maintenance costs. 

<img src="../../doc/_static/microservice_hero.png" alt="Microservice architecture." style="width: 60%;">

### Seamless Hardware Acceleration:

Pyxu is designed to execute computational imaging (CI) tasks efficiently on both CPU and GPU architectures, including distributed arrays for multi-node environments. Built around the concept of duck arrays, Pyxu enables running CI pipelines on different platforms by simply supplying the appropriate array type to the operators. This design allows Pyxu to smoothly switch between CPU-bound NumPy arrays and GPU-bound array backends—employing NumPy [@Harris:2020] for CPU operations and CuPy [@Nishino:2017] for GPU tasks—utilizing standardized array APIs to achieve modularity and portability.

When GPU execution is desired, Pyxu's architecture minimizes or completely avoids data transfers between GPUs and CPUs, addressing a common efficiency bottleneck in many computational imaging frameworks. By maintaining data exclusively on the GPU throughout the processing pipeline, Pyxu significantly reduces runtimes and enhances its capability to handle the extensive computations in iterative algorithms.

### High-Performance Computing (HPC) Features:

Pyxu's supports distributed computing workloads to address the unique challenges presented by large-scale datasets which surpass single-machine memory limits. Built around Dask [@Rocklin:2015], large computations are broken down into a computational graph operating (in parallel) on *chunks* of data at any point in time. This enables Pyxu to execute imaging computations effectively across a diverse range of hardware setups, from multi-core CPUs to distributed GPU arrays.

Pyxu's operators are inherently vectorized, optimizing the processing of batches of data in parallel to exploit data-level parallelism. This, combined with Just-in-Time (JIT) compilation via Numba [@Lam:2015] for compute-intensive operations, allows Pyxu to achieve performance on par with codes written in compiled languages.

### Advanced Algorithmic Suite

Pyxu features an extensive library of state-of-the-art algorithms for solving inverse problems in computational imaging. This includes efficient approaches for traditional direct inversion techniques, advanced primal-dual splitting algorithms with acceleration and relaxation for iterative schemes, and recent deep learning-based methods such as unrolled networks, deep priors, and plug-and-play priors. Pyxu provides a comprehensive toolkit for tackling a wide range of imaging challenges.

Using Pyxu does not require rewriting existing processing pipelines from scratch. Instead, users can leverage Pyxu's interoperability with the PyData stack and leading deep learning frameworks like JAX [@Bradbury:2018] and PyTorch [@Paszke:2019], allowing them to integrate and use the components best suited for their specific tasks.


### Quality Assurance and Control:

Pyxu ensures the correctness and reliability of its components through rigorous testing practices using Pytest-based test suites. These suites comprehensively test the API and components, ensuring functional and logical correctness across all configurations.

Key quality assurance measures include:

- Mathematical Testing: Conducts mathematical tests for the correctness of the adjoint (for linear operators, the accuracy of Lipschitz constant and other estimations to ensure algorihms behave as expected and that convergence guarantees are respected.
- Backend testing: Tests all functions with NumPy, CuPy, and Dask as array backends, ensuring compatibility with major array libraries and architectures. Includes launching a Dask-distributed local cluster to comprehensively test distributed computing capabilities in a controlled environment.
- Precision testing: Ensures that precision is consistently maintained to the user needs during processing.

These rigorous testing practices ensure that Pyxu remains a reliable and robust framework for computational imaging.

### Community-Driven Development:
Pyxu is characterized by its application-agnostic nature, versatility, and open-source foundation with minimal dependencies. Central to enhancing Pyxu's adaptability and usability for specific applications is the Pyxu FAIR, a platform which streamlines the development, sharing, and integration of specialized plugins. By fostering community contributions and a collaborative environment, Pyxu FAIR significantly contributes to the expansion of the framework's capabilities, making sure it responds to the diverse needs of its user community.

Addressing the framework's broad applicability, Pyxu FAIR ensures Pyxu's adaptability to the requirements of various imaging communities. This initiative has facilitated the establishment of a user-friendly catalogue website, serving as a centralized hub for plugin discovery, exploration, and utilization, thereby enhancing Pyxu's adaptability and customization capabilities. Moreover, the introduction of a meta-programming framework (the Pyxu-Cookiecutter) alongside an interoperability protocol based on Python's Entrypoints, streamlines the plugin development process, ensuring seamless integration into Pyxu's core framework.

### Comparison with other Frameworks
This section provides a comparison between Pyxu and its main contenders in computational imaging frameworks, focusing on features, maturity, ease-of-use, and support for distributed and GPU computing.

#### Main Contenders
- PyLops [@Ravasi:2020]: An open-source Python library for matrix-free linear operators and related computations.
- PyProximal [@Ravasi:2024]: Proximal operators and algorithms in Python.
- ODL [@Adler:2017]: Enables research in inverse problems on realistic or real data.
- SCICO [@Balke:2022]: A JAX-powered Python package for solving inverse problems in scientific imaging.
- DeepInv [@Tachella:2023]: A PyTorch-based library for solving imaging inverse problems with deep learning.
- SigPy [@Ong:2019]: Focuses on signal processing, built on NumPy and CuPy.
- GlobalBioIm [@Unser:2017]: A Matlab framework for developing reconstruction algorithms in computational imaging.

#### Comparative Analysis
| Package Name  | Operator Types                           | Operator Algebra | Algorithmic Suite                 | Application Focus         | Auto Diff/Prox | GPU Computing | Out-of-core Computing | JIT Compiling |
|---------------|------------------------------------------|------------------|-----------------------------------|---------------------------|----------------|---------------|-----------------------|---------------|
| PyLops        | Linear operators                         | Partial          | Least-squares & sparse reconstructions | Wave-processing, geophysics | No             | Yes (CuPy)    | No                    | Partial (LLVM via Numba) |
| PyProximal    | Proximable functionals                   | None             | Non-smooth convex optimization    | None                      | No             | No            | No                    | No            |
| ODL           | (Non)linear operators, differentiable/proximable functionals | Full             | Smooth, non-smooth & hybrid (non-)convex optimization | None                      | Yes            | Very limited (CUDA) | No                    | No            |
| GlobalBioIm   | (Non)linear operators, differentiable/proximable functionals | Full             | Smooth, non-smooth & hybrid convex optimization | None                      | Yes            | Yes (MATLAB)  | No                    | No            |
| SigPy         | Linear operators, proximable functionals | Partial          | Smooth & non-smooth convex optimization | MRI                       | No             | Yes (CuPy)    | Manual (MPI)          | No            |
| SCICO         | (Non)linear operators, differentiable/proximable functionals | Full             | Smooth, non-smooth & hybrid (non-)


Note: This comparison excludes traditional medical imaging frameworks like TomoPy and ASTRA, as well as general-purpose optimization frameworks and deep learning frameworks. These exclusions are due to their architectural limitations or the narrow focus of their optimization methods, which are unsuitable for the diverse requirements of modern computational imaging.

# Conclusion

Pyxu represents a significant advancement in the field of computational imaging, offering a robust and modular solution to the challenges posed by processing large-scale image datasets. Its comprehensive suite of tools ax~nd features, combined with support for distributed computing and hardware acceleration, makes Pyxu an essential framework for researchers and practitioners in computational imaging.

Through the integration of Dask's distributed and out-of-core computing functionalities, along with advanced strategies for minimizing unnecessary data CPU/GPU transfers and leveraging vectorization and JIT compilation, Pyxu presents a robust framework for computational imaging. It is designed to simplify complex imaging tasks, ensuring scalability, ease-of-use, and efficiency in handling extensive imaging data.

# Acknowledgements

We extend our gratitude to all Pyxu contributors as detailed in the contributors list on GitHub. Special acknowledgment is given to the EPFL Center for Imaging and the EPFL LCAV research group for their invaluable support during the inception of this project. Additionally, we express our appreciation for the financial backing received from Meta, Carl ZEISS, and various programs facilitated by ETH and SNF.

# References
