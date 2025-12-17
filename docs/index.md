![leaspy logo](./_static/images/leaspy_logo.png)

## LEArning Spatiotemporal Patterns in Python

### Description

**Leaspy** is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.

```{figure} ./_static/images/leaspy_front.png
:name: fig-leaspy-front
:alt: Leaspy Front Illustration

Figure 1: Illustration of the Disease Progression Model underlying Leaspy. (a) illustrates how sparse individual observations (colored dots) can be recombined into a common population trajectory (in black), while accounting for the fact that each subject may progress earlier/later or faster/slower than the average (the green individual progresses earlier, and the blue one slower than the average). (b) shows the corresponding geometric interpretation: individual data points are mapped onto a shared trajectory on a Riemannian manifold. Individual trajectories are then computed through subject-specific temporal transformations (time-shift $\tau_i$ and acceleration $\alpha_i$), and spatial transformations ($w_i$) capturing differences in the ordering and pattern of deterioration across outcomes.
```

Figure 1 provides an intuitive overview of the objective of the software. By estimating the temporal and spatial transformations, Leaspy positions each patient relative to the group-average disease timeline, enabling meaningful comparison across individuals despite heterogeneous follow-ups.

The framework further allows:
- Quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal;
- Imputing missing values;
- Predicting future observations;
- Simulating virtual patients to un-bias the initial cohort or mimic its characteristics.

The software package can be used with scalar multivariate data whose progression can be described by a logistic, linear, joint, or mixture model.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers) but also imaging-derived data that are rescaled, for instance, between 0 and 1 to describe a logistic progression.

#### Getting started

Information to install, test, and contribute to the package are available [here](./install.md).

#### API Documentation

The exact [API](./glossary.md#api) of all functions and classes, as given in the docstrings.
The [API](./glossary.md#api) documents expected types and allowed features for all functions, and all parameters available for the algorithms.

#### User Guide

The main documentation. This contains an in-depth description of all algorithms and how to apply them.

#### License

The package is distributed under the BSD-3-Clause-Clear license.

### Further information

More detailed explanations about the models themselves and about the estimation procedure can be found in the following articles :

- **Mathematical framework**: *A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations.* Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman. The Journal of Machine Learning Research, 18:1–33, December 2017. [Open Access](https://hal.archives-ouvertes.fr/hal-01540367)
- **Mixture Model**: *A mixture model for subtype identification: Application to CADASIL* Sofia Kaisaridi, Juliette Ortholand, Caglayan Tuna, Nicolas Gensollen, and Sophie Tezenas Du Montcel
  ISCB 46-46th Annual Conference of the International Society for Clinical Biostatistics, August 2025. [Open Access](https://hal.science/hal-05266776v1)
- **Joint Model**: *Joint model with latent disease age: overcoming the need for reference time* Juliette Ortholand, Nicolas Gensollen, Stanley Durrleman, Sophie Tezenas du Montcel. arXiv preprint arXiv:2401.17249. 2024 [Open Access](https://arxiv.org/abs/2401.17249)
- [www.digital-brain.org](https://project.inria.fr/digitalbrain/) : Website related to the application of the model for Alzheimer's disease.

```{toctree}
 :hidden:
:caption: Getting Started
:maxdepth: 2
install
```

```{toctree}
:hidden:
:caption: Examples
:maxdepth: 2

auto_examples/index

```

```{toctree}
:hidden:
:caption: Documentation
:maxdepth: 3

user_guide
glossary
notations
references
changelog
license
```
