![leaspy logo](./_static/images/leaspy_logo.png)


## LEArning Spatiotemporal Patterns in Python

### Description

**Leaspy** is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.

![leaspy front](./_static/images/leaspy_front.png)

Considering these series of short-term data, the software aims at :

- Recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- Positioning each patient observations relatively to the group-average timeline, in term of both temporal differences (time shift and acceleration factor) and spatial differences (different sequences of events, spatial pattern of progression, ...)
- Quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- Imputing missing values
- Predicting future observations
- Simulating virtual patients to un-bias the initial cohort or mimic its characteristics

The software package can be used with scalar multivariate data whose progression can be described by a logistic model, linear, joint or mixture model.
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

The package is distributed under the GNU LESSER GENERAL PUBLIC LICENSE.

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