[![Tests](https://github.com/aramis-lab/leaspy/actions/workflows/test.yaml/badge.svg)](https://github.com/aramis-lab/leaspy/actions/workflows/test.yaml)
[![documentation status](https://readthedocs.org/projects/leaspy/badge/)](https://leaspy.readthedocs.io)
[![PyPI - license](https://img.shields.io/pypi/l/leaspy)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI - version](https://img.shields.io/pypi/v/leaspy)](https://pypi.org/project/leaspy/)
[![PyPI - downloads](https://img.shields.io/pypi/dm/leaspy)](https://pypi.org/project/leaspy/)
[![PyPI - versions](https://img.shields.io/pypi/pyversions/leaspy)](https://pypi.org/project/leaspy/)

# Leaspy - LEArning Spatiotemporal Patterns in Python

Leaspy is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.

## Get started Leaspy

### Installation

Leaspy requires Python >= 3.9, <= 3.13.

Whether you wish to install a released version of Leaspy, or to install its development version, it is **highly recommended** to use a virtual environment to install the project and its dependencies.

There exists multiple solutions for that, the most common option is to use `conda`:

```bash
conda create --name leaspy python=3.10
conda activate leaspy
```

#### Install a released version

To install the latest version of Leaspy:

```bash
pip install leaspy
```

#### Install in development mode

If you haven't done it already, create and activate a dedicated environment.

**Clone the repository**

To install the project in development mode, you first need to get the source code by cloning the project's repository:

```bash
git clone git@github.com:aramis-lab/leaspy.git
cd leaspy
```

**Install poetry**

This project relies on [poetry](https://python-poetry.org) that you would need to install (see the [official instructions](https://python-poetry.org/docs/#installation)).

It is recommended install it in a dedicated environment, separated from the one in which you will install Leaspy and its dependencies. One possibility is to install it with a tool called [pipx](https://pipx.pypa.io/stable/).

If you don't have `pipx` installed, already, you can follow the [official installation guidelines](https://pipx.pypa.io/stable/installation/).

In short, you can do:

```bash
pip install pipx
pipx ensurepath
pipx install poetry
```

**Install Leaspy and its dependencies**

Install leaspy in development mode:

```bash
poetry install
```

**Install the pre-commit hook**

Once you have installed Leaspy in development mode, do not forget to install the [pre-commit](https://pre-commit.com) hook in order to automatically format and lint your commits:

```bash
pipx install pre-commit
pre-commit install
```

### Documentation

[Available online at _Readthedocs.io_](https://leaspy.readthedocs.io)

### Examples & Tutorials

The `examples` folder contains a starting point if you want to launch your first scripts and notebook with the Leaspy package.

## Description
Leaspy is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.
Considering these series of short-term data, the software aims at :
- recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- positioning each patient observations relatively to the group-average timeline, in terms of both temporal differences (time shift and acceleration factor) and spatial differences (different sequences of events, spatial pattern of progression, ...)
- quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- imputing missing values
- predicting future observations
- simulating virtual patients to un-bias the initial cohort or mimics its characteristics

The software package can be used with scalar multivariate data whose progression can be described by a logistic model, linear, joint or mixture model.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers) but also imaging-derived data that are rescaled, for instance, between 0 and 1 to describe a logistic progression.

### Main features
- `fit` : determine the **population parameters** that describe the disease progression at the population level
- `personalize` : determine the **individual parameters** that characterize the individual scenario of feature progression
- `estimate` : evaluate the feature values of a patient at any age, either for missing value imputation or future prediction
- `simulate` : generate synthetic data from the model

### Further information
More detailed explanations about the models themselves and about the estimation procedure can be found in the following articles :

- **Mathematical framework**: *A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations*. Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman. The Journal of Machine Learning Research, 18:1–33, December 2017. [Open Access](https://hal.archives-ouvertes.fr/hal-01540367).
- **Mixture Model**: *A mixture model for subtype identification: Application to CADASIL* Sofia Kaisaridi, Juliette Ortholand, Caglayan Tuna, Nicolas Gensollen, and Sophie Tezenas Du Montcel. ISCB 46-46th Annual Conference of the International Society for Clinical Biostatistics, August 2025. [Open Access](https://hal.science/hal-05266776v1)
- **Joint Model**: *Joint model with latent disease age: overcoming the need for reference time* Juliette Ortholand, Nicolas Gensollen, Stanley Durrleman, Sophie Tezenas du Montcel. arXiv preprint arXiv:2401.17249. 2024 [Open Access](https://arxiv.org/abs/2401.17249)

## License
The package is distributed under the GNU LESSER GENERAL PUBLIC LICENSE.

## Support
The development of this software has been supported by the European Union H2020 program (project EuroPOND, grant number 666992, project HBP SGA1 grant number 720270), by the European Research Council (to Stanley Durrleman project LEASP, grant number 678304) and by the ICM Big Brain Theory Program (project DYNAMO).

Additional support has been provided by the REWIND project (pRecision mEdicine WIth loNgitudinal Data), as part of the PEPR Santé Numérique (grant number 2023000354-03), dedicated to advancing precision medicine with longitudinal data.

## Contact
[ARAMIS Lab](https://www.aramislab.fr/)
