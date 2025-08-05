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

Leaspy requires Python >= 3.9, < 3.13.

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

If you haven't done it already, create and activate a dedicated environment (see the beginning of the installation section). 

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

The `example/start/` folder contains a starting point if you want to launch your first scripts and notebook with the Leaspy package.
You can find additional walkthroughs in:
- this series of [online tutorials](https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/disease_course_mapping.html) from 2020
- this [Medium post](https://medium.com/@igoroa/analysis-of-longitudinal-data-made-easy-with-leaspy-f8d529fcb5f8) of 2019 (_warning_: the plotter and the individual parameters described there have been deprecated since then)

## Description
Leaspy is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.
Considering these series of short-term data, the software aims at :
- recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- positioning each patient observations relatively to the group-average timeline, in terms of both temporal differences (time shift and acceleration factor) and spatial differences (different sequences of events, spatial pattern of progression, ...)
- quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- imputing missing values
- predicting future observations
- simulating virtual patients to un-bias the initial cohort or mimics its characteristics

The software package can be used with scalar multivariate data whose progression can be modelled by a logistic shape, an exponential decay or a linear progression.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers) but also imaging-derived data that are rescaled, for instance, between 0 and 1 to describe a logistic progression.

### Main features
- `fit` : determine the **population parameters** that describe the disease progression at the population level
- `personalize` : determine the **individual parameters** that characterize the individual scenario of biomarker progression
- `estimate` : evaluate the biomarker values of a patient at any age, either for missing value imputation or future prediction
- `simulate` : generate synthetic data from the model

### Further information
More detailed explanations about the models themselves and about the estimation procedure can be found in the following articles :

- **Mathematical framework**: *A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations*. Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman. The Journal of Machine Learning Research, 18:1–33, December 2017. [Open Access](https://hal.archives-ouvertes.fr/hal-01540367).
- **Application to imaging data**: *Statistical learning of spatiotemporal patterns from longitudinal manifold-valued networks*. I. Koval, J.-B. Schiratti, A. Routier, M. Bacci, O. Colliot, S. Allassonnière and S. Durrleman. MICCAI, September 2017. [Open Access](https://hal.archives-ouvertes.fr/hal-01540828)
- **Application to imaging data**: *Spatiotemporal Propagation of the Cortical Atrophy: Population and Individual Patterns*. Igor Koval, Jean-Baptiste Schiratti, Alexandre Routier, Michael Bacci, Olivier Colliot, Stéphanie Allassonnière, and Stanley Durrleman. Front Neurol. 2018 May 4;9:235. [Open Access](https://hal.inria.fr/hal-01910400)
- **Application to data with missing values**: *Learning disease progression models with longitudinal data and missing values*. R. Couronne, M. Vidailhet, JC. Corvol, S. Lehéricy, S. Durrleman. ISBI, April 2019. [Open Access](https://hal.archives-ouvertes.fr/hal-02091571)
- **Intensive application for Alzheimer's Disease progression**: *AD Course Map charts Alzheimer's disease progression*, I. Koval, A. Bone, M. Louis, S. Bottani, A. Marcoux, J. Samper-Gonzalez, N. Burgos, B. Charlier, A. Bertrand, S. Epelbaum, O. Colliot, S. Allassonniere & S. Durrleman, Scientific Reports, 2021. 11(1):1-16 [Open Access](https://hal.inria.fr/hal-01964821)
- [www.digital-brain.org](https://www.digital-brain.org): website related to the application of the model for Alzheimer's disease
- [Disease Course Mapping](https://disease-progression-modelling.github.io/pages/models/disease_course_mapping.html) webpage by Igor Koval

## License
The package is distributed under the BSD 3-Clause license.

## Support
The development of this software has been supported by the European Union H2020 program (project EuroPOND, grant number 666992, project HBP SGA1 grant number 720270), by the European Research Council (to Stanley Durrleman project LEASP, grant number 678304) and by the ICM Big Brain Theory Program (project DYNAMO).

Additional support has been provided by the REWIND project (pRecision mEdicine WIth loNgitudinal Data), as part of the PEPR Santé Numérique (grant number 2023000354-03), dedicated to advancing precision medicine with longitudinal data.

## Contact
[ARAMIS Lab](https://www.aramislab.fr/)
