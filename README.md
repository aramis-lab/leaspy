# Leaspy - LEArning Spatiotemporal Patterns in Python


[![pipeline status](https://gitlab.com/icm-institute/aramislab/leaspy/badges/master/pipeline.svg)](https://gitlab.com/icm-institute/aramislab/leaspy/commits/master)


>  DISCLAIMER : This version is intended for beta-testing yet. There might remain bugs, therefore, use it precautiously, especially the results that you obtain.


## Description
Leasp is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points. 
Considering these series of short-term data, the software aims at : 
- recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- positioning each patient observations relatively to the group-average timeline, in term of both temporal differences (time shift and acceleration factor) and spatial differences (diffent sequences of events, spatial pattern of progression, ...)
- quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- imputing missing values
- predicting future observations
- simulating virtual patients to unbias the initial cohort or mimis its characteristics


#### Scalar inputs
The software package can be used with scalar multivariate data whose progression can be modeled by a logistic shape, an exponential decay or a linear progression.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers).


#### Further information
More detailed explanations about the models themselves and  about the estimation procedure can be found in the following articles : 

- A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations. Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman.  The Journal of Machine Learning Research, 18:1–33, December 2017. [Open Access PDF](https: //hal.archives-ouvertes.fr/hal-01540367).
- Statistical learning of spatiotemporal patterns from longitudinal manifold-valued networks. I. Koval, J.-B. Schiratti, A. Routier, M. Bacci, O. Colliot, S. Allassonnière and S. Durrleman. MICCAI, September 2017. [Open Access PDF](https://arxiv.org/pdf/1709.08491.pdf)
- Spatiotemporal Propagation of the Cortical Atrophy: Population and Individual Patterns. Igor Koval, Jean-Baptiste Schiratti, Alexandre Routier, Michael Bacci, Olivier Colliot, Stéphanie Allassonnière, and Stanley Durrleman. Front Neurol. 2018 May 4;9:235. Open Access PDF

## Supported features
- `fit` : estimate **population parameters** of the longitudinal progression model
- `personalize` : estimate **individual parameters** of patients
- `simulate` : generate synthetic data from the model

## Get started Leaspy
#### Requirements
- Mac and Linux - check for windows

#### Installation
1. Obtain the sources by :

`git clone https://gitlab.com/icm-institute/aramislab/leaspy.git`

2. Install the dependencies:

`conda install pytorch torchvision -c pytorch`

`conda install matplotlib pandas scikit-learn scipy`

### Examples & Tutorials
The `example` folder contains a relevant number of example to start with. The `start` should be your first introduction, especially as it is better described in a [Medium post](https://medium.com/@igoroa/analysis-of-longitudinal-data-made-easy-with-leaspy-f8d529fcb5f8)

### Documentation
[Coming soon]

### Website
[Coming soon]

## Support

The development of this software has been supported by the European Union H2020 program (project EuroPOND, grant number 666992, project HBP SGA1 grant number 720270), by the European Research Council (to Stanley Durrleman project LEASP, grant number 678304) and by the ICM Big Brain Theory Program (project DYNAMO).

## Licence

[Coming soon]

## Contacts
http://www.aramislab.fr/
