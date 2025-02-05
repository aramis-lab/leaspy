# Glossary

The Glossary provides short definitions of concepts as well as Leaspy specific vocabulary.

If you wish to add a missing term, please create an issue or open a Merge Request.

## API

An [Application Programming Interface](https://en.wikipedia.org/wiki/API) is a way for two or more computer programs to communicate with each other.
It is a type of software interface, offering a service to other pieces of software.
In the case of Leaspy, the API describes the public classes and functions that are exposed to client code.

## Attachment

The attachment is a diminutive for attachment to data.
It is a term of the [log-likelihood](#likelihood) that describes how close to the real data the model is.

## Biomarker

In biomedical contexts, a [biomarker](https://en.wikipedia.org/wiki/Biomarker), or biological marker, is a measurable indicator of some biological state or condition.
Biomarkers are often measured and evaluated using blood, urine, or soft tissues to examine normal biological processes, pathogenic processes, or pharmacologic responses to a therapeutic intervention.

## Calibration

Is the process that computes the population parameters, the [fixed effects](#fixed-effects-model) of the model.
It is done by a [likelihood](#likelihood) maximisation using an [MCMC-SAEM](#mcmc-saem) algorithm.

## Estimation

The estimation consists in computing the trajectory of a given patient thanks to population parameters (computed during the [calibration](#calibration) step) and individual parameters (computed during [personalization](#personalization) step).

## Fixed effects model

In statistics, a [fixed effects model](https://en.wikipedia.org/wiki/Fixed_effects_model) is a statistical model in which the model parameters are fixed or non-random quantities.
This is in contrast to [random effects models](#random-effects-model) and [mixed models](#mixed-model) in which all or some of the model parameters are random variables.

## JSON

[JavaScript Object Notation](https://en.wikipedia.org/wiki/JSON) (JSON) is an open standard file format and data interchange format that uses human-readable text to store and transmit data objects consisting of attribute–value pairs and arrays (or other serializable values).
JSON is a language-independent data format. It was derived from JavaScript, but many modern programming languages include code to generate and parse JSON-format data. JSON filenames use the extension ``.json``.

## Likelihood

Is the probability of the data knowing the model parameters.
The log-likelihood could be separated into two parts:

- the data [attachment](#attachment) term which describes how close the model is to the data.
- the [regularity](#regularity) term which corresponds to how far from the priors the model is.

## MCMC

In statistics, [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) methods comprise a class of algorithms for sampling from a probability distribution.
By constructing a Markov chain that has the desired distribution as its equilibrium distribution, one can obtain a sample of the desired distribution by recording states from the chain.
The more steps that are included, the more closely the distribution of the sample matches the actual desired distribution.
Various algorithms exist for constructing chains, including the Metropolis–Hastings algorithm.

## MCMC-SAEM

The [MCMC-SAEM](https://hal.science/hal-00189580/document) is a powerful algorithm used to estimate maximum [likelihood](#likelihood) in the wide class of exponential non-linear mixed effects models.

## Mixed model

A [mixed model](https://en.wikipedia.org/wiki/Mixed_model), "mixed-effects model" or "mixed error-component model" is a statistical model containing both [fixed effects](#fixed-effects-model) and [random effects](#random-effects-model).

## Mixing matrix

The matrix involved in the dimensionality reduction of the [space shifts](#space-shifts) through an [Independant Component Analysis (ICA)](https://en.wikipedia.org/wiki/Independent_component_analysis).
This matrix is used to construct the [space shifts](#space-shifts) by applying it to a vector of [sources](#sources).

## MMSE

The [Mini Mental State Examination](https://en.wikipedia.org/wiki/Mini–mental_state_examination) (or Folstein test) is a 30-point questionnaire that is used extensively in clinical and research settings to measure cognitive impairment.
It is commonly used in medicine and allied health to screen for dementia.

## Overfitting

In mathematical modeling, [overfitting](https://en.wikipedia.org/wiki/Overfitting) is the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit to additional data or predict future observations reliably.
An overfitted model is a mathematical model that contains more parameters than can be justified by the data.

## Personalization

Once the population parameters ([fixed effects](#fixed-effects-model)) have been computed, the user might want to know the individual parameters ([random effects](#random-effects-model)) of a given patient that are necessary in order to compute his/her individual trajectory ([estimation](#estimation) step).

## Random effects model

In statistics, a [random effects model](https://en.wikipedia.org/wiki/Random_effects_model), also called a "variance components model", is a statistical model where the model parameters are random variables.
A random effects model is a special case of a [mixed model](#mixed-model).

## Regularity

Is a term of the [log-likelihood](#likelihood) that describes to how far from the priors the parameters of the model are.
It is sort of a bayesian version of the machine learning [regularization](#regularization).

## Regularization

In mathematics, statistics, finance, computer science, particularly in machine learning and inverse problems, [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) is a process that changes the result answer to be "simpler".
It is often used to obtain results for ill-posed problems or to prevent [overfitting](#overfitting).

## Sources

Vector stemming from the [dimensionality reduction](#mixing-matrix) of the [space shifts](#space-shifts).
The sources are one of the [random effects](#random-effects-model) that are estimated by the model to fit the reference disease trajectory to each patient.

## Spatial effects

See [space shifts](#space-shifts).

## Space shifts

Also referred to as the [spatial effects](#spatial-effects) of our model, the space shifts capture the variability in the disease that is orthogonal to the temporal reparametrization.
That is to say that two subjects could be modelled at the same stage of the disease and with the same progression rate but still exhibit different patterns (for instance in the ordering of the abnormal features). They are usually denoted `w_i`.

## Sufficient statistics

In statistics, a statistic is [sufficient](https://en.wikipedia.org/wiki/Sufficient_statistic) with respect to a statistical model and its associated unknown parameter if "no other statistic that can be calculated from the same sample provides any additional information as to the value of the parameter".
In particular, a statistic is sufficient for a family of probability distributions if the sample from which it is calculated gives no additional information than the statistic, as to which of those probability distributions is the sampling distribution.
