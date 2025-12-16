# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Leaspy version 2

These are the releases of the version 2 of Leaspy.

### [2.0.0] - 2025-12-16

This is the official release of Leaspy version 2.

- **The new structure is a result of a global refactoring.**
  
  - This breaking change is the result of a major refactoring of how model parameters and variables are handled. 
  - The new architecture is more modular and is designed to simplify the definition and extension of models compared to v1. 
  - The codebase has been structured to closely mirror the mathematical formulation of the models. 
  - At its core, distinct classes are implemented, following a well-defined hierarchy of dependencies and inheritance, for the model and the algorithm class, using a Directed Acyclic Graph (DAG) and a 'family' concept. 
  - This modular and transparent architecture ensures clarity, extensibility, and consistency, while its straightforward structure greatly simplifies the development and integration of new model or algorithm variants. 
  - A State builds on this structure: it is a DAG with an additional mapping between node names and their current values, effectively holding both the model's blueprint and the values currently loaded.

- **New models are added:**
  - `joint`: for the joint modelisation of patient outcomes and events
  - `mixture`: for the clustering of patients with similar spatiotemporal profiles

- **Documentation is updated and completed.**

  The documentation page that comes with this release provides a detailed description of the mathematical intuition behind leaspy, the models and the algorithms implemented and the interpretation of the results. It also comes with an example gallery with complete notebooks fitting different models with synthetic data.

```{warning}
Leaspy v2 is not backward compatible with v1. The public API has changed, and models serialized with Leaspy v1 cannot be loaded in v2 (and vice versa).
```

### [2.0.0rc3] - 2025-09-18

This is the final release candidate before the official release of Leaspy version 2.

This version includes some improvements over 2.0.0rc2.

Please report any issues or unexpected behaviors to help us finalize the stable v2.0.0 release.

### [2.0.0rc2] - 2025-07-02

This is the second release candidate.

This version includes some improvements over 2.0.0rc1.

Please report any issues or unexpected behaviors to help us finalize the stable v2.0.0 release.

### [2.0.0rc1] - 2025-06-02

This is the first release candidate of the version 2 of Leaspy.

```{warning}
There is no backward compatibility with the version 1 of Leaspy.
This means that the public API is different and that models serialized with the version 1 of Leaspy won't be compatible with version 2 and vice-versa.
```

Leaspy v2 introduces a whole new logic in the way model parameters and variables are handled, with the objective of being more modular than version 1 and ease the definition of new models.

The main idea behind the new model specification scheme lies in the concepts of `State` and `DAG` which are two core components of Leaspy v2:

- A `DAG` is a directed acyclic graph defining the symbolic relationships between parameters, hyperparameter, and variables in a model. It is basically the blueprint of your model.
- A `State` is a `DAG` with an additional mapping between node names and current values. A `State` basically holds the model's blueprint as well as the values that are currently loaded.

The fact that a model in Leaspy v2 is almost equivalent to its `DAG` makes it very easy for users to define a new model as relationships between variables are translated in a very transparent way.

```{note}
The actual release of Leaspy `v2.0.0` is an ongoing work that would benefit from user inputs. Please open issues if you had problems using this version of Leaspy or if you encountered bugs doing so.
```

## Leaspy version 1

These are the releases of the version 1 of Leaspy.

```{warning}
If you are a user of Leaspy v1, it is recommended to upgrade to v2 since the maintenance of the version 1 of Leaspy will end in 2025.
```

### [1.5.0] - 2024-06-03

- [COMPAT] No more support for Python 3.7 and 3.8 (see MR !99)
- [COMPAT] Support for Python 3.10 and 3.11 (see MR !99 and !121)
- [DOC] New glossary page added to the documentation (see MR !95)
- [REF] Refactoring of Samplers (see MR !93)
- [REF] Refactoring of Noise Models (see MR !91)
- [REF] Refactoring of the Univariate Model (see MR !89)
- [REF] Refactoring to decouple Algorithms and Models (see MR !90)
- [REF] Refactoring of Realizations (see MR !92)

### [1.4.0] - 2022-11-21

- [FEAT] New ordinal models (see MR !73)
- [FEAT] New `"Metropolis-Hastings"` and `"FastGibbs"` samplers for population variables in MCMC algorithms
- [REFACT-FEAT] Refact and improve `Data` and `IndividualData` classes

### [1.3.1] - 2022-03-11

- [FIX] Inconsistent warning about annealing.n_iter / n_iter_frac
- [FIX/CHORE] Fix minor typos without functional impact
- [COMPAT] PyTorch 1.11 was tested and is compatible

### [1.3.0] - 2022-03-09

- [FIX] Fix critical regression on `AlgorithmSettings.save`
- [FIX] Fix some presentation issues with convergence plots
- [FIX/FEAT] Raise a `LeaspyConvergenceError` if variances of params collapse to zero
- [FIX] Improve and robustify `AlgorithmSettings.set_logs`
- [FIX] `AlgorithmSettings` dict parameters are now recursively merged with defaults parameters
- [FIX] Fix 'mode_real' and 'mean_real' personalization algorithms (bug with initial temperature / annealing)
- [PERF] Slightly improve performance of Gibbs samplers and fit algorithm
- [PERF] Initial adaptative std-dev used in Gibbs samplers is now parameter dependent (i.e. scaled) to speed-up convergence
- [FEAT] In `ScipyMinimize` algorithm, user can now tune parameters sent to `scipy.optimize.minimize` and customize how convergence issues are logged
- [FEAT] Hyper-parameters of the samplers can now be tuned in the 'mcmc_saem', 'mode_real' and 'mean_real' algorithms
- [FEAT] The `n_burn_in_iter_frac` and `annealing.n_iter_frac` parameters were introduced to dynamically adjust the fraction of iterations independently of `n_iter` parameter (for 'mcmc_saem', 'mode_real' and 'mean_real')
- [FEAT] The computed RMSE at end of fit with Bernoulli noise model is now per feature
- [FEAT] More models / individual parameters in `leaspy.Loader`
- [DOC] Improve documentation & add a docstrings validator
- [CHORE] Clean-up code & tests
- [TESTS] Much more functional tests for all the models & personalization algorithms supported
- [COMPAT] PyTorch >=1.7.1 is now required

### [1.2.0] - 2021-12-17

- [CODE] Broad use of type annotations in Leaspy package
- [COMPAT] As a consequence support of Python 3.6 is dropped
- [COMPAT] PyTorch >=1.7 is now supported, as well as Python 3.9
- [FEAT] Custom Leaspy exceptions raised in code: cf. `leaspy.exceptions` in documentation
- [FEAT] Implementation of model _inverse_ in API to get age according to model for a given feature value: `Leaspy.estimate_ages_from_biomarker_values`
- [FIX] Simulation algorithm is fixed (shape issue with noise and bad behavior for random visits <= 0)
- [REFACT] Configuration of noise structure was reshaped:
  - models do not have a `loss` hyperparameter any more, it was replaced by `noise_model` (backward-compatibility is ensured)
  - algorithms do not have a `loss` parameter any more, it is fully controlled by noise structure of model
- [FEAT] Simulation algorithm now supports new keywords (cf. `SimulationAlgorithm`) to better control:
  - delay between simulated visits (can be random, constant or defined with a custom function)
  - number of simulated visits (possible to set min & max number of visits when random)
  - noise structure, in line with new `noise_model` implementation
- [DEFAULTS] Some default configuration values changed for some algorithms & models:
  - LME model now has `with_random_slope_age` = True by default
  - `mcmc_saem` and `scipy_minimize` now have `progress_bar` = True by default
  - `scipy_minimize` now has `use_jacobian` = True by default (fallback to False when not implemented, with a warning)
  - Multivariate models now have `gaussian_diagonal` noise structure by default
  - `simulation` algorithm now has constant delay of 6 months between simulated visits by default
- [DEPRECATION] Some classes were deprecated:
  - `VisualizationToolbox` class was removed from code
  - `Plotting` class was deprecated and removed from Leaspy API
  - Some already deprecated algorithms are not supported any more (moved in `algo/_legacy` folder)
- [BROWSER] Browser web-app was improved
- [LICENSE] Changing GPL license to BSD 3-Clause license - active only for current and future releases
- [REFACT] Readers now implement more checks on input data; CSVDataReader now calls `pandas.read_csv` internally
- [TESTS] Refactoring of tests with a new `LeaspyTestCase`, have a look at `tests/README.md` if you want to add or modify them

### [1.1.2] - 2021-04-13

- **Fix computation of orthonormal basis for leaspy multivariate models:**
  - **<!> this modification is breaking old multivariate models for which orthogonality**
    **between space-shifts and time-derivative of geodesic at initial time was not respected.**
  - To prevent misusing old saved models (with betas and sources being related to the old wrong
    orthonormal basis) with new code, we added the leaspy version in the model parameters when saved,
    you'll have to use leaspy 1.0.* to run old erroneous models and use leaspy >= 1.1 to run new models
- Change of the sequence `epsilon_k` in the updates of the sufficient statistics (after burn-in phase)
  to ensure the theoretical convergence
- No use of v0_mean hyperprior for multivariate models
- Uniformize tiny MCMC std priors on population parameters
- Better initialization of velocities for multivariate models
- New method for initialization `lme` (not used by default)
- In `AlgorithmSettings`:
  - `initialization_method` is renamed `algorithm_initialization_method`
  - for fit algorithms, you can define `model_initialization_method`
- Refactorization of models attributes
- Clean-up in code and in tests

### [1.0.3] - 2021-03-03

- Fix multivariate linear model
- Fix multivariate linear & logistic_parallel jacobian computation
- Update requirements.txt and add a `__watermark__`
- Add support for new torch versions (1.2.* and >1.4 but <1.7)
- Tiny fixes on starter notebook
- Tiny fix on `Plotting`
- Clean-up in documentation

### [1.0.2] - 2021-01-05

- Jacobian for all models
- Clean univariate models
- More coherent initializations

### [1.0.1] - 2021-01-04

- First released version
