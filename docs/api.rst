.. _api:

=================
API Documentation
=================

Full API documentation of the *Leaspy* Python package.


:mod:`leaspy.api`: **Main API**
===============================
The main class, from which you can instantiate and calibrate a model, personalize it to
a given set a subjects, estimate trajectories and simulate synthetic data.

.. currentmodule:: leaspy.api

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Leaspy

.. autoclass:: leaspy.api.Leaspy
   :no-members:
   :no-inherited-members:
   :no-undoc-members:
   :no-private-members:
   :noindex:
   :member-order: bysource

:mod:`leaspy.models`: **Models**
================================
Available models in `Leaspy`.

.. currentmodule:: leaspy.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AbstractModel
    AbstractMultivariateModel
    BaseModel
    ConstantModel
    GenericModel
    LMEModel
    ModelFactory
    MultivariateModel
    MultivariateParallelModel
    UnivariateModel

:mod:`leaspy.models.noise_models`: **Noise Models**
---------------------------------------------------
Available noise models in `Leaspy`.

.. currentmodule:: leaspy.models.noise_models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DistributionFamily
    AbstractGaussianNoiseModel
    AbstractOrdinalNoiseModel
    BaseNoiseModel
    BernoulliNoiseModel
    GaussianDiagonalNoiseModel
    GaussianScalarNoiseModel
    OrdinalNoiseModel
    OrdinalRankingNoiseModel

.. autosummary::
    :toctree: generated/
    :template: function.rst

    noise_model_factory
    export_noise_model

:mod:`leaspy.models.utils.attributes`: **Models' attributes**
-------------------------------------------------------------
Attributes used by the models.

.. currentmodule:: leaspy.models.utils.attributes

.. autosummary::
    :toctree: generated/
    :template: class.rst

    attributes_factory.AttributesFactory
    abstract_attributes.AbstractAttributes
    abstract_manifold_model_attributes.AbstractManifoldModelAttributes
    linear_attributes.LinearAttributes
    logistic_attributes.LogisticAttributes
    logistic_parallel_attributes.LogisticParallelAttributes

:mod:`leaspy.models.utils.initialization`: **Initialization methods**
---------------------------------------------------------------------
Available methods to initialize model parameters before a fit.

.. currentmodule:: leaspy.models.utils.initialization

.. autosummary::
  :toctree: generated/
  :template: function.rst

  model_initialization.initialize_parameters
  .. model_initialization.initialize_linear
  .. model_initialization.initialize_logistic
  .. model_initialization.initialize_logistic_parallel

:mod:`leaspy.algo`: **Algorithms**
==================================
Contains all algorithms used in the package.

.. currentmodule:: leaspy.algo

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_algo.AbstractAlgo
    algo_factory.AlgoFactory

:mod:`leaspy.algo.fit`: **Fit algorithms**
------------------------------------------
Algorithms used to calibrate (fit) a model.

.. currentmodule:: leaspy.algo.fit

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_fit_algo.AbstractFitAlgo
    abstract_mcmc.AbstractFitMCMC
    tensor_mcmcsaem.TensorMCMCSAEM

:mod:`leaspy.algo.personalize`: **Personalization algorithms**
--------------------------------------------------------------
Algorithms used to personalize a model to given subjects.

.. currentmodule:: leaspy.algo.personalize

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_personalize_algo.AbstractPersonalizeAlgo
    scipy_minimize.ScipyMinimize

:mod:`leaspy.algo.simulate`: **Simulation algorithms**
------------------------------------------------------
Algorithm to simulate synthetic observations and individual parameters.

.. currentmodule:: leaspy.algo.simulate

.. autosummary::
    :toctree: generated/
    :template: class.rst

    simulate.SimulationAlgorithm

:mod:`leaspy.algo.others`: **Other algorithms**
-----------------------------------------------
Reference algorithms to use with reference models (for benchmarks).

.. currentmodule:: leaspy.algo.others

.. autosummary::
    :toctree: generated/
    :template: class.rst

    constant_prediction_algo.ConstantPredictionAlgorithm
    lme_fit.LMEFitAlgorithm
    lme_personalize.LMEPersonalizeAlgorithm

:mod:`leaspy.samplers`: **Samplers**
====================================
Samplers used by the :term:`MCMC` algorithms.

.. currentmodule:: leaspy.samplers

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AbstractSampler
    AbstractPopulationSampler
    AbstractIndividualSampler
    IndividualGibbsSampler
    PopulationGibbsSampler
    PopulationFastGibbsSampler
    PopulationMetropolisHastingsSampler

.. autosummary::
    :toctree: generated/
    :template: function.rst

    sampler_factory

:mod:`leaspy.dataset`: **Datasets**
===================================
Give access to some synthetic longitudinal observations mimicking cohort of subjects with neurodegenerative disorders,
as well as calibrated models and computed individual parameters.

.. currentmodule:: leaspy.datasets

.. autosummary::
    :toctree: generated/
    :template: class.rst

    loader.Loader

:mod:`leaspy.io`: **Inputs / Outputs**
======================================
Containers classes used as input / outputs in the `Leaspy` package.

:mod:`leaspy.io.data`: **Data containers**
------------------------------------------

.. currentmodule:: leaspy.io.data

.. autosummary::
    :toctree: generated/
    :template: class.rst

    data.Data
    dataset.Dataset
    .. individual_data.IndividualData
    .. csv_data_reader.CSVDataReader
    .. dataframe_data_reader.DataframeDataReader

.. autoclass:: leaspy.io.data.data.Data
   :no-members:
   :no-inherited-members:
   :no-undoc-members:
   :no-private-members:
   :noindex:
   :member-order: bysource

:mod:`leaspy.io.settings`: **Settings classes**
-----------------------------------------------

.. currentmodule:: leaspy.io.settings

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_settings.ModelSettings
    algorithm_settings.AlgorithmSettings
    outputs_settings.OutputsSettings

.. autoclass:: leaspy.io.settings.algorithm_settings.AlgorithmSettings
   :no-members:
   :no-inherited-members:
   :no-undoc-members:
   :no-private-members:
   :noindex:
   :member-order: bysource

:mod:`leaspy.io.outputs`: **Outputs classes**
---------------------------------------------

.. currentmodule:: leaspy.io.outputs

.. autosummary::
    :toctree: generated/
    :template: class.rst

    individual_parameters.IndividualParameters
    .. result.Result

:mod:`leaspy.io.realizations`: **Realizations classes**
-------------------------------------------------------
Internal classes used for random variables in :term:`MCMC` algorithms.

.. currentmodule:: leaspy.io.realizations

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AbstractRealization
    IndividualRealization
    PopulationRealization
    DictRealizations
    CollectionRealization
    VariableType

.. autosummary::
    :toctree: generated/
    :template: function.rst

    realization_factory

:mod:`leaspy.exceptions`: **Exceptions**
========================================

.. automodule:: leaspy.exceptions
   :show-inheritance:
   :no-members:
   :no-inherited-members:
   :no-undoc-members:
   :no-private-members:
   :noindex:
   :member-order: bysource

.. currentmodule:: leaspy.exceptions

.. autosummary::
    :toctree: generated/
    :template: exception.rst

    LeaspyException
    LeaspyTypeError
    LeaspyInputError
    LeaspyDataInputError
    LeaspyModelInputError
    LeaspyAlgoInputError
    LeaspyIndividualParamsInputError
    LeaspyConvergenceError
