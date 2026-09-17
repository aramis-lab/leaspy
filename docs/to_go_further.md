# To Go Further

## Model Implementation

This section aims to provide a survival guide for any potential future developer who wishes to contribute to `leaspy` by implementing a new model. In order to implement a new model you should be able to answer the following questions :

- Data
    - What kind of data my model aims to analyze?
- Parameters
    - What parameters my model needs to be fully specified?
- Algorithm
    - Does my model require a new/modified version of the MCMC-SAEM algorithm to estimate the parameters?

First, you must decide which model class to branch from. An overview of the class architecture is available in the [Developer's Guide](docdev/codesource/architecture.md). The remainder of this guide provides a thorough explanation of each class's characteristics. As a best practice, try to branch new models as low as possible in the inheritance chain. This makes it easier to preserve structure during future refactoring. However, if you need to modify a structural characteristic to make your model work, you can branch higher up. For example, if you need a time reparametrization equation that differs from the **latent disease age** currently implemented, you can create your own `MY_TimeReparametrizedModel` and implement your desired characteristics.

In the ideal case that you want to inherit from all the existing implementations and branch your model from `LogisticModel` you can create a *my_model.py* to hold your model's basic structure. Inside this file, you will need to define a `MyModel(LogisticModel)` new methods to define your model's specific behavior or override parent methods to customize them as needed.

**Tip:** We recommend reviewing the existing class implementations before writing your own code.

### Data

If other than longitudinal, events, covariates you should buid a *_data_reader.py* 

### Parameters

- **DAG**
    For models to be computationally sound, we needed to have a flow of dependency: a variable cannot depend on a variable that itself depends on it. Said differently, every model should be laid out as a Directed Acyclic Graph (DAG). The graph specifies the properties of each variable, as well as the dependency flow and the update rules. Such structure can optimise computations and has the advantage of being close to the mathematical formulation. You can view the DAG of the `LogisticModel` in [The Variables DAG](docdev/codesource/logistic/DAG.md) section. Because the inheritance rules are handled by the `leaspy.variables.dag` module you do not need to code your own DAG from scratch. However, you should sketch a DAG that reflects your model's structure to ensure seamless integration into the existing `leaspy` architecture. Ultimately, your most important task is deciding which parameter class to assign to each variable.

- **Parameter classes**
    The most critical function to define in your model class is `get_variables_specs`. This is where you specify the class assigned to each variable, choosing from the available types in `leaspy.variables.specs`. You do not need to rewrite the full characteristics of existing variables; you only need to declare instances of new or modified variables following the rules outlined in the `leaspy.variables.specs` module. For example if you are defining a `PopulationLatentVariable` you must provide a valid distribution argument that depends on the appropriate model parameters.

- **Likelihood**
    All models in `leaspy` are currently estimated using the **MCMC-SAEM** algorithm, which is a stochastic extension of the standard EM maximization algorithm. Due to its stochastic nature, certain parameters, known as *latent parameters*, must be sampled during estimation iterations. These parameters follow specific distributions, and you can view pre-implemented options in the `leaspy.variables.distributions` module. If you need to introduce a new distribution for your latent parameters, you should define a new class by following the structure of existing ones. To ensure full compatibility with `leaspy` your new class must define several key methods: `dist_factory`, `sample`, `_nll`, `mean`, `mode`, and `std`.

- **Sufficient Statistics**
    The **MCMC-SAEM** algorithm uses parts of the complete log-likelihood to estimate model parameters, which are known as **sufficient statistics**. In essence, these represent the calculations your algorithm performs to update model parameters during the fit iterations. In the `ModelParameter` class within `leaspy.variables.specs`, you will notice that different types of model parameters invoke different functions via the `update_rule` argument. You can find these functions in the `leaspy.models.utilities` module, where you can also add your own custom sufficient statistics. For a detailed mathematical derivation of the sufficient statistics, please refer to the appendices of recent theses from the Aramis lab {cite}`kaisaridi_2026` and {cite}`ortholand_joint_2024`.

### Algortihm

Thanks to its modular architecture, `leaspy` is designed for seamless extensibility, allowing developers and advanced users to implement and experiment with custom estimation routines. Whether you want to introduce targeted variations of the standard **MCMC-SAEM** algorithm or build an entirely new estimation algorithm from scratch, the library provides a robust structural foundation to support your needs. 

The foundational structures for all algorithmic tasks in `leaspy` are centrally organized within the `leaspy.algo` module. This modular design supplies the core base classes and modules required for essential tasks like model fitting, personalization, and simulation. You can easily define your custom algorithm within the appropriate subsection of `leaspy.algo` and seamlessly register it to drive your longitudinal progression models.

If your specific model introduces unique data properties, constraints, or distinct distribution structures that require adjustments to the standard **MCMC-SAEM** pipeline, you can dive directly into the sampling mechanics. To modify how latent variables are simulated or updated during the algorithmic iterations, start by exploring the `leaspy.samplers.gibbs` module. This module governs the sampling behaviors and provides the foundational hooks required to tailor or optimize the sampling process for complex, specialized models.

## What kind of scientific question could be answered with leaspy? 

Different papers have been published trying to answer different scientific questions using the software.

### Used in different context

__Different chronic diseases:__ The model has been used to describe very different chronic diseases as Hungtington {cite}`koval_forecasting_2022`, Alzheimer {cite}`maheux_forecasting_2023`, Cerebral Autosomal Dominant Arteriopathy with Subcortical Infarcts Leukoencephalopathy (CADASIL) {cite}`kaisaridi_determining_2025`, Amyothrophic Lateral Sclerosis {cite}`ortholand_interaction_2023`, Ataxia {cite}`moulaire_temporal_2023`, Parkinson {cite}`poulet_multivariate_2023, couronne_learning_2019`.
ALS

__Many types of data:__ Different types of data have been analysed from clinical scores to biomarkers such as clinical scores and brain markers {cite}`koval_ad_2021` and events {cite}`ortholand_joint_2025`. For longitudinal data progression where used from linear [REF?] and logistic {cite}`kaisaridi_determining_2025, ortholand_interaction_2023` to ordinal {cite}`moulaire_temporal_2023`. The model has been shown quite robust to missing data {cite}`couronne_learning_2019`.

### Used for different tasks

__Describe the joint progression of multiple outcomes:__ This package has been extensively used to describe the progression of multiple outcomes {cite}`koval_ad_2021, ortholand_interaction_2023` up to 14 clinical outcomes have been studied at the same time {cite}`kaisaridi_determining_2025`.

__Describe disease heterogeneity:__ Post-hoc analysis of the individual variability to describe disease heterogeneity were conducted using a supervised approach for Amyotrophic Lateral Sclerosis {cite}`ortholand_interaction_2023` and Ataxia {cite}`moulaire_temporal_2023` as well as an unsupervised approach for CADASIL {cite}`kaisaridi_determining_2025`.

__Improve clinical trials:__ The model has been shown useful to select patients for clinical trials in order to increase the sensibility of the trial {cite}`maheux_forecasting_2023`. The model's predictions could also be integrated to clinical trials, through prognostic score's methods (such as Prognostic Covariate Adjustment or Prediction-Powered inference for Clinical Trials) to increase the statistical power of the trials {cite}`poulet_prediction-powered_2025`.

__Make predictions:__ Leaspy outperformed the 56 alternative methods for predicting cognitive decline in the framework of the TADPOLE challenge {cite}`marinescu_tadpole_2019` and was more generally used for diverse applications {cite}`maheux_forecasting_2023, koval_forecasting_2022`
