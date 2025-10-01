# To Go Further

## Model Implementation

This section aims to provide a survival guide for any potential futue developper who wishes to contribute to leaspy by implementing a new model. In order to implement a new model you should be able to answer the following questions :

- Data
    - What kind of data my model aims to analyze?
- Parameters
    - What parameters my model needs to be fully specified?
- Algorithm
    - Does my model require a new/modified version of the MCMC-SAEM algortihm to estimate the parameters?

You could start by creating a *my_model.py* with the basic structure of you model.
What model class should be branched?

### Data

If other than longitudinal, events, covariates you should buid a *_data_reader.py* 

### Parameters

- **DAG**
    For models to be computationally sound, we needed to have a flow of dependency: a variable cannot depend on a variable that itself depends on it. Said differently, every model
    should be laid out as a Directed Acyclic Graph (DAG). The graph specifies the properties of each variable, as well as the dependency flow and the update rules. Such structure can
    optimise computations and has the advantage of being close to the mathematical formulation.
- **Parameter classes**
    Decide on what class to attribute each parameter, *get_variable_specs*
- **Likelihood**
    *distributions* for the latent parameters
- **Sufficient Statistics**
    *utils* functions for new model parameters

### Algortihm

Probably *samplers* because *algo/mcmc_saem* provides the basic structure.

### DAG Intuition and Structure for Models

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

## References

```{bibliography}
:filter: docname in docnames
```