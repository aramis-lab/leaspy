# Algorithms

## Fit
### Prerequisites
   - Data format and preprocessing steps.
### Running Task
   - How to run the fit algorithm.
   - Example commands and code snippets.
### Output
   - What results to expect from the fitting process.
   - DataFrame object 
   - Plotting object 
### Setting Options (Different Models)
   - How to set specific options for different types of models.
   - Customizing parameters for logistic, joint, mixture, and covariate models.

## Personalize

The idea of this section is to describe how the inference over new unseen patients could be done. For this random effects of new patients should be estimated using some follow-up visits to be able to describe their progression and make predictions about their future progression. Note here that we assume that fixed effects have already been estimated and are fixed. Two main approach for estimating the random effects exist in Leaspy. 

__ more of a frequentist one:__ random effects are estimated using the solver [_minimise_](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from the package Scipy (\cite{2020SciPy_NMeth}) to maximize the likelihood knowing the fixed effects.

```python
>>> personalize_settings = AlgorithmSettings("scipy_minimize", seed=0)
>>> algorithm = algorithm_factory(personalize_settings)
>>> individual_parameters = algorithm.run(model, dataset)
==> Setting seed to 0
|##################################################|   200/200 subjects
Personalize with `AlgorithmName.PERSONALIZE_SCIPY_MINIMIZE` took: 29s
The standard-deviation of the noise at the end of the AlgorithmType.PERSONALIZE is: 6.85%
>>> individual_parameters.to_dataframe()
        sources_0  sources_1        tau        xi
ID
GS-001   0.027233  -0.423354  76.399582 -0.386364
GS-002  -0.262680   0.665351  75.137474 -0.577525
GS-003   0.409585   0.844824  75.840012  0.102436
GS-004   0.195366   0.056577  78.110085  0.433171
GS-005   1.424637   1.054663  84.183098 -0.051317
...           ...        ...        ...       ...
GS-196   0.961941   0.389468  72.528786  0.354126
GS-197  -0.561685  -0.720041  79.006042 -0.624100
GS-198  -0.061995   0.177671  83.596138  0.201686
GS-199   1.932454   1.820023  92.275978 -0.136224
GS-200   1.152407  -0.171888  76.504517  0.770118

[200 rows x 4 columns]
```

__ more a bayesian one:__ random effects are estimated using a Gibbs sampler with an option on the burn-in phase [REF part?] and temperature scheme [REF part?]. Nevertheless, for now the package enable only to extract the mean or the mode of the posterior distribution. 

[TODO ADD CODE HERE]


## Estimate
## Simulate

## Data Generalities
- monotonicity
- NaN 
- number of visits 
- outliers
- not enough patients 
- parameters don't converge 
- score don't progress