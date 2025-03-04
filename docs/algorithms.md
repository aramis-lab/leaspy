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

The idea of this section is to describe how the inference over new patients could be done. For this, random effects of new patients should be estimated using some follow-up visits to be able to describe their progression and make predictions about their future progression. We assume that fixed effects have already been estimated. To estimate the random effects, two main approaches exist in Leaspy. 

__more of a frequentist one:__ random effects are estimated using the solver [_minimise_](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from the package Scipy {cite}`2020SciPy_NMeth` to maximize the likelihood knowing the fixed effects.

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

__more a bayesian one:__ random effects are estimated using a Gibbs sampler with an option on the burn-in phase (see [fit description](##Fit))and temperature scheme [fit description](##Fit). Currently, the package enables to extract the mean or the mode of the posterior distribution. They can be used with the same procedure using `mean_posterior` or `mode_posterior` flag. 

```python
>>> personalize_settings = AlgorithmSettings("mean_posterior", seed=0)
>>> algorithm = algorithm_factory(personalize_settings)
>>> individual_parameters = algorithm.run(model, dataset)
==> Setting seed to 0
|##################################################|   1000/1000 iterations

Personalize with `mean_posterior` took: 1s
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


## Estimate
## Simulate

This section describes the procedure for simulating new patient data under the Spatiotemporal model structure. The simulation method, relying on a fitted Leaspy model and user-defined parameters, involves the following steps:

**Step 1: Generation of Individual Parameters** <br>
For each simulated patient, individual parameters ($\tau_i$, $\xi_i$, and the sources) are sampled from normal distributions defined by the model’s mean and standard deviation:<br>
- $\xi_i \sim \mathcal{N}\left(0, \sigma^2_{\xi}\right)$,
- $\tau_i \sim \mathcal{N}\left(t_0, \sigma^2_{\tau}\right)$,
- $N_s \text{ sources } s_{i,m} \sim \mathcal{N}\left(\bar{s}, \sigma_s\right)$

These model parameters come from a previously fitted Leaspy model, provided by the user. 

**Step 2: Generation of Visit Times** <br>
Visit times are generated based on user-specified visit parameters, such as the number of visits, spacing between visits, and follow-up duration. These parameters are provided through a dictionary.

**Step 3: Estimation of Observations** <br>
The estimate function from Leaspy is used to compute the patients observation at the generated visit time $t_{i,j,k}$, based on the individual parameters:<br>
$$
y_{i,j,k} = \gamma_{i,k}(t_{i,j,k}) + \epsilon_{i,j,k}, \quad \epsilon_{i,j,k} \sim \mathcal{N}(0, \sigma^2_k)
$$

To reflect variability in the observations, beta-distributed noise is added, appropriate for modeling outcomes in a logistic framework.

### Prerequisites
To run a simulation, the following variables are required:
- A fitted Leaspy model (see the `fit` function), used for both parameter sampling (step 1) and the estimate function (step 3).
- A dictionary of visit parameters, specifying the number, type, and timing of visits (used in step 2).
- An `AlgorithmSettings` object, configured for simulation and including:
  - The name of the features to simulate.
  - The visit parameter dictionary.

### Running the Task

```python
>>> from leaspy.algo import AlgorithmSettings
>>> visits_params = {
        'patient_nb': 200,
        'visit_type': "regular",
        'regular_visit': 1,
        'first_visit_mean': 0.,
        'first_visit_std': 0.4,
        'time_follow_up_mean': 11,
        'time_follow_up_std': 0.5,
        'distance_visit_mean': 2 / 12,
        'distance_visit_std': 0.75 / 12,
    }
>>> simulated_data = model.simulate( 
         algorithm="simulate", 
         features=["MDS1_total", "MDS2_total", "MDS3_off_total"],
         visit_parameters= visits_params
    )
Simulate with `simulation` took: 3s
>>> print(simulated_data.data.to_dataframe().set_index(['ID', 'TIME']).head())
              MDS1_total   MDS2_total   MDS3_total
ID     TIME
 0   48.092     0.113991     0.344018     0.228870
     49.092     0.177138     0.291440     0.332267
     50.092     0.256084     0.253093     0.314241
     51.092     0.174059     0.344921     0.313776
     52.092     0.245289     0.400363     0.310065
```

### Output

The output is a Data object with ID, TIME and simulated values of each feature. 

### Setting options

There are three options to simulate the visit times in Leaspy, which can be specified in visit_param dictionary: 
- `random`: Visit times and intervals are sampled from normal distributions.
- `regular`: Visits occur at regular intervals, defined by regular_visit. 
- `dataframe`: Custom visit times are provided directly via a DataFrame.

Refer to the docstring for further details.


## Data Generalities
- monotonicity
- NaN 
- number of visits 
- outliers
- not enough patients 
- parameters don't converge 
- score don't progress

## References

```{bibliography}
:filter: docname in docnames
```
>>>>>>> dc1e1354 (plan proposition for the user guide)
