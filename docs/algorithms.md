# Algorithms

## Fit

In this section we describe how to fit a `leaspy` model with your data. Leaspy uses the [MCMC-SAEM algorithm](./glossary.md#mcmc-saem) to fit a model by jointly estimating the fixed effects and the distribution of the random effects. It is particularly well suited to this kind of models where the likelihood involves latent variables and is not available in closed form.

The algorithm is an adaptation of the Expectation-Maximisation (EM) algorithm that relies on an iterative procedure that alternates between the following main steps:

- Expectation/Stochastic Approximation Step: the algorithm uses [Markov Chain Monte Carlo (MCMC)](./glossary.md#mcmc) to generate samples of the latent variables (random effects) conditional on the current parameter estimates, and compute the sufficient statistics of the complete-data log-likelihood using a stochastic approximation scheme. 
- Maximization Step: Given the updated sufficient statistics, the fixed effects and variance components are re-estimated by maximizing the approximate complete-data log-likelihood.

By iterating these steps, the MCMC-SAEM algorithm converges to the maximum likelihood estimates of the model parameters.

### Prerequisites

Depending on the model you want to fit, you need a dataframe with a specific structure (see [logistic](./models.md#logistic-data), [joint](./models.md#joint-data), and [mixture](./models.md#mixture-data) models).

### Running Task

First, choose one of the the existing models in `leaspy.models`. The model you select determines the expected structure of your dataset (see [Prerequisites](#prerequisites)).

Let's use the logistic model as an example.

```python
from leaspy.models import LogisticModel
```

We need to specify the arguments `name`, `dimension` (the number of features $K$ in your dataset) and the `obs_models` (valid choices for the logistic model are 'gaussian-diagonal' to estimate one noise coefficient per feature or 'gaussian-scalar' to estimate one noise coefficient for all the features). When we fit a multivariate model we also need to specify `source_dimension` that corresponds to the degrees of freedom of intermarker spacing parameters. We refer you to the [mathematical background section](./mathematics.md#individual-trajectory--spatial-random-effects) for more details. We generally suggest a number of sources close to the square root of the number of features ($\sqrt{dimension}$).

You can also add a `seed` or control other arguments for the output and the logs like `save_periodicity`, `path`, etc.

```python
model = LogisticModel(name="my-model", source_dimension=1, dimension=2, obs_models='gaussian-diagonal')
model.fit(data_leaspy, "mcmc_saem", n_iter=20000)
```

Note that the joint and mixture models require additional model-specific arguments. Please refer to their respective documentation for details: [joint model](./models.md#model-summary) and [mixture model](./models.md#model-summary-1).

### Output

Once the iterations are done we can see the parameters that were estimated by the model and save them in a dedicated file.

```python
model.save('my_path/my_model/model_parameters.json')
model.parameters
```

And we can also plot the estimated average trajectory.

```python
import matplotlib.pyplot as plt
from leaspy.io.logs.visualization.plotting import Plotting
leaspy_plotting = Plotting(model)

ax = leaspy_plotting.average_trajectory(
    alpha=1, figsize=(14, 6), n_std_left=2, n_std_right=8
)
plt.show()
```


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

This sections describes the procedure for estimating a patient's trajectory.
Once the personalization is performed then we can estimate and visualize the trajectory of a specific subject using its individual parameters.

```python
import numpy as np

observations = dataframe.loc["GS-187"]
print(f"Seen ages: {observations.index.values}")
print("Individual Parameters : ", ip["GS-187"])

timepoints = np.linspace(60, 100, 100)
reconstruction = model.estimate({"GS-187": timepoints}, ip)
```

The reconstruction object contains the estimated feature values for the individual and we can plot them along with the actual observations.

```python
ax = leaspy_plotting.patient_trajectories(
    data_leaspy,
    ip,
    patients_idx=["GS-187"],
    labels=["FEATURE_1", "FEATURE_2"],
    alpha=1,
    linestyle="-",
    linewidth=2,
    markersize=8,
    obs_alpha=0.5,
    figsize=(16, 6),
    factor_past=0.5,
    factor_future=5,
)
ax.set_xlim(45, 80)
plt.show()
```

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
$y_{i,j,k} = \gamma_{i,k}(t_{i,j,k}) + \epsilon_{i,j,k}, \quad \epsilon_{i,j,k} \sim \mathcal{N}(0, \sigma^2_k)$

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
        'visit_type': "random",
        'first_visit_mean': 0.,
        'first_visit_std': 0.4,
        'time_follow_up_mean': 11,
        'time_follow_up_std': 0.5,
        'distance_visit_mean': 2 / 12,
        'distance_visit_std': 0.75 / 12,
        'distance_visit_min': 1/365
    }
>>> simulated_data = model.simulate( 
         algorithm="simulate", 
         features=["MDS1_total", "MDS2_total", "MDS3_off_total"],
         visit_parameters= visits_params
    )
>>> print(simulated_data.data.to_dataframe().set_index(['ID', 'TIME']).head())
 ID  TIME  MDS1_total  MDS2_total  MDS3_off_total  SCOPA_total  MOCA_total  \
  0  63.0    0.130888    0.220548        0.186086     0.083651    0.088756   
     64.0    0.138080    0.039211        0.289588     0.034846    0.047147   
     65.0    0.228149    0.068744        0.151979     0.141604    0.131976   
     66.0    0.208679    0.112899        0.202224     0.192716    0.067183   
     67.0    0.290484    0.252141        0.255622     0.240425    0.115898   

   REM_total  PUTAMEN_R  PUTAMEN_L  CAUDATE_R  CAUDATE_L  
   0.555283   0.808789   0.685063   0.546174   0.467885  
   0.660931   0.758014   0.640209   0.541839   0.474202  
   0.766028   0.941519   0.738120   0.643509   0.549832  
   0.671021   0.796510   0.930209   0.657473   0.622322  
   0.791594   0.955246   0.844813   0.677306   0.638281  
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