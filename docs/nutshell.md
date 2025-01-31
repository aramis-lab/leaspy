# Leaspy in a nutshell

## Comprehensive example

We first load synthetic data to get of a grasp of longitudinal data:

```python
>>> from leaspy.datasets import load_dataset
>>> alzheimer_df = load_dataset('alzheimer-multivariate')
>>> print(alzheimer_df.columns)
Index(['E-Cog Subject', 'E-Cog Study-partner', 'MMSE', 'RAVLT', 'FAQ',
       'FDG PET', 'Hippocampus volume ratio'],
      dtype='object')
>>> alzheimer_df = alzheimer_df[['MMSE', 'RAVLT', 'FAQ', 'FDG PET']]
>>> print(alzheimer_df.head())
                      MMSE     RAVLT       FAQ   FDG PET
ID     TIME
GS-001 73.973183  0.111998  0.510524  0.178827  0.454605
       74.573181  0.029991  0.749223  0.181327  0.450064
       75.173180  0.121922  0.779680  0.026179  0.662006
       75.773186  0.092102  0.649391  0.156153  0.585949
       75.973183  0.203874  0.612311  0.320484  0.634809
```

The data correspond to repeated visits (`TIME` index) of different participants (`ID` index).

Each visit corresponds to the measurement of 4 different variables : the [MMSE](./glossary.md#mmse), the RAVLT, the FAQ and the FDG PET.

If plotted, the data would look like the following:

![alzheimer-observations](./_static/images/alzheimer-observations.png)

Where each color corresponds to a variable, and the connected dots corresponds to the repeated visits of a single participant.

Not very engaging, right ? To go a step further, let's first encapsulate the data into the main `Data` container:

```python
>>> from leaspy.io.data import Data, Dataset
>>> data = Data.from_dataframe(alzheimer_df)
>>> dataset = Dataset(data)
```

Leaspy core functionality is to estimate the group-average trajectory of the different variables that are measured in a population.

Let's initialize a multivariate logistic model:

```python
>>> from leaspy.models import LogisticMultivariateModel
>>> model = LogisticMultivariateModel(name="test-model", source_dimension=2)
```

As well as the algorithm needed to estimate the group-average trajectory:

```python
>>> from leaspy.algo import AlgorithmSettings, algorithm_factory
>>> fit_settings = AlgorithmSettings("mcmc_saem", seed=0, n_iter=8000)
>>> algorithm = algorithm_factory(fit_settings)
>>> model.initialize(dataset, fit_settings.model_initialization_method)
>>> algorithm.run(model, dataset)
==> Setting seed to 0
|##################################################|   8000/8000 iterations

Fit with `AlgorithmName.FIT_MCMC_SAEM` took: 1m 32s
The standard-deviation of the noise at the end of the AlgorithmType.FIT is: -100.00%
<leaspy.variables.state.State object at 0x305b7ff90>
```

If we were to plot the measured average progression of the variables, see [started example notebook](https://gitlab.com/icm-institute/aramislab/leaspy) for details, it would look like the following:

![alzheimer-model](./_static/images/alzheimer-model.png)

We can also derive the individual trajectory of each subject.

To do this, we use a `personalization` algorithm called `scipy_minimize`:

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

Plotting the input participant data against its personalization would give the following, see [started example notebook](https://gitlab.com/icm-institute/aramislab/leaspy) for details.

![alzheimer-subject_trajectories](./_static/images/alzheimer-subject_trajectories.png)

### Using my own data

#### Data format

`Leaspy` uses its own data container. To use it properly, you need to provide a `csv` file or a `pandas.DataFrame` in the right format.

Let's have a look at the data used in the previous example:

```python
>>> print(alzheimer_df.head())
                   MMSE      RAVLT     FAQ       FDG PET
ID      TIME
GS-001  73.973183  0.111998  0.510524  0.178827  0.454605
        74.573181  0.029991  0.749223  0.181327  0.450064
        75.173180  0.121922  0.779680  0.026179  0.662006
        75.773186  0.092102  0.649391  0.156153  0.585949
        75.973183  0.203874  0.612311  0.320484  0.634809
```

You **MUST** have `ID` and `TIME`, either in index or in the columns. The other columns must be the observed variables (also named *features* or *endpoints*). In this fashion, you have one column per *feature* and one line per *visit*.

#### Data scale & constraints

`Leaspy` uses *linear* and *logistic* models. The features **MUST** be increasing with time. For the *logistic* model, you need to rescale your data between 0 and 1.

#### Missing data

`Leaspy` automatically handles missing data as long as they are encoded as ``nan`` in your `pandas.DataFrame`, or as empty values in your `csv` file.

### Going further

You can check the [user_guide](./user_guide.md) and the full API documentation.

You can also dive into the [started example](https://gitlab.com/icm-institute/aramislab/leaspy) of the Leaspy repository.

The [Disease Progression Modelling](https://disease-progression-modelling.github.io/) website also hosts a [mathematical introduction](https://disease-progression-modelling.github.io/pages/models/disease_course_mapping.html) and [tutorials](https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/disease_course_mapping.html) for Leaspy.