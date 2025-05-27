"""
Quickstart with Leaspy
======================

This example shows how to quickly use Leaspy.
"""

# %%
# Comprehensive example
# ---------------------
# We first load synthetic data in a long format to get of a grasp of longitudinal data:

from leaspy.datasets import load_dataset

alzheimer_df = load_dataset("alzheimer")
print(alzheimer_df.columns)
alzheimer_df = alzheimer_df[["MMSE", "RAVLT", "FAQ", "FDG PET"]]
print(alzheimer_df.head())

# %%
# The data correspond to repeated visits (`TIME` index) of different participants (`ID` index).
# Each visit corresponds to the measurement of 4 different outcomes : the MMSE, the RAVLT, the FAQ and the FDG PET.
# If plotted, the data would look like the following:
#
# .. image:: ../alzheimer-observations.png
#     :width: 600
#     :alt: Alzeimer observations

# %%
# Where each color corresponds to a variable, and the connected dots corresponds to the repeated visits of a single participant.
# Not very engaging, right ? To go a step further, let's first encapsulate the data into the main `Data` container:

from leaspy.io.data import Data, Dataset

data = Data.from_dataframe(alzheimer_df)
dataset = Dataset(data)

# %%
# Leaspy core functionality is to estimate the group-average trajectory of the different variables that are measured in a population.
# Clinical scores often have a ceiling and a floor effect, so let's initialize a multivariate logistic model:

from leaspy.models import LogisticModel

model = LogisticModel(name="test-model", source_dimension=2)

# %%
# As well as the algorithm needed to estimate the group-average trajectory:

from leaspy.algo import AlgorithmSettings, algorithm_factory

fit_settings = AlgorithmSettings(
    "mcmc_saem", seed=42, n_iter=100, progress_bar=False, save_periodicity=None
)
algorithm = algorithm_factory(fit_settings)
model.initialize(dataset, fit_settings.model_initialization_method)
algorithm.run(model, dataset)

# %%
# If we were to plot the measured average progression of the variables, see started example notebook for details, it would look like the following:
#
# .. image:: ../alzheimer-model.png
#     :width: 600
#     :alt: Alzeimer model

# %%
# We can also derive the individual trajectory of each subject.
# To do this, we use a personalization algorithm called `scipy_minimize`:

personalize_settings = AlgorithmSettings("scipy_minimize", seed=0, progress_bar=False)
algorithm = algorithm_factory(personalize_settings)
individual_parameters = algorithm.run(model, dataset)
print(individual_parameters.to_dataframe())

# %%
# Plotting the input participant data against its personalization would give the following, see started example notebook for details.
#
# .. image:: ../alzheimer-subject_trajectories.png
#     :width: 600
#     :alt: Alzeimer subject trajectories

# %%
# Using my own data
# -----------------
#
# Data format
# ...........
#
# Leaspy uses its own data container. To use it properly, you need to provide a csv file or a pandas.DataFrame in the right format: longitudinal data in a long format.
# Let's have a look at the data used in the previous example:

print(alzheimer_df.head())

# %% [markdown]
# ```{warning}
# You **MUST** have `ID` and `TIME`, either in index or in the columns. The other columns must be the observed variables (also named features or endpoints). In this fashion, you have one column per feature and one line per visit.
# ```

# %%
# Data scale & constraints
# ........................
#

# %% [markdwon]
# ```{warning}
# - Leaspy uses linear and logistic models.
# - The features **MUST** be increasing with time.
# - For the logistic model, you need to rescale your data between 0 and 1.
# ```

# %%
# Missing data
# ............
# Leaspy automatically handles missing data as long as they are encoded as nan in your pandas.DataFrame, or as empty values in your csv file.

# %%
# Going further
# .............
# You can check the [user_guide](../user_guide.md) and the full API documentation.
#
# You can also dive into the [examples](./index.rst).
#
# The [Disease Progression Modelling](https://disease-progression-modelling.github.io/) website also hosts a [mathematical introduction](https://disease-progression-modelling.github.io/pages/models/disease_course_mapping.html) and [tutorials](https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/disease_course_mapping.html) for Leaspy.
