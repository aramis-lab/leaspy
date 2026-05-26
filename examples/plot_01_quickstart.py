"""
Quickstart with Leaspy
======================

This example demonstrates how to quickly use Leaspy with properly formatted data.
"""

# %%
# Leaspy uses its own data container. To use it correctly, you need to provide either
# a CSV file or a pandas.DataFrame in *long format*.
#
# Below is an example of synthetic longitudinal data illustrating how to use Leaspy:

from leaspy.datasets import load_dataset

alzheimer_df = load_dataset("alzheimer")
print(alzheimer_df.columns)
alzheimer_df = alzheimer_df[["MMSE", "RAVLT", "FAQ", "FDG PET"]]
print(alzheimer_df.head())

# %%
# The data correspond to repeated visits (`TIME` index) of different participants (`ID` index).
# Each visit corresponds to the measurement of 4 different outcomes : the MMSE, the RAVLT, the FAQ and the FDG PET.


# %%
# ```{warning}
# You **MUST** include both `ID` and `TIME`, either as indices or as columns.
# The remaining columns should correspond to the observed variables
# (also called features or endpoints).
# Each feature should have its own column, and each visit should occupy one row.
# ```


# %%
# ```{warning}
# - Leaspy supports *linear* and *logistic* models.
# - The features **MUST** be increasing over time.
# - For logistic models, data must be rescaled between 0 and 1.
# ```

from leaspy.io.data import Data, Dataset

data = Data.from_dataframe(alzheimer_df)
dataset = Dataset(data)

# %%
# The core functionality of Leaspy is to estimate the group-average trajectory
# of the variables measured in a population.  To do this, you need to choose a model.
# For example, a logistic model can be initialized and fitted as follows:

from leaspy.models import LogisticModel

model = LogisticModel(name="test-model", source_dimension=2, obs_models="gaussian-scalar")
model.fit(
    dataset,
    "mcmc_saem",
    seed=42,
    n_iter=100,
    progress_bar=False,
    overwrite_logs_folder=True,
    save_periodicity=10,
    plot_periodicity=10,
)
# %%
# The save_periodicity and plot_periodicity arguments are optional, and control how often the 
# model parameters are saved and plotted during the fitting process. By setting them to an
# integer value, an output folder is created under the name `_outputs`in the working directory,
# where the convergence plots and csv are saved. You can also control the target folder by
# providing a string to the `path` argument.
# %%
# Leaspy can also estimate the *individual trajectories* of each participant.
# This is done using a personalization algorithm, here `scipy_minimize`:

individual_parameters = model.personalize(
    dataset, "scipy_minimize", seed=0, progress_bar=False
)
print(individual_parameters.to_dataframe())


# %%
# To go further;
#
# 1. See the [User Guide](../user_guide.md) and full API documentation.
# 2. Explore additional [examples](./index.rst).
