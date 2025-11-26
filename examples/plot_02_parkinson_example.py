"""
Parkinson's Disease Progression Modeling with Leaspy
=====================================================

This example demonstrates how to use Leaspy to model the progression of Parkinson's disease using synthetic data.
"""

# %%
# The following imports bring in the required modules and load the synthetic dataset from Leaspy.
# The dataset contains repeated measurements for multiple subjects over time.
from leaspy.datasets import load_dataset
from leaspy.io.data import Data

df = load_dataset("parkinson")

# %%
# The first few rows of the dataset provide an overview of its structure.
df.head()

# %%
# The total number of unique subjects present in the dataset is shown below.
n_subjects = df.index.get_level_values("ID").unique().shape[0]
print(f"{n_subjects} subjects in the dataset.")

# %%
# The dataset is separated into a training set and a test set.
# The first portion of the data is used for training and the remaining portion for testing.
df_train = df.loc[:"GS-160"][["MDS1_total", "MDS2_total", "MDS3_off_total"]]
df_test = df.loc["GS-161":][["MDS1_total", "MDS2_total", "MDS3_off_total"]]

# %%
# The pandas DataFrames are converted to Leaspy `Data` objects for further modeling.
data_train = Data.from_dataframe(df_train)
data_test = Data.from_dataframe(df_test)


# %%
# The logistic model is imported and initialized.
# A two-dimensional source space is chosen to represent disease progression trajectories.
from leaspy.models import LogisticModel

model = LogisticModel(name="test-model", source_dimension=2)

# %%
# Visualization utilities from Leaspy and Matplotlib are imported.
import matplotlib.pyplot as plt

from leaspy.io.logs.visualization.plotting import Plotting

leaspy_plotting = Plotting(model)

# %%
# Data that will be used to fit the model can be illustrated as follows:

ax = leaspy_plotting.patient_observations(data_train, alpha=0.7, figsize=(14, 6))
ax.set_ylim(0, 0.8)  # The y-axis is adjusted for better visibility.
plt.show()


# %%
# The model is fitted to the training data using the MCMC-SAEM algorithm.
# A fixed seed is used for reproducibility and 100 iterations are performed.
model.fit(
    data_train,
    "mcmc_saem",
    seed=0,
    n_iter=100,
    progress_bar=False,
)


# %%
# The average trajectory estimated by the model is displayed.
# This figure shows the mean disease progression curves for all features.
ax = leaspy_plotting.average_trajectory(
    alpha=1, figsize=(14, 6), n_std_left=2, n_std_right=8
)
plt.show()

# %%
# Individual parameters are obtained for the test data using the personalization step.
ip = model.personalize(data_test, "scipy_minimize", seed=0, progress_bar=False)

# %%
# The test data with individually re-parametrized ages is plotted below.
ax = leaspy_plotting.patient_observations_reparametrized(
    data_test, ip, alpha=0.7, linestyle="-", figsize=(14, 6)
)
plt.show()

# %%
# The test data with the true ages (without re-parametrization) is plotted below.
ax = leaspy_plotting.patient_observations(
    data_test,
    alpha=0.7,
    linestyle="-",
    figsize=(14, 6),
)
plt.show()

# %%
# Observations for a specific subject are extracted to demonstrate reconstruction.
import numpy as np

observations = df_test.loc["GS-187"]
print(f"Seen ages: {observations.index.values}")
print("Individual Parameters : ", ip["GS-187"])

timepoints = np.linspace(60, 100, 100)
reconstruction = model.estimate({"GS-187": timepoints}, ip)

# %%
# The reconstructed trajectory along with the actual observations for selected subjects is displayed.
ax = leaspy_plotting.patient_trajectories(
    data_test,
    ip,
    patients_idx=["GS-187"],
    labels=["MDS1", "MDS2", "MDS3 (off)"],
    alpha=1,
    linestyle="-",
    linewidth=2,
    markersize=8,
    obs_alpha=0.5,
    figsize=(16, 6),
    factor_past=0.5,
    factor_future=5,
)
ax.set_xlim(45, 120)
plt.show()

# %%
# This concludes the Parkinson's disease progression modeling example using Leaspy.
# Leaspy is also capable of handling various other types of models, as the Joint Models,
# which will be explored in the [next section](./plot_03_joint).