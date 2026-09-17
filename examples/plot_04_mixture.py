"""
Mixture Model
=====================================================
This notebook contains the code for a simple implementation of the Leaspy Mixture model on synthetic data.
Before implementing the model take a look at the relevant mathematical framework in the user guide.
"""

# %%
# The following imports are required libraries for numerical computation, data manipulation, and visualization.
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import leaspy
from leaspy.io.data import Data

# %%
# This toy example is part of a simulation study, carried out by Sofia Kaisaridi that will be included in
# an article to be submitted in a biostatistics journal. The dataset contains 1000 individuals each with 6 visits and 6 scores.

leaspy_root = os.path.dirname(leaspy.__file__)
data_path = os.path.join(leaspy_root, "datasets/data/simulated_data_for_mixture.csv")

all_data = pd.read_csv(data_path, sep=";", decimal=",")
all_data["ID"] = all_data["ID"].ffill()
all_data = all_data.set_index(["ID", "TIME"])
all_data.head()
# %%
# We load the Mixture Model from the leaspy library and transform the dataset in a leaspy-compatible form with the built-in functions.
from leaspy.models import LogisticMultivariateMixtureModel

leaspy_data = Data.from_dataframe(all_data)

# %%
# Then we fit a model with 3 clusters and 2 sources. Note that we have an extra argument `n_clusters` than the
# standard model that has to be specified in order for the mixture model to run.

model = LogisticMultivariateMixtureModel(
    name="multi",
    source_dimension=2,
    dimension=6,
    n_clusters=3,
    obs_models="gaussian-diagonal",
)

model.fit(leaspy_data, "mcmc_saem", seed=1312, n_iter=100, progress_bar=False)
model.summary()

# %%
# First we take a look at the population parameters.
# With the mixture model we obtain separate values for the `tau_mean`, `xi_mean` and the `sources_mean` for each cluster,
# as well as the cluster probabilities (`probs`).

model.info()

# %%
# Then we can plot the average trajectory for each cluster. When having many scores the default 
# option is to provide graphs for every three scores for readability. We can change this by 
# providing the `n_features_per_plot` argument to the following function.
# The default option is to plot all the clusters. We can control this by providing 
# a list of cluster indices to plot at the argument `clusters`.

from leaspy.io.logs.visualization.plotting import Plotting

leaspy_plotting = Plotting(model)
leaspy_plotting.average_trajectory_cluster(
    alpha=0.8, 
    linewidth=3, 
    figsize=(14, 6), 
    n_std_left=4, 
    n_std_right=5
    )
plt.show()

# %%
# We can also access the individual parameters of each patient with the `personalize` method.
# Then the `get_individual_probabilities` method can be used to obtain the probability of each patient to belong to each cluster.
# The individual cluster assignment can be obtained by taking the cluster with the highest probability for each patient.

ip = model.personalize(leaspy_data, "scipy_minimize", seed=0, progress_bar=False)
ip_df = ip.to_dataframe()
ip_with_probs = model.get_individual_probabilities(ip_df)
ip_with_probs.head()

# %%
# We can also plot the patient observations.

ax = leaspy_plotting.patient_observations_reparametrized(
    leaspy_data, ip_with_probs, figsize=(14, 6),
    patients_idx=['subj_1','subj_2','subj_3','subj_4','subj_5'],
)
plt.show()

# %%
# Or the patient trajectories.

ax = leaspy_plotting.patient_trajectories(
    leaspy_data,
    ip_with_probs,
    patients_idx=['subj_12']
)
ax.set_xlim(45, 100)
plt.show()

# %%
# This concludes the Mixture Model example. Next, we introduce the
# :doc:`Covariate Model <plot_05_covariate>`.
