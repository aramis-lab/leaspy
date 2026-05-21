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
# .. warning::
#
#    You **MUST** include both ``ID`` and ``TIME``, either as indices or as columns.
#    The remaining columns should correspond to the observed variables
#    (also called features or endpoints).
#    Each feature should have its own column, and each visit should occupy one row.


# %%
# .. warning::
#
#    - Leaspy supports *linear* and *logistic* models.
#    - Features should follow an overall increasing trend over time. Individual observations may decrease due to noise or measurement variability — what matters is that the general progression goes upward.
#    - For logistic models, data must be rescaled between 0 and 1.

from leaspy.io.data import Data

data = Data.from_dataframe(alzheimer_df)

# %%
# .. seealso::
#
#    For a deeper understanding of the ``Data`` and ``Dataset`` classes, including
#    iteration, cofactors, and best practices, refer to the Data Containers Guide
#    in the documentation.

# %%
# The core functionality of Leaspy is to estimate the group-average trajectory
# of the variables measured in a population.  To do this, you need to choose a model.
# For example, a logistic model can be initialized and fitted as follows:

from leaspy.models import LogisticModel

model = LogisticModel(name="test-model", source_dimension=2)
model.fit(
    data,
    "mcmc_saem",
    seed=42,
    n_iter=100,
    progress_bar=False,
)
model.summary()

# %%
# **Interpreting the population parameters.** The summary above describes the
# cohort-average disease trajectory through three population-level quantities:
#
# * ``tau_mean`` — the reference age (in years) at which the cohort, on
#   average, reaches the reference state ``p0``. It anchors the shared disease
#   clock.
# * ``v0`` — the per-feature velocity at ``tau_mean`` (per year, on the
#   ``[0, 1]`` scale). Features with larger ``v0`` change faster around the
#   reference age.
# * ``p0`` — the per-feature value at ``tau_mean``, on ``[0, 1]``. Read it as
#   the average impairment level when the cohort reaches ``tau_mean``.
#
# ``v0`` and ``p0`` appear under "Derived Parameters" in the summary. They are
# returned in interpretable scale by ``model.compute_derived_parameters()`` —
# the raw fitted values ``log_v0_mean`` and ``log_g_mean`` live in log / logit
# space and are not meant to be read directly.

derived = model.compute_derived_parameters()
for k, name in enumerate(model.features):
    v0_k = derived["v0"][k].item()
    p0_k = derived["p0"][k].item()
    print(f"  {name:<8}  v0 = {v0_k: .4f} / yr     p0 = {p0_k:.3f}")
print(f"  tau_mean = {float(model.parameters['tau_mean']):.2f} yr")

# %%
# The ``fit`` method estimates the parameters of the model, which are then accessible
# through the ``summary`` method. The parameters are also stored in the ``parameters`` attribute of the model.

model.info()

# %%
# The method ``info`` provides the model configuration and the settings used for the fit,
# as well as the dataset information and the training information.
#
# Leaspy can also estimate the *individual trajectories* of each participant.
# This is done using a personalization algorithm, here `scipy_minimize`:

individual_parameters = model.personalize(
    data, "scipy_minimize", seed=0, progress_bar=False, use_jacobian=False
)
print(individual_parameters.to_dataframe())

# %%
# We have seen how to fit a model and personalize it to individuals.
# Leaspy also provides various plotting functions to visualize the results.
# Let's  go to the next :doc:`section <plot_02_parkinson_example>` to see how to plot
# the group-average trajectory and the individual trajectories using the Parkinson's disease dataset.

# %%
# To go further:
#
# 1. See the :doc:`User Guide <../user_guide>` and full API documentation.
# 2. Explore additional :doc:`examples <./index>`.
