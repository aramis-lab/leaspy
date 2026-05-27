"""
Personalization: Parkinson's disease progression and inference modeling with Leaspy
===================================================================================

This example walks through the core Leaspy workflow on a synthetic Parkinson's disease dataset:

1. Fit a shared progression model on a training cohort.
2. Personalize the model to new patients — estimating where each one sits on the shared disease timeline and how fast they are progressing.
3. Reconstruct and predict individual trajectories from those two numbers alone.

The key concept is **personalization**: once a model is trained, a new patient needs only a handful
of visits for Leaspy to estimate their individual parameters (τ, ξ) and predict their future course.
"""

# %%
# We load a synthetic dataset of Parkinson's patients with repeated measurements
# of three MDS-UPDRS motor subscores over time.
from leaspy.datasets import load_dataset
from leaspy.io.data import Data

df = load_dataset("parkinson")

# %%
df.head()

# %%
n_subjects = df.index.get_level_values("ID").unique().shape[0]
print(f"{n_subjects} subjects in the dataset.")

# %%
# We split into a training cohort and a held-out test cohort. The model is fitted
# on training subjects; we then personalize it to test subjects as if they were
# new patients arriving at a clinic.
df_train = df.loc[:"GS-160"][["MDS1_total", "MDS2_total", "MDS3_off_total"]]
df_test  = df.loc["GS-161":][["MDS1_total", "MDS2_total", "MDS3_off_total"]]

data_train = Data.from_dataframe(df_train)
data_test  = Data.from_dataframe(df_test)

# %%
# We use a multivariate logistic model: all three scores share a single sigmoidal
# trajectory, and patients differ only in *when* and *how fast* they travel along it.
from leaspy.models import LogisticModel

model = LogisticModel(name="test-model", source_dimension=2)

# %%
import matplotlib.pyplot as plt
from leaspy.io.logs.visualization.plotting import Plotting

leaspy_plotting = Plotting(model)

# %%
# Raw training observations. The scores look heterogeneous because each patient
# is at a different disease stage and progresses at a different pace.
ax = leaspy_plotting.patient_observations(data_train, alpha=0.7, figsize=(14, 6))
ax.set_ylim(0, 0.8)
plt.show()

# %%
# Fitting learns a single population-level sigmoidal curve that best explains
# all training subjects simultaneously.
model.fit(data_train, "mcmc_saem", seed=0, n_iter=100, progress_bar=False)

# %%
# The average trajectory is the shared progression curve. Every patient is assumed
# to follow this same curve, shifted and rescaled in time.
ax = leaspy_plotting.average_trajectory(alpha=1, figsize=(14, 6), n_std_left=2, n_std_right=8)
plt.show()

# %%
# Personalization estimates two individual parameters per test patient from their visits:
#   τ (tau) — disease onset age (position on the timeline)
#   ξ (xi)  — log-acceleration (pace of progression)
ip = model.personalize(data_test, "scipy_minimize", seed=0, progress_bar=False, use_jacobian=False)

# %%
# After time reparametrization ψᵢ(t) = exp(ξᵢ)·(t − τᵢ), all patients align onto
# the same curve — confirming the model has captured their individual stages and speeds.
ax = leaspy_plotting.patient_observations_reparametrized(
    data_test, ip, alpha=0.7, linestyle="-", figsize=(14, 6)
)
plt.show()

# %%
# Without reparametrization the same data looks scattered: patients of the same
# chronological age may be at very different disease stages.
ax = leaspy_plotting.patient_observations(data_test, alpha=0.7, linestyle="-", figsize=(14, 6))
plt.show()

# %%
# To illustrate prediction we pick one test patient. `model.estimate` is the low-level
# API that returns predicted scores at arbitrary timepoints — useful for custom analyses.
import numpy as np

print(f"Seen ages: {df_test.loc['GS-187'].index.values}")
print("Individual parameters:", ip["GS-187"])

timepoints = np.linspace(60, 100, 100)
reconstruction = model.estimate({"GS-187": timepoints}, ip)
print(f"Predicted scores at age 80: {reconstruction['GS-187'][40]}")  # index 40 ≈ age 80

# %%
# `patient_trajectories` wraps that same call and overlays the predicted curve on
# the observed visits, extrapolating beyond the last observation.
ax = leaspy_plotting.patient_trajectories(
    data_test, ip,
    patients_idx=["GS-187"],
    labels=["MDS1", "MDS2", "MDS3 (off)"],
    figsize=(16, 6),
    factor_future=5,
)
ax.set_xlim(45, 120)
plt.show()

# %%
# From a fitted model and just a few visits, Leaspy reduces each patient to (τ, ξ) —
# two numbers that place them on a shared disease timeline and predict their future
# trajectory across all scores simultaneously.

# %%
# **Interpreting the individual space shifts.** Beyond (τᵢ, ξᵢ), each patient
# also has a *spatial* signature — per-feature offsets that describe whether
# they are more or less affected on certain features than the average
# trajectory at ``tau_mean``. These offsets are the
# **space shifts** ``wᵢ,ₖ``:
#
# * ``wᵢ,ₖ`` has one entry per feature, returned as columns ``w_<feature>``.
# * A *positive* ``w_MDS1`` for patient *i* means "given this patient's
#   (τ, ξ), they are *more* impaired on MDS1 than the average trajectory
#   predicts"; *negative* means *less* impaired.
# * By construction, the average ``wᵢ,ₖ`` is approximately zero.

ip.compute_space_shifts(model).head()

# %%
# Large ``|wᵢ,ₖ|`` flags patients whose feature *k* is
# atypically ahead or behind their overall stage — a signal worth a closer
# look (alternative diagnosis, treatment response, comorbidity). This parameter
# can be also interpreted "reverting" the features normalization, i.e. for the
# MMSE that goes from 0 to 30, a ``wᵢ,MMSE = -0.1`` means that patient *i* is 3 
# points better than the average patient at their stage.

# %%
# The next example extends this to joint models that also incorporate time-to-event
# outcomes: see :doc:`plot_03_joint`.
