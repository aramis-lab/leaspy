"""
Covariate Model
===============

This example fits a logistic model with a patient-level covariate.
"""

# %%
# .. warning::
#
#    The covariate model is in beta. We will not provide support for this model.
#    Its methods may change in the next version, which is expected very shortly.

# %%
# We load synthetic longitudinal data containing one binary covariate, ``cov_A``.
import os

import pandas as pd

import leaspy
from leaspy.io.data import Data

data_path = os.path.join(
    os.path.dirname(leaspy.__file__),
    "datasets/data/simulated_data_for_covariate.csv",
)
df = pd.read_csv(data_path).set_index(["ID", "TIME"])
df.head()

# %%
# We identify ``cov_A`` as the covariate when creating the Leaspy data container.
data = Data.from_dataframe(
    df,
    "covariate",
    factory_kws={"covariate_names": ["cov_A"]},
)

# %%
# First, we fit a standard logistic model to initialize the shared parameters.
# This is faster than fitting the covariate model directly, even if it is not strictly necessary.
from leaspy.models import CovariateLogisticModel, LogisticModelSchiratti

logistic_model = LogisticModelSchiratti(
    name="logistic-initialization",
    source_dimension=2,
)
logistic_model.fit(
    data,
    "mcmc_saem",
    seed=0,
    n_iter=100,
    progress_bar=False,
)

# %%
# We then fit the covariate model, initialized from the logistic model.
covariate_model = CovariateLogisticModel(
    name="covariate-model",
    source_dimension=2,
    init_from_model=logistic_model,
)
covariate_model.fit(
    data,
    "mcmc_saem",
    seed=42,
    n_iter=1000,
    progress_bar=False,
)

# %%
# The fitted population and covariate-effect parameters are available below.
covariate_model.parameters

# %%
# The next example shows how to simulate data from a fitted model:
# see :doc:`plot_06_simulate`.
