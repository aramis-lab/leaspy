"""
Joint Model
=====================================================
This notebook contains the code for a simple implementation of the Leaspy Joint model on synthetic data.
"""

# %%
# The following imports are required libraries for numerical computation and data manipulation.
import os

import pandas as pd

import leaspy
from leaspy.io.data import Data

leaspy_root = os.path.dirname(leaspy.__file__)

data_path = os.path.join(leaspy_root, "datasets/data/simulated_data_for_joint.csv")

df = pd.read_csv(data_path, dtype={"ID": str}, sep=";")
print(df.head())

# %%
# To use the Joint Model in Leaspy, your dataset must include the following columns:
#
# 1. **ID** : Patient identifier
# 2. **TIME** : Time of measurement
# 3. **EVENT_TIME** : Time of the event
# 4. **EVENT_BOOL** : Event indicator:
#    - `1` if the event occurred
#    - `0` if censored
#    - `2` if a competing event occurred
#
# For one patient, the event time and event bool are the same for each row.


# %%
# We load the Joint Model from the `leaspy.models` and transform the dataset in a leaspy-compatible form with the built-in functions.
from leaspy.models import JointModel

data = Data.from_dataframe(df, "joint")
model = JointModel(name="test_model", nb_events=1, source_dimension=2)

# %%
# The parameter `nb_events` should match the number of distinct event types
# present in the `EVENT_BOOL` column:
#
# - If `EVENT_BOOL` contains values {0, 1}, then `nb_events=1`.
# - If it contains values {0, 1, 2}, then `nb_events=2`.
#
# Once the model is initialized, we can fit it to the data.

model.fit(data, "mcmc_saem", seed=1312, n_iter=100, progress_bar=False)


# %%
# The Joint Model includes specific parameters such as `log_rho_mean` and `zeta_mean`.
print(model.parameters)

# %%
# We have seend how to fit a Joint Model using Leaspy. It also provides other models as the
# [Mixture Model](./plot_04_mixture) that can be explored in the next examples.