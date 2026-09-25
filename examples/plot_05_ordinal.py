"""
Ordinal Model
=============

This example illustrates the main steps for using the ``OrdinalModel``:

1. loading and inspecting ordinal longitudinal data;
2. fitting the population model;
3. interpreting the fitted population parameters and visualizing ordinal transitions.
"""

# %%
# Load the example longitudinal dataset. Each feature is measured on an ordered discrete scale.

from pathlib import Path

import leaspy.datasets as _leaspy_datasets
from leaspy.io.data import Data

data_path = Path(_leaspy_datasets.__file__).parent / "data" / "ordinal_example.csv"
data = Data.from_csv_file(data_path)

# Each row represents a patient visit; columns after ID and TIME are ordinal items.
print(data.to_dataframe().head())

# %%
# Show how often each ordinal score occurs for each item.
# Rows are ordinal scores, and columns are items (Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8).
# Each cell gives the number of patient visits with that score for that item.
# Items may have different ranges of scores.

import pandas as pd

raw_df = data.to_dataframe()
features = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"]

level_counts = pd.DataFrame({
    item: raw_df[item].value_counts(dropna=False)
    for item in features
}).fillna(0).astype(int)

level_counts.columns.name = "Item"
level_counts.index.name = "Ordinal score"

print(level_counts.sort_index().to_string())

# %%
# Fit the population model to the patients' longitudinal observations.
# ``source_dimension`` sets the number of latent sources used to capture variation across items.
# A practical starting point is roughly the square root of the number of items. For eight items, we choose three sources.

from leaspy.models import OrdinalModel

model = OrdinalModel(name="ordinal", source_dimension=3)
model.fit(
    data,
    "mcmc_saem",
    seed=42,
    n_iter=100,
    progress_bar=False,
)


# %%
# Inspect the fitted parameters describing the population trajectory and the effects of latent sources.

for name in ("log_g_mean", "log_v0_mean", "tau_mean", "betas_mean"):
    print(f"{name}:\n{model.parameters[name]}\n")

# In betas_mean, columns correspond to latent sources; rows give the coefficients for the items.

# %%
# In addition, the ordinal model estimates ``log_deltas_mean``.
# We exponentiate its finite values to display the delays on their original scale.

import numpy as np
log_deltas = model.parameters["log_deltas_mean"].detach().cpu().numpy()

deltas = np.full_like(log_deltas, np.nan)
valid = np.isfinite(log_deltas)
deltas[valid] = np.exp(log_deltas[valid])


# %%
# Each row represents an item. ``delta_h`` describes the spacing between
# the transitions into levels ``h-1`` and ``h``; ``delta_1`` is fixed to zero and is omitted.
# Larger deltas indicate longer intervals between consecutive score transitions.
# Positions without a corresponding transition are masked and will appear as blank cells in the table.

delta_table = pd.DataFrame(
    deltas,
    index=pd.Index(features, name="Item"),
    columns=[f"delta_{h}" for h in range(2, log_deltas.shape[1] + 2)],
)

print(delta_table.round(3).to_string(na_rep="—"))



# %%
# Visualize how the fitted delays space score transitions along the disease progression timeline.
# The first transition of each item is aligned at time zero.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 3.5))

for feature, row in zip(features, deltas):
    x = np.r_[0, np.cumsum(row[np.isfinite(row)])]
    ax.step(np.r_[x, 7.5], np.r_[1:len(x) + 1, len(x)], where="post", label=feature)

ax.set(xlim=(0, 7.5), xticks=np.arange(8), xlabel="Time since first transition (years)", ylabel="Ordinal score")
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()

# %%
# For an example of the simulation workflow, see :doc:`plot_06_simulate`.
