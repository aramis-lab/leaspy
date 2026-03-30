"""
Ordinal Model
======================

Quick example of how to use the OrdinalModel in Leaspy.
"""

# %%
# Let's start by importing the Data class from the leaspy library and loading the csv into a Data object.

from pathlib import Path
import leaspy.datasets as _leaspy_datasets
from leaspy.io.data import Data

data_path = Path(_leaspy_datasets.__file__).parent / "data" / "ordinal.csv"
data = Data.from_csv_file(data_path)
print(data.to_dataframe().head())

# %%
# We can also take a look at the distribution of the values in the dataset. The OrdinalModel expects ordinal data, so we should see that the values are integers with a certain order.

import pandas as pd

raw_df = data.to_dataframe()
table = pd.DataFrame(
    {col: raw_df[col].value_counts(dropna=False).sort_index() for col in ["Y0", "Y1", "Y2", "Y3"]}
).fillna(0).astype(int)
table

# %%
# Then we can instantiate the OrdinalModel and fit it to the data. The model will automatically use the correct observation model for ordinal data, so we don't need to specify it.

from leaspy.models import OrdinalModel
model = OrdinalModel(name="ordinal", source_dimension=2)
model.fit(data, "mcmc_saem", seed=42, n_iter=100, progress_bar=False)

# %%
# After fitting the model, we can retrieve the population parameters and the individual parameters.

model.parameters

