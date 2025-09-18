"""
Simulating Data with Leaspy
=====================================================

This example demonstrates how to use Leaspy to simulate longitudinal data based on a fitted model.
"""

# %%
# The following imports bring in the required modules and load the synthetic Parkinson dataset from Leaspy.
# A logistic model will be fitted on this dataset and then used to simulate new longitudinal data.
from leaspy.datasets import load_dataset
from leaspy.io.data import Data

df = load_dataset("parkinson")

# %%
# The clinical and imaging features of interest are selected and the DataFrame is converted
# into a Leaspy `Data` object that can be used for model fitting.
data = Data.from_dataframe(
    df[
        [
            "MDS1_total",
            "MDS2_total",
            "MDS3_off_total",
            "SCOPA_total",
            "MOCA_total",
            "REM_total",
            "PUTAMEN_R",
            "PUTAMEN_L",
            "CAUDATE_R",
            "CAUDATE_L",
        ]
    ]
)

# %%
# A logistic model with a two-dimensional latent space is initialized.
from leaspy.models import LogisticModel

model = LogisticModel(name="test-model", source_dimension=2)

# %%
# The model is fitted to the data using the MCMC-SAEM algorithm.
# A fixed seed is used for reproducibility and 100 iterations are performed.
model.fit(
    data,
    "mcmc_saem",
    n_iter=100,
    progress_bar=False,
)

# %%
# The parameters for simulating patient visits are defined.
# These parameters specify the number of patients, the visit spacing, and the timing variability.
visit_params = {
    "patient_number": 5,
    "visit_type": "random",  # The visit type could also be 'dataframe' with df_visits.
    # "df_visits": df_test           # Example for custom visit schedule.
    "first_visit_mean": 0.0,  # The mean of the first visit age/time.
    "first_visit_std": 0.4,  # The standard deviation of the first visit age/time.
    "time_follow_up_mean": 11,  # The mean follow-up time.
    "time_follow_up_std": 0.5,  # The standard deviation of the follow-up time.
    "distance_visit_mean": 2 / 12,  # The mean spacing between visits in years.
    "distance_visit_std": 0.75
    / 12,  # The standard deviation of the spacing between visits in years.
    "min_spacing_between_visits": 1,  # The minimum allowed spacing between visits.
}

# %%
# A new longitudinal dataset is simulated from the fitted model using the specified parameters.
df_sim = model.simulate(
    algorithm="simulate",
    features=[
        "MDS1_total",
        "MDS2_total",
        "MDS3_off_total",
        "SCOPA_total",
        "MOCA_total",
        "REM_total",
        "PUTAMEN_R",
        "PUTAMEN_L",
        "CAUDATE_R",
        "CAUDATE_L",
    ],
    visit_parameters=visit_params,
)

# %%
# The simulated data is converted back to a pandas DataFrame for inspection.
df_sim = df_sim.data.to_dataframe()

# %%
# The simulated longitudinal dataset is displayed below.
print(df_sim)
