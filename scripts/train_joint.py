import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os

from leaspy.io.data import Data
from leaspy.datasets import load_dataset

df = load_dataset("simulated_data_for_joint")

from leaspy.models import JointModel

data = Data.from_dataframe(df, "joint")
model = JointModel(name="joint", nb_events=1)

model.fit(data, "mcmc_saem", seed=1312, n_iter=100000, progress_bar=True) 

os.makedirs("models", exist_ok=True)

model.save("models/model_on_simulated_data_for_joint_100000_iterations.json")