import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from leaspy.algo import AlgorithmSettings, algorithm_factory
from leaspy.algo.base import AlgorithmType, BaseAlgorithm
from leaspy.algo.simulate import JointSimulationAlgorithm
from leaspy.datasets import load_dataset
from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data.data import Data
from leaspy.io.outputs import IndividualParameters
from leaspy.io.outputs.result import Result
from leaspy.models import JointModel, ModelName, ModelSettings, model_factory
from tests import LeaspyTestCase


class JointSimulateAlgoTest(LeaspyTestCase):
    @classmethod
    def setUpClass(cls):
        temp_instance = cls()

        joint_df = load_dataset("simulated_data_for_joint")
        data = Data.from_dataframe(joint_df, "joint")

        cls.model_loaded = JointModel(name="test_model", nb_events=1)
        auto_path_logs = temp_instance.get_test_tmp_path("model-logs")
        cls.model_loaded.fit(
            data,
            "mcmc_saem",
            seed=0,
            n_iter=100,
            progress_bar=False,
            path=auto_path_logs,
            overwrite_logs_folder=True,
        )

    def test_random_visits(self):
        model = self.model_loaded

        visit_params = {
            "patient_number": 5,
            "visit_type": "random",
            # 'visit_type': "dataframe",
            # "df_visits": df_test
            "first_visit_mean": 0.0,  # OK
            "first_visit_std": 5.7,  # OK
            "time_follow_up_mean": 6.4,  # OK
            "time_follow_up_std": 1.2,  # OK
            "distance_visit_mean": 0.7,  # OK 
            "distance_visit_std": 0.3,  # OK
            "min_spacing_between_visits": 0.3,
        }

        df_sim = model.simulate(
            algorithm="joint_simulate",
            features=["Y0", "Y1", "Y2", "Y3"],
            visit_parameters=visit_params,
        )

        df_sim = df_sim.data.to_dataframe()

        self.assertFalse(df_sim.empty)
        self.assertEqual(len(df_sim["ID"].unique()), 5)

        # Check times are increasing with variability
        for id in df_sim["ID"].unique():
            times = df_sim.loc[df_sim["ID"] == id, "TIME"].values
            diffs = np.diff(times)
            self.assertTrue((diffs > 0).all())

    def test_dataframe_visits(self):
        df_input = pd.DataFrame({"ID": ["p1", "p1", "p2"], "TIME": [79.0, 80.0, 81.0]})

        visits_param = {"visit_type": "dataframe", "df_visits": df_input}

        model = self.model_loaded
        df_sim = model.simulate(
            algorithm="joint_simulate",
            features=["Y0", "Y1", "Y2", "Y3"],
            visit_parameters=visits_param,
        )

        df_sim = df_sim.data.to_dataframe()
        self.assertFalse(df_sim.empty)

        # Simulated visits must be a *subset* of the input visits: the algorithm may
        # legitimately drop visits that occur after a simulated event, but it must
        # never introduce times that were not in the input dataframe.
        df_sim = df_sim.set_index(["ID", "TIME"])
        input_indices = df_input.set_index(["ID", "TIME"]).index
        simulated_indices = df_sim.index

        for idx in simulated_indices:
            self.assertIn(idx, input_indices)

        # Check features exist
        for feature in model.features:
            self.assertIn(feature, df_sim.columns)
