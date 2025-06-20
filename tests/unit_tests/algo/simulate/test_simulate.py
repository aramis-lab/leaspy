import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from leaspy.algo import AlgorithmSettings, algorithm_factory
from leaspy.algo.base import AlgorithmType, BaseAlgorithm
from leaspy.algo.simulate import SimulationAlgorithm
from leaspy.datasets import load_dataset
from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data.data import Data
from leaspy.io.outputs import IndividualParameters
from leaspy.io.outputs.result import Result
from leaspy.models import LogisticModel, ModelName, ModelSettings, model_factory
from tests import LeaspyTestCase


class SimulateAlgoTest(LeaspyTestCase):
    @classmethod
    def setUpClass(cls):
        temp_instance = cls()

        putamen_df = load_dataset("parkinson")
        data = Data.from_dataframe(
            putamen_df[
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

        cls.model_loaded = LogisticModel(name="test-model", source_dimension=2)
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
            "first_visit_mean": 0.0,  # OK1
            "first_visit_std": 0.4,  # OK2
            "time_follow_up_mean": 11,  # OK
            "time_follow_up_std": 0.5,  # OK
            "distance_visit_mean": 2 / 12,  # OK # 1.
            "distance_visit_std": 0.75 / 12,  # OK # 6
            "min_spacing_between_visits": 1,
        }

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

        df_sim = df_sim.data.to_dataframe()

        self.assertFalse(df_sim.empty)
        self.assertEqual(len(df_sim["ID"].unique()), 5)

        # Check times are increasing with variability
        for id in df_sim["ID"].unique():
            times = df_sim.loc[df_sim["ID"] == id, "TIME"].values
            diffs = np.diff(times)
            self.assertTrue((diffs > 0).all())

    def test_dataframe_visits(self):
        df_input = pd.DataFrame({"ID": ["p1", "p1", "p2"], "TIME": [50.0, 51.0, 52.0]})

        visits_param = {"visit_type": "dataframe", "df_visits": df_input}

        model = self.model_loaded
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
            visit_parameters=visits_param,
        )

        df_sim = df_sim.data.to_dataframe()

        # Check all input times are present
        df_sim = df_sim.set_index(["ID", "TIME"])
        input_indices = df_input.set_index(["ID", "TIME"]).index
        simulated_indices = df_sim.index

        for idx in input_indices:
            self.assertIn(idx, simulated_indices)

        # Check features exist
        for feature in model.features:
            self.assertIn(feature, df_sim.columns)
