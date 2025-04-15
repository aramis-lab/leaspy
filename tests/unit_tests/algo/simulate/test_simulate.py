import unittest
import pandas as pd
import numpy as np
from tests import LeaspyTestCase
from unittest.mock import Mock, patch
from leaspy.api import Leaspy
from leaspy.datasets import load_dataset
from leaspy.io.outputs import IndividualParameters
from leaspy.algo import AlgorithmSettings, algorithm_factory
from leaspy.algo.base import AbstractAlgo, AlgorithmType
from leaspy.models import ModelName, ModelSettings, model_factory
from leaspy.io.outputs.result import Result
from leaspy.io.data.data import Data
from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.algo.simulate import SimulationAlgorithm

class SimulateAlgoTest(LeaspyTestCase):
    
    @classmethod
    def setUpClass(cls):

        temp_instance = cls()

        putamen_df = load_dataset("parkinson-multivariate")
        data = Data.from_dataframe(putamen_df[['MDS1_total','MDS2_total','MDS3_off_total', 'SCOPA_total','MOCA_total','REM_total','PUTAMEN_R','PUTAMEN_L','CAUDATE_R','CAUDATE_L']])
        
        cls.model_loaded = Leaspy('logistic', dimension=10, source_dimension=2)
        fit_settings = AlgorithmSettings('mcmc_saem', seed=0, n_iter=50)
        
        auto_path_logs = temp_instance.get_test_tmp_path("model-logs")
        fit_settings.set_logs(path=auto_path_logs)
        
        cls.model_loaded.fit(data, fit_settings)

    def test_regular_visits(self):
        model_loaded = self.model_loaded
        settings = AlgorithmSettings(
                                    "simulation",
                                    seed=0,
                                    features=["MDS1_total", "MDS2_total", "MDS3_off_total", 'SCOPA_total','MOCA_total','REM_total','PUTAMEN_R','PUTAMEN_L','CAUDATE_R','CAUDATE_L'], 
                                    visit_parameters={
                                        "visit_type": "regular",
                                        "pat_nb": 5,
                                        "regular_visit": 1.0,
                                        "fv_mean": 50.0,
                                        "fv_std": 2.0,
                                        "tf_mean": 10.0,
                                        "tf_std": 1.0,
                                    }
                                    )

        algo = algorithm_factory(settings)
        df_sim= algo.run_impl(model_loaded.model)
        df_sim = df_sim.data.to_dataframe()

                
        expected_columns = ["ID", "TIME"] + settings.parameters["features"]
        assert all(col in df_sim.columns for col in expected_columns)
        self.assertFalse(df_sim.empty)
        self.assertEqual(len(df_sim["ID"].unique()), 5)

        for id in df_sim["ID"].unique():
            times = df_sim[df_sim["ID"] == id]["TIME"].values
            assert np.allclose(np.diff(times), 1.0, atol=0.1)
    
    def test_random_visits(self):
        model_loaded = self.model_loaded
        settings = AlgorithmSettings(
                                    "simulation",
                                    seed=0,
                                    features=["MDS1_total", "MDS2_total", "MDS3_off_total", 'SCOPA_total','MOCA_total','REM_total','PUTAMEN_R','PUTAMEN_L','CAUDATE_R','CAUDATE_L'], 
                                    visit_parameters={
                                    "visit_type": "random",
                                    "pat_nb": 5,
                                    "fv_mean": 50.0,
                                    "fv_std": 2.0,
                                    "tf_mean": 10.0,
                                    "tf_std": 1.0,
                                    "distv_mean": 1.0,
                                    "distv_std": 0.2,
                                }
                            )
        algo = algorithm_factory(settings)
        df_sim= algo.run_impl(model_loaded.model)
        df_sim = df_sim.data.to_dataframe()

        
        self.assertFalse(df_sim.empty)
        self.assertEqual(len(df_sim["ID"].unique()), 5)

        # Check times are increasing with variability
        for id in df_sim["ID"].unique():
            times = df_sim.loc[df_sim["ID"] == id, "TIME"].values
            diffs = np.diff(times)
            self.assertTrue((diffs > 0).all())
            self.assertGreater(np.std(diffs), 0)

    def test_dataframe_visits(self):
        model_loaded = self.model_loaded
        df_input = pd.DataFrame({
            "ID": ["p1", "p1", "p2"],
            "TIME": [50.0, 51.0, 52.0]
        })

        settings = AlgorithmSettings(
                                    "simulation",
                                    seed=0,
                                    features=["MDS1_total", "MDS2_total", "MDS3_off_total", 'SCOPA_total','MOCA_total','REM_total','PUTAMEN_R','PUTAMEN_L','CAUDATE_R','CAUDATE_L'], 
                                    visit_parameters={
                                        "visit_type": "dataframe",
                                        "df_visits": df_input
                                    }
                                    )
        
        algo = algorithm_factory(settings)
        df_sim= algo.run_impl(model_loaded.model)
        df_sim = df_sim.data.to_dataframe()
        
        # Check all input times are present
        df_sim = df_sim.set_index(["ID", "TIME"])        
        input_indices = df_input.set_index(["ID", "TIME"]).index
        simulated_indices = df_sim.index

        for idx in input_indices:
            self.assertIn(idx, simulated_indices)

        # Check features exist
        for feature in settings.parameters["features"]:
            self.assertIn(feature, df_sim.columns)