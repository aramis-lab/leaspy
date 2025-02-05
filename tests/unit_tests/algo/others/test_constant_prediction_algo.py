from math import isnan

import numpy as np
import pandas as pd

from leaspy.algo import AlgorithmSettings
from leaspy.algo.others import ConstantPredictionAlgorithm
from leaspy.io.data import Data, Dataset
from leaspy.models.constant import ConstantModel
from tests import LeaspyTestCase


class ConstantPredictionAlgorithmTest(LeaspyTestCase):
    @classmethod
    def setUpClass(cls):
        # for tmp handling
        super().setUpClass()

        arr = [
            ["1", 1.0, 2.0, 1.0],
            ["1", 3.0, 3.0, float("nan")],  # non-sorted
            ["1", 2.0, 4.0, 3.0],
        ]

        df = pd.DataFrame(data=arr, columns=["ID", "TIME", "A", "B"]).set_index(
            ["ID", "TIME"]
        )
        data = Data.from_dataframe(df)
        cls.dataset = Dataset(data)

    def test_constructor(self):
        settings = AlgorithmSettings("constant_prediction")
        algo = ConstantPredictionAlgorithm(settings)
        self.assertEqual(algo.name, "constant_prediction")
        self.assertEqual(algo.prediction_type, "last")
        self.assertTrue(algo.deterministic)
        self.assertEqual(algo.family, "personalize")

        for prediction_type in ["last", "last_known", "max", "mean"]:
            settings = AlgorithmSettings(
                "constant_prediction", prediction_type=prediction_type
            )
            algo = ConstantPredictionAlgorithm(settings)
            self.assertEqual(algo.prediction_type, prediction_type)

    def test_get_individual_last_values(self):
        times = [31, 32, 34, 33]
        values = np.array(
            [[1.0, 0.5], [2.0, 0.5], [float("nan"), 2.0], [3.0, float("nan")]]
        )

        results = [
            ("last", {"A": None, "B": 2.0}),
            ("last_known", {"A": 3.0, "B": 2.0}),
            ("max", {"A": 3.0, "B": 2.0}),
            ("mean", {"A": 2.0, "B": 1.0}),
        ]

        for prediction_type, res in results:
            settings = AlgorithmSettings(
                "constant_prediction", prediction_type=prediction_type
            )
            algo = ConstantPredictionAlgorithm(settings)
            ind_ip = algo._get_individual_last_values(times, values, fts=["A", "B"])

            # replace nans by None for comparisons
            ind_ip = {k: None if isnan(v) else v for k, v in ind_ip.items()}

            self.assertEqual(ind_ip, res)

    def test_run_last(self):
        results = [
            ("last", {"1": {"A": 3.0, "B": float("nan")}}),
            ("last_known", {"1": {"A": 3, "B": 3.0}}),
            ("max", {"1": {"A": 4.0, "B": 3.0}}),
            ("mean", {"1": {"A": 3.0, "B": 2.0}}),
        ]

        for pred_type, res in results:
            settings = AlgorithmSettings(
                "constant_prediction", prediction_type=pred_type
            )
            algo = ConstantPredictionAlgorithm(settings)
            model = ConstantModel("constant")

            ip = algo.run(model, self.dataset, return_loss=True)
            self.assertListEqual(ip._indices, ["1"])
            self.assertDictEqual(ip._parameters_shape, {"A": (), "B": ()})
            self.assertEqual(ip._default_saving_type, "csv")

            dict_ip = ip._individual_parameters

            if pred_type == "last":
                self.assertEqual(dict_ip.keys(), {"1": 0}.keys())
                self.assertEqual(dict_ip["1"]["A"], 3.0)
                self.assertTrue(np.isnan(dict_ip["1"]["B"]))
            else:
                self.assertDictEqual(ip._individual_parameters, res)
