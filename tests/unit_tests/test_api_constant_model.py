import numpy as np
import pandas as pd

from leaspy.algo import AlgorithmSettings
from leaspy.api import Leaspy
from leaspy.models import ConstantModel
from tests import LeaspyTestCase


class ConstantModelAPITest(LeaspyTestCase):
    def test_run(self):
        # Data
        data = self.get_suited_test_data_for_model("constant_multivariate")

        # Settings
        # The `prediction_type` could be `last`, `last_known`, `max` or `mean`
        settings = AlgorithmSettings("constant_prediction", prediction_type="last")

        self.assertEqual(settings.parameters["prediction_type"], "last")

        # Leaspy
        lsp = Leaspy("constant")
        model = lsp.model
        self.assertIsInstance(model, ConstantModel)

        # Personalize
        ip = lsp.personalize(data, settings)

        self.assertListEqual(model.features, ["Y0", "Y1", "Y2", "Y3"])
        self.assertEqual(model.dimension, 4)

        # Estimate
        timepoints = {"178": [30, 31]}
        cst_data_for_178 = [0.73333, 0.0, 0.2, 0.4]
        results = lsp.estimate(timepoints, ip)
        self.assertEqual(results.keys(), {"178"})
        self.assertEqual(results["178"].shape, (2, 4))
        for i, v in enumerate(cst_data_for_178):
            self.assertAlmostEqual(results["178"][0].tolist()[i], v, delta=10e-5)
            self.assertAlmostEqual(results["178"][1].tolist()[i], v, delta=10e-5)

        # Estimate with a pandas.MultiIndex
        ix = pd.MultiIndex.from_tuples(
            [(-1, "178", "XXX", 30, True), (-42, "178", "YYY", 31, False)],
            names=["extra_level_1", "ID", "extra_level_2", "TIME", "extra_level_3"],
        )

        expected_res = pd.DataFrame(
            [cst_data_for_178] * 2, index=ix, columns=model.features, dtype=np.float32
        )

        results = lsp.estimate(ix, ip)
        self.assertIsInstance(results, pd.DataFrame)
        pd.testing.assert_frame_equal(results, expected_res)

        # Estimate with regular dict but request a dataframe
        results = lsp.estimate(timepoints, ip, to_dataframe=True)
        self.assertIsInstance(results, pd.DataFrame)
        pd.testing.assert_frame_equal(
            results,
            expected_res.droplevel(["extra_level_1", "extra_level_2", "extra_level_3"]),
        )
