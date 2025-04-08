from unittest import skip

import torch

from leaspy.models.obs_models import OrdinalObservationModel
from tests import LeaspyTestCase


class TestMultivariateModelOrdinal(LeaspyTestCase):
    @skip("Broken: Ordinal models are currently broken")
    def test_reload_model(self):
        model = self.get_hardcoded_model("logistic_ordinal").model
        self.assertIsInstance(model.obs_models[0], OrdinalObservationModel)
        ordinal_infos = model.ordinal_infos
        ordinal_mask = ordinal_infos.pop("mask")  # test after
        self.assertEqual(
            ordinal_infos,
            {
                "batch_deltas": False,
                "max_levels": {
                    "Y0": 3,
                    "Y1": 4,
                    "Y2": 6,
                    "Y3": 10,
                },
                "max_level": 10,
            },
        )
        expected_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(ordinal_mask, expected_mask))  # not approximate

    @skip("Broken: Ordinal models are currently broken")
    def test_reload_model_batched(self):
        model = self.get_hardcoded_model("logistic_ordinal_b").model
        self.assertIsInstance(model.obs_models[0], OrdinalObservationModel)
        ordinal_infos = model.ordinal_infos
        ordinal_mask = ordinal_infos.pop("mask")  # test after
        self.assertEqual(
            ordinal_infos,
            {
                "batch_deltas": True,
                "max_levels": {
                    "Y0": 3,
                    "Y1": 4,
                    "Y2": 6,
                    "Y3": 10,
                },
                "max_level": 10,
            },
        )
        expected_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(ordinal_mask, expected_mask))  # not approximate

    @skip("Broken: Ordinal models are currently broken")
    def test_reload_model_ordinal_ranking(self):
        model = self.get_hardcoded_model("logistic_ordinal_ranking").model
        self.assertIsInstance(model.obs_models[0], OrdinalObservationModel)
        ordinal_infos = model.ordinal_infos
        ordinal_mask = ordinal_infos.pop("mask")  # test after
        self.assertEqual(
            ordinal_infos,
            {
                "batch_deltas": False,
                "max_levels": {
                    "Y0": 3,
                    "Y1": 4,
                    "Y2": 6,
                    "Y3": 10,
                },
                "max_level": 10,
            },
        )
        expected_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(ordinal_mask, expected_mask))  # not approximate
