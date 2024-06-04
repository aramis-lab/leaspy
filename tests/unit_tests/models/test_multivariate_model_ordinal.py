import torch

from leaspy.models.noise_models import NOISE_MODELS
from tests import LeaspyTestCase


class TestMultivariateModelOrdinal(LeaspyTestCase):

    def test_reload_model(self):

        model = self.get_hardcoded_model('logistic_ordinal').model

        self.assertIsInstance(model.noise_model, NOISE_MODELS['ordinal'])

        ordinal_infos = model.ordinal_infos
        ordinal_mask = ordinal_infos.pop('mask')  # test after
        self.assertEqual(ordinal_infos, {
            'batch_deltas': False,
            'max_levels': {
                'Y0': 3,
                'Y1': 4,
                'Y2': 6,
                'Y3': 10,
            },
            'max_level': 10,
        })

        expected_mask = torch.tensor([
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ])
        self.assertTrue(torch.equal(ordinal_mask, expected_mask))  # not approximate

    def test_reload_model_batched(self):

        model = self.get_hardcoded_model('logistic_ordinal_b').model

        self.assertIsInstance(model.noise_model, NOISE_MODELS['ordinal'])

        ordinal_infos = model.ordinal_infos
        ordinal_mask = ordinal_infos.pop('mask')  # test after
        self.assertEqual(ordinal_infos, {
            'batch_deltas': True,
            'max_levels': {
                'Y0': 3,
                'Y1': 4,
                'Y2': 6,
                'Y3': 10,
            },
            'max_level': 10,
        })

        expected_mask = torch.tensor([
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ])
        self.assertTrue(torch.equal(ordinal_mask, expected_mask))  # not approximate

    def test_reload_model_ordinal_ranking(self):

        model = self.get_hardcoded_model('logistic_ordinal_ranking').model

        self.assertIsInstance(model.noise_model, NOISE_MODELS['ordinal-ranking'])

        ordinal_infos = model.ordinal_infos
        ordinal_mask = ordinal_infos.pop('mask')  # test after
        self.assertEqual(ordinal_infos, {
            'batch_deltas': False,
            'max_levels': {
                'Y0': 3,
                'Y1': 4,
                'Y2': 6,
                'Y3': 10,
            },
            'max_level': 10,
        })

        expected_mask = torch.tensor([
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ])
        self.assertTrue(torch.equal(ordinal_mask, expected_mask))  # not approximate
