from dataclasses import dataclass
from typing import List

from leaspy.models.lme import LMEModel

from tests import LeaspyTestCase


@dataclass
class MockDataset:
    headers: List[str]

    def __post_init__(self):
        self.dimension = len(self.headers)

class LMEModelTest(LeaspyTestCase):

    def test_constructor(self):
        model = LMEModel('lme')
        self.assertEqual(model.name, 'lme')
        self.assertFalse(model.is_initialized)  # new: more coherent (needs a fit)
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, 1)

    def test_str_repr(self):
        model = LMEModel('lme')
        self.assertIsInstance(str(model), str)

    def test_bad_hyperparams(self):
        with self.assertRaisesRegex(ValueError, 'hyperparameter'):
            LMEModel('lme', unknown_hyperparameter=False)

    def test_bad_properties_consistency(self):
        with self.assertRaisesRegex(ValueError, 'hyperparameter'):
            LMEModel('lme', features=['a', 'b'], dimension=1)

    def test_init_reinit(self):
        model = LMEModel('lme')

        mock_dataset = MockDataset(['ft_1'])
        model.initialize(mock_dataset)

        self.assertTrue(model.hyperparameters_ok())

        mock_dataset_new = MockDataset(['ft_other'])
        with self.assertWarnsRegex(UserWarning, 'features'):
            model.initialize(mock_dataset_new)
