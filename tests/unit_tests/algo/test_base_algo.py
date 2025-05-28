from dataclasses import dataclass

import torch

from leaspy.algo import BaseAlgo
from tests import LeaspyTestCase


@dataclass
class FakeAlgorithmSettings:
    name: str = None
    seed: int = None
    parameters: dict = None


class TestAbstractAlgo(LeaspyTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling
        super().setUpClass()

        cls.fake_algo_settings = FakeAlgorithmSettings()

    @LeaspyTestCase.allow_abstract_class_init(BaseAlgo)
    def test_constructor(self):
        algo = BaseAlgo(self.fake_algo_settings)
        self.assertEqual(algo.algo_parameters, None)
        self.assertEqual(algo.name, None)
        self.assertEqual(algo.output_manager, None)
        self.assertEqual(algo.seed, None)
        self.assertIsNone(algo.family)
        self.assertFalse(algo.deterministic)

    @LeaspyTestCase.allow_abstract_class_init(BaseAlgo)
    def test_initialize_seed(self):
        algo = BaseAlgo(self.fake_algo_settings)
        seed = torch.randint(10000, (1,)).item()
        algo._initialize_seed(seed)
        self.assertEqual(seed, torch.random.initial_seed())

    @LeaspyTestCase.allow_abstract_class_init(BaseAlgo)
    def test_load_parameters(self):
        algo = BaseAlgo(self.fake_algo_settings)
        algo.algo_parameters = {"param1": 1, "param2": 2}
        parameters = {"param1": 10, "param3": 3}
        algo.load_parameters(parameters)
        self.assertEqual(
            list(algo.algo_parameters.keys()), ["param1", "param2", "param3"]
        )
        self.assertEqual(algo.algo_parameters["param1"], 10)
        self.assertEqual(algo.algo_parameters["param2"], 2)
        self.assertEqual(algo.algo_parameters["param3"], 3)
