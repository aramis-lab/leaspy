from leaspy.algo import AlgorithmSettings
from leaspy.algo.personalize import PersonalizeAlgorithm
from tests import LeaspyTestCase


class AbstractPersonalizeAlgoTest(LeaspyTestCase):
    @LeaspyTestCase.allow_abstract_class_init(PersonalizeAlgorithm)
    def test_constructor(self):
        settings = AlgorithmSettings("scipy_minimize")
        PersonalizeAlgorithm.name = "scipy_minimize"
        algo = PersonalizeAlgorithm(settings)

        self.assertEqual(algo.name, "scipy_minimize")
        self.assertEqual(algo.seed, None)
        self.assertEqual(algo.family.value, "personalize")
