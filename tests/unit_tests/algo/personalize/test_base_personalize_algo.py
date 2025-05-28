from leaspy.algo import AlgorithmSettings
from leaspy.algo.personalize import PersonalizeAlgo
from tests import LeaspyTestCase


class AbstractPersonalizeAlgoTest(LeaspyTestCase):
    @LeaspyTestCase.allow_abstract_class_init(PersonalizeAlgo)
    def test_constructor(self):
        settings = AlgorithmSettings("scipy_minimize")
        PersonalizeAlgo.name = "scipy_minimize"
        algo = PersonalizeAlgo(settings)

        self.assertEqual(algo.name, "scipy_minimize")
        self.assertEqual(algo.seed, None)
        self.assertEqual(algo.family.value, "personalize")
