from leaspy.models.utils.attributes import AttributesFactory, LogisticAttributes
from tests import LeaspyTestCase


class AttributesFactoryTest(LeaspyTestCase):
    def test_attributes(self):
        """Test attributes static method"""
        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ["lgistic", "blabla"]
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(
                ValueError,
                lambda name: AttributesFactory.attributes(name, 4, 2),
                wrong_arg,
            )

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_exemples = [3.8, {"truc": 0.1}]
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(
                ValueError,
                lambda name: AttributesFactory.attributes(name, 4, 2),
                wrong_arg,
            )

        # Test if lower name:
        name_exemples = ["logistic", "LogIStiC", "LOGISTIC"]
        for name in name_exemples:
            self.assertIsInstance(
                AttributesFactory.attributes(name, 4, 2), LogisticAttributes
            )

    def test_bad_consistency_univariate_dim(self):
        with self.assertRaisesRegex(ValueError, "univariate"):
            AttributesFactory.attributes(
                name="univariate_logistic", dimension=2, source_dimension=0
            )

        with self.assertRaisesRegex(ValueError, "univariate"):
            AttributesFactory.attributes(
                name="logistic", dimension=1, source_dimension=0
            )
