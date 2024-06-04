from leaspy.models.multivariate import MultivariateModel

from tests import LeaspyTestCase


class TestMultivariateModel(LeaspyTestCase):

    def test_wrong_name(self):

        with self.assertRaises(ValueError):
            MultivariateModel('unknown-suffix')
