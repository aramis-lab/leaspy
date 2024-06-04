from leaspy.models.constant import ConstantModel

from tests import LeaspyTestCase


class ConstantModelTest(LeaspyTestCase):

    def test_constructor(self):
        model = ConstantModel('constant')
        self.assertEqual(model.name, 'constant')
        self.assertTrue(model.is_initialized)  # no need for a fit
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)




