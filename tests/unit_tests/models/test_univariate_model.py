from leaspy.models.abstract_model import AbstractModel
from leaspy.models.univariate import UnivariateModel
from leaspy.models.noise_models import NOISE_MODELS

from tests import LeaspyTestCase


class ManifoldModelTest_Mixin(LeaspyTestCase):

    def check_common_attrs(self, model):
        self.assertIsInstance(model, AbstractModel)

        self.assertIsNone(model.attributes)
        self.assertIsNone(model.bayesian_priors)

        self.assertIsNone(model.parameters['g'])
        self.assertIsNone(model.parameters['tau_mean'])
        self.assertIsNone(model.parameters['tau_std'])
        self.assertIsNone(model.parameters['xi_mean'])
        self.assertIsNone(model.parameters['xi_std'])
        #self.assertIsNone(model.noise_model.parameters['scale'])
        self.assertIsNone(model.noise_model.parameters)

        self.assertIsNone(model.MCMC_toolbox['attributes'])
        self.assertIsNone(model.MCMC_toolbox['priors']['g_std'])

class UnivariateModelTest(ManifoldModelTest_Mixin):

    def test_univariate_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        """
        for name in ['univariate_linear', 'univariate_logistic']:

            model = UnivariateModel(name)
            self.assertIsInstance(model, UnivariateModel)
            self.assertEqual(model.name, name)
            self.assertEqual(model.dimension, 1)
            self.assertEqual(model.source_dimension, 0)
            self.assertIsInstance(model.noise_model, NOISE_MODELS['gaussian-scalar'])

            self.check_common_attrs(model)

    def test_wrong_name(self):

        with self.assertRaises(ValueError):
            UnivariateModel('univariate_unknown-suffix')

    def test_get_attributes(self):

        m = UnivariateModel('univariate_logistic')

        # not supported attributes (only None & 'MCMC' are)
        with self.assertRaises(ValueError):
            m._get_attributes(False)
        with self.assertRaises(ValueError):
            m._get_attributes(True)
        with self.assertRaises(ValueError):
            m._get_attributes('toolbox')
