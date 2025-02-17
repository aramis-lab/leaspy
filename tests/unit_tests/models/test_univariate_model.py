from unittest import skip

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.obs_models import FullGaussianObservationModel
from leaspy.models.univariate import LinearUnivariateModel, LogisticUnivariateModel
from tests import LeaspyTestCase


class ManifoldModelTestMixin(LeaspyTestCase):
    def check_common_attrs(self, model: AbstractModel):
        self.assertIsInstance(model, AbstractModel)
        for variable in ("g", "tau_mean", "tau_std", "xi_mean", "xi_std"):
            self.assertIn(variable, model.state.dag)


class UnivariateModelTest(ManifoldModelTestMixin):
    def _generic_testing(self, model: AbstractModel):
        self.assertEqual(model.name, "test_model")
        self.assertEqual(model.dimension, 1)
        self.assertEqual(model.source_dimension, 0)
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        model.initialize()
        self.check_common_attrs(model)

    def test_univariate_logistic_constructor(self):
        """
        Test attribute's initialization of leaspy univariate logistic model.
        """
        model = LogisticUnivariateModel("test_model")
        self.assertIsInstance(model, LogisticUnivariateModel)
        self._generic_testing(model)

    # @skip("Linear models are currently broken")
    def test_univariate_linear_constructor(self):
        """
        Test attribute's initialization of leaspy univariate linear model.
        """
        model = LinearUnivariateModel("test_model")
        self.assertIsInstance(model, LinearUnivariateModel)
        self._generic_testing(model)
