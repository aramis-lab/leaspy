from leaspy.models import LinearModel, LogisticModel, McmcSaemCompatibleModel
from leaspy.models.obs_models import FullGaussianObservationModel
from tests import LeaspyTestCase


class ManifoldModelTestMixin(LeaspyTestCase):
    def check_common_attrs(self, model: McmcSaemCompatibleModel):
        self.assertIsInstance(model, McmcSaemCompatibleModel)
        for variable in ("g", "tau_mean", "tau_std", "xi_mean", "xi_std"):
            self.assertIn(variable, model.state.dag)


class UnivariateModelTest(ManifoldModelTestMixin):
    def _generic_testing(self, model: McmcSaemCompatibleModel):
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
        model = LogisticModel("test_model", dimension=1)
        self.assertIsInstance(model, LogisticModel)
        self._generic_testing(model)

    # @skip("Linear models are currently broken")
    def test_univariate_linear_constructor(self):
        """
        Test attribute's initialization of leaspy univariate linear model.
        """
        model = LinearModel("test_model", dimension=1)
        self.assertIsInstance(model, LinearModel)
        self._generic_testing(model)
