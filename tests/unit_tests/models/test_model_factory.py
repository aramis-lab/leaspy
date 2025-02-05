from unittest import skipIf

from leaspy.models import LogisticUnivariateModel, ModelName, model_factory
from leaspy.models.obs_models import FullGaussianObservationModel
from tests import LeaspyTestCase

TEST_LINEAR_MODELS = True
SKIP_LINEAR_MODELS = "Linear models are currently broken"

TEST_LOGISTIC_PARALLEL_MODELS = False
SKIP_LOGISTIC_PARALLEL_MODELS = "Logistic parallel models are currently broken"


class ModelFactoryTestMixin(LeaspyTestCase):
    def check_model_factory_constructor(self, model):
        """
        Test initialization of leaspy model.

        Parameters
        ----------
        model : str, optional (default None)
            Name of the model
        """
        # valid name (preconditon)
        self.assertEqual(ModelName(model.name), model.name)


class ModelFactoryTest(ModelFactoryTestMixin):
    def test_wrong_arg(self):
        """Test if raise error for wrong argument"""
        # Test if raise ValueError if wrong string arg for name
        wrong_arg_examples = ("lgistic", "blabla")
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, model_factory, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_examples = [3.8, {"truc": 0.1}]
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, model_factory, wrong_arg)

    def _generic_univariate_hyperparameters_checker(self, model_name: str) -> None:
        model = model_factory(model_name, features=["t1"])
        self.assertEqual(model.features, ["t1"])
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        self.assertEqual(model.dimension, 1)
        self.assertEqual(model.source_dimension, 0)
        # inconsistent features for a univariate model (dimension=1)
        with self.assertRaisesRegex(ValueError, r"(?i)\bdimension\b.+\bfeatures\b"):
            model_factory(model_name, features=["t1", "t2", "t3"])

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_load_hyperparameters_univariate_linear(self):
        self._generic_univariate_hyperparameters_checker("univariate_linear")

    def test_load_hyperparameters_univariate_logistic(self):
        self._generic_univariate_hyperparameters_checker("univariate_logistic")

    def _generic_multivariate_hyperparameters_checker(self, model_name: str) -> None:
        model = model_factory(
            model_name,
            features=["t1", "t2", "t3"],
            source_dimension=2,
            dimension=3,
        )
        self.assertEqual(model.features, ["t1", "t2", "t3"])
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        self.assertEqual(model.dimension, 3)  # TODO: automatic from length of features?
        self.assertEqual(model.source_dimension, 2)
        with self.assertRaisesRegex(ValueError, r"(?i)\bhyperparameters\b.+\bblabla\b"):
            model_factory(model_name, blabla=2)

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_load_hyperparameters_multivariate_linear(self):
        self._generic_multivariate_hyperparameters_checker("linear")

    def test_load_hyperparameters_multivariate_logistic(self):
        self._generic_multivariate_hyperparameters_checker("logistic")

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_load_hyperparameters_multivariate_logistic_parallel(self):
        self._generic_multivariate_hyperparameters_checker("logistic_parallel")

    def test_bad_noise_model_or_old_loss(self):
        # raise if invalid loss
        with self.assertRaises(ValueError):
            model_factory("logistic", noise_model="bad_noise_model")

        # NO MORE BACKWARD COMPAT -> raises about old loss kw
        with self.assertRaises(ValueError):
            model_factory("logistic", loss="MSE_diag_noise")
