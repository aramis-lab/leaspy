import re
from unittest import skipIf

from leaspy.models import ALL_MODELS, LogisticUnivariateModel, BaseModel
from leaspy.models.factory import ModelFactory
from leaspy.models.obs_models import FullGaussianObservationModel
from leaspy.exceptions import LeaspyModelInputError

from tests import LeaspyTestCase

TEST_LOGISTIC_PARALLEL_MODELS = False
SKIP_LOGISTIC_PARALLEL_MODELS = "Logistic parallel models are currently broken"


class ModelFactoryTestMixin(LeaspyTestCase):

    def check_model_factory_constructor(self, model: BaseModel):
        """
        Test initialization of leaspy model.

        Parameters
        ----------
        model : str, optional (default None)
            Name of the model
        """
        self.assertIn(model.name, ALL_MODELS)
        self.assertEqual(type(model), ALL_MODELS[model.name])


class ModelFactoryTest(ModelFactoryTestMixin):

    def test_model_factory_constructor(self):
        for name in ALL_MODELS.keys():
            with self.subTest(model_name=name):
                self.check_model_factory_constructor(model=ModelFactory.model(name))

    def test_lower_case(self):
        """Test lower case"""
        for name in ("univariate_logistic", "uNIVariaTE_LogIsTIc", "UNIVARIATE_LOGISTIC"):
            model = ModelFactory.model(name)
            self.assertEqual(type(model), LogisticUnivariateModel)

    def test_wrong_arg(self):
        """Test if raise error for wrong argument"""
        # Test if raise ValueError if wrong string arg for name
        wrong_arg_examples = ("lgistic", "blabla")
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_examples = [3.8, {"truc": .1}]
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

    def _generic_univariate_hyperparameters_checker(self, model_name: str) -> None:
        model = ModelFactory.model(model_name, features=["t1"])
        self.assertEqual(model.features, ["t1"])
        self.assertTrue(isinstance(model.obs_models, tuple))
        self.assertEqual(len(model.obs_models), 1)
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        self.assertEqual(model.dimension, 1)
        self.assertEqual(model.source_dimension, 0)
        with self.assertRaisesRegex(
            LeaspyModelInputError,
            re.escape(
                "Cannot set the model's features to ['t1', 't2', 't3'], "
                "because the model has been configured with a dimension of 1."
            ),
        ):
            ModelFactory.model(model_name, features=["t1", "t2", "t3"])

    def test_load_hyperparameters_univariate_linear(self):
        self._generic_univariate_hyperparameters_checker("univariate_linear")

    def test_load_hyperparameters_univariate_logistic(self):
        self._generic_univariate_hyperparameters_checker("univariate_logistic")

    def _generic_multivariate_hyperparameters_checker(self, model_name: str) -> None:
        model = ModelFactory.model(
            model_name,
            features=["t1", "t2", "t3"],
            source_dimension=2,
            dimension=3,
        )
        self.assertEqual(model.features, ["t1", "t2", "t3"])
        self.assertTrue(isinstance(model.obs_models, tuple))
        self.assertEqual(len(model.obs_models), 1)
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        self.assertEqual(model.dimension, 3)
        self.assertEqual(model.source_dimension, 2)

    def test_load_hyperparameters_multivariate_linear(self):
        self._generic_multivariate_hyperparameters_checker("linear")

    def test_load_hyperparameters_multivariate_logistic(self):
        self._generic_multivariate_hyperparameters_checker("logistic")

    #@skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_load_hyperparameters_multivariate_logistic_parallel(self):
        self._generic_multivariate_hyperparameters_checker("logistic_parallel")

    def test_bad_observation_model(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "The requested ObservationModel bad_noise_model is not implemented. "
                "Valid observation model names are: "
                "['gaussian-diagonal', 'gaussian-scalar', 'bernoulli', 'ordinal', 'weibull-right-censored']."
            ),
        ):
            ModelFactory.model("logistic", obs_models="bad_noise_model")
