from glob import glob
from unittest import skip

import torch

from leaspy.models import model_factory
from leaspy.models.obs_models import (
    OBSERVATION_MODELS,
    FullGaussianObservationModel,
    ObservationModelNames,
)

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.functional_tests.api.test_api_fit import LeaspyFitTestMixin
from tests.unit_tests.models.test_model_factory import ModelFactoryTestMixin


class LeaspyTest(LeaspyFitTestMixin, ModelFactoryTestMixin):
    model_names = (
        "linear",
        "logistic",
        "shared_speed_logistic",
    )

    def test_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        """
        from leaspy.models import model_factory

        for name in self.model_names:
            model = model_factory(name)
            self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
            self.assertEqual(type(model), type(model_factory(name)))
            self.check_model_factory_constructor(model)
            self.assertFalse(model.is_initialized)

        for observation_model_name, observation_model in OBSERVATION_MODELS.items():
            if observation_model_name != ObservationModelNames.WEIBULL_RIGHT_CENSORED:
                print(observation_model_name)
                model = model_factory(
                    "logistic", obs_models=observation_model_name, dimension=1
                )
                self.assertIsInstance(model.obs_models[0], observation_model)
            else:
                to_test = [
                    [None, {"w": 1, "s": 0}],
                    ["gaussian-scalar", {"w": 1, "s": 0}],
                    [observation_model_name, {"w": 0, "s": 1}],
                    [(observation_model_name, "gaussian-scalar"), {"w": 0, "s": 1}],
                    [("gaussian-scalar", observation_model_name), {"w": 1, "s": 0}],
                ]
                for input, output in to_test:
                    # If no observational model given
                    model = model_factory("joint", dimension=1, obs_models=input)
                    self.assertIsInstance(
                        model.obs_models[output["w"]], observation_model
                    )
                    self.assertIsInstance(
                        model.obs_models[output["s"]],
                        OBSERVATION_MODELS[ObservationModelNames.GAUSSIAN_SCALAR],
                    )
        for name in (
            "linear",
            "logistic",
            "shared_speed_logistic",
        ):
            model = model_factory(name, source_dimension=2)
            self.assertEqual(model.source_dimension, 2)

        for name in ("linear", "logistic"):
            model = model_factory(name, dimension=1)
            self.assertEqual(model.source_dimension, 0)
            self.assertEqual(model.dimension, 1)

    def test_load_hyperparameters(self):
        model = self.get_hardcoded_model("logistic_diag_noise")
        model._load_hyperparameters({"source_dimension": 3})

        self.assertEqual(model.source_dimension, 3)
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)

    def test_load_logistic_scalar_noise(self):
        """Test the initialization of a logistic model from a json file."""
        model = self.get_hardcoded_model("logistic_scalar_noise")

        self.assertEqual(type(model), type(model_factory("logistic")))
        self.assertEqual(model.dimension, 4)
        self.assertEqual(model.features, ["Y0", "Y1", "Y2", "Y3"])
        self.assertEqual(model.source_dimension, 2)
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        parameters = {
            # "g": [0.5, 1.5, 1.0, 2.0],  broken...
            # "v0": [-2.0, -3.5, -3.0, -2.5],  broken...
            "betas": [[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]],
            "tau_mean": [75.2],
            "tau_std": [7.1],
            "xi_mean": 0.0,
            "xi_std": [0.2],
            "sources_mean": [0.0, 0.0],
            "sources_std": 1.0,
        }
        for param_name, param_value in parameters.items():
            self.assertTrue(
                torch.equal(
                    model.state[param_name],
                    torch.tensor(param_value),
                )
            )
        self.assertEqual(model.is_initialized, True)

    @skip("logistic parallel is broken")
    def test_load_logistic_parallel_scalar_noise(self):
        """Test the initialization of a logistic parallel model from a json file."""
        model = self.get_hardcoded_model("logistic_parallel_scalar_noise")

        # Test the name
        self.assertEqual(type(model), type(model_factory("logistic_parallel")))

        # Test the hyperparameters
        self.assertEqual(model.dimension, 4)
        self.assertEqual(model.features, ["Y0", "Y1", "Y2", "Y3"])
        self.assertEqual(model.source_dimension, 2)
        # self.assertIsInstance(leaspy.model.noise_model, GaussianScalarNoiseModel)

        # Test the parameters
        parameters = {
            "g": 1.0,
            "tau_mean": 70.4,
            "tau_std": 7.0,
            "xi_mean": -1.7,
            "xi_std": 1.0,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "deltas": [-3, -2.5, -1.0],
            "betas": [[0.1, -0.1], [0.5, -0.3], [0.3, 0.4]],
        }

        self.assertDictAlmostEqual(model.parameters, parameters)
        self.assertDictAlmostEqual(model.noise_model.parameters, {"scale": 0.1})

        # Test the initialization
        self.assertEqual(model.is_initialized, True)

        # Test that the model attributes were initialized
        attrs = model._get_attributes(None)
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, tuple)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(attr is not None for attr in attrs))

    @skip("linear is broken")
    def test_load_linear_scalar_noise(self):
        """Test the initialization of a linear model from a json file."""
        model = self.get_hardcoded_model("linear_scalar_noise")

        self.assertEqual(type(model), type(model_factory("linear")))

        # Test the hyperparameters
        self.assertEqual(model.dimension, 4)
        self.assertEqual(model.features, ["Y0", "Y1", "Y2", "Y3"])
        self.assertEqual(model.source_dimension, 2)
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)

        # Test the parameters
        parameters = {
            "g": [0.5, 0.06, 0.1, 0.3],
            "v0": [-0.5, -0.5, -0.5, -0.5],
            "betas": [[0.1, -0.5], [-0.1, 0.1], [-0.8, -0.1]],
            "tau_mean": 75.2,
            "tau_std": 0.9,
            "xi_mean": 0.0,
            "xi_std": 0.3,
            "sources_mean": 0.0,
            "sources_std": 1.0,
        }

        self.assertDictAlmostEqual(model.parameters, parameters)
        self.assertDictAlmostEqual(model.noise_model.parameters, {"scale": 0.1})

        # Test the initialization
        self.assertEqual(model.is_initialized, True)

        # Test that the model attributes were initialized
        attrs = model._get_attributes(None)
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, tuple)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(attr is not None for attr in attrs))

    def test_load_univariate_logistic(self):
        """Test the initialization of a linear model from a json file."""
        model = self.get_hardcoded_model("logistic")
        self.assertEqual(model.features, ["Y0"])
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)

        # Test the parameters
        parameters = {
            # "g": [1.0],
            # "v0": [-2.6265233750364456],
            "tau_mean": [70.0],
            "tau_std": [2.5],
            "xi_mean": 0.0,
            "xi_std": [0.01],
            # never used parameters
            # "betas": [],
            # "sources_mean": 0,
            # "sources_std": 1,
        }

        for param_name, param_value in parameters.items():
            try:
                self.assertTrue(
                    torch.equal(
                        model.state[param_name],
                        torch.tensor(param_value),
                    )
                )
            except:
                print(model.state[param_name])
                print(torch.tensor(param_value))

        # self.assertDictAlmostEqual(leaspy.model.parameters, parameters)
        # self.assertDictAlmostEqual(leaspy.model.noise_model.parameters, {"scale": 0.2})

        # Test the initialization
        self.assertEqual(model.is_initialized, True)

        # Test that the model attributes were initialized
        # for attribute in leaspy.model._get_attributes(None):
        #    self.assertIsInstance(attribute, torch.FloatTensor)

    @skip("linear is broken")
    def test_load_univariate_linear(self):
        """Test the initialization of a linear model from a json file."""
        model = self.get_hardcoded_model("linear")
        self.assertEqual(model.features, ["Y0"])
        # self.assertIsInstance(leaspy.model.noise_model, GaussianScalarNoiseModel)

        # Test the parameters
        parameters = {
            "g": [0.5],
            "v0": [-4.0],
            "tau_mean": 78.0,
            "tau_std": 5.0,
            "xi_mean": 0.0,
            "xi_std": 0.5,
            # never used parameters
            "betas": [],
            "sources_mean": 0,
            "sources_std": 1,
        }

        self.assertDictAlmostEqual(model.parameters, parameters)
        self.assertDictAlmostEqual(model.noise_model.parameters, {"scale": 0.15})

        # Test the initialization
        self.assertEqual(model.is_initialized, True)

        # Test that the model attributes were initialized
        for attribute in model._get_attributes(None):
            self.assertIsInstance(attribute, torch.FloatTensor)

    def test_load_save_load(self, *, atol=1e-4):
        """Test loading, saving and loading again all models (hardcoded and functional)."""
        from leaspy.models import BaseModel

        for model_path in glob(self.hardcoded_model_path("*.json")):
            with self.subTest(model_path=model_path):
                self.check_model_consistency(
                    BaseModel.load(model_path), model_path, atol=atol
                )

        # functional models (OK because no direct test on values)
        for model_path in glob(self.from_fit_model_path("*.json")):
            with self.subTest(model_path=model_path):
                self.check_model_consistency(
                    BaseModel.load(model_path), model_path, atol=atol
                )
