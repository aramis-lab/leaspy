from leaspy.io.settings.model_settings import ModelSettings

from tests import LeaspyTestCase


class ModelSettingsTest(LeaspyTestCase):

    def _load_settings(self, model_type: str) -> ModelSettings:
        return ModelSettings(
            self.get_test_data_path(
                "settings", "models", f"model_settings_{model_type}.json"
            )
        )

    @property
    def univariate_model_settings(self) -> ModelSettings:
        return self._load_settings("univariate")

    @property
    def multivariate_model_settings(self) -> ModelSettings:
        return self._load_settings("multivariate")

    def test_model_settings_univariate(self):
        self.assertEqual(self.univariate_model_settings.name, "univariate")
        self.assertEqual(self.univariate_model_settings.parameters['p0'], 0.3)
        self.assertEqual(self.univariate_model_settings.parameters['tau_mean'], 50)
        self.assertEqual(self.univariate_model_settings.parameters['tau_var'], 2)
        self.assertEqual(self.univariate_model_settings.parameters['xi_var'], 0.8)
        self.assertEqual(self.univariate_model_settings.hyperparameters, {"xi_mean": -10})

    def test_model_settings_multivariate(self):
        self.assertEqual(self.multivariate_model_settings.name, "multivariate")
        self.assertEqual(
            self.multivariate_model_settings.parameters,
            {
                "p0": 0.3,
                "beta": [[0.1, 0.2, 0.3], [0.5, 0.6, 0.9]],
                "tau_mean": 70,
                "tau_var": 50,
                # "xi_mean": -2,
                "xi_var": 4,
                # "sources_mean": [0.0, 0.1],
                "sources_var": [1.1, 0.9],
                "noise_var": 0.02
            }
        )
        self.assertEqual(
            self.multivariate_model_settings.constructor_parameters,
            {
                "dimension": 3,
                "source_dimension": 2
            }
        )
        self.assertEqual(
            self.multivariate_model_settings.hyperparameters,
        {'xi_mean': -2, 'sources_mean': [0.0, 0.1]},
        )
