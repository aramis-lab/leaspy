import torch

from leaspy.models import LogisticModel
from tests import LeaspyTestCase


class TestMultivariateModel(LeaspyTestCase):
    def test_load_parameters(self):
        """
        Test the method load_parameters.
        """
        model_ref = self.get_hardcoded_model("logistic_scalar_noise")
        model = LogisticModel("test_model", obs_models="gaussian-scalar")
        model.source_dimension = 2
        model.dimension = 4
        model.load_parameters(model_ref.parameters)

        expected_parameters = {
            # "g": [0.5, 1.5, 1.0, 2.0],
            # "v0": [-2.0, -3.5, -3.0, -2.5],
            "betas": [[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]],
            "tau_mean": [75.2],
            "tau_std": [7.1],
            "xi_mean": 0.0,
            "xi_std": [0.2],
            "sources_mean": [0.0, 0.0],
            "sources_std": 1.0,
        }
        for param_name, param_value in expected_parameters.items():
            self.assertTrue(
                torch.equal(
                    model.state[param_name],
                    torch.tensor(param_value),
                )
            )
