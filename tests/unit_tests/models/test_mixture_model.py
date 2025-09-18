import torch

from leaspy.models import LogisticMultivariateMixtureModel
from tests import LeaspyTestCase


class TestMultivariateModel(LeaspyTestCase):
    def test_load_parameters(self):
        """
        Test the method load_parameters.
        """
        model_ref = self.get_hardcoded_model("mixture")
        model = LogisticMultivariateMixtureModel("test_model", obs_models="gaussian-diagonal", dimension=4, source_dimension=2, n_clusters=2)
        model.source_dimension = 2
        model.dimension = 4
        model.load_parameters(model_ref.parameters)

        expected_parameters = {
    "betas": [
        [0.021, -0.063],
        [0.009, -0.032],
        [-0.040, -0.026]
    ],
    "log_g_mean": [
        0.123,
        2.952,
        2.556,
        1.245
    ],
    "log_v0_mean": [
        -3.539,
        -4.306,
        -4.253,
        -3.283
    ],
    "noise_std": [
        0.080,
        0.041,
        0.078,
        0.159
    ],
    "probs": [
        0.569,
        0.431
    ],
    "sources_mean": [
        [-1.799, 0.585],
        [-0.318, 2.210]
    ],
    "sources_std": 1.0,
    "tau_mean": [
        78.121,
        86.194
    ],
    "tau_std": [
        7.994,
        8.368
    ],
    "xi_mean": [
        0.002,
        -0.002
    ],
    "xi_std": [
        0.631,
        0.631
    ]
}

        for param_name, param_value in expected_parameters.items():
            self.assertTrue(
                torch.equal(
                    model.state[param_name],
                    torch.tensor(param_value),
                )
            )
