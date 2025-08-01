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
                [0.021020518615841866, -0.06325671076774597],
                [0.008699563331902027, -0.032000601291656494],
                [-0.04013106971979141, -0.026052488014101982]
            ],
            "log_g_mean": [
                0.12327956408262253,
                2.9518003463745117,
                2.5558218955993652,
                1.2449039220809937
            ],
            "log_v0_mean": [
                -3.538581609725952,
                -4.306070804595947,
                -4.2525529861450195,
                -3.2830135822296143
            ],
            "noise_std": [
                0.07965857320295525,
                0.04058105511749294,
                0.07811478503238264,
                0.1589867754849439
            ],
            "probs": [
                0.5690345813433885,
                0.4309654186566116
            ],
            "sources_mean": [
                [-1.7990124298623547, 0.585131786451524],
                [-0.3175997992029286, 2.2095831827727266]
            ],
            "sources_std": 1.0,
            "tau_mean": [
                78.12053640331543,
                86.19382326208037
            ],
            "tau_std": [
                7.9941298929276545,
                8.367593287845276
            ],
            "xi_mean": [
                0.0015176474587179957,
                -0.0020038588919509756
            ],
            "xi_std": [
                0.6305820482664299,
                0.6311062048367313
            ],
            "mixing_matrix": [
                [-0.015071430243551731, 0.008749578148126602, 0.0022491805721074343, -0.04265308380126953],
                [0.07423201203346252, -0.002818079199641943, -0.0002302309439983219, -0.013630702160298824]
            ]
        }
        for param_name, param_value in expected_parameters.items():
            self.assertTrue(
                torch.equal(
                    model.state[param_name],
                    torch.tensor(param_value),
                )
            )
