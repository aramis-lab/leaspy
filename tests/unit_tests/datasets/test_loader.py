import torch

from leaspy.datasets import (
    DatasetName,
    load_dataset,
    load_individual_parameters,
    load_model,
)
from leaspy.models.obs_models import FullGaussianObservationModel
from tests import LeaspyTestCase

# TODO: regenerate example models + individual parameters


class LoaderTest(LeaspyTestCase):
    def test_load_dataset(self):
        """
        Check ID and dtype of ID, TIME and values.
        """
        for name in DatasetName:
            df = load_dataset(name)
            expected_index = (
                ["ID", "TIME", "SPLIT"] if "train_and_test" in name else ["ID", "TIME"]
            )
            self.assertEqual(df.index.names, expected_index)
            self.assertTrue(all(df.dtypes.values == "float64"))
            self.assertEqual(
                df.index.get_level_values("ID").unique().tolist(),
                ["GS-" + "0" * (3 - len(str(i))) + str(i) for i in range(1, 201)],
            )
            self.assertIn(
                df.index.get_level_values("TIME").dtype, ("float64", "float32")
            )

    def test_load_model(self):
        """Check that all models are loadable, and check parameter values for one model."""
        from leaspy.models import LogisticModel

        for name in DatasetName:
            if name != DatasetName.PARKINSON_PUTAMEN_TRAIN_TEST:
                model = load_model(name)
                self.assertTrue(isinstance(model, LogisticModel))
        model = load_model(DatasetName.PARKINSON_PUTAMEN)
        self.assertEqual(model.features, ["PUTAMEN"])
        self.assertIsInstance(model.obs_models[0], FullGaussianObservationModel)
        self.assertAlmostEqual(model.parameters["noise_std"].item(), 0.02122, places=6)
        parameters = {
            "log_g_mean": torch.tensor([-1.1862]),
            "log_v0_mean": torch.tensor([-4.0517]),
            "noise_std": torch.tensor([0.0212]),
            "tau_mean": torch.tensor([68.7493]),
            "tau_std": torch.tensor([10.0295]),
            "xi_std": torch.tensor([0.5543]),
        }
        hyperparameters = {
            "log_g_std": torch.tensor(0.0100),
            "log_v0_std": torch.tensor(0.0100),
            "xi_mean": torch.tensor(0.0),
        }
        self.assertDictAlmostEqual(model.parameters, parameters, atol=1e-4)
        self.assertDictAlmostEqual(model.hyperparameters, hyperparameters, atol=1e-4)

    def test_load_individual_parameters(self):
        """Check that all ips are loadable, and check values for one individual_parameters instance."""
        for name in DatasetName:
            if name != DatasetName.PARKINSON_PUTAMEN_TRAIN_TEST:
                individual_parameters = load_individual_parameters(name)
        individual_parameters = load_individual_parameters("alzheimer")

        self.assertAlmostEqual(
            individual_parameters.get_mean("tau")[0], 76.9612, delta=1e-4
        )
        self.assertAlmostEqual(
            individual_parameters.get_mean("xi")[0], 0.0629, delta=1e-4
        )
        self.assertAllClose(
            individual_parameters.get_mean("sources"),
            [0.00315, -0.02109],
            atol=1e-5,
            rtol=1e-4,
            what="sources.mean",
        )
