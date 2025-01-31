import unittest

import torch

from leaspy.algo import AlgorithmSettings
from leaspy.api import Leaspy
from tests import LeaspyTestCase


@unittest.skipIf(
    not torch.cuda.is_available(),
    "GPU calibration tests need an available CUDA environment",
)
class GPUModelFit(LeaspyTestCase):
    def test_all_model_gpu_run(self):
        """
        Check if the following models run with the following algorithms, using
        a GPU for calibration.
        """
        for model_name in (
            "linear",
            "univariate_logistic",
            "univariate_linear",
            "logistic",
            "logistic_parallel",
        ):
            with self.subTest(model_name=model_name):
                extra_kws = {}
                if "univariate" not in model_name:
                    extra_kws["source_dimension"] = 2  # force so not to get a warning

                leaspy = Leaspy(model_name, **extra_kws)
                settings = AlgorithmSettings(
                    "mcmc_saem", n_iter=200, seed=0, device=torch.device("cuda")
                )

                data = self.get_suited_test_data_for_model(model_name)

                leaspy.fit(data, settings)

                methods = ["mode_real", "mean_real", "scipy_minimize"]

                for method in methods:
                    extra_kws = dict()  # not for all algos
                    if "_real" in method:
                        extra_kws = dict(n_iter=100)
                    settings = AlgorithmSettings(method, seed=0, **extra_kws)
                    result = leaspy.personalize(data, settings)

    def test_all_model_gpu_run_crossentropy(self):
        """
        Check if the following models run with the following algorithms, using
        a GPU for calibration.
        """
        for model_name in (
            "linear",
            "univariate_logistic",
            "univariate_linear",
            "logistic",
            "logistic_parallel",
        ):
            with self.subTest(model_name=model_name):
                extra_kws = {}
                if "univariate" not in model_name:
                    extra_kws["source_dimension"] = 2  # force so not to get a warning

                leaspy = Leaspy(model_name, noise_model="bernoulli", **extra_kws)
                settings = AlgorithmSettings(
                    "mcmc_saem", n_iter=200, seed=0, device=torch.device("cuda")
                )

                data = self.get_suited_test_data_for_model(model_name + "_binary")

                leaspy.fit(data, settings)

                for method in ["scipy_minimize"]:
                    extra_kws = dict()  # not for all algos
                    if "_real" in method:
                        extra_kws = dict(n_iter=100)
                    settings = AlgorithmSettings(method, seed=0, **extra_kws)
                    result = leaspy.personalize(data, settings)
