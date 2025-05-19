import json
import os
import warnings
from typing import Optional
from unittest import skip, skipIf

import torch

from leaspy.api import Leaspy
from leaspy.models.obs_models import observation_model_factory
from tests.unit_tests.plots.test_plotter import MatplotlibTestCase


class LeaspyFitTestMixin(MatplotlibTestCase):
    """Mixin holding generic fit methods that may be safely reused in other tests (no actual test here)."""

    def generic_fit(
        self,
        model_name: str,
        model_codename: str,
        *,
        algo_name: Optional[str] = "mcmc_saem",
        algo_params: Optional[dict] = None,
        # change default parameters for logs so everything is tested despite the very few iterations in tests
        # TODO reactivate plotting once FitOutputManager & Plotter are ready
        # logs_kws: dict = dict(console_print_periodicity=50, save_periodicity=20, plot_periodicity=100),
        logs_kws: Optional[dict] = None,
        print_model: Optional[bool] = False,
        check_model: Optional[bool] = True,
        check_kws: Optional[dict] = None,
        save_model: Optional[bool] = False,
        **model_hyperparams,
    ):
        """Helper for a generic calibration in following tests.

        Parameters
        ----------
        model_name : str
            The name of the model.
        model_codename : str
            The name of the model used to retrieve the expected model filename.
        algo_name : str, optional
            The name of the fitting algorithm to use.
        algo_params : dict, optional
            Parameters to be used with the fitting algorithm.
            Default={"n_iter": 100, "seed": 0}
        logs_kws : dict, optional
            Parameters for logging.
        print_model : bool, optional
            If True, the model parameters are printed after calibration.
            This can be helpful for debugging.
            Default=False.
        check_model : bool, optional
            If True, the calibrated model will be checked against a reference model.
            Default=True.
        check_kws : dict, optional
            Additional parameters for model consistency checking.
            Default={}.
        save_model : bool, optional
            If True, the computed model will be saved to disk.
            Default=False.
        """
        algo_params = algo_params or {"n_iter": 100, "seed": 0}
        check_kws = check_kws or {}
        logs_kws = logs_kws or {
            "console_print_periodicity": 50,
            "save_periodicity": None,
            "plot_periodicity": None,
        }

        data = self.get_suited_test_data_for_model(model_codename)
        leaspy = Leaspy(model_name, **model_hyperparams)
        algo_settings = self.get_algo_settings(name=algo_name, **algo_params)

        # set logs by default
        if logs_kws is not None:
            auto_path_logs = self.get_test_tmp_path(f"{model_codename}-logs")
            with self.assertWarnsRegex(
                UserWarning, r" does not exist\. Needed paths will be created"
            ):
                algo_settings.set_logs(path=auto_path_logs, **logs_kws)

        leaspy.fit(data, settings=algo_settings)

        if print_model:
            print(leaspy.model.parameters)

        # path to expected
        expected_model_path = self.from_fit_model_path(model_codename)
        inexistant_model = not os.path.exists(expected_model_path)

        # check that values in already saved file are same than the ones in fitted model
        if check_model:
            if inexistant_model:
                warnings.warn(
                    f"<!> Consistency of model could not be checked since '{model_codename}' did not exist..."
                )
            else:
                self.check_model_consistency(leaspy, expected_model_path, **check_kws)

        # set `save_model=True` to re-generate example model
        # <!> use carefully (only when needed following breaking changes in model / calibration)
        if save_model or inexistant_model:
            leaspy.save(expected_model_path)
            if save_model:
                warnings.warn(f"<!> You overwrote previous '{model_codename}' model...")

        return leaspy, data

    def check_model_consistency(
        self, leaspy: Leaspy, path_to_backup_model: str, **allclose_kwds
    ):
        # Temporary save parameters and check consistency with previously saved model

        allclose_kwds = {"atol": 1e-5, "rtol": 1e-4, **allclose_kwds}

        path_to_tmp_saved_model = self.get_test_tmp_path(
            os.path.basename(path_to_backup_model)
        )
        leaspy.save(path_to_tmp_saved_model)

        with open(path_to_backup_model, "r") as f1:
            expected_model_parameters = json.load(f1)
        with open(path_to_tmp_saved_model, "r") as f2:
            model_parameters_new = json.load(f2)

        # Remove the temporary file saved (before asserts since they may fail!)
        os.remove(path_to_tmp_saved_model)

        # don't compare leaspy exact version...
        expected_model_parameters["leaspy_version"] = None
        new_model_version, model_parameters_new["leaspy_version"] = (
            model_parameters_new["leaspy_version"],
            None,
        )

        self.assertDictAlmostEqual(
            model_parameters_new, expected_model_parameters, **allclose_kwds
        )

        # the reloading of model parameters will test consistency of model derived variables
        # (only mixing matrix here)
        # TODO: use `.load(expected_dict_adapted)` instead of `.load(expected_file_not_adapted)`
        #  until expected file are regenerated
        # expected_model = Leaspy.load(path_to_backup_model).model
        expected_model_parameters["obs_models"] = model_parameters_new["obs_models"] = {
            obs_model.name: obs_model.to_string()
            for obs_model in leaspy.model.obs_models
        }  # WIP: not properly serialized for now
        expected_model_parameters["leaspy_version"] = model_parameters_new[
            "leaspy_version"
        ] = new_model_version
        Leaspy.load(expected_model_parameters)
        Leaspy.load(model_parameters_new)


# some noticeable reproducibility errors btw MacOS and Linux here...
ALLCLOSE_CUSTOM = dict(
    nll_regul_ind_sum=dict(atol=5),
    nll_attach=dict(atol=10),
    nll_tot=dict(atol=15),
    tau_mean=dict(atol=0.2),
    tau_std=dict(atol=0.2),
)
DEFAULT_CHECK_KWS = dict(atol=0.1, rtol=1e-2, allclose_custom=ALLCLOSE_CUSTOM)


class LeaspyFitTest(LeaspyFitTestMixin):
    # <!> reproducibility gap for PyTorch >= 1.7, only those are supported now

    def test_fit_logistic_scalar_noise(self):
        """Test MCMC-SAEM."""
        self.generic_fit(
            "logistic",
            "logistic_scalar_noise",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
            check_kws=DEFAULT_CHECK_KWS,
        )

    def test_fit_logistic_diagonal_noise(self):
        """Test MCMC-SAEM (1 noise per feature)."""
        # TODO: dimension should not be needed at this point...
        self.generic_fit(
            "logistic",
            "logistic_diag_noise",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
            check_kws=DEFAULT_CHECK_KWS,
        )

    def test_fit_logistic_diagonal_noise_fast_gibbs(self):
        # TODO: dimension should not be needed at this point...
        self.generic_fit(
            "logistic",
            "logistic_diag_noise_fast_gibbs",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
            algo_params={"n_iter": 100, "seed": 0, "sampler_pop": "FastGibbs"},
            check_kws=DEFAULT_CHECK_KWS,
        )

    def test_fit_logistic_diagonal_noise_mh(self):
        # TODO: dimension should not be needed at this point...
        self.generic_fit(
            "logistic",
            "logistic_diag_noise_mh",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
            algo_params={
                "n_iter": 100,
                "seed": 0,
                "sampler_pop": "Metropolis-Hastings",
            },
            check_kws=DEFAULT_CHECK_KWS,
        )

    def test_fit_logistic_diagonal_noise_with_custom_tuning_no_sources(self):
        # TODO: dimension should not be needed at this point...
        self.generic_fit(
            "logistic",
            "logistic_diag_noise_custom",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=0,
            algo_params={
                "n_iter": 100,
                "burn_in_step_power": 0.65,
                "sampler_pop_params": {
                    "acceptation_history_length": 10,
                    "mean_acceptation_rate_target_bounds": (0.1, 0.5),
                    "adaptive_std_factor": 0.1,
                },
                "sampler_ind_params": {
                    "acceptation_history_length": 10,
                    "mean_acceptation_rate_target_bounds": (0.1, 0.5),
                    "adaptive_std_factor": 0.1,
                },
                "annealing": {
                    "initial_temperature": 5.0,
                    "do_annealing": True,
                    "n_plateau": 2,
                },
                "seed": 0,
            },
            check_kws=DEFAULT_CHECK_KWS,
        )

    # @skip("Logistic parallel models are currently broken.")
    def test_fit_logistic_parallel(self):
        self.generic_fit(
            "logistic_parallel",
            "logistic_parallel_scalar_noise",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
        )

    # @skip("Logistic parallel models are currently broken.")
    def test_fit_logistic_parallel_diagonal_noise(self):
        self.generic_fit(
            "logistic_parallel",
            "logistic_parallel_diag_noise",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
        )

    # @skip("Logistic parallel models are currently broken.")
    def test_fit_logistic_parallel_diagonal_noise_no_source(self):
        self.generic_fit(
            "logistic_parallel",
            "logistic_parallel_diag_noise_no_source",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=0,
        )

    def test_fit_univariate_logistic(self):
        self.generic_fit(
            "univariate_logistic",
            "univariate_logistic",
            check_kws=DEFAULT_CHECK_KWS,
        )

    def test_fit_univariate_joint(self):
        self.generic_fit(
            "univariate_joint",
            "univariate_joint",
            check_kws=DEFAULT_CHECK_KWS,
            check_model=True,
        )

    def test_fit_joint_no_sources(self):
        self.generic_fit(
            "joint", "joint_no_sources", check_kws=DEFAULT_CHECK_KWS, check_model=True
        )

    def test_fit_joint_diagonal(self):
        self.generic_fit(
            "joint",
            "joint_diagonal",
            check_kws=DEFAULT_CHECK_KWS,
            check_model=True,
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
        )

    def test_fit_joint_scalar(self):
        self.generic_fit(
            "joint",
            "joint_scalar",
            check_kws=DEFAULT_CHECK_KWS,
            check_model=True,
            source_dimension=0,
        )

    # @skip("Linear models are currently broken.")
    def test_fit_univariate_linear(self):
        self.generic_fit("univariate_linear", "univariate_linear")

    # @skip("Linear models are currently broken.")
    def test_fit_linear(self):
        self.generic_fit(
            "linear",
            "linear_scalar_noise",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
        )

    # @skip("Linear models are currently broken.")
    def test_fit_linear_diagonal_noise(self):
        self.generic_fit(
            "linear",
            "linear_diag_noise",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
        )

    def test_fit_logistic_binary(self):
        self.generic_fit(
            "logistic",
            "logistic_binary",
            obs_models=observation_model_factory("bernoulli"),
            source_dimension=2,
        )

    # @skip("Logistic parallel models are currently broken.")
    def test_fit_logistic_parallel_binary(self):
        self.generic_fit(
            "logistic_parallel",
            "logistic_parallel_binary",
            obs_models=observation_model_factory("bernoulli"),
            source_dimension=2,
        )


@skipIf(
    not torch.cuda.is_available(),
    "GPU calibration tests need an available CUDA environment",
)
class LeaspyFitGPUTest(LeaspyFitTestMixin):
    def test_fit_logistic_scalar_noise(self):
        """Test MCMC-SAEM."""
        self.generic_fit(
            "logistic",
            "logistic_scalar_noise_gpu",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )

    def test_fit_logistic_diagonal_noise(self):
        """Test MCMC-SAEM (1 noise per feature)."""
        self.generic_fit(
            "logistic",
            "logistic_diag_noise_gpu",
            obs_models=observation_model_factory("gaussian-diagonal", dimension=4),
            source_dimension=2,
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )

    def test_fit_logistic_parallel(self):
        self.generic_fit(
            "logistic_parallel",
            "logistic_parallel_scalar_noise_gpu",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )

    def test_fit_logistic_parallel_diagonal_noise(self):
        self.generic_fit(
            "logistic_parallel",
            "logistic_parallel_diag_noise_gpu",
            obs_models=observation_model_factory("gaussian-diagonal"),
            source_dimension=2,
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )

    def test_fit_univariate_logistic(self):
        self.generic_fit(
            "univariate_logistic",
            "univariate_logistic_gpu",
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )

    def test_fit_univariate_linear(self):
        self.generic_fit(
            "univariate_linear",
            "univariate_linear_gpu",
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )

    def test_fit_linear(self):
        self.generic_fit(
            "linear",
            "linear_scalar_noise_gpu",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
            algo_params={"n_iter": 100, "seed": 0, "device": "cuda"},
        )
