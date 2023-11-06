# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from unittest import skip
from typing import Optional, Union, List

from .test_api_fit import LeaspyFitTestMixin
from .test_api_personalize import LeaspyPersonalizeTestMixin
from .test_api_simulate import LeaspySimulateTest_Mixin

# Simulation algos are broken for now due to new observation models
# Flip this to True once they are adapted to the new paradigm
RUN_SIMULATION_TESTS = False


class LeaspyAPITest(LeaspyFitTestMixin, LeaspyPersonalizeTestMixin, LeaspySimulateTest_Mixin):

    def generic_usecase(
        self,
        model_name: str,
        model_codename: str,
        *,
        expected_loss_personalization: Union[float, List[float]],
        personalization_algo: str,
        fit_algo: Optional[str] = "mcmc_saem",
        simulate_algo: Optional[str] = "simulation",
        fit_check_kws: Optional[dict] = None,
        fit_algo_params: Optional[dict] = None,
        personalization_algo_params: Optional[dict] = None,
        simulate_algo_params: Optional[dict] = None,
        simulate_tol: Optional[float] = 1e-4,
        tol_loss: Optional[float] = 1e-2,
        **model_hyperparams,
    ):
        """
        Functional test of a basic analysis using leaspy package

        1 - Data loading
        2 - Fit logistic model with MCMC algorithm
        3 - Save parameters & reload (remove created files to keep the repo clean)
        4 - Personalize model with 'mode_real' algorithm
        (5 - Plot results)
        6 - Simulate new patients

        Parameters
        ----------
        model_name : str
            The name of the model.
        model_codename : str
            The name of the model used to retrieve the expected model filename.
        expected_loss_personalization : float or list of floats
            The expected loss value for personalization algorithm.
        personalization_algo : str
            The algorithm to use for personalization.
        fit_algo : str, optional
            The algorithm to use for fitting the model.
            Default="mcmc_saem"
        simulate_algo : str, optional
            The algorithm to use for simulation.
            Default="simulation"
        fit_check_kws : dict, optional
        fit_algo_params : dict, optional
            Parameters for the fitting algorithm.
            Default is {n_iter: 200, seed: 0}
        personalization_algo_params : dict, optional
            Parameters for the personalization algorithm.
            Default is {seed: 0}
        simulate_algo_params : dict, optional
            Parameters for the simulation algorithm.
            Default is {seed: 0}
        simulate_tol : float, optional
            Tolerance for consistency checks of simulation results.
        tol_loss : float, optional
            Tolerance for consistency checks of personalization loss result.
        """
        fit_algo_params = fit_algo_params or {"n_iter": 200, "seed": 0}
        fit_check_kws = fit_check_kws or {"atol": 1e-3}
        personalization_algo_params = personalization_algo_params or {"seed": 0}
        simulate_algo_params = simulate_algo_params or {"seed": 0}
        filename_expected_model = f"{model_codename}_for_test_api"

        # no loss returned for fit for now
        leaspy, data = self.generic_fit(
            model_name,
            filename_expected_model,
            **model_hyperparams,
            algo_name=fit_algo,
            algo_params=fit_algo_params,
            check_model=True,
            check_kws=fit_check_kws,
        )

        # unlink 1st functional fit test from next steps...
        leaspy = self.get_from_fit_model(filename_expected_model)
        self.assertTrue(leaspy.model.is_initialized)

        # Personalize
        algo_personalize_settings = self.get_algo_settings(
            name=personalization_algo,
            **personalization_algo_params,
        )
        individual_parameters, loss = leaspy.personalize(
            data,
            settings=algo_personalize_settings,
            return_loss=True,
        )
        self.check_consistency_of_personalization_outputs(
            individual_parameters,
            loss,
            expected_loss=expected_loss_personalization,
            tol_loss=tol_loss,
        )

        if RUN_SIMULATION_TESTS:
            # Simulate
            simulation_settings = self.get_algo_settings(name=simulate_algo, **simulate_algo_params)
            simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)

            self.check_consistency_of_simulation_results(
                simulation_settings,
                simulation_results,
                data,
                expected_results_file=f"simulation_results_{model_codename}.csv",
                model=leaspy.model,
                tol=simulate_tol,
            )

    def test_usecase_logistic_scalar_noise(self):
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": lambda n: [.5] * min(n, 2) + [1.] * max(0, n - 2),
            "number_of_subjects": 100,
        }
        self.generic_usecase(
            "logistic",
            model_codename="logistic_scalar_noise",
            obs_models="gaussian-scalar",
            source_dimension=2,
            fit_check_kws={"atol": 1e-2, "rtol": 1e-2},
            personalization_algo="mode_real",
            personalization_algo_params={"n_iter": 200, "seed": 0},
            expected_loss_personalization=0.0857,  # scalar RMSE
            simulate_algo_params=simulation_parameters,
        )

    def test_usecase_univariate_joint(self):
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": lambda n: [.5] * min(n, 2) + [1.] * max(0, n - 2),
            "number_of_subjects": 100,
        }
        self.generic_usecase(
            "univariate_joint",
            model_codename="univariate_joint",
            fit_check_kws={"atol": 1e-2, "rtol": 1e-2},
            personalization_algo="mode_real",
            personalization_algo_params={"n_iter": 200, "seed": 0},
            expected_loss_personalization=0.06622593104839325,  # scalar RMSE
            simulate_algo_params=simulation_parameters,
        )

    def test_usecase_logistic_diagonal_noise(self):
        custom_delays_vis = {
            "mean": 1.,
            "min": .2,
            "max": 2.,
            "std": 1.,
        }
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 100,
        }
        # some noticeable reproducibility errors btw MacOS and Linux here...
        all_close_custom = {
            # "nll_regul_tau": dict(atol=1),
            # "nll_regul_xi": dict(atol=5),
            # "nll_regul_sources": dict(atol=1),
            "nll_regul_ind_sum": {"atol": 5},
            "nll_attach": {"atol": 6},
            "nll_tot": {"atol": 5},
            "tau_mean": {"atol": 0.3},
            "tau_std": {"atol": 0.3},
        }
        self.generic_usecase(
            "logistic",
            model_codename="logistic_diag_noise",
            obs_models="gaussian-diagonal",
            source_dimension=2,
            dimension=4,  # WIP
            fit_check_kws={
                "atol": 0.1,
                "rtol": 1e-2,
                "allclose_custom": all_close_custom,
            },
            personalization_algo="scipy_minimize",
            expected_loss_personalization=[0.064, 0.037, 0.066, 0.142],  # per-ft RMSE
            simulate_algo_params=simulation_parameters,
            simulate_tol=2e-3,  # Not fully reproducible on Linux below this tol...
        )

    def test_usecase_logistic_binary(self):
        custom_delays_vis = .5
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 100,
            "reparametrized_age_bounds": (50, 85),
        }

        self.generic_usecase(
            'logistic',
            model_codename="logistic_binary",
            obs_models="bernoulli",
            source_dimension=2,
            personalization_algo="mean_real",
            expected_loss_personalization=105.18,  # logLL, not noise_std
            simulate_algo_params=simulation_parameters,
        )

    @skip("Ordinal observation models not implemented")
    def test_usecase_logistic_ordinal(self):
        custom_delays_vis = .5
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 100,
            "reparametrized_age_bounds": (50, 85),
        }

        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal",
            obs_models="ordinal",
            source_dimension=2,
            personalization_algo="mean_real",
            expected_loss_personalization=1029.1,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simulation_parameters,
        )

    @skip("Ordinal observation models not implemented")
    def test_usecase_logistic_ordinal_batched(self):
        # <!> Ordinal simulation may not be fully reproducible on different machines
        #     due to rounding errors when computing MultinomialDistribution.cdf that
        #     can lead to ±1 differences on MLE outcomes in rare cases...
        #     (changing seed, reducing subjects & increasing tol to avoid the problem)
        custom_delays_vis = .5
        simulation_parameters = {
            "seed": 123,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 10,
            "reparametrized_age_bounds": (50, 85),
        }

        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal_b",
            obs_models="ordinal",
            source_dimension=2,
            fit_check_kws={"atol": 0.005},
            personalization_algo="mean_real",
            expected_loss_personalization=1132.6,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simulation_parameters,
            simulate_tol=5e-2,
            batch_deltas_ordinal=True,
        )

    @skip("Ordinal observation models not implemented")
    def test_usecase_univariate_logistic_ordinal(self):
        custom_delays_vis = .5
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 100,
            "reparametrized_age_bounds": (50, 85),
        }

        self.generic_usecase(
            "univariate_logistic",
            model_codename="univariate_logistic_ordinal",
            obs_models="ordinal",
            personalization_algo="mean_real",
            expected_loss_personalization=169.8,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simulation_parameters,
        )

    @skip("Ordinal observation models not implemented")
    def test_usecase_logistic_ordinal_ranking(self):
        custom_delays_vis = .5
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 100,
            "reparametrized_age_bounds": (50, 85),
        }

        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal_ranking",
            obs_models="ordinal-ranking",
            source_dimension=2,
            personalization_algo="mean_real",
            expected_loss_personalization=974.15,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simulation_parameters,
        )

    @skip("Ordinal observation models not implemented")
    def test_usecase_logistic_ordinal_ranking_batched(self):
        custom_delays_vis = .5
        simulation_parameters = {
            "seed": 0,
            "delay_btw_visits": custom_delays_vis,
            "number_of_subjects": 100,
            "reparametrized_age_bounds": (50, 85),
        }

        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal_ranking_b",
            obs_models="ordinal-ranking",
            source_dimension=2,
            personalization_algo="mode_real",
            expected_loss_personalization=971.95,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simulation_parameters,
            batch_deltas_ordinal=True,
        )
