# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
import os
import platform
from leaspy.utils.typing import Optional, Dict
from .test_api_fit import LeaspyFitTest_Mixin
from .test_api_personalize import LeaspyPersonalizeTest_Mixin
from .test_api_simulate import LeaspySimulateTest_Mixin


class LeaspyAPITest(LeaspyFitTest_Mixin, LeaspyPersonalizeTest_Mixin, LeaspySimulateTest_Mixin):

    def generic_usecase(
        self,
        model_name: str,
        model_codename: str,
        *,
        expected_loss_perso,
        perso_algo: str,
        fit_algo: str = "mcmc_saem",
        simulate_algo: str = "simulation",
        fit_check_kws: Optional[Dict] = None,
        fit_algo_params: Optional[Dict] = None,
        perso_algo_params: Optional[Dict] = None,
        simulate_algo_params: Optional[Dict] = None,
        simulate_tol: float = 1e-4,
        tol_loss: float = 1e-2,
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
        """
        fit_check_kws = fit_check_kws or {"atol": 1e-3}
        fit_algo_params = fit_algo_params or {"seed": 0}
        perso_algo_params = perso_algo_params or {"seed": 0}
        simulate_algo_params = simulate_algo_params or dict(seed=0)
        filename_expected_model = model_codename + '_for_test_api'
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
        algo_personalize_settings = self.get_algo_settings(name=perso_algo, **perso_algo_params)
        individual_parameters, loss = leaspy.personalize(
            data,
            settings=algo_personalize_settings,
            return_loss=True,
        )
        self.check_consistency_of_personalization_outputs(
            individual_parameters,
            loss,
            expected_loss=expected_loss_perso,
            tol_loss=tol_loss,
        )
        # Simulate
        simulation_settings = self.get_algo_settings(name=simulate_algo, **simulate_algo_params)
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)
        if (
            platform.system() == "Linux"
            and model_codename in ("logistic_ordinal_b", "logistic_ordinal", "logistic_diag_noise")
        ):
            model_codename = f"{model_codename}_linux"
        self.check_consistency_of_simulation_results(
            simulation_settings,
            simulation_results,
            data,
            expected_results_file=f"simulation_results_{model_codename}{'_arm' if os.uname()[4][:3] == 'arm' else ''}.csv",
            model=leaspy.model,
            tol=simulate_tol,
        )

    def test_usecase_logistic_scalar_noise(self):
        self.generic_usecase(
            "logistic",
            model_codename="logistic_scalar_noise",
            noise_model="gaussian_scalar",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            perso_algo="mode_real",
            perso_algo_params={"n_iter": 200, "seed": 0},
            expected_loss_perso=0.0857,  # scalar RMSE
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": lambda n: [.5]*min(n, 2) + [1.]*max(0, n-2),
                "number_of_subjects": 100,
            },
        )

    def test_usecase_logistic_diag_noise(self):
        self.generic_usecase(
            "logistic",
            model_codename="logistic_diag_noise",
            noise_model="gaussian_diagonal",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            fit_check_kws={
                "atol": 0.1,
                "rtol": 1e-2,
                "allclose_custom": {
                    "nll_regul_tau": {"atol": 1},
                    "nll_regul_xi": {"atol": 5},
                    "nll_regul_sources": {"atol": 1},
                    "nll_regul_tot": {"atol": 5},
                    "nll_attach": {"atol": 6},
                    "nll_tot": {"atol": 5},
                    "tau_mean": {"atol": 0.3},
                    "tau_std": {"atol": 0.3},
                }
            },
            perso_algo="scipy_minimize",
            expected_loss_perso=[0.064, 0.037, 0.066, 0.142],  # per-ft RMSE
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": {
                    "mean": 1.,
                    "min": .2,
                    "max": 2.,
                    "std": 1.,
                },
                "number_of_subjects": 100,
            },
            simulate_tol=2e-3,  # Not fully reproducible on Linux below this tol...
        )

    def test_usecase_logistic_binary(self):
        self.generic_usecase(
            "logistic",
            model_codename="logistic_binary",
            noise_model="bernoulli",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            perso_algo="mean_real",
            expected_loss_perso=105.18,  # logLL, not noise_std
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": .5,
                "number_of_subjects": 100,
                "reparametrized_age_bounds": (50, 85),
            },
        )

    def test_usecase_logistic_ordinal(self):
        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal",
            noise_model="ordinal",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            perso_algo="mean_real",
            expected_loss_perso=1064.9 if os.uname()[4][:3] == "arm" else 1065.0,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": .5,
                "number_of_subjects": 100,
                "reparametrized_age_bounds": (50, 85),
            },
        )

    def test_usecase_logistic_ordinal_batched(self):
        """
        Ordinal simulation may not be fully reproducible on different machines
        due to rounding errors when computing MultinomialDistribution.cdf that
        can lead to ±1 differences on MLE outcomes in rare cases...
        (changing seed, reducing subjects & increasing tol to avoid the problem).
        """
        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal_b",
            noise_model="ordinal",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            fit_check_kws={"atol": 0.005},
            perso_algo="mean_real",
            expected_loss_perso=1045.989,
            tol_loss=0.1,
            simulate_algo_params={
                "seed": 123,
                "delay_btw_visits": .5,
                "number_of_subjects": 10,
                "reparametrized_age_bounds": (50, 85),
            },
            simulate_tol=5e-2,
            batch_deltas_ordinal=True,
        )

    def test_usecase_univariate_logistic_ordinal(self):
        self.generic_usecase(
            "univariate_logistic",
            model_codename="univariate_logistic_ordinal",
            noise_model="ordinal",
            fit_algo_params={"n_iter": 200, "seed": 0},
            perso_algo="mean_real",
            expected_loss_perso=169.8,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": .5,
                "number_of_subjects": 100,
                "reparametrized_age_bounds": (50, 85),
            },
        )

    def test_usecase_logistic_ordinal_ranking(self):
        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal_ranking",
            noise_model="ordinal_ranking",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            perso_algo="mean_real",
            expected_loss_perso=976.4 if os.uname()[4][:3] == "arm" else 977.3,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": .5,
                "number_of_subjects": 100,
                "reparametrized_age_bounds": (50, 85),
            },
        )

    def test_usecase_logistic_ordinal_ranking_batched(self):
        self.generic_usecase(
            "logistic",
            model_codename="logistic_ordinal_ranking_b",
            noise_model="ordinal_ranking",
            source_dimension=2,
            fit_algo_params={"n_iter": 200, "seed": 0},
            perso_algo="mode_real",
            expected_loss_perso=971.95,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params={
                "seed": 0,
                "delay_btw_visits": .5,
                "number_of_subjects": 100,
                "reparametrized_age_bounds": (50, 85),
            },
            batch_deltas_ordinal=True,
        )
