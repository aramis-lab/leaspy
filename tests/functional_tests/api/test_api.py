# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from .test_api_fit import LeaspyFitTest_Mixin
from .test_api_personalize import LeaspyPersonalizeTest_Mixin
from .test_api_simulate import LeaspySimulateTest_Mixin


class LeaspyAPITest(LeaspyFitTest_Mixin, LeaspyPersonalizeTest_Mixin, LeaspySimulateTest_Mixin):

    def generic_usecase(self, model_name: str, model_codename: str, *,
                        expected_noise_std, # in perso
                        perso_algo: str, fit_algo='mcmc_saem', simulate_algo='simulation',
                        fit_check_kws = dict(atol=1e-3),
                        fit_algo_params=dict(seed=0), perso_algo_params=dict(seed=0),
                        simulate_algo_params=dict(seed=0), simulate_tol=1e-4,
                        **model_hyperparams):
        """
        Functional test of a basic analysis using leaspy package

        1 - Data loading
        2 - Fit logistic model with MCMC algorithm
        3 - Save parameters & reload (remove created files to keep the repo clean)
        4 - Personalize model with 'mode_real' algorithm
        (5 - Plot results)
        6 - Simulate new patients
        """
        filename_expected_model = model_codename + '_for_test_api'

        leaspy, data = self.generic_fit(model_name, filename_expected_model, **model_hyperparams,
                                        algo_name=fit_algo, algo_params=fit_algo_params,
                                        check_model=True, check_kws=fit_check_kws)

        # unlink 1st functional fit test from next steps...
        leaspy = self.get_from_fit_model(filename_expected_model)
        self.assertTrue(leaspy.model.is_initialized)

        # Personalize
        algo_personalize_settings = self.get_algo_settings(name=perso_algo, **perso_algo_params)
        individual_parameters, noise_std = leaspy.personalize(data, settings=algo_personalize_settings, return_noise=True)
        self.check_consistency_of_personalization_outputs(
                individual_parameters, noise_std,
                expected_noise_std=expected_noise_std, tol_noise=1e-2)

        # Simulate
        simulation_settings = self.get_algo_settings(name=simulate_algo, **simulate_algo_params)
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)

        self.check_consistency_of_simulation_results(simulation_settings, simulation_results, data,
                expected_results_file=f'simulation_results_{model_codename}.csv', tol=simulate_tol)

    def test_usecase_logistic_scalar_noise(self):

        # Simulation parameters
        custom_delays_vis = lambda n: [.5]*min(n, 2) + [1.]*max(0, n-2)  # OLD weird delays between visits
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100)  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_scalar_noise',
            noise_model='gaussian_scalar', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mode_real',
            perso_algo_params=dict(n_iter=200, seed=0),
            expected_noise_std=0.0857, # in perso
            simulate_algo_params=simul_params,
        )

    def test_usecase_logistic_diag_noise(self):

        # Simulation parameters
        custom_delays_vis = dict(mean=1., min=.2, max=2., std=1.)
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100)  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_diag_noise',
            noise_model='gaussian_diagonal', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='scipy_minimize',
            expected_noise_std=[0.064, 0.037, 0.066, 0.142],  # in perso
            simulate_algo_params=simul_params, simulate_tol=2e-3, # Not fully reproducible on Linux below this tol...
        )

    def test_usecase_logistic_binary(self):

        # Simulation parameters
        custom_delays_vis = .5
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100,
                            reparametrized_age_bounds=(50, 85))  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_binary',
            noise_model='bernoulli', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mean_real',
            perso_algo_params=dict(n_iter=200, seed=0),
            expected_noise_std=[0.343, 0.091, 0.125, 0.243],  # in perso
            simulate_algo_params=simul_params,
        )
