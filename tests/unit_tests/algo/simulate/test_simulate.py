import os
import copy
import json

import numpy as np
import pandas as pd
import torch

from leaspy import AlgorithmSettings, Data
from leaspy.algo.simulate.simulate import SimulationAlgorithm
from leaspy.io.outputs.result import Result
from leaspy.io.settings import algo_default_data_dir

from tests import LeaspyTestCase


class SimulationAlgorithmTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls):
        # for tmp handling
        super().setUpClass()

        with open(os.path.join(algo_default_data_dir, 'default_simulation.json'), 'r') as jf:
            cls.default_params = json.load(jf)['parameters']

        cls.settings = AlgorithmSettings('simulation')
        cls.algo = SimulationAlgorithm(cls.settings)

        # reused data, model, individual parameters
        cls.data = cls.get_suited_test_data_for_model('logistic_scalar_noise')
        cofactors = pd.read_csv(cls.example_data_covars_path, dtype={'ID': str}).set_index('ID')
        cls.data.load_cofactors(cofactors, cofactors=["Treatments"])

        cls.lsp = cls.get_hardcoded_model('logistic_scalar_noise')

        # make data features match model ones (this way because we also use other models with same features in tests)
        cls.data.headers = cls.lsp.model.features

        perso_settings = AlgorithmSettings('mode_real')
        cls.individual_parameters = cls.lsp.personalize(cls.data, perso_settings)

    def test_constructor(self):
        """
        Test the initialization.
        """
        self.assertEqual(self.settings.parameters, self.default_params)

        for k, v in self.settings.parameters.items():
            self.assertEqual(v, getattr(self.algo, k), msg=f'Parameter {k}')

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', sources_method="bad_method"))

    def test_bad_nb_of_visits(self):
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', min_number_of_visits=0))
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', min_number_of_visits=3.))  # bad type
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', min_number_of_visits=7, mean_number_of_visits=6))
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', max_number_of_visits=5, mean_number_of_visits=6))

    def test_bad_delay_between_visits(self):
        with self.assertRaises(ValueError):
            # < 0
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits=-1))
        with self.assertRaises(ValueError):
            # missing mandatory keys min & std
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits={'mean': 0.5}))
        with self.assertRaises(ValueError):
            # bad order mean < min
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits={'min': 0.5, 'mean': .4, 'std': .1}))
        with self.assertRaises(ValueError):
            # bad extra key
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits={'min': .1, 'mean': .2, 'std': .1, 'yop': 'unknown_key'}))
        with self.assertRaises(ValueError):
            # std <= 0
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits={'min': 0.5, 'mean': .4, 'std': 0}))
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits='bad_kwd'))
        with self.assertRaises(ValueError):
            # bad length of result
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits=lambda n: [0.4]*(n+1)))
        with self.assertRaises(ValueError):
            # length OK but < 0
            SimulationAlgorithm(AlgorithmSettings('simulation', delay_btw_visits=lambda n: [0.]*n))

    def test_bad_nb_of_subj(self):
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', number_of_subjects=0))
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', number_of_subjects=-34))
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', number_of_subjects=None))
        with self.assertRaises(ValueError):
            SimulationAlgorithm(AlgorithmSettings('simulation', number_of_subjects=[24]))

    def test_get_number_of_visits(self):
        for _ in range(100):
            n_visit = self.algo._get_number_of_visits()
            self.assertIsInstance(n_visit, int)
            self.assertGreaterEqual(n_visit, 1)

    def test_get_mean_and_covariance_matrix(self):
        """
        Test the result given by the calculus with torch vs the dedicated function of numpy.
        """
        values = np.random.rand(100, 5)
        t_mean = torch.tensor(values).mean(dim=0)
        self.assertAllClose(values.mean(axis=0), t_mean, what='matrix.mean')
        t_cov = torch.tensor(values) - t_mean[None, :]
        t_cov = 1. / (t_cov.size(0) - 1) * t_cov.t() @ t_cov
        self.assertAllClose(np.cov(values.T), t_cov, what='matrix.cov')

    def test_check_cofactors(self):
        """
        Test Leaspy.simulate return a ``ValueError`` if the ``cofactor`` and ``cofactor_state`` parameters given
        in the ``AlgorithmSettings`` are invalid.
        """
        lsp, individual_parameters, data = self.lsp, self.individual_parameters, self.data

        # cofactor not None but cofactor_state None...
        settings = AlgorithmSettings('simulation', cofactor=["Treatments"])
        self.assertRaises(ValueError, lsp.simulate, individual_parameters, data, settings)

        # bad type for cofactor / cofactor state
        settings = AlgorithmSettings('simulation', cofactor="Treatments", cofactor_state=["Treatment_A"])
        self.assertRaises(ValueError, lsp.simulate, individual_parameters, data, settings)

        settings = AlgorithmSettings('simulation', cofactor=["Treatments"], cofactor_state="Treatment_A")
        self.assertRaises(ValueError, lsp.simulate, individual_parameters, data, settings)

        # bad length for cofactor_state
        settings = AlgorithmSettings('simulation', cofactor=["Treatments"], cofactor_state=["Treatment_A", "Treatment_B"])
        self.assertRaises(ValueError, lsp.simulate, individual_parameters, data, settings)

        # invalid cofactor name
        settings = AlgorithmSettings('simulation', cofactor=["dummy"], cofactor_state=["dummy"])
        self.assertRaises(ValueError, lsp.simulate, individual_parameters, data, settings)

        # invalid cofactor state
        settings = AlgorithmSettings('simulation', cofactor=["Treatments"], cofactor_state=["dummy"])
        self.assertRaises(ValueError, lsp.simulate, individual_parameters, data, settings)

    def _check_bin_values(self, result):
        vals = [set(np.unique(np.around(idata.observations, 6)))
                for idata in result.data.individuals.values()]
        self.assertEqual(set.union(*vals), {0., 1.})

    def test_simulation_noises(self):
        # define data & models
        lsp_scal, individual_parameters, data = self.lsp, self.individual_parameters, self.data

        lsp_diag = self.get_hardcoded_model('logistic_diag_noise')

        lsp_bin = copy.deepcopy(lsp_diag)
        lsp_bin.model.noise_model = 'bernoulli'

        # noise: value (scalar)
        settings = AlgorithmSettings('simulation', seed=0, noise=.12)
        r = lsp_diag.simulate(individual_parameters, data, settings)
        self.assertAllClose(r.noise_std, .12, what='noise')

        # noise: value (diagonal)
        diag_noise = .08 + .02*np.arange(lsp_bin.model.dimension)
        settings = AlgorithmSettings('simulation', seed=0, noise=diag_noise)
        r = lsp_bin.simulate(individual_parameters, data, settings)
        self.assertAllClose(r.noise_std, diag_noise, what='noise')

        # noise: Bernoulli
        settings = AlgorithmSettings('simulation', seed=0, noise='bernoulli')
        r = lsp_scal.simulate(individual_parameters, data, settings)
        self._check_bin_values(r)

        # noise: inherit_struct
        settings = AlgorithmSettings('simulation', seed=0)
        self.assertEqual(settings.parameters['noise'], 'inherit_struct')

        for lsp_obj in [lsp_scal, lsp_diag, lsp_bin]:
            r = lsp_obj.simulate(individual_parameters, data, settings)
            if lsp_obj != lsp_bin:
                model_noise_scale = lsp_obj.model.noise_model.parameters['scale']
                self.assertEqual(r.noise_std.numel(), model_noise_scale.numel())
            else:
                self.assertIsNone(r.noise_std)
                self._check_bin_values(r)

        # noise: removed 'default' (replaced by 'inherit_struct')
        with self.assertRaises(ValueError):
            settings = AlgorithmSettings('simulation', seed=0, noise='default')
            lsp_diag.simulate(individual_parameters, data, settings)

        # noise: model
        settings = AlgorithmSettings('simulation', seed=0, noise='model')

        for lsp_obj in [lsp_scal, lsp_diag, lsp_bin]:
            r = lsp_obj.simulate(individual_parameters, data, settings)
            if lsp_obj != lsp_bin:
                model_noise_scale = lsp_obj.model.noise_model.parameters['scale']
                self.assertAllClose(r.noise_std, model_noise_scale, what='noise')
            else:
                self.assertIsNone(r.noise_std)
                self._check_bin_values(r)

    @staticmethod
    def _get_nb_visits(result):
        return [len(idata.timepoints) for idata in result.data.individuals.values()]

    @staticmethod
    def _get_delays_btw_visits(result):
        return [np.diff(idata.timepoints) for idata in result.data.individuals.values()]

    def test_mean_nb_of_visits(self, tol=1e-1):
        lsp, individual_parameters, data = self.lsp, self.individual_parameters, self.data
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=10000,
                                     mean_number_of_visits=6, std_number_of_visits=3)
        new_results = lsp.simulate(individual_parameters, data, settings)
        self.assertIsInstance(new_results, Result)
        nb_visits = self._get_nb_visits(new_results)
        self.assertAlmostEqual(np.mean(nb_visits), 6, delta=tol) # deterministic

    def test_simulation_run(self, tol=1e-4):
        """
        Test if the simulation run properly with different settings.
        """
        lsp, individual_parameters, data = self.lsp, self.individual_parameters, self.data

        # full kde
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2)
        new_results = lsp.simulate(individual_parameters, data, settings)  # just test if run without error
        nb_visits = self._get_nb_visits(new_results)
        self.assertEqual(set(nb_visits), {3}) # deterministic

        # normal sources
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=None, sources_method="normal_sources", bandwidth_method=.2)
        new_results = lsp.simulate(individual_parameters, data, settings)  # just test if run without error
        nb_visits = self._get_nb_visits(new_results)
        self.assertEqual(set(nb_visits), {3}) # deterministic

        # min/max nb visits
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100, mean_number_of_visits=4,
                                     std_number_of_visits=6, min_number_of_visits=2, max_number_of_visits=6)
        new_results = lsp.simulate(individual_parameters, data, settings)  # just test if run without error
        nb_visits = self._get_nb_visits(new_results)
        self.assertGreaterEqual(min(nb_visits), 2)
        self.assertLessEqual(max(nb_visits), 6)

        # delay btw visits
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100, mean_number_of_visits=4,
                                     std_number_of_visits=3, delay_btw_visits={'min': .4, 'mean': .8, 'max': 1.2, 'std': 1.})
        new_results = lsp.simulate(individual_parameters, data, settings)  # just test if run without error
        delays_visits = self._get_delays_btw_visits(new_results)
        min_delays = [min(d_visits) for d_visits in delays_visits if len(d_visits) >= 1]
        max_delays = [max(d_visits) for d_visits in delays_visits if len(d_visits) >= 1]
        self.assertGreaterEqual(min(min_delays), .4 - tol)
        self.assertLessEqual(max(max_delays), 1.2 + tol)

        custom_vis_delays = lambda n_delays: [.5]*min(n_delays, 2) + [1.]*max(n_delays-2, 0)
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100, mean_number_of_visits=6,
                                     std_number_of_visits=3, delay_btw_visits=custom_vis_delays)
        new_results = lsp.simulate(individual_parameters, data, settings)  # just test if run without error
        delays_visits = [np.diff(idata.timepoints) for idata in new_results.data.individuals.values()
                         if len(idata.timepoints) > 1]  # empty delays when only 1 timepoint (was raising a numpy warning)
        for delays_vis in delays_visits:
            self.assertAllClose(delays_vis, custom_vis_delays(len(delays_vis)), what='delay_vis')

        # features bounds (auto)
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=200, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     features_bounds=True)  # idem + test scores bounds
        new_results = self._bounds_behaviour(lsp, individual_parameters, data, settings)
        nb_visits = self._get_nb_visits(new_results)
        self.assertEqual(set(nb_visits), {3}) # deterministic

        bounds = dict(zip(self.lsp.model.features, [(0., .5), (0., .1), (0., .1), (0., .1)]))
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=200, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     features_bounds=bounds)  # idem + test scores bounds
        new_results = self._bounds_behaviour(lsp, individual_parameters, data, settings)
        nb_visits = self._get_nb_visits(new_results)
        self.assertEqual(set(nb_visits), {3}) # deterministic

        # reparametrized_age_bounds
        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=200, mean_number_of_visits=4,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     reparametrized_age_bounds=(65, 75))
        new_results = lsp.simulate(individual_parameters, data, settings)  # just test if run without error
        nb_visits = self._get_nb_visits(new_results)
        self.assertEqual(set(nb_visits), {4}) # deterministic
        # Test if the reparametrized ages are within (65, 75) up to a tolerance of 2.
        repam_age = new_results.data.to_dataframe().groupby('ID').first()['TIME'].values
        repam_age -= new_results.individual_parameters['tau'].squeeze().numpy()
        repam_age *= np.exp(new_results.individual_parameters['xi'].squeeze().numpy())
        repam_age += lsp.model.parameters['tau_mean'].item()
        self.assertTrue(all(repam_age > 63) & all(repam_age < 77))  # "soft" bounds compared to (65, 75)

    def test_simulation_cofactors_run(self):
        """
        Test if the simulation run properly with different settings (no result check, only unit test).
        """
        lsp, individual_parameters, data = self.lsp, self.individual_parameters, self.data

        settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=1000, mean_number_of_visits=3,
                                     std_number_of_visits=0, sources_method="full_kde", bandwidth_method=.2,
                                     cofactor=['Treatments'], cofactor_state=['Treatment_A'])
        lsp.simulate(individual_parameters, data, settings)  # just test if run without error


    def _bounds_behaviour(self, lsp, individual_parameters, data, settings, *, tol=1e-4):
        """
        Test the good behaviour of the ``features_bounds`` parameter.

        Parameters
        ----------
        lsp : :class:`.Leaspy`
        results : :class:`~.io.outputs.result.Result`
        settings : :class:`.AlgorithmSettings`
            Contains the ``features_bounds`` parameter.
        """

        new_results = lsp.simulate(individual_parameters, data, settings)
        simulated_data_bl = new_results.data.to_dataframe().groupby('ID').first()
        simulated_max_bounds: np.ndarray = simulated_data_bl.max().values[1:]
        simulated_min_bounds: np.ndarray = simulated_data_bl.min().values[1:]

        if isinstance(settings.parameters['features_bounds'], dict):
            constraint_max_bounds = np.array([val[1] for val in settings.parameters["features_bounds"].values()])
            constraint_min_bounds = np.array([val[0] for val in settings.parameters["features_bounds"].values()])
        elif settings.parameters['features_bounds']:
            data_bl = data.to_dataframe().groupby('ID').first()
            constraint_max_bounds: np.ndarray = data_bl.max().values[1:]
            constraint_min_bounds: np.ndarray = data_bl.min().values[1:]
        else:
            raise ValueError('features_bounds is ill-defined')

        self.assertTrue(all(simulated_max_bounds <= constraint_max_bounds + tol),
                        "Generated scores contain scores outside the bounds")
        self.assertTrue(all(simulated_min_bounds >= constraint_min_bounds - tol),
                        "Generated scores contain scores outside the bounds")

        return new_results

    def test_simulate_univariate(self):
        from leaspy.datasets import Loader

        putamen_df = Loader.load_dataset('parkinson-putamen-train_and_test')
        data = Data.from_dataframe(putamen_df.xs('train', level='SPLIT'))
        leaspy_logistic = Loader.load_leaspy_instance('parkinson-putamen-train')
        individual_parameters = Loader.load_individual_parameters('parkinson-putamen-train')

        simulation_settings = AlgorithmSettings('simulation', seed=0)
        simulated_data = leaspy_logistic.simulate(individual_parameters, data, simulation_settings)

        simu_df = simulated_data.data.to_dataframe(reset_index=False)
        self.assertEqual(simu_df.index.names, ['ID', 'TIME'])
        self.assertEqual(list(simu_df.columns), ['PUTAMEN'])
        simu_df.reset_index('TIME', inplace=True)
        self.assertListEqual(list(simu_df.dtypes), ['float64']*2)
        self.assertTrue(all(simu_df['PUTAMEN'].values <= 1))
        self.assertTrue(all(simu_df['PUTAMEN'].values >= 0))
        self.assertTrue(all(simu_df['TIME'].values <= 150))
        self.assertTrue(all(simu_df['TIME'].values >= 10))

    # global behaviour of SimulationAlgorithm class is tested in the functional test test_api.py
