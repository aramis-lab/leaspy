import numpy as np
import os
import unittest

import matplotlib.pyplot as plt
import torch
from pandas import DataFrame

from tests import test_data_dir
from leaspy import Leaspy, Data, AlgorithmSettings, Plotter
from tests import example_data_path
from leaspy.inputs.data.result import Result
from leaspy.models.univariate_model import UnivariateModel


class LeaspyTest(unittest.TestCase):

    def test_usecase(self):
        """
        Fonctional test of a basic analysis using leaspy package
        1 - Data loading
        2 - Fit logistic model with MCMC algorithm
        3 - Save paramaters & reload (remove created files to keep the repo clean)
        4 - Personalize model with 'mode_real' algorithm
        5 - Plot results
        6 - Simulate new patients (TODO)
        :return: exit code
        """

        data = Data.from_csv_file(example_data_path)

        # Fit
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.2842, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 77.9887, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 1.0315, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1439, delta=0.001)
        diff_g = leaspy.model.parameters['g'] - torch.Tensor([1.9557, 2.5899, 2.5184, 2.2369])
        diff_v = leaspy.model.parameters['v0'] - torch.Tensor([-3.5714, -3.5820, -3.5811, -3.5886])
        self.assertAlmostEqual(torch.sum(diff_g ** 2), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v ** 2), 0.0, delta=0.01)

        # Save parameters and reload
        path_to_saved_model = os.path.join(test_data_dir,
                                           "model_parameters",
                                           'fitted_multivariate_model_testusecase-copy.json')
        leaspy.save(path_to_saved_model)
        leaspy = Leaspy.load(path_to_saved_model)
        os.remove(path_to_saved_model)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.2842, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 77.9887, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 1.0315, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1439, delta=0.001)
        diff_g = leaspy.model.parameters['g'] - torch.Tensor([1.9557, 2.5899, 2.5184, 2.2369])
        diff_v = leaspy.model.parameters['v0'] - torch.Tensor([-3.5714, -3.5820, -3.5811, -3.5886])
        self.assertAlmostEqual(torch.sum(diff_g ** 2), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v ** 2), 0.0, delta=0.01)

        # Personalize
        algo_personalize_settings = AlgorithmSettings('mode_real', seed=0)
        result = leaspy.personalize(data, settings=algo_personalize_settings)
        self.assertAlmostEqual(result.noise_std, 0.0923, delta=0.01)

        # Get error distribution
        error_distribution = result.get_error_distribution(leaspy.model)
        self.assertTrue(list(error_distribution.keys()), list(data.individuals.keys()))
        self.assertTrue(np.array(error_distribution['116']).shape,
                        np.array(data.individuals['116'].observations).shape)
        error_distribution = result.get_error_distribution(leaspy.model, aggregate_subscores=True)
        self.assertTrue(len(error_distribution['116']),
                        np.array(data.individuals['116'].observations).shape[0])
        error_distribution = result.get_error_distribution(leaspy.model, aggregate_visits=True)
        self.assertTrue(len(error_distribution['116']),
                        np.array(data.individuals['116'].observations).shape[1])
        error_distribution = result.get_error_distribution(leaspy.model, aggregate_visits=True, aggregate_subscores=True)
        self.assertTrue(type(error_distribution['116']) == float)

        # Plot TODO
        path_output = os.path.join(os.path.dirname(__file__), '../../_data', "_outputs")
        plotter = Plotter(path_output)
        # plotter.plot_mean_trajectory(leaspy.model, save_as="mean_trajectory_plot")
        plt.close()

        # Simulate
        simulation_settings = AlgorithmSettings('simulation')
        simulation_results = leaspy.simulate(result, simulation_settings)
        self.assertTrue(type(simulation_results) == Result)
        self.assertTrue(type(simulation_results.data.to_dataframe()) == DataFrame)
        self.assertTrue(simulation_results.data.headers == data.headers)
        n = simulation_settings.parameters['number_of_subjects']
        self.assertEqual(simulation_results.data.n_individuals, n)
        self.assertEqual(len(simulation_results.get_parameter_distribution('xi')), n)
        self.assertEqual(len(simulation_results.get_parameter_distribution('tau')), n)
        self.assertEqual(len(simulation_results.get_parameter_distribution('sources')['sources0']), n)



        """
        
            def test_constructor(self):
        leaspy = Leaspy('univariate')
        self.assertEqual(leaspy.type, 'univariate')
        self.assertEqual(type(leaspy.model), UnivariateModel)
        self.assertEqual(leaspy.model.model_parameters['p0'], None)
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], None)
        self.assertEqual(leaspy.model.model_parameters['tau_var'], None)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], None)
        self.assertEqual(leaspy.model.model_parameters['xi_var'], None)

    def test_constructor_from_parameters(self):
        path_to_model_parameters = os.path.join(test_data_dir, 'model_settings_univariate.json')
        leaspy = Leaspy.load(path_to_model_parameters)
        self.assertEqual(leaspy.type, "univariate")
        self.assertEqual(leaspy.model.model_parameters['p0'], [0.3])
        self.assertEqual(leaspy.model.model_parameters['tau_mean'], 50)
        self.assertEqual(leaspy.model.model_parameters['tau_var'], 2)
        self.assertEqual(leaspy.model.model_parameters['xi_mean'], -10)
        self.assertEqual(leaspy.model.model_parameters['xi_var'], 0.8)

        
        """








