from json import load
from os import path
import unittest

from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings
from tests import test_data_dir
from leaspy.algo.simulate.simulate import SimulationAlgorithm
from tests.unit_tests.inputs.data.test_result import ResultTest


class SimulationAlgorithmTest(unittest.TestCase):

    def setUp(self):
        self.settings = AlgorithmSettings('simulation')
        self.algo = SimulationAlgorithm(self.settings)

    def test_construtor(self):
        self.assertEqual(self.settings.parameters['bandwidth_method'], self.algo.bandwidth_method)
        self.assertEqual(self.settings.parameters['noise'], self.algo.noise)
        self.assertEqual(self.settings.parameters['number_of_subjects'], self.algo.number_of_subjects)
        self.assertEqual(self.settings.parameters['mean_number_of_visits'], self.algo.mean_number_of_visits)
        self.assertEqual(self.settings.parameters['std_number_of_visits'], self.algo.std_number_of_visits)
        self.assertEqual(self.settings.parameters['cofactor'], self.algo.cofactor)
        self.assertEqual(self.settings.parameters['cofactor_state'], self.algo.cofactor_state)

    def test_get_number_of_visits(self):
        n_visit = self.algo._get_number_of_visits()
        self.assertTrue(type(n_visit) == int)
        self.assertTrue(n_visit >= 1)

    def test_get_xi_tau_sources_bl(self):
        results = ResultTest().setUp(get=True)
        self.algo.cofactor = "Treatments"
        self.algo.cofactor_state = "Treatment_A"
        xi, tau, sources, bl = self.algo._get_xi_tau_sources_bl(results)

        self.assertEqual(type(bl), list)
        self.assertEqual(len(bl), len([_ for _ in results.get_cofactor_distribution('Treatments') if _ == 'Treatment_A']))

        individual_parameters_treatment_A_path = path.join(test_data_dir, "individual_parameters",
                                                           "data_tiny-individual_parameters-Treatment_A.json")
        with open(individual_parameters_treatment_A_path, 'r') as f:
            individual_parameters_treatment_A = load(f)
        self.assertEqual(xi, individual_parameters_treatment_A['xi'])
        self.assertEqual(tau, individual_parameters_treatment_A['tau'])
        self.assertEqual(sources.T.tolist(), individual_parameters_treatment_A['sources'])
