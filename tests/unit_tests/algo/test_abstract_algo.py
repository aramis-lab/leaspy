from numpy.random import get_state, random_integers
from os import path
from subprocess import Popen
import unittest

from torch.random import initial_seed

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings
from leaspy.utils.output.fit_output_manager import FitOutputManager
from tests import test_data_dir


class Test(unittest.TestCase):

    def test_initialize_seed(self):
        algo = AbstractAlgo()
        seed = random_integers(10000)
        algo._initialize_seed(seed)
        self.assertEqual(seed, get_state()[1][0])
        self.assertEqual(seed, initial_seed())

    def test_load_parameters(self):
        algo = AbstractAlgo()
        algo.algo_parameters = {'param1': 1, 'param2': 2}
        parameters = {'param1': 10, 'param3': 3}
        algo.load_parameters(parameters)
        self.assertEqual(list(algo.algo_parameters.keys()), ['param1', 'param2', 'param3'])
        self.assertEqual(algo.algo_parameters['param1'], 10)
        self.assertEqual(algo.algo_parameters['param2'], 2)
        self.assertEqual(algo.algo_parameters['param3'], 3)

    def test_set_output_manager(self):
        algo = AbstractAlgo()
        algo.set_output_manager(None)
        self.assertEqual(algo.output_manager, None)

        # TODO: capture question & answer yes automatically (use subprocess.Popen.communicate ?) or remove interactive mode
        # settings = AlgorithmSettings('mcmc_saem')
        # output_path = path.join(test_data_dir, "_outputs", 'output_manager')
        # settings.set_logs(output_path)
        # algo.set_output_manager(settings.logs)
        # self.assertTrue(type(algo.output_manager) == FitOutputManager)
