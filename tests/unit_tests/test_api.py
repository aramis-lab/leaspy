import filecmp
import os
import unittest

from torch import tensor

from leaspy.api import Leaspy
from leaspy.models.model_factory import ModelFactory

from tests import test_data_dir
from tests.unit_tests.models.test_model_factory import ModelFactoryTest


class LeaspyTest(unittest.TestCase):

    def test_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        :return: exit code
        """
        print("Unit-test constructor Leaspy")
        for name in ['univariate', 'linear', 'logistic', 'logistic_parallel']:
            leaspy = Leaspy(name)
            self.assertEqual(leaspy.type, name)
            self.assertEqual(type(leaspy.model), type(ModelFactory.model(name)))
            ModelFactoryTest().test_model_factory_constructor(leaspy.model)

    def test_load_individual_parameters(self, path=None):
        """
        Test load individual parameters
        :param path: string - optional - data path
        :return: exit code
        """
        if path is None:
            data_path_torch = os.path.join(test_data_dir,
                                           'individual_parameters/individual_parameters-unit_tests-torch.pt')
            data_path_json = os.path.join(test_data_dir,
                                          'individual_parameters/individual_parameters-unit_tests-json.json')
            self.test_load_individual_parameters(data_path_torch)
            self.test_load_individual_parameters(data_path_json)
        else:
            individual_parameters = Leaspy.load_individual_parameters(path)
            self.assertTrue((individual_parameters['xi'] == tensor([[1], [2], [3]])).min().item() == 1)
            self.assertTrue((individual_parameters['tau'] ==  tensor([[2], [3], [4]])).min().item() == 1)
            self.assertTrue((individual_parameters['sources'] ==
                             tensor([[1, 2], [2, 3], [3, 4]])).min().item() == 1)

    def test_save_individual_parameters(self):
        individual_parameters = {'xi': tensor([[1], [2], [3]]),
                                 'tau': tensor([[2], [3], [4]]),
                                 'sources': tensor([[1, 2], [2, 3], [3, 4]])}

        data_path_torch = os.path.join(test_data_dir,
                                       'individual_parameters/individual_parameters-unit_tests-torch.pt')
        data_path_json = os.path.join(test_data_dir,
                                      'individual_parameters/individual_parameters-unit_tests-json.json')

        data_path_torch_copy = data_path_torch[:-3] + '-Copy.pt'
        data_path_json_copy = data_path_json[:-5] + '-Copy.json'

        # Test torch file saving
        Leaspy.save_individual_parameters(data_path_torch_copy,
                                          individual_parameters,
                                          human_readable=False)
        try:
            self.test_load_individual_parameters(data_path_torch_copy)
            # filecmp does not work on torch file object - two different file can encode the same object
            os.remove(data_path_torch_copy)
        except AssertionError:
            os.remove(data_path_torch_copy)
            raise AssertionError("Leaspy.save_individual_parameters did not produce the expected torch file")

        # Test json file saving
        Leaspy.save_individual_parameters(data_path_json_copy,
                                          individual_parameters,
                                          human_readable=True)
        try:
            self.assertTrue(filecmp.cmp(data_path_json, data_path_json_copy))
            os.remove(data_path_json_copy)
        except AssertionError:
            os.remove(data_path_json_copy)
            raise AssertionError("Leaspy.save_individual_parameters did not produce the expected json file")

