import os
import unittest
import pandas as pd

from leaspy import Leaspy
from leaspy import Data
from leaspy.inputs.data.result import Result
from tests import example_data_path
from tests import test_data_dir


class ResultTest(unittest.TestCase):

    def setUp(self, get=False):
        # Inputs
        data = Data.from_csv_file(example_data_path)

        cofactors_path = os.path.join(test_data_dir,
                                      "inputs",
                                      "data_tiny_covariate.csv")
        self.cofactors = pd.read_csv(cofactors_path, index_col=0)
        self.cofactors.index = [str(i) for i in self.cofactors.index.values]
        data.load_cofactors(self.cofactors, ['Treatments'])

        load_individual_parameters_path = os.path.join(test_data_dir,
                                                       "individual_parameters",
                                                       "data_tiny-individual_parameters.json")
        individual_parameters = Leaspy.load_individual_parameters(load_individual_parameters_path)

        self.results = Result(data, individual_parameters)
        if get:
            return self.results

    def test_constructor(self):
        self.assertTrue(type(self.results.data) == Data)
        self.assertTrue(type(self.results.individual_parameters) == dict)
        self.assertEqual(list(self.results.individual_parameters.keys()), ['tau', 'xi', 'sources'])
        for key in self.results.individual_parameters.keys():
            self.assertEqual(len(self.results.individual_parameters[key]), 17)
        self.assertEqual(self.results.noise_std, None)

    def test_get_cofactor_distribution(self):
        self.assertEqual(self.results.get_cofactor_distribution('Treatments'),
                         self.cofactors.values.reshape(self.cofactors.shape[0]).tolist())

    def test_get_patient_individual_parameters(self):
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['tau'].tolist()[0],
                               79.124, delta=1e-3)
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['xi'].tolist()[0],
                               0.5355, delta=1e-3)
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['sources'].tolist()[0],
                               3.7742, delta=1e-3)
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['sources'].tolist()[1],
                               5.0088, delta=1e-3)

    def test_get_parameter_distribution(self):
        self.assertEqual(self.results.get_parameter_distribution('xi'),
                         self.results.individual_parameters['xi'].numpy().ravel().tolist())
        self.assertEqual(self.results.get_parameter_distribution('tau'),
                         self.results.individual_parameters['tau'].numpy().ravel().tolist())
        self.assertEqual(self.results.get_parameter_distribution('sources'),
                         {'sources' + str(i): val
                          for i, val in
                          enumerate(self.results.individual_parameters['sources'].numpy().T.tolist())})

        xi_treatment_param = self.results.get_parameter_distribution('xi', 'Treatments')
        self.assertEqual(list(xi_treatment_param.keys()), ["Treatment_A", "Treatment_B"])
        self.assertEqual(len(xi_treatment_param['Treatment_A']),
                         self.cofactors[self.cofactors['Treatments'] == 'Treatment_A'].shape[0])

