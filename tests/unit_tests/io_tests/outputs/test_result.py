import os
import filecmp
import inspect
import warnings

import pandas as pd
import torch

from leaspy import Data, Result

from tests import LeaspyTestCase
from unittest import skip


class ResultTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls):

        # for tmp handling
        super().setUpClass()

        # The list of individuals that were pre-saved for tests on subset of individuals
        cls.idx_sub = ['116', '142', '169']

        # Starting from Torch 1.6.0 a new serialization method is used
        sig = inspect.signature(torch.save).parameters
        if '_use_new_zipfile_serialization' in sig and sig['_use_new_zipfile_serialization'].default is True:
            cls.torch_save_suffix = '_v2.pt'
        else:
            cls.torch_save_suffix = '.pt' #'_v1.pt'

        # Inputs
        cls.data = Data.from_csv_file(cls.example_data_path)
        cls.covariates = pd.read_csv(cls.example_data_covars_path, dtype={'ID': str}).set_index('ID')
        cls.data.load_covariates(cls.covariates)

        cls.df = cls.data.to_dataframe()

        load_individual_parameters_path = cls.from_personalize_ip_path("data_tiny-individual_parameters.json")
        cls.results = Result.load_result(cls.data, load_individual_parameters_path)

    def setUp(self):
        # ignore deprecation warnings in all tests (does not work in `setUpClass`)
        warnings.simplefilter('ignore', DeprecationWarning)

    def test_constructor(self):
        self.assertIsInstance(self.results.data, Data)
        self.assertIsInstance(self.results.individual_parameters, dict)
        self.assertEqual(list(self.results.individual_parameters.keys()), ['tau', 'xi', 'sources'])
        for key in self.results.individual_parameters.keys():
            self.assertEqual(len(self.results.individual_parameters[key]), 17)
        self.assertEqual(self.results.noise_std, None)

    def test_get_parameter_name_and_dim(self):
        f = Result._get_parameter_name_and_dim
        self.assertEqual(f('sources_12'), ('sources', 12))
        self.assertEqual(f('abc_def'), ('abc_def', None))
        self.assertEqual(f('tau'), ('tau', None))
        self.assertEqual(f('abc_def_5'), ('abc_def', 5))

    def test_save_individual_parameters(self):

        to_test = [
            # method, path, args, kwargs

            ## JSON
            # test save default
            (self.results.save_individual_parameters_json, 'data_tiny-individual_parameters.json', [], dict(indent=None)),
            # test to save only a subset of subjects
            (self.results.save_individual_parameters_json, 'data_tiny-individual_parameters-3subjects.json', [self.idx_sub], dict(indent=None)),
            # test if run with an **args of json.dump
            (self.results.save_individual_parameters_json, 'data_tiny-individual_parameters-indent_4.json', [self.idx_sub], dict(indent=4)),

            ## CSV
            # test save default
            (self.results.save_individual_parameters_csv, 'data_tiny-individual_parameters.csv', [], {}),
            # test to save only a subset of subjects
            (self.results.save_individual_parameters_csv, 'data_tiny-individual_parameters-3subjects.csv', [self.idx_sub], {}),

            ## Torch (suffix depend on PyTorch version)
            # test save default
            (self.results.save_individual_parameters_torch,
             f'data_tiny-individual_parameters{self.torch_save_suffix}', [], {}),
            # test to save only a subset of subjects
            (self.results.save_individual_parameters_torch,
             f'data_tiny-individual_parameters-3subjects{self.torch_save_suffix}', [self.idx_sub], {}),
        ]

        # We check that each re-saved file (with options) is same as expected
        for saving_method, ip_original, args, kwargs in to_test:
            with self.subTest(ip_original=ip_original):
                path_original = self.from_personalize_ip_path(ip_original)
                path_copy = self.get_test_tmp_path(ip_original)
                saving_method(path_copy, *args, **kwargs)
                self.assertTrue(filecmp.cmp(path_original, path_copy, shallow=False))
                os.unlink(path_copy)

        # Bad type: a list of indexes is expected for `idx` keyword (no tuple nor scalar!)
        bad_idx = ['116', 116, ('116',), ('116', '142')]
        fake_save = {
            'should_not_be_saved_due_to_error.json': self.results.save_individual_parameters_json,
            'should_not_be_saved_due_to_error.csv': self.results.save_individual_parameters_csv,
            f'should_not_be_saved_due_to_error{self.torch_save_suffix}': self.results.save_individual_parameters_torch,
        }
        for fake_path, saving_method in fake_save.items():
            for idx in bad_idx:
                with self.assertRaises(ValueError, msg=dict(idx=idx, path=fake_path)):
                    saving_method(self.get_test_tmp_path(fake_path), idx=idx)

    def generic_check_individual_parameters(self, ind_param, *, nb_individuals: int):
        self.assertEqual(type(ind_param), dict)
        self.assertEqual(list(ind_param.keys()), ['tau', 'xi', 'sources'])
        for key in ind_param.keys():
            self.assertEqual(type(ind_param[key]), torch.Tensor)
            self.assertEqual(ind_param[key].dtype, torch.float32)
            self.assertEqual(ind_param[key].dim(), 2)
            self.assertEqual(ind_param[key].shape[0], nb_individuals)

    def test_load_individual_parameters(self):
        self.generic_check_individual_parameters(self.results.individual_parameters, nb_individuals=17)

    def test_load_result(self):

        ind_param_input_list = [
            self.from_personalize_ip_path("data_tiny-individual_parameters.json"),
            self.from_personalize_ip_path("data_tiny-individual_parameters.csv"),
            self.from_personalize_ip_path(f"data_tiny-individual_parameters{self.torch_save_suffix}")
        ]

        data_input_list = [self.data, self.df, self.example_data_path]

        def load_result_and_check_same_as_expected(ind_param, data):
            results = Result.load_result(data, ind_param, covariates=self.example_data_covars_path)
            new_df = results.data.to_dataframe()
            pd.testing.assert_frame_equal(new_df, self.df)
            self.generic_check_individual_parameters(ind_param=results.individual_parameters, nb_individuals=17)

        for data_input in data_input_list:
            for ind_param_input in ind_param_input_list:
                with self.subTest(ip_path=ind_param_input, data=data_input):
                    load_result_and_check_same_as_expected(ind_param_input, data_input)

    @skip("Broken: 'MultivariateModel' object has no attribute 'compute_individual_tensorized'")
    def test_get_error_distribution_dataframe(self):
        leaspy_session = self.get_hardcoded_model('logistic_scalar_noise')
        self.results.get_error_distribution_dataframe(leaspy_session.model)

    ###############################################################
    # DEPRECATION WARNINGS
    # The corresponding methods will be removed in a future release
    ###############################################################

    def test_get_covariate_distribution(self):
        self.assertEqual(self.results.get_covariate_distribution('Treatments'),
                         self.covariates.values.reshape(self.covariates.shape[0]).tolist())

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
                         self.results.individual_parameters['xi'].view(-1).tolist())
        self.assertEqual(self.results.get_parameter_distribution('tau'),
                         self.results.individual_parameters['tau'].view(-1).tolist())
        self.assertEqual(self.results.get_parameter_distribution('sources'),
                         {'sources' + str(i): val
                          for i, val in
                          enumerate(self.results.individual_parameters['sources'].transpose(1, 0).tolist())})

        xi_treatment_param = self.results.get_parameter_distribution('xi', 'Treatments')
        self.assertEqual(list(xi_treatment_param.keys()), ["Treatment_A", "Treatment_B"])
        self.assertEqual(len(xi_treatment_param['Treatment_A']),
                         self.covariates[self.covariates['Treatments'] == 'Treatment_A'].shape[0])
