import os
import torch
import warnings
import pandas as pd
from enum import Enum

from leaspy import Data, Result
from leaspy.utils.typing import Callable

from tests import LeaspyTestCase


class Method(str, Enum):
    JSON = "json"
    CSV = "csv"
    TORCH = "torch"


class ResultTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The list of individuals that were pre-saved for tests on subset of individuals
        cls.idx_sub = ["116", "142", "169"]
        cls.data = Data.from_csv_file(cls.example_data_path)
        cls.cofactors = pd.read_csv(cls.example_data_covars_path, dtype={'ID': str}).set_index('ID')
        cls.data.load_cofactors(cls.cofactors)
        cls.df = cls.data.to_dataframe()
        load_individual_parameters_path = cls.from_personalize_ip_path("data_tiny-individual_parameters.json")
        cls.results = Result.load_result(cls.data, load_individual_parameters_path)

    def setUp(self):
        # ignore deprecation warnings in all tests (does not work in `setUpClass`)
        warnings.simplefilter('ignore', DeprecationWarning)

    def test_constructor(self):
        self.assertIsInstance(self.results.data, Data)
        self.assertIsInstance(self.results.individual_parameters, dict)
        self.assertEqual(list(self.results.individual_parameters.keys()), ["tau", "xi", "sources"])
        for key in self.results.individual_parameters.keys():
            self.assertEqual(len(self.results.individual_parameters[key]), 17)
        self.assertEqual(self.results.noise_std, None)

    def test_get_parameter_name_and_dim(self):
        f = Result._get_parameter_name_and_dim
        self.assertEqual(f('sources_12'), ('sources', 12))
        self.assertEqual(f('abc_def'), ('abc_def', None))
        self.assertEqual(f('tau'), ('tau', None))
        self.assertEqual(f('abc_def_5'), ('abc_def', 5))

    def _saving_method_factory(self, method: Method) -> Callable:
        if method == Method.JSON:
            return self.results.save_individual_parameters_json
        if method == Method.CSV:
            return self.results.save_individual_parameters_csv
        if method == Method.TORCH:
            return self.results.save_individual_parameters_torch

    def _loading_method_factory(self, method: Method) -> Callable:
        if method == Method.JSON:
            return self.results.load_individual_parameters_from_json
        if method == Method.CSV:
            return self.results.load_individual_parameters_from_csv
        if method == Method.TORCH:
            return self.results.load_individual_parameters_from_torch

    def _save_and_compare(self, individual_parameter_filename: str, method: Method, *args, **kwargs):
        path_to_expected_individual_parameters = self.from_personalize_ip_path(individual_parameter_filename)
        path_to_computed_individual_parameters = self.get_test_tmp_path(individual_parameter_filename)
        self._saving_method_factory(method)(path_to_computed_individual_parameters, *args, **kwargs)
        self.assertDictAlmostEqual(
            self._loading_method_factory(method)(path_to_expected_individual_parameters),
            self._loading_method_factory(method)(path_to_computed_individual_parameters),
        )
        os.unlink(path_to_computed_individual_parameters)

    def test_save_individual_parameters_default_json(self):
        self._save_and_compare("data_tiny-individual_parameters.json", Method.JSON, indent=None)

    def test_save_individual_parameters_subset_of_subjects_json(self):
        self._save_and_compare(
            "data_tiny-individual_parameters-3subjects.json", Method.JSON, self.idx_sub, indent=None
        )

    def test_save_individual_parameters_with_json_dump_args(self):
        self._save_and_compare(
            "data_tiny-individual_parameters-indent_4.json", Method.JSON, self.idx_sub, indent=4
        )

    def test_save_individual_parameters_default_csv(self):
        self._save_and_compare("data_tiny-individual_parameters.csv", Method.CSV)

    def test_save_individual_parameters_subset_of_subjects_csv(self):
        self._save_and_compare("data_tiny-individual_parameters-3subjects.csv", Method.CSV, self.idx_sub)

    def test_save_individual_parameters_default_torch(self):
        self._save_and_compare("data_tiny-individual_parameters.pt", Method.TORCH)

    def test_save_individual_parameters_subset_of_subjects_torch(self):
        self._save_and_compare(
            "data_tiny-individual_parameters-3subjects.pt", Method.TORCH, self.idx_sub
        )

    def test_save_individual_parameters_bad_type(self):
        """Bad type: a list of indexes is expected for `idx` keyword (no tuple nor scalar!)."""
        bad_idx = ["116", 116, ("116",), ("116", "142")]
        fake_save = {
            "should_not_be_saved_due_to_error.json": self.results.save_individual_parameters_json,
            "should_not_be_saved_due_to_error.csv": self.results.save_individual_parameters_csv,
            f"should_not_be_saved_due_to_error.pt": self.results.save_individual_parameters_torch,
        }
        for fake_path, saving_method in fake_save.items():
            for idx in bad_idx:
                with self.assertRaises(ValueError, msg=dict(idx=idx, path=fake_path)):
                    saving_method(self.get_test_tmp_path(fake_path), idx=idx)

    def generic_check_individual_parameters(self, ind_param, *, nb_individuals: int):
        self.assertEqual(type(ind_param), dict)
        self.assertEqual(list(ind_param.keys()), ["tau", "xi", "sources"])
        for key in ind_param.keys():
            self.assertEqual(type(ind_param[key]), torch.Tensor)
            self.assertEqual(ind_param[key].dtype, torch.float32)
            self.assertEqual(ind_param[key].dim(), 2)
            self.assertEqual(ind_param[key].shape[0], nb_individuals)

    def test_load_individual_parameters(self):
        self.generic_check_individual_parameters(self.results.individual_parameters, nb_individuals=17)

    def test_load_result(self):
        individual_parameter_input_list = [
            self.from_personalize_ip_path(f"data_tiny-individual_parameters.{extension}")
            for extension in ("json", "csv", "pt")
        ]
        for data_input in (self.data, self.df, self.example_data_path):
            for ind_param_input in individual_parameter_input_list:
                with self.subTest(ip_path=ind_param_input, data=data_input):
                    self._load_result_and_check_same_as_expected(ind_param_input, data_input)

    def _load_result_and_check_same_as_expected(self, ind_param, data):
        results = Result.load_result(data, ind_param, cofactors=self.example_data_covars_path)
        new_df = results.data.to_dataframe()
        pd.testing.assert_frame_equal(new_df, self.df)
        self.generic_check_individual_parameters(ind_param=results.individual_parameters, nb_individuals=17)

    def test_get_error_distribution_dataframe(self):
        leaspy_session = self.get_hardcoded_model('logistic_scalar_noise')
        self.results.get_error_distribution_dataframe(leaspy_session.model)

    ###############################################################
    # DEPRECATION WARNINGS
    # The corresponding methods will be removed in a future release
    ###############################################################

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
                         self.cofactors[self.cofactors['Treatments'] == 'Treatment_A'].shape[0])
