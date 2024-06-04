import os
import unittest
import warnings

import numpy as np
import pandas as pd

from leaspy.io.outputs.result import Result

from tests import LeaspyTestCase


class LeaspySimulateTest_Mixin(LeaspyTestCase):
    """Mixin holding generic simulation methods that may be safely reused in other tests (no actual test here)."""

    @classmethod
    def generic_simulate(
        cls,
        hardcoded_model_name: str,
        hardcoded_ip_name: str,
        *,
        algo_name: str = "simulation",
        **algo_params,
    ):
        """Helper for a generic simulation in following tests."""

        # load saved model (hardcoded values)
        leaspy = cls.get_hardcoded_model(hardcoded_model_name)
        # load the right data
        data = cls.get_suited_test_data_for_model(hardcoded_model_name)
        # load saved individual parameters
        individual_parameters = cls.get_hardcoded_individual_params(hardcoded_ip_name)
        # create the simulate algo settings
        simulation_settings = cls.get_algo_settings(name=algo_name, **algo_params)
        # simulate new subjects and their data
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)

        return simulation_results

    def check_consistency_of_simulation_results(
        self,
        simulation_settings,
        simulation_results: Result,
        data,
        *,
        model,
        expected_results_file,
        tol: float = 1e-5,
    ):
        # TODO: refact, so dirty!

        self.assertIsInstance(simulation_results, Result)
        self.assertEqual(simulation_results.data.headers, data.headers)
        n = simulation_settings.parameters['number_of_subjects']
        self.assertEqual(simulation_results.data.n_individuals, n)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self.assertEqual(len(simulation_results.get_parameter_distribution('xi')), n)
            self.assertEqual(len(simulation_results.get_parameter_distribution('tau')), n)
            if 'sources' in simulation_results.individual_parameters:
                n_sources = simulation_results.individual_parameters['sources'].shape[1]
                for i in range(n_sources):
                    self.assertEqual(len(simulation_results.get_parameter_distribution(f'sources_{i}')), n)

        df_simulation = simulation_results.data.to_dataframe().set_index('ID')

        # First consistency check on range of values
        max_level_fts = 1  # set np.inf for linear models?
        if getattr(model, 'is_ordinal', False):
            max_level_fts = pd.Series(model.ordinal_infos["max_levels"])

        df_vals = df_simulation.set_index('TIME', append=True)
        range_simulated_nok = ~((0 <= df_vals) & (df_vals <= max_level_fts))
        bad_rows = df_vals.loc[range_simulated_nok.any(axis=1), range_simulated_nok.any(axis=0)]
        if len(bad_rows):
            self.fail(f"Some simulated values are out of expected range:\n{bad_rows}")

        path_expected_sim_res = self.get_test_data_path("simulation", expected_results_file)
        # uncomment to re-generate simulation results
        if not os.path.exists(path_expected_sim_res):
            warnings.warn(f"Generating missing results for '{expected_results_file}'...")
            df_simulation.to_csv(path_expected_sim_res, float_format='{:.6g}'.format)

        # Test the reproducibility of simulate
        # round is necessary, writing and reading induces numerical errors of magnitude ~ 1e-13
        # BUT ON DIFFERENT MACHINE I CAN SEE ERROR OF MAGNITUDE 1e-5 !!!
        # TODO: Can we improve this??
        expected_df_simulation = pd.read_csv(path_expected_sim_res, index_col=['ID'])

        # Check ID before (str doesn't work with numpy.allclose)
        id_simulation_is_reproducible = expected_df_simulation.index.equals(df_simulation.index)
        self.assertTrue(id_simulation_is_reproducible)

        simulation_is_reproducible = np.allclose(
            expected_df_simulation.values,
            df_simulation.values,
            atol=tol,
            rtol=tol,
        )
        # Use of numpy.allclose instead of pandas.testing.assert_frame_equal because of buggy behaviour reported
        # in https://github.com/pandas-dev/pandas/issues/22052

        # If reproducibility error > 1e-5 => display it + visit with the biggest reproducibility error
        error_message = ''
        if not simulation_is_reproducible:
            round_decimal = 5
            # expected_df_simulation = pd.read_csv(path_expected_sim_res)
            max_diff = 0.
            value_v1 = 0.
            value_v2 = 0.
            count = 0
            for v1, v2 in zip(
                expected_df_simulation.values.tolist(),
                df_simulation.values.tolist()
            ):
                diff = [abs(val1 - val2) for val1, val2 in zip(v1, v2)]
                if max(diff) > tol:
                    count += 1
                if max(diff) > max_diff:
                    value_v1 = v1
                    value_v2 = v2
                    max_diff = max(diff)
            error_message += '\nTolerance error = %.1e' % tol
            error_message += '\nMaximum error = %.3e' % max_diff
            error_message += '\n' + str([round(v, round_decimal+1) for v in value_v1])
            error_message += '\n' + str([round(v, round_decimal+1) for v in value_v2])
            error_message += '\nNumber of simulated visits above tolerance error = %d / %d \n' \
                             % (count, expected_df_simulation.shape[0])
        # For loop before the last self.assert - otherwise no display is made
        self.assertTrue(simulation_is_reproducible, error_message)


class LeaspySimulateTest(LeaspySimulateTest_Mixin):

    @unittest.skip('TODO')
    def test_simulate_for_some_models(self):

        # TODO: hardcode a file with individuals parameters for each individual from data tiny!
        # Complete functional tests are done in test_api.py

        for model_codename, hardcoded_ip_file, simulation_params in [
            ('logistic_scalar_noise', ..., dict(number_of_subjects=100)),
            ('logistic_diag_noise', ..., dict(number_of_subjects=100)),
            ('logistic_binary', ..., dict(number_of_subjects=100)),
        ]:
            with self.subTest(model_codename=model_codename, **simulation_params):
                simulation_results = self.generic_simulate(model_codename, hardcoded_ip_file, **simulation_params)
