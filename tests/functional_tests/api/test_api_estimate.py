from typing import Dict, Iterable, List

import pandas as pd

from leaspy.io.outputs.individual_parameters import IndividualParameters
from tests import LeaspyTestCase


class LeaspyEstimateTestMixin(LeaspyTestCase):
    def check_almost_equal_for_all_individual_timepoints(
        self,
        estimation_1: Dict[str, List],
        estimation_2: Dict[str, List],
        *,
        tol: float = 1e-5,
    ) -> None:
        self.assertDictAlmostEqual(estimation_1, estimation_2, atol=tol)

    def batch_checks(
        self,
        individual_parameters: IndividualParameters,
        timepoints: Dict[str, List],
        model_names: Iterable[str],
        expected_estimations: Dict[str, List],
    ) -> None:
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                leaspy = self.get_hardcoded_model(model_name)
                estimations = leaspy.estimate(
                    timepoints,
                    individual_parameters,
                )
                self.check_almost_equal_for_all_individual_timepoints(
                    estimations,
                    expected_estimations,
                    tol=1e-4,
                )


class LeaspyEstimateTest(LeaspyEstimateTestMixin):
    logistic_models = (
        "logistic_scalar_noise",
        "logistic_diag_noise_id",
        "logistic_diag_noise",
    )

    @property
    def individual_parameters(self):
        return self.get_hardcoded_individual_params("ip_save.json")

    def test_estimate_multivariate(self):
        timepoints = {"idx1": [78, 81], "idx2": [71]}
        expected_estimations = {
            "idx1": [
                [0.99641526, 0.34549406, 0.67467, 0.98959327],
                [0.9994672, 0.5080943, 0.8276345, 0.99921334],
            ],
            "idx2": [
                [0.13964376, 0.1367586, 0.23170303, 0.01551363],
            ],
        }

        self.batch_checks(
            self.individual_parameters,
            timepoints,
            self.logistic_models,
            expected_estimations,
        )

        # TODO logistic parallel?

        # TODO linear model?

    def test_estimate_univariate(self):
        individual_parameters = self.get_hardcoded_individual_params(
            "ip_univariate_save.json"
        )
        timepoints = {"idx1": [78, 81], "idx2": [71]}
        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ("logistic",)
        expected_ests = {"idx1": [[0.999607], [0.9999857]], "idx2": [[0.03098414]]}
        self.batch_checks(individual_parameters, timepoints, models, expected_ests)

    def test_estimate_joint_univariate(self):
        individual_parameters = self.get_hardcoded_individual_params(
            "ip_univariate_save.json"
        )
        timepoints = {"idx1": [71, 74, 81], "idx2": [72, 77.5]}
        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ("joint",)
        expected_ests = {
            "idx1": [[0.8602, 1.0], [0.9995, 0.9337], [1.0, 0.0]],
            "idx2": [[0.0973, 1.0], [0.9999, 0.4957]],
        }
        self.batch_checks(individual_parameters, timepoints, models, expected_ests)

    def test_estimate_joint_univariate_to_dataframe(self):
        """estimate() with to_dataframe=True returns a DataFrame whose columns are
        model.features + model.event_features and whose values match the dict output."""
        import numpy as np

        individual_parameters = self.get_hardcoded_individual_params(
            "ip_univariate_save.json"
        )
        timepoints = {"idx1": [71, 74, 81], "idx2": [72, 77.5]}
        model = self.get_hardcoded_model("joint")

        df = model.estimate(timepoints, individual_parameters, to_dataframe=True)
        raw = model.estimate(timepoints, individual_parameters, to_dataframe=False)

        # DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        expected_columns = list(model.features) + model.event_features
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(list(df.index.names), ["ID", "TIME"])

        # Values must match the dict-based output
        for subj_id, tpts in timepoints.items():
            for t_idx, t in enumerate(tpts):
                row = df.loc[(subj_id, t)]
                np.testing.assert_allclose(
                    row.values, raw[subj_id][t_idx], rtol=1e-5
                )

    def test_estimate_joint_univariate_multiindex(self):
        """estimate() with a pd.MultiIndex input auto-enables to_dataframe=True and
        returns a correctly indexed DataFrame with longitudinal + event columns."""
        import numpy as np

        individual_parameters = self.get_hardcoded_individual_params(
            "ip_univariate_save.json"
        )
        timepoints_dict = {"idx1": [71, 74, 81], "idx2": [72, 77.5]}
        ix = pd.MultiIndex.from_tuples(
            [(sid, t) for sid, tpts in timepoints_dict.items() for t in tpts],
            names=["ID", "TIME"],
        )
        model = self.get_hardcoded_model("joint")

        df = model.estimate(ix, individual_parameters)

        self.assertIsInstance(df, pd.DataFrame)
        expected_columns = list(model.features) + model.event_features
        self.assertEqual(list(df.columns), expected_columns)
        # Index must be the original MultiIndex (same order)
        self.assertTrue(df.index.equals(ix))

    def test_estimate_joint_multivariate_to_dataframe(self):
        """estimate() on a multivariate joint model returns a DataFrame with
        n_longitudinal_features + nb_events columns."""
        individual_parameters = self.get_hardcoded_individual_params("ip_save.json")
        timepoints = {"idx1": [71, 74], "idx2": [72]}
        model = self.get_hardcoded_model("joint_diagonal")

        df = model.estimate(timepoints, individual_parameters, to_dataframe=True)

        self.assertIsInstance(df, pd.DataFrame)
        expected_columns = list(model.features) + model.event_features
        self.assertEqual(list(df.columns), expected_columns)
        # Shape: total timepoints × (n_features + nb_events)
        total_tpts = sum(len(tpts) for tpts in timepoints.values())
        self.assertEqual(df.shape, (total_tpts, len(model.features) + model.nb_events))
