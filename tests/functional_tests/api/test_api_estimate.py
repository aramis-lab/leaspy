from typing import Dict, Iterable, List
from unittest import skip

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
        *,
        ordinal_method: str = "MLE",
    ) -> None:
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                leaspy = self.get_hardcoded_model(model_name)
                estimations = leaspy.estimate(
                    timepoints,
                    individual_parameters,
                    ordinal_method=ordinal_method,
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
    ordinal_models = (
        "logistic_ordinal_b",
    )  # "logistic_ordinal_ranking_same",) # not implemented anymore

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

    @skip("Broken: Ordinal models are broken")
    def test_estimate_ordinal_mle(self):
        """Loss is not involved in estimation so all expected outputs are the same
        for the 2 ordinal models.
        """
        timepoints = {"idx1": [71, 81], "idx2": [71]}
        expected_estimations = {
            "idx1": [
                [2, 0, 1, 4],  # before saturation
                [3, 4, 6, 10],  # saturation
            ],
            "idx2": [
                [0, 0, 0, 0],
            ],
        }
        self.batch_checks(
            self.individual_parameters,
            timepoints,
            self.ordinal_models,
            expected_estimations,
            ordinal_method="MLE",
        )

    @skip("Broken: Ordinal models are broken")
    def test_estimate_ordinal_expectation(self):
        timepoints = {"idx1": [71, 81], "idx2": [71]}
        expected_estimations = {
            "idx1": [
                [1.7107, 0.7169, 1.8397, 3.9503],
                [3.0, 3.9864, 5.9822, 10.0],
            ],
            "idx2": [
                [1.8477e-05, 1.3745e-01, 4.5545e-01, 1.7405e-05],
            ],
        }
        self.batch_checks(
            self.individual_parameters,
            timepoints,
            self.ordinal_models,
            expected_estimations,
            ordinal_method="expectation",
        )

    @skip("Broken: Ordinal models are broken")
    def test_estimate_ordinal_probabilities(self):
        timepoints = {"idx1": [71, 81], "idx2": [71]}
        expected_estimations = {
            "idx1": {
                ("Y0", 0): [0.0, 0.0],
                ("Y0", 1): [0.2893, 0.0],
                ("Y0", 2): [0.7107, 0.0],
                ("Y0", 3): [0.0, 1.0],
                ("Y1", 0): [0.6068, 0.0008],
                ("Y1", 1): [0.2569, 0.0024],
                ("Y1", 2): [0.0427, 0.0017],
                ("Y1", 3): [0.0, 0.0],  # too rare
                ("Y1", 4): [0.0937, 0.9952],
                ("Y2", 0): [0.2375, 0.0003],
                ("Y2", 1): [0.4202, 0.0014],
                ("Y2", 2): [0.1227, 0.0014],
                ("Y2", 3): [0.0478, 0.0011],
                ("Y2", 4): [0.0, 0.0],  # too rare
                ("Y2", 5): [0.0, 0.0],  # too rare
                ("Y2", 6): [0.1718, 0.9958],
                ("Y3", 0): [0.0, 0.0],
                ("Y3", 1): [0.0, 0.0],
                ("Y3", 2): [0.0305, 0.0],
                ("Y3", 3): [0.0, 0.0],
                ("Y3", 4): [0.9638, 0.0],
                ("Y3", 5): [0.0, 0.0],
                ("Y3", 6): [0.0057, 0.0],
                ("Y3", 7): [0.0, 0.0],
                ("Y3", 8): [0.0, 0.0],
                ("Y3", 9): [0.0, 0.0],
                ("Y3", 10): [0.0, 1.0],
            },
            "idx2": {
                ("Y0", 0): [1.0],
                ("Y0", 1): [0.0],
                ("Y0", 2): [0.0],
                ("Y0", 3): [0.0],
                ("Y1", 0): [0.9143],
                ("Y1", 1): [0.0634],
                ("Y1", 2): [0.0076],
                ("Y1", 3): [0.0],  # too rare
                ("Y1", 4): [0.0147],
                ("Y2", 0): [0.7146],
                ("Y2", 1): [0.2246],
                ("Y2", 2): [0.0270],
                ("Y2", 3): [0.0087],
                ("Y2", 4): [0.0],
                ("Y2", 5): [0.0],
                ("Y2", 6): [0.0252],
                ("Y3", 0): [1.0],
                ("Y3", 1): [0.0],
                ("Y3", 2): [0.0],
                ("Y3", 3): [0.0],
                ("Y3", 4): [0.0],
                ("Y3", 5): [0.0],
                ("Y3", 6): [0.0],
                ("Y3", 7): [0.0],
                ("Y3", 8): [0.0],
                ("Y3", 9): [0.0],
                ("Y3", 10): [0.0],
            },
        }
        self.batch_checks(
            self.individual_parameters,
            timepoints,
            self.ordinal_models,
            expected_estimations,
            ordinal_method="probabilities",
        )

    @skip("Broken: Ordinal models are broken")
    def test_estimate_ordinal_dataframe(self):
        lsp = self.get_hardcoded_model("logistic_ordinal_ranking")

        timepoints = pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "ID": ["idx1", "idx2", "idx1"],
                    "TIME": [71, 71, 81],
                }
            )
        )

        # no precise check of values (part of previous check): just a check of shape & index here
        df_ests = lsp.estimate(
            timepoints, self.individual_parameters, ordinal_method="MLE"
        )
        self.assertIsInstance(df_ests, pd.DataFrame)
        self.assertTrue(df_ests.index.equals(timepoints))
        self.assertEqual(df_ests.columns.tolist(), lsp.model.features)
        self.assertTrue(
            ((0 <= df_ests) & (df_ests <= lsp.model.noise_model.max_level)).all(
                axis=None
            )
        )

        df_ests = lsp.estimate(
            timepoints, self.individual_parameters, ordinal_method="E"
        )
        self.assertIsInstance(df_ests, pd.DataFrame)
        self.assertTrue(df_ests.index.equals(timepoints))
        self.assertEqual(df_ests.columns.tolist(), lsp.model.features)
        self.assertTrue(
            ((0 <= df_ests) & (df_ests <= lsp.model.noise_model.max_level)).all(
                axis=None
            )
        )

        df_ests = lsp.estimate(
            timepoints, self.individual_parameters, ordinal_method="P"
        )
        self.assertIsInstance(df_ests, pd.DataFrame)
        self.assertTrue(df_ests.index.equals(timepoints))
        self.assertEqual(df_ests.columns.nlevels, 2)  # 2D columns
        expected_cols = [
            (ft, lvl)
            for ft, ft_max_level in lsp.model.noise_model.max_levels.items()
            for lvl in range(0, ft_max_level + 1)
        ]
        self.assertEqual(df_ests.columns.tolist(), expected_cols)
        self.assertTrue(((0 <= df_ests) & (df_ests <= 1)).all(axis=None))

    def test_estimate_univariate(self):
        individual_parameters = self.get_hardcoded_individual_params(
            "ip_univariate_save.json"
        )
        timepoints = {"idx1": [78, 81], "idx2": [71]}
        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ("univariate_logistic",)
        expected_ests = {"idx1": [[0.999607], [0.9999857]], "idx2": [[0.03098414]]}
        self.batch_checks(individual_parameters, timepoints, models, expected_ests)

    def test_estimate_joint_univariate(self):
        individual_parameters = self.get_hardcoded_individual_params(
            "ip_univariate_save.json"
        )
        timepoints = {"idx1": [71, 74, 81], "idx2": [72, 77.5]}
        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ("univariate_joint",)
        expected_ests = {
            "idx1": [[0.8602, 1.0], [0.9995, 0.9337], [1.0, 0.0]],
            "idx2": [[0.0973, 1.0], [0.9999, 0.4957]],
        }
        self.batch_checks(individual_parameters, timepoints, models, expected_ests)
