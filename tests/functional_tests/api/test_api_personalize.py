import math
import warnings
from typing import Optional
from unittest import skipIf

import pandas as pd
from numpy import nan

from leaspy.io.data import Data, Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from tests import LeaspyTestCase

# Logistic parallel models are currently broken in Leaspy v2.
# Flip to True to test with them.
TEST_LOGISTIC_PARALLEL_MODELS = False
SKIP_LOGISTIC_PARALLEL_MODELS = "Logistic parallel models are currently broken."

TEST_LOGISTIC_MODELS_WITH_JACOBIAN = False
SKIP_LOGISTIC_MODELS_WITH_JACOBIAN = "Jacobian not implemented for logistic model."

TEST_LOGISTIC_BINARY_MODELS = False
SKIP_LOGISTIC_BINARY_MODELS = "Logistic binary models are currently broken."

# Linear models are currently broken in Leaspy v2.
# Flip to True to test with them.
TEST_LINEAR_MODELS = True
SKIP_LINEAR_MODELS = "Linear models are currently broken."

TEST_LINEAR_MODELS_WITH_JACOBIAN = False
SKIP_LINEAR_MODELS_WITH_JACOBIAN = "Jacobian not implemented for linear model."


class LeaspyPersonalizeTestMixin(LeaspyTestCase):
    """
    Mixin holding generic personalization methods that may be safely
    reused in other tests (no actual test here).
    """

    @classmethod
    def generic_personalization(
        cls,
        hardcoded_model_name: str,
        *,
        data_path: Optional[str] = None,
        data_kws: Optional[dict] = None,
        algo_path: Optional[str] = None,
        algo_name: Optional[str] = None,
        **algo_params,
    ):
        """Helper for a generic personalization in following tests."""
        data_kws = data_kws or {}
        model = cls.get_hardcoded_model(hardcoded_model_name)
        if data_path is None:
            # automatic (main test data)
            data = cls.get_suited_test_data_for_model(hardcoded_model_name)
        else:
            # relative path to data (csv expected)
            data_full_path = cls.get_test_data_path("data_mock", data_path)
            data = Data.from_csv_file(data_full_path, **data_kws)

        ips = model.personalize(
            data, algo_name, algorithm_settings_path=algo_path, **algo_params
        )

        return ips, model

    def check_consistency_of_personalization_outputs(
        self,
        ips: IndividualParameters,
        msg=None,
    ):
        self.assertIsInstance(ips, IndividualParameters)


class LeaspyPersonalizeTest(LeaspyPersonalizeTestMixin):
    def test_personalize_mean_posterior_logistic_old(self):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # There was a bug previously in mode & mean real: initial temperature = 10 was used even if
        # no real annealing is implemented for those perso algos. As a consequence regularity term
        # was not equally weighted during all the sampling of individual variables.
        # We test this old "buggy" behavior to check past consistency (but we raise a warning now)
        path_settings = self.get_test_data_path(
            "settings", "algo", "settings_mean_posterior_old_with_annealing.json"
        )
        with self.assertWarnsRegex(UserWarning, r"[Aa]nnealing"):
            ips, _ = self.generic_personalization(
                "logistic_scalar_noise", algo_path=path_settings
            )
        self.check_consistency_of_personalization_outputs(ips)

    def test_personalize_mode_posterior_logistic_old(self):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # cf. mean_real notice
        path_settings = self.get_test_data_path(
            "settings", "algo", "settings_mode_posterior_old_with_annealing.json"
        )
        with self.assertWarnsRegex(UserWarning, r"[Aa]nnealing"):
            ips, _ = self.generic_personalization(
                "logistic_scalar_noise", algo_path=path_settings
            )
        self.check_consistency_of_personalization_outputs(ips)

    def _personalize_generic(
        self,
        model_name: str,
        algo_name: str,
        algo_kws: Optional[dict] = None,
    ):
        algo_kws = algo_kws or {}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            # only look at loss to detect any regression in personalization
            ips, _ = self.generic_personalization(
                model_name, algo_name=algo_name, seed=0, **algo_kws
            )

        ws = [str(w.message) for w in ws]

        self.assertEqual(
            ws,
            [
                "Outputs will be saved in '_outputs' relative to the current working directory"
            ],
        )

        self.check_consistency_of_personalization_outputs(
            ips,
            msg={
                "model_name": model_name,
                "perso_name": algo_name,
                "perso_kws": algo_kws,
            },
        )

    def test_multivariate_logistic_scipy_minimize(self):
        self._personalize_generic(
            "logistic_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_mode_posterior(self):
        self._personalize_generic("logistic_scalar_noise", "mode_posterior")

    def test_multivariate_logistic_mean_posterior(self):
        self._personalize_generic("logistic_scalar_noise", "mean_posterior")

    def test_multivariate_logistic_diagonal_id_scipy_minimize(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_id_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_id_mode_posterior(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "mode_posterior",
        )

    def test_multivariate_logistic_diagonal_id_mean_posterior(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "mean_posterior",
        )

    def test_multivariate_logistic_diagonal_scipy_minimize(self):
        self._personalize_generic(
            "logistic_diag_noise", "scipy_minimize", {"use_jacobian": False}
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_mode_erior(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "mode_posterior",
        )

    def test_multivariate_logistic_diagonal_mean_posterior(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "mean_posterior",
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_no_source_scipy_minimize_with_jacobian(
        self,
    ):
        self._personalize_generic(
            "logistic_diag_noise_no_source",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_no_source_mode_posterior(self):
        self._personalize_generic(
            "logistic_diag_noise_no_source",
            "mode_posterior",
        )

    def test_multivariate_logistic_diagonal_no_source_mean_posterior(self):
        self._personalize_generic(
            "logistic_diag_noise_no_source",
            "mean_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_scipy_minimize(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS
        if TEST_LOGISTIC_MODELS_WITH_JACOBIAN
        else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN,
    )
    def test_multivariate_logistic_parallel_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mode_posterior(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "mode_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mean_posterior(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "mean_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS
        if TEST_LOGISTIC_MODELS_WITH_JACOBIAN
        else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN,
    )
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mode_posterior(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "mode_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mean_posterior(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "mean_posterior",
        )

    def test_univariate_logistic_scipy_minimize(self):
        self._personalize_generic(
            "logistic",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_univariate_logistic_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_univariate_logistic_mode_posterior(self):
        self._personalize_generic("logistic", "mode_posterior")

    def test_univariate_logistic_mean_posterior(self):
        self._personalize_generic("logistic", "mean_posterior")

    def test_univariate_joint_scipy_minimize(self):
        self._personalize_generic(
            "joint",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_univariate_joint_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "joint",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_univariate_joint_mode_posterior(self):
        self._personalize_generic(
            "joint",
            "mode_posterior",
        )

    def test_univariate_joint_mean_posterior(self):
        self._personalize_generic(
            "joint",
            "mean_posterior",
        )

    def test_joint_scipy_minimize(self):
        self._personalize_generic(
            "joint_diagonal",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_joint_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "joint_diagonal",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_joint_mode_erior(self):
        self._personalize_generic("joint_diagonal", "mode_posterior")

    def test_joint_mean_posterior(self):
        self._personalize_generic("joint_diagonal", "mean_posterior")

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_scipy_minimize(self):
        self._personalize_generic(
            "linear",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS_WITH_JACOBIAN, SKIP_LINEAR_MODELS_WITH_JACOBIAN)
    def test_univariate_linear_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "linear",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_mode_posterior(self):
        self._personalize_generic("linear", "mode_posterior")

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_mean_posterior(self):
        self._personalize_generic("linear", "mean_posterior")

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_scipy_minimize(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS_WITH_JACOBIAN, SKIP_LINEAR_MODELS_WITH_JACOBIAN)
    def test_multivariate_linear_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_mode_posterior(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "mode_posterior",
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_mean_posterior(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "mean_posterior",
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_scipy_minimize(self):
        self._personalize_generic(
            "linear_diag_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS_WITH_JACOBIAN, SKIP_LINEAR_MODELS_WITH_JACOBIAN)
    def test_multivariate_linear_diagonal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "linear_diag_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_mode_posterior(self):
        self._personalize_generic(
            "linear_diag_noise",
            "mode_posterior",
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_mean_posterior(self):
        self._personalize_generic(
            "linear_diag_noise",
            "mean_posterior",
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_multivariate_binary_scipy_minimize(self):
        self._personalize_generic(
            "logistic_binary",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_binary_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_binary",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_multivariate_binary_mode_posterior(self):
        self._personalize_generic(
            "logistic_binary",
            "mode_posterior",
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_multivariate_binary_mean_posterior(self):
        self._personalize_generic(
            "logistic_binary",
            "mean_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_parallel_binary_scipy_minimize(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS
        if TEST_LOGISTIC_MODELS_WITH_JACOBIAN
        else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN,
    )
    def test_multivariate_parallel_binary_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_parallel_binary_mode_posterior(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "mode_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_parallel_binary_mean_posterior(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "mean_posterior",
        )


class LeaspyPersonalizeRobustnessDataSparsityTest(LeaspyPersonalizeTestMixin):
    """
    In this test, we check that estimated individual parameters are almost the same
    no matter if data is sparse (i.e. multiple really close visits with many missing
    values) or data is 'merged' in a rounded visit.

    TODO? we could build a mock dataset to also check same property for calibration :)
    """

    def _robustness_to_data_sparsity(
        self,
        model_name: str,
        algo_name: str,
        algo_kws: Optional[dict] = None,
        rtol: float = 2e-2,
        atol: float = 5e-3,
    ) -> None:
        algo_kws = algo_kws or {}
        subtest = {
            "model_name": model_name,
            "perso_name": algo_name,
            "perso_kws": algo_kws,
        }
        common_params = dict(algo_name=algo_name, seed=0, **algo_kws)

        ips_sparse, _ = self.generic_personalization(
            model_name,
            **common_params,
            data_path="missing_data/sparse_data.csv",
            data_kws={"drop_full_nan": False},
        )
        ips_merged, _ = self.generic_personalization(
            model_name,
            **common_params,
            data_path="missing_data/merged_data.csv",
        )
        indices_sparse, ips_sparse_torch = ips_sparse.to_pytorch()
        indices_merged, ips_merged_torch = ips_merged.to_pytorch()

        # same individuals
        self.assertEqual(indices_sparse, indices_merged, msg=subtest)

        # same individual parameters (up to rounding errors)
        self.assertDictAlmostEqual(
            ips_sparse_torch,
            ips_merged_torch,
            left_desc="sparse",
            right_desc="merged",
            rtol=rtol,
            atol=atol,
            msg=subtest,
        )

    def test_multivariate_logistic_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_id_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_id",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_id_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_id",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_mode_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "mode_posterior",
            {"n_iter": 100},
        )

    def test_multivariate_logistic_diagonal_mean_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "mean_posterior",
            {"n_iter": 100},
        )

    def test_multivariate_logistic_diagonal_no_source_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_no_source_scipy_minimize_with_jacobian(
        self,
    ):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_no_source_mode_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "mode_posterior",
        )

    def test_multivariate_logistic_diagonal_no_source_mean_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "mean_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS
        if TEST_LOGISTIC_MODELS_WITH_JACOBIAN
        else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN,
    )
    def test_multivariate_logistic_parallel_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mode_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_scalar_noise", "mode_posterior", 0.1517
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mean_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_scalar_noise", "mean_posterior", 0.2079
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS
        if TEST_LOGISTIC_MODELS_WITH_JACOBIAN
        else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN,
    )
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mode_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "mode_posterior",
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mean_posterior(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "mean_posterior",
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_linear_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "linear_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS_WITH_JACOBIAN, SKIP_LINEAR_MODELS_WITH_JACOBIAN)
    def test_linear_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "linear_scalar_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_linear_diagonal_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "linear_diag_noise",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS_WITH_JACOBIAN, SKIP_LINEAR_MODELS_WITH_JACOBIAN)
    def test_linear_diagonal_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "linear_diag_noise",
            "scipy_minimize",
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_binary_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_binary",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_binary_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_binary",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_parallel_binary_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_binary",
            "scipy_minimize",
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_parallel_binary_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_binary",
            "scipy_minimize",
            {"use_jacobian": False},
        )


class LeaspyPersonalizeWithNansTest(LeaspyPersonalizeTestMixin):
    def test_personalize_full_nan(self, *, general_tol=1e-3):
        # test result of personalization with no data at all
        df = pd.DataFrame(
            {
                "ID": ["SUBJ1", "SUBJ1"],
                "TIME": [75.12, 78.9],
                "Y0": [nan] * 2,
                "Y1": [nan] * 2,
                "Y2": [nan] * 2,
                "Y3": [nan] * 2,
            }
        ).set_index(["ID", "TIME"])

        model = self.get_hardcoded_model("logistic_diag_noise")

        for perso_algo, perso_kws, coeff_tol_per_param_std in [
            ("scipy_minimize", dict(use_jacobian=False), general_tol),
            # ('scipy_minimize', dict(use_jacobian=True), general_tol),
            # the LL landscape is quite flat so tolerance is high here...
            # we may deviate from tau_mean / xi_mean / sources_mean when no data at all
            # (intrinsically represent the incertitude on those individual parameters)
            ("mode_posterior", {}, 0.4),
            ("mean_posterior", {}, 0.4),
        ]:
            subtest = dict(perso_algo=perso_algo, perso_kws=perso_kws)
            with self.subTest(**subtest):
                with self.assertRaisesRegex(
                    ValueError, "Dataframe should have at least "
                ):
                    # drop rows full of nans, nothing is left...
                    Data.from_dataframe(df)
                with self.assertWarnsRegex(
                    UserWarning,
                    r"These columns only contain nans: \['Y0', 'Y1', 'Y2', 'Y3'\]",
                ):
                    data_1 = Data.from_dataframe(df.head(1), drop_full_nan=False)
                    data_2 = Data.from_dataframe(df, drop_full_nan=False)
                dataset_1 = Dataset(data_1)
                dataset_2 = Dataset(data_2)

                self.assertEqual(data_1.n_individuals, 1)
                self.assertEqual(data_1.n_visits, 1)
                self.assertEqual(dataset_1.n_observations_per_ft.tolist(), [0, 0, 0, 0])
                self.assertEqual(dataset_1.n_observations, 0)

                self.assertEqual(data_2.n_individuals, 1)
                self.assertEqual(data_2.n_visits, 2)
                self.assertEqual(dataset_2.n_observations_per_ft.tolist(), [0, 0, 0, 0])
                self.assertEqual(dataset_2.n_observations, 0)

                ips_1 = model.personalize(
                    data_1, perso_algo, seed=0, progress_bar=False, **perso_kws
                )
                ips_2 = model.personalize(
                    data_2, perso_algo, seed=0, progress_bar=False, **perso_kws
                )

                indices_1, dict_1 = ips_1.to_pytorch()
                indices_2, dict_2 = ips_2.to_pytorch()

                self.assertEqual(indices_1, ["SUBJ1"])
                self.assertEqual(indices_1, indices_2)

                # replication is OK
                self.assertDictAlmostEqual(
                    dict_1, dict_2, atol=general_tol, msg=subtest
                )

                # we have no information so high incertitude when stochastic perso algo
                from leaspy.variables.specs import IndividualLatentVariable

                all_params = model.parameters | model.hyperparameters
                allclose_custom = {
                    p: dict(
                        atol=(
                            math.ceil(
                                coeff_tol_per_param_std
                                * all_params[f"{p}_std"].item()
                                / general_tol
                            )
                            * general_tol
                        )
                    )
                    for p in model.dag.sorted_variables_by_type[
                        IndividualLatentVariable
                    ]
                }
                self.assertDictAlmostEqual(
                    dict_1,
                    {
                        "tau": [model.parameters["tau_mean"]],
                        "xi": [[0.0]],
                        "sources": [model.source_dimension * [0.0]],
                    },
                    allclose_custom=allclose_custom,
                    msg=subtest,
                )

    def test_personalize_same_if_extra_totally_nan_visits(self):
        df = pd.DataFrame(
            {
                "ID": ["SUBJ1"] * 4,
                "TIME": [75.12, 78.9, 67.1, 76.1],
                "Y0": [nan, 0.6, nan, 0.2],
                "Y1": [nan, 0.4, nan, nan],
                "Y2": [nan, 0.5, nan, 0.2],
                "Y3": [nan, 0.3, nan, 0.2],
            }
        ).set_index(["ID", "TIME"])

        model = self.get_hardcoded_model("logistic_diag_noise")

        for perso_algo, perso_kws, tol in [
            ("scipy_minimize", dict(use_jacobian=False), 1e-3),
            # ('scipy_minimize', dict(use_jacobian=True), 1e-3),
            ("mode_posterior", {}, 1e-3),
            ("mean_posterior", {}, 1e-3),
        ]:
            subtest = dict(perso_algo=perso_algo, perso_kws=perso_kws)
            with self.subTest(**subtest):
                data_without_empty_visits = Data.from_dataframe(df)
                data_with_empty_visits = Data.from_dataframe(df, drop_full_nan=False)

                dataset_without_empty_visits = Dataset(data_without_empty_visits)
                dataset_with_empty_visits = Dataset(data_with_empty_visits)

                self.assertEqual(data_without_empty_visits.n_individuals, 1)
                self.assertEqual(data_without_empty_visits.n_visits, 2)
                self.assertEqual(
                    dataset_without_empty_visits.n_observations_per_ft.tolist(),
                    [2, 1, 2, 2],
                )
                self.assertEqual(dataset_without_empty_visits.n_observations, 7)

                self.assertEqual(data_with_empty_visits.n_individuals, 1)
                self.assertEqual(data_with_empty_visits.n_visits, 4)
                self.assertEqual(
                    dataset_with_empty_visits.n_observations_per_ft.tolist(),
                    [2, 1, 2, 2],
                )
                self.assertEqual(dataset_with_empty_visits.n_observations, 7)

                ips_without_empty_visits = model.personalize(
                    data_without_empty_visits,
                    perso_algo,
                    seed=0,
                    progress_bar=False,
                    **perso_kws,
                )
                ips_with_empty_visits = model.personalize(
                    data_with_empty_visits,
                    perso_algo,
                    seed=0,
                    progress_bar=False,
                    **perso_kws,
                )

                indices_1, dict_1 = ips_without_empty_visits.to_pytorch()
                indices_2, dict_2 = ips_with_empty_visits.to_pytorch()

                self.assertEqual(indices_1, ["SUBJ1"])
                self.assertEqual(indices_1, indices_2)

                self.assertDictAlmostEqual(dict_1, dict_2, atol=tol, msg=subtest)
