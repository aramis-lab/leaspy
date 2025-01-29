from typing import Optional
from unittest import skip

import numpy as np
import pandas as pd
import torch

from leaspy.algo import AlgorithmSettings
from leaspy.algo.personalize import ScipyMinimize
from leaspy.io.data import Data, Dataset
from leaspy.models import BaseModel
from leaspy.models.obs_models import (
    BernoulliObservationModel,
    FullGaussianObservationModel,
    OrdinalObservationModel,
    observation_model_factory,
)
from tests import LeaspyTestCase

# test tolerance, lack of precision btw different machines...
# (no exact reproducibility in scipy.optimize.minimize?)
default_tolerance: float = 3e-3
default_tolerance_tau: float = 1e-2


class ScipyMinimizeTest(LeaspyTestCase):
    @property
    def default_settings(self) -> AlgorithmSettings:
        return AlgorithmSettings("scipy_minimize")

    @property
    def default_algorithm(self) -> ScipyMinimize:
        return ScipyMinimize(self.default_settings)

    def check_individual_parameters(
        self,
        ips,
        *,
        tau,
        xi,
        tolerance_tau: float,
        tolerance_xi: float,
        sources=None,
        tolerance_sources: Optional[float] = None,
    ) -> None:
        self.assertAlmostEqual(ips["tau"].item(), tau, delta=tolerance_tau)
        self.assertAlmostEqual(ips["xi"].item(), xi, delta=tolerance_xi)
        if sources is not None:
            res_sources = ips["sources"].squeeze().tolist()
            self.assertEqual(len(res_sources), len(sources))
            for s, s_expected in zip(res_sources, sources):
                self.assertAlmostEqual(s, s_expected, delta=tolerance_sources)

    def test_default_settings(self):
        self.assertEqual(
            self.default_settings.parameters,
            {
                "use_jacobian": True,
                "n_jobs": 1,
                "progress_bar": True,
                "custom_scipy_minimize_params": None,
                "custom_format_convergence_issues": None,
            },
        )

    def test_default_constructor(self):
        self.assertEqual(self.default_algorithm.name, "scipy_minimize")
        self.assertEqual(self.default_algorithm.seed, None)
        self.assertEqual(
            self.default_algorithm.scipy_minimize_params,
            ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN,
        )
        self.assertEqual(
            self.default_algorithm.format_convergence_issues,
            ScipyMinimize.DEFAULT_FORMAT_CONVERGENCE_ISSUES,
        )
        self.assertEqual(self.default_algorithm.logger.__name__, "_default_logger")

    def test_default_constructor_no_jacobian(self):
        settings = AlgorithmSettings("scipy_minimize", use_jacobian=False)

        self.assertEqual(
            settings.parameters,
            {
                "use_jacobian": False,
                "n_jobs": 1,
                "progress_bar": True,
                "custom_scipy_minimize_params": None,
                "custom_format_convergence_issues": None,
            },
        )

        algo = ScipyMinimize(settings)

        self.assertEqual(algo.name, "scipy_minimize")
        self.assertEqual(algo.seed, None)
        self.assertEqual(
            algo.scipy_minimize_params,
            ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN,
        )
        self.assertEqual(
            algo.format_convergence_issues,
            ScipyMinimize.DEFAULT_FORMAT_CONVERGENCE_ISSUES,
        )
        self.assertEqual(algo.logger, algo._default_logger)

    def test_custom_constructor(self):
        def custom_logger(msg: str):
            pass

        custom_format_convergence_issues = (
            "{patient_id}: {optimization_result_pformat}..."
        )
        custom_scipy_minimize_params = {
            "method": "BFGS",
            "options": {
                "gtol": 5e-2,
                "maxiter": 100,
            },
        }
        settings = AlgorithmSettings(
            "scipy_minimize",
            seed=24,
            custom_format_convergence_issues=custom_format_convergence_issues,
            custom_scipy_minimize_params=custom_scipy_minimize_params,
        )
        settings.logger = custom_logger
        algo = ScipyMinimize(settings)
        self.assertEqual(algo.name, "scipy_minimize")
        self.assertEqual(algo.seed, 24)
        self.assertEqual(
            algo.format_convergence_issues,
            custom_format_convergence_issues,
        )
        self.assertEqual(
            algo.scipy_minimize_params,
            custom_scipy_minimize_params,
        )
        self.assertIs(algo.logger, custom_logger)

    @skip("Broken: ScipyMinimize has no _initialize_parameters method")
    def test_initialize_parameters_univariate(self):
        model = self.get_hardcoded_model("univariate_logistic")
        parameters = self.default_algorithm._initialize_parameters(model.model)

        self.assertEqual(
            parameters,
            [
                torch.tensor([0.0 / 0.01]),
                torch.tensor([70.0 / 2.5]),
            ],
        )

    @skip("Broken: ScipyMinimize has no _initialize_parameters method")
    def test_initialize_parameters_multivariate(self):
        model = self.get_hardcoded_model("logistic_scalar_noise")
        parameters = self.default_algorithm._initialize_parameters(model.model)

        self.assertEqual(
            parameters,
            [
                torch.tensor([0.0]),
                torch.tensor([75.2 / 7.1]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
            ],
        )

    def test_fallback_without_jacobian(self):
        model = self.get_hardcoded_model("logistic_scalar_noise").model

        # pretend as if compute_jacobian_tensorized was not implemented
        def not_implemented_compute_jacobian_tensorized(tpts, ips, **kws):
            raise NotImplementedError

        model.compute_jacobian_tensorized = not_implemented_compute_jacobian_tensorized
        mini_dataset = Dataset(
            self.get_suited_test_data_for_model("logistic_scalar_noise"),
            no_warning=True,
        )
        with self.assertWarnsRegex(UserWarning, r"`use_jacobian\s?=\s?False`"):
            self.default_algorithm._get_individual_parameters(model, mini_dataset)

    @skip("Broken : ScipyMinimize has not _pull_individual_parameters method")
    def test_get_reconstruction_error(self):
        """This method is not part of scipy minimize anymore but is a Gaussian noise-model class method."""
        leaspy = self.get_hardcoded_model("logistic_scalar_noise")
        times = torch.tensor([70, 80])
        values = torch.tensor(
            [
                [0.5, 0.4, 0.4, 0.45],
                [0.3, 0.3, 0.2, 0.4],
            ]
        )
        individual_parameters = self.default_algorithm._pull_individual_parameters(
            [0.0, 75.2 / 7.1, 0.0, 0.0],
            leaspy.model,
        )
        dataset = self._get_individual_dataset_from_times_values(
            leaspy.model, np.array(times), np.array(values)
        )
        predictions = leaspy.model.compute_individual_tensorized(
            dataset.timepoints, individual_parameters
        )
        residuals = leaspy.model.noise_model.compute_residuals(dataset, predictions)

        expected_residuals = torch.tensor(
            [
                [-0.4705, -0.3278, -0.3103, -0.4477],
                [0.6059, 0.0709, 0.3537, 0.4523],
            ]
        )
        self.assertIsInstance(residuals, torch.Tensor)
        self.assertAlmostEqual(
            torch.sum((residuals - expected_residuals) ** 2).item(),
            0,
            delta=1e-8,
        )

    @skip("Broken : ScipyMinimize does not implement _pull_individual_parameters")
    def test_get_regularity(self):
        leaspy = self.get_hardcoded_model("logistic_scalar_noise")

        for obs_model in (
            None,
            "bernoulli",
        ):  # 'ordinal'
            if obs_model is not None:
                leaspy.model.obs_models = (observation_model_factory(obs_model),)

            # regularity constant is not added anymore (useless)
            z0 = [
                0.0,
                75.2 / 7.1,
                0.0,
                0.0,
            ]  # for all individual parameters we set `mean/std`

            # Broken : ScipyMinimize does not implement _pull_individual_parameters
            individual_parameters = self.default_algorithm._pull_individual_parameters(
                z0, leaspy.model
            )
            expected_regularity = torch.tensor([0.0])
            expected_regularity_grads = {
                "tau": torch.tensor([[0.0]]),
                "xi": torch.tensor([[0.0]]),
                "sources": torch.tensor([[0.0, 0.0]]),
            }
            regularity, regularity_grads = self.default_algorithm._get_regularity(
                leaspy.model, individual_parameters
            )
            self.assertTrue(torch.is_tensor(regularity))
            self.assertEqual(regularity.shape, expected_regularity.shape)
            self.assertAllClose(regularity, expected_regularity)

            # gradients
            self.assertIsInstance(regularity_grads, dict)
            self.assertEqual(
                regularity_grads.keys(),
                expected_regularity_grads.keys(),
            )
            # types & dimensions
            for ip, expected_regularity_grad in expected_regularity_grads.items():
                self.assertTrue(torch.is_tensor(regularity_grads[ip]))
                self.assertEqual(
                    regularity_grads[ip].shape, expected_regularity_grad.shape
                )
            # nice check for all values
            self.assertDictAlmostEqual(regularity_grads, expected_regularity_grads)

            # second test with a non-zero regularity term
            s = [
                0.33,
                -0.59,
                0.72,
                -0.14,
            ]  # random shifts to test (in normalized space)
            z = [
                si + z0i for si, z0i in zip(s, z0)
            ]  # we have to add the z0 by design of `_pull_individual_parameters`
            individual_parameters = self.default_algorithm._pull_individual_parameters(
                z, leaspy.model
            )
            expected_reg = (
                0.5 * (torch.tensor(s) ** 2).sum()
            )  # gaussian regularity (without constant)
            reg, _ = self.default_algorithm._get_regularity(
                leaspy.model, individual_parameters
            )
            self.assertAllClose(reg, [expected_reg])

    def _get_individual_dataset_from_times_values(
        self,
        model: BaseModel,
        times: np.ndarray,
        values: np.ndarray,
    ) -> Dataset:
        df = pd.DataFrame(
            {
                "ID": ["ID1"] * len(times),
                "TIME": times,
                **{feature: values[:, i] for i, feature in enumerate(model.features)},
            }
        )
        return Dataset(Data.from_dataframe(df), no_warning=True)

    @skip("ScipyMinimize does not implement obj")
    def test_obj(self):
        leaspy = self.get_hardcoded_model("logistic_scalar_noise")

        z0 = [
            0.0,
            75.2 / 7.1,
            0.0,
            0.0,
        ]  # for all individual parameters we set `mean/std`
        times = np.array([70, 80])

        # previously we did not add the "constant" in NLL for Gaussian noise
        from leaspy.models.noise_models.gaussian import TWO_PI

        normal_cst = (
            0.5 * torch.log(TWO_PI * leaspy.model.state["noise_std"] ** 2).item() * 8
        )
        for obs_model, (values, expected_obj, expected_obj_grads) in {
            None: (
                np.array(
                    [
                        [0.5, 0.4, 0.4, 0.45],
                        [0.3, 0.3, 0.2, 0.4],
                    ]
                ),
                16.590890884399414 + normal_cst,
                [2.8990, -14.2579, -0.8748, 0.2912],
            ),
            "bernoulli": (
                np.array(
                    [
                        [0, 1, 0, 1],
                        [0, 1, 1, 1],
                    ]
                ),
                12.92530632019043,
                [1.1245, 5.4126, -2.6562, -10.9062],
            ),
            #'ordinal': (0., []),
        }.items():
            if obs_model is not None:
                leaspy.model.obs_models = (observation_model_factory(obs_model),)

            dataset = self._get_individual_dataset_from_times_values(
                leaspy.model, times, values
            )

            # Broken
            # ScipyMinimize implements obj_no_jac and obj_with_jac (not yet implemented)
            obj, obj_grads = self.default_algorithm.obj(
                z0, leaspy.model, dataset, with_gradient=True
            )

            self.assertIsInstance(obj, float)
            self.assertAlmostEqual(obj, expected_obj, delta=1e-4)
            self.assertIsInstance(obj_grads, torch.Tensor)
            self.assertEqual(obj_grads.shape, (2 + leaspy.model.source_dimension,))
            self.assertAllClose(obj_grads, expected_obj_grads, atol=1e-4)

    def get_individual_parameters_patient(
        self,
        model_name: str,
        times: np.ndarray,
        values: np.ndarray,
        *,
        obs_model: Optional[str] = None,
        **algo_kwargs,
    ):
        leaspy = self.get_hardcoded_model(model_name)
        if obs_model is not None:
            leaspy.model.obs_model = (observation_model_factory(obs_model),)

        settings = AlgorithmSettings("scipy_minimize", seed=0, **algo_kwargs)
        algo = ScipyMinimize(settings)

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        dataset = self._get_individual_dataset_from_times_values(
            leaspy.model, times, values
        )
        pyt_ips = algo._get_individual_parameters(
            leaspy.model,
            dataset,
            # with_jac=algo_kwargs["use_jacobian"],
        )
        nll_regularity = algo._get_regularity(leaspy.model, pyt_ips)[0]
        predictions = leaspy.model.compute_individual_tensorized(
            dataset.timepoints, pyt_ips
        )

        loss = None
        # residuals_getter = getattr(leaspy.model.noise_model, 'compute_residuals', None)
        residuals = None
        # if residuals_getter:
        #    res = residuals_getter(dataset, preds)

        return (
            leaspy.model.obs_models[0],
            pyt_ips,
            (dataset, predictions),
            (loss, nll_regularity),
            residuals,
        )

    @skip(
        "Broken: MultivariateModel has no compute_regularity_individual_parameters method"
    )
    def test_get_individual_parameters_patient_univariate_models(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = np.array(
            [[0.5], [0.4]]
        )  # no test with nans (done in multivariate models)

        for (model_name, use_jacobian), expected_dict in {
            ("univariate_logistic", False): {
                "tau": 69.2868,
                "xi": -0.0002,
                "err": [[-0.1765], [0.5498]],
            },
            ("univariate_logistic", True): {
                "tau": 69.2868,
                "xi": -0.0002,
                "err": [[-0.1765], [0.5498]],
            },
            ("univariate_linear", False): {
                "tau": 78.1131,
                "xi": -0.2035,
                "err": [[-0.1212], [0.1282]],
            },
            ("univariate_linear", True): {
                "tau": 78.0821,
                "xi": -0.2035,
                "err": [[-0.1210], [0.1287]],
            },
        }.items():
            (
                obs_model,
                individual_parameters,
                _,
                (rmse, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
            )
            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
            )
            # gaussian noise for those models
            self.assertIsInstance(obs_model, FullGaussianObservationModel)
            self.assertIsInstance(res, torch.Tensor)
            expected_res = torch.tensor(expected_dict["err"])
            self.assertAlmostEqual(
                torch.sum((res - expected_res) ** 2).item(), 0, delta=tolerance**2
            )
            # scalar noise
            expected_rmse = (expected_res**2).mean() ** 0.5
            self.assertAlmostEqual(rmse.item(), expected_rmse.item(), delta=1e-4)

    @skip(
        "Broken: MultivariateModel has no compute_regularity_individual_parameters method"
    )
    def test_get_individual_parameters_patient_multivariate_models(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = np.array(
            [
                [0.5, 0.4, 0.4, 0.45],
                [0.3, 0.3, 0.2, 0.4],
            ]
        )
        for (model_name, use_jacobian), expected_dict in {
            ("logistic_scalar_noise", False): {
                "tau": 78.5750,
                "xi": -0.0919,
                "sources": [0.3517, 0.1662],
                "err": [
                    [-0.49765, -0.31615, -0.351310, -0.44945],
                    [0.00825, 0.06638, 0.139204, 0.00413],
                ],
            },
            ("logistic_scalar_noise", True): {
                "tau": 78.5750,
                "xi": -0.0918,
                "sources": [0.3483, 0.1678],
                "err": [
                    [-0.4976, -0.3158, -0.3510, -0.4494],
                    [0.0079, 0.0673, 0.1403, 0.0041],
                ],
            },
            # TODO? linear, logistic_parallel
        }.items():
            (
                obs_model,
                individual_parameters,
                _,
                (rmse, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
            )
            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
                sources=expected_dict["sources"],
                tolerance_sources=tolerance,
            )

            # gaussian noise for those models
            self.assertIsInstance(obs_model, FullGaussianObservationModel)
            self.assertIsInstance(res, torch.Tensor)
            expected_res = torch.tensor(expected_dict["err"])
            self.assertAlmostEqual(
                torch.sum((res - expected_res) ** 2).item(), 0, delta=tolerance**2
            )
            # scalar noise
            expected_rmse = (expected_res**2).mean() ** 0.5
            self.assertAlmostEqual(rmse.item(), expected_rmse.item(), delta=1e-4)

    @skip(
        "Broken: MultivariateModel has no compute_regularity_individual_parameters method"
    )
    def test_get_individual_parameters_patient_multivariate_models_with_nans(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = np.array(
            [
                [0.5, 0.4, 0.4, np.nan],
                [0.3, np.nan, np.nan, 0.4],
            ]
        )
        nan_positions = torch.tensor(
            [[False, False, False, True], [False, True, True, False]]
        )

        for (model_name, use_jacobian), expected_dict in {
            ("logistic_scalar_noise", False): {
                "tau": 77.5558,
                "xi": -0.0989,
                "sources": [-0.9805, 0.7745],
                "err": [[-0.4981, -0.0895, -0.1161, 0.0], [-0.0398, 0.0, 0.0, 0.0863]],
            },
            ("logistic_scalar_noise", True): {
                "tau": 77.5555,
                "xi": -0.0990,
                "sources": [-0.9799, 0.7743],
                "err": [[-0.4981, -0.0896, -0.1162, 0.0], [-0.0397, 0.0, 0.0, 0.0863]],
            },
            # TODO? linear, logistic_parallel
        }.items():
            (
                obs_model,
                individual_parameters,
                _,
                (rmse, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
            )
            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
                sources=expected_dict["sources"],
                tolerance_sources=tolerance,
            )

            # gaussian noise for those models
            self.assertIsInstance(obs_model, FullGaussianObservationModel)
            self.assertIsInstance(res, torch.Tensor)
            self.assertTrue(torch.equal(res.squeeze(0) == 0, nan_positions))
            expected_res = torch.tensor(expected_dict["err"])
            self.assertAlmostEqual(
                torch.sum((res - expected_res) ** 2).item(), 0, delta=tolerance**2
            )
            # scalar noise with nans
            expected_rmse = ((expected_res**2).sum() / (~nan_positions).sum()) ** 0.5
            self.assertAlmostEqual(rmse.item(), expected_rmse.item(), delta=1e-4)

    @skip(
        "Broken: MultivariateModel has no compute_regularity_individual_parameters method"
    )
    def test_get_individual_parameters_patient_multivariate_models_crossentropy(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = np.array(
            [
                [0, 1, 0, 1],
                [0, 1, 1, 1],
            ]
        )
        for (model_name, use_jacobian), expected_dict in {
            ("logistic_scalar_noise", False): {
                "tau": 70.6041,
                "xi": -0.0458,
                "sources": [0.9961, 1.2044],
                "err": [
                    [1.2993e-03, -6.2189e-02, 4.8657e-01, -4.1219e-01],
                    [2.4171e-01, -9.4945e-03, -8.5862e-02, -4.0013e-04],
                ],
                "nll": (1.6398087739944458, 1.4571648836135864)
                if os.uname()[4][:3] == "arm"
                else (1.6396453380584717, 1.457330584526062),
            },
            ("logistic_scalar_noise", True): {
                "tau": 70.5971,
                "xi": -0.0471,
                "sources": [0.9984, 1.2037],
                "err": [
                    [1.3049e-03, -6.2177e-02, 4.8639e-01, -4.1050e-01],
                    [2.4118e-01, -9.5164e-03, -8.6166e-02, -4.0120e-04],
                ],
                "nll": (1.6363288164138794, 1.460662841796875),
            },
        }.items():
            (
                obs_model,
                individual_parameters,
                dataset_and_preds,
                (nll_attach, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
                obs_model="bernoulli",
            )
            # Bernoulli noise for those models
            self.assertIsInstance(obs_model, BernoulliObservationModel)
            self.assertIsNone(res)

            # check overall nll (no need for dataset...)
            self.assertAlmostEqual(
                nll_attach.item(), expected_dict["nll"][0], delta=1e-3
            )
            self.assertAlmostEqual(
                nll_regul.item(), expected_dict["nll"][1], delta=1e-3
            )

            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
                sources=expected_dict["sources"],
                tolerance_sources=tolerance,
            )
            # we compute residuals anyway even if not really relevant (only for the test)
            res = FullGaussianObservationModel().compute_residuals(*dataset_and_preds)
            self.assertAlmostEqual(
                torch.sum((res - torch.tensor(expected_dict["err"])) ** 2).item(),
                0,
                delta=tolerance**2,
            )

    @skip(
        "Broken: MultivariateModel has no compute_regularity_individual_parameters method"
    )
    def test_get_individual_parameters_patient_multivariate_models_with_nans_crossentropy(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = np.array(
            [
                [0, 1, 0, np.nan],
                [0, np.nan, np.nan, 1],
            ]
        )
        nan_positions = torch.tensor(
            [[False, False, False, True], [False, True, True, False]]
        )
        for (model_name, use_jacobian), expected_dict in {
            ("logistic_scalar_noise", False): {
                "tau": 75.7494,
                "xi": -0.0043,
                "sources": [0.4151, 1.0180],
                "err": [
                    [2.7332e-04, -0.30629, 0.22526, 0.0],
                    [0.077974, 0.0, 0.0, -0.036894],
                ],
                "nll": (0.740096390247345, 0.6074170470237732)
                if os.uname()[4][:3] == "arm"
                else (0.7398623824119568, 0.6076518297195435),
            },
            ("logistic_scalar_noise", True): {
                "tau": 75.7363,
                "xi": -0.0038,
                "sources": [0.4146, 1.0160],
                "err": [
                    [2.7735e-04, -0.30730, 0.22526, 0.0],
                    [0.079207, 0.0, 0.0, -0.036614],
                ],
                "nll": (0.7424823045730591, 0.6050525903701782),
            },
            # TODO? linear, logistic_parallel
        }.items():
            (
                obs_model,
                individual_parameters,
                dataset_and_preds,
                (nll_attach, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
                obs_model="bernoulli",
            )
            # Bernoulli noise for those models
            self.assertIsInstance(obs_model, BernoulliObservationModel)
            self.assertIsNone(res)

            # check overall nll (no need for dataset...)
            self.assertAlmostEqual(
                nll_attach.item(), expected_dict["nll"][0], delta=1e-3
            )
            self.assertAlmostEqual(
                nll_regul.item(), expected_dict["nll"][1], delta=1e-3
            )

            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
                sources=expected_dict["sources"],
                tolerance_sources=tolerance,
            )
            # we compute residuals anyway even if not really relevant (only for the test)
            res = FullGaussianObservationModel().compute_residuals(*dataset_and_preds)
            self.assertTrue(torch.equal(res.squeeze(0) == 0, nan_positions))
            self.assertAlmostEqual(
                torch.sum((res - torch.tensor(expected_dict["err"])) ** 2).item(),
                0,
                delta=tolerance**2,
            )

    @skip("Broken: Ordinal models are currently broken")
    def test_get_individual_parameters_patient_multivariate_models_ordinal(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = np.array(
            [
                [0, 1, 0, 2],
                [1, 2, 2, 4],
            ]
        )
        for (model_name, use_jacobian), expected_dict in {
            ("logistic_ordinal", False): {
                "tau": 74.0180,
                "xi": -1.7808,
                "sources": [-0.3543, 0.5963],
                "nll": (7.399204730987549, 1.113590955734253),
            },
            ("logistic_ordinal", True): {
                "tau": 73.9865,
                "xi": -1.7801,
                "sources": [-0.3506, 0.5917],
                "nll": (7.402340412139893, 1.1103485822677612),
            },
        }.items():
            (
                obs_model,
                individual_parameters,
                _,
                (nll_attach, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
            )
            # Ordinal noise for those models
            self.assertIsInstance(obs_model, OrdinalObservationModel)
            self.assertIsNone(res)

            # check overall nll (no need for dataset...)
            self.assertAlmostEqual(
                nll_attach.item(), expected_dict["nll"][0], delta=5e-4
            )
            self.assertAlmostEqual(
                nll_regul.item(), expected_dict["nll"][1], delta=5e-4
            )

            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
                sources=expected_dict["sources"],
                tolerance_sources=tolerance,
            )

    @skip("Broken: Ordinal models are currently broken")
    def test_get_individual_parameters_patient_multivariate_models_with_nans_ordinal(
        self,
        tolerance: float = default_tolerance,
        tolerance_tau: float = default_tolerance_tau,
    ):
        times = np.array([70, 80])
        values = torch.tensor(
            [
                [0, 1, 2, np.nan],
                [1, np.nan, 2, 4],
            ]
        )
        for (model_name, use_jacobian), expected_dict in {
            ("logistic_ordinal", False): {
                "tau": 74.2849,
                "xi": -1.7475,
                "sources": [0.0763, 0.8606],
                "nll": (5.821434497833252, 1.2080579996109009)
                if os.uname()[4][:3] == "arm"
                else (5.82178258895874, 1.2077088356018066),
            },
            ("logistic_ordinal", True): {
                "tau": 74.2808,
                "xi": -1.7487,
                "sources": [0.0754, 0.8551],
                "nll": (5.825297832489014, 1.2041646242141724),
            },
        }.items():
            (
                obs_model,
                individual_parameters,
                _,
                (nll_attach, nll_regul),
                res,
            ) = self.get_individual_parameters_patient(
                model_name,
                times,
                values,
                use_jacobian=use_jacobian,
            )
            # Ordinal noise for those models
            self.assertIsInstance(obs_model, OrdinalObservationModel)
            self.assertIsNone(res)

            # check overall nll (no need for dataset...)
            self.assertAlmostEqual(
                nll_attach.item(), expected_dict["nll"][0], delta=1e-4
            )
            self.assertAlmostEqual(
                nll_regul.item(), expected_dict["nll"][1], delta=1e-4
            )

            self.check_individual_parameters(
                individual_parameters,
                tau=expected_dict["tau"],
                tolerance_tau=tolerance_tau,
                xi=expected_dict["xi"],
                tolerance_xi=tolerance,
                sources=expected_dict["sources"],
                tolerance_sources=tolerance,
            )
