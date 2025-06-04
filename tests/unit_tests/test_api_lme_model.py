import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMParams

from leaspy.algo import AlgorithmSettings
from leaspy.algo.fit import LMEFitAlgorithm
from leaspy.io.data import Data
from leaspy.models import BaseModel, model_factory
from tests import LeaspyTestCase


class LMEModelAPITest(LeaspyTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling
        super().setUpClass()
        cls.raw_data_df = pd.read_csv(cls.example_data_path, dtype={"ID": str})
        cls.raw_data_df["TIME"] = round(cls.raw_data_df["TIME"], 3)
        cls.raw_data_df.iloc[30, 2] = np.nan

        ages = cls.raw_data_df.dropna(subset=["Y0"])["TIME"]
        cls.ages_mean, cls.ages_std = ages.mean(), ages.std(ddof=0)
        cls.raw_data_df["TIME_norm"] = (
            cls.raw_data_df["TIME"] - cls.ages_mean
        ) / cls.ages_std

        # Data must have only one feature:
        data_df = cls.raw_data_df[["ID", "TIME", "Y0"]]
        cls.data = Data.from_dataframe(data_df)

        data_df_others_ix = data_df.copy()
        data_df_others_ix["ID"] += "_new"  # but same data to test easily...

        cls.data_new_ix = Data.from_dataframe(data_df_others_ix)

        cls.default_lme_fit_params = {
            "with_random_slope_age": None,  # to be completely removed at a point
            "force_independent_random_effects": False,
            "method": ["lbfgs", "bfgs", "powell"],
        }

    def test_bivariate_data(self):
        bivariate_data = Data.from_dataframe(
            pd.DataFrame(
                {
                    "ID": [1, 1, 1, 2, 2, 2],
                    "TIME": [50, 51, 52, 60, 62, 64],
                    "FT_1": [0.4] * 6,
                    "FT_2": [0.6] * 6,
                }
            )
        )

        model = model_factory("lme")
        with self.assertRaises(ValueError):
            model.fit(bivariate_data, "lme_fit")

    def test_run(self):
        model = model_factory("lme")
        self.assertIsNone(model.features)
        self.assertEqual(model.with_random_slope_age, True)  # new default
        model.load_hyperparameters(dict(with_random_slope_age=False))
        self.assertEqual(model.with_random_slope_age, False)

        settings = AlgorithmSettings("lme_fit")
        self.assertDictEqual(settings.parameters, self.default_lme_fit_params)

        model.fit(self.data, "lme_fit")

        self.assertListEqual(model.features, ["Y0"])
        self.assertEqual(model.with_random_slope_age, False)
        self.assertEqual(model.dimension, 1)

        self.assertAlmostEqual(self.ages_mean, model.parameters["ages_mean"], places=3)
        self.assertAlmostEqual(self.ages_std, model.parameters["ages_std"], places=3)

        # fit that should not work (not multivariate!)
        with self.assertRaises(ValueError):
            model.fit(Data.from_dataframe(self.raw_data_df), "lme_fit")

        ip = model.personalize(self.data_new_ix, "lme_personalize")

        # check statsmodels consistency
        self.check_consistency_sm(model.parameters, ip, re_formula="~1")

        # Personalize that shouldn't work (different feature)
        with self.assertRaises(ValueError):
            model.personalize(
                Data.from_dataframe(self.raw_data_df[["ID", "TIME", "Y1"]]),
                "lme_personalize",
            )

        # # Estimate
        timepoints = {"709_new": [80]}
        results = model.estimate(timepoints, ip)
        self.assertEqual(results.keys(), timepoints.keys())
        self.assertEqual(results["709_new"].shape, (1, 1))
        self.assertAlmostEqual(results["709_new"][0, 0], 0.57, places=2)

    def test_fake_data(self):
        # try to see when fitting on df and personalizing on unseen_df
        df = pd.DataFrame.from_records(
            (
                np.arange(3, 3 + 10, 1),
                np.arange(15, 15 + 10, 1),
                np.arange(6, 6 + 10, 1),
            ),
            index=["pat1", "pat2", "pat3"],
        ).T.stack()
        df = pd.DataFrame(df)
        df.index.names = ["TIME", "ID"]
        df = df.rename(columns={0: "feat1"})
        df = df.swaplevel()
        df.loc[("pat1", 0)] = np.nan

        unseen_df = pd.DataFrame.from_records(
            (np.arange(2, 2 + 10, 1), np.arange(18, 18 + 10, 1)),
            index=["pat4", "pat5"],
        ).T.stack()
        unseen_df = pd.DataFrame(unseen_df)
        unseen_df.index.names = ["TIME", "ID"]
        unseen_df = unseen_df.rename(columns={0: "feat1"})
        unseen_df = unseen_df.swaplevel()

        easy_data = Data.from_dataframe(df)
        model = model_factory("lme", with_random_slope_age=False)
        model.fit(easy_data, "lme_fit")

        unseen_easy_data = Data.from_dataframe(unseen_df)
        ip = model.personalize(unseen_easy_data, "lme_personalize")

        # # Estimate
        easy_timepoints = {"pat4": [15, 16]}
        easy_results = model.estimate(easy_timepoints, ip)
        self.assertEqual(easy_results.keys(), easy_timepoints.keys())
        self.assertEqual(easy_results["pat4"].shape, (2, 1))
        self.assertAlmostEqual(easy_results["pat4"][0, 0], 17, delta=10e-1)
        self.assertAlmostEqual(easy_results["pat4"][1, 0], 18, delta=10e-1)

    def check_consistency_sm(self, model_params, ip, re_formula, **fit_kws):
        # compare
        lmm_test = smf.mixedlm(
            formula="Y0 ~ 1 + TIME_norm",
            data=self.raw_data_df.dropna(subset=["Y0"]),
            re_formula=re_formula,
            groups="ID",
        ).fit(method="lbfgs", **fit_kws)

        # pop effects
        self.assertAllClose(
            lmm_test.fe_params, model_params["fe_params"], what="fe_params"
        )
        self.assertAllClose(lmm_test.cov_re, model_params["cov_re"], what="cov_re")
        self.assertAllClose(
            lmm_test.scale**0.5, model_params["noise_std"], what="scale"
        )

        # ind effects
        sm_ranef = lmm_test.random_effects
        for pat_id, ind_ip in ip.items():
            exp_ranef = sm_ranef[pat_id[: -len("_new")]]
            self.assertAlmostEqual(ind_ip["random_intercept"], exp_ranef[0], places=5)
            if "TIME" in re_formula:
                self.assertAlmostEqual(
                    ind_ip["random_slope_age"], exp_ranef[1], places=5
                )

        return lmm_test

    def test_with_random_slope_age(self):
        model = model_factory("lme")
        self.assertTrue(model.with_random_slope_age)
        settings = AlgorithmSettings("lme_fit")
        self.assertDictEqual(settings.parameters, self.default_lme_fit_params)

        model.fit(self.data, "lme_fit")

        self.assertListEqual(model.features, ["Y0"])
        self.assertEqual(model.dimension, 1)

        self.assertEqual(model.with_random_slope_age, True)
        self.assertGreater(
            np.abs(model.parameters["cov_re"][0, 1]), 0
        )  # not forced independent

        # + test save/load
        model_path = self.get_test_tmp_path("lme_model_1.json")
        model.save(model_path)
        del model

        model = BaseModel.load(model_path)
        os.unlink(model_path)

        # Personalize
        settings = AlgorithmSettings("lme_personalize")
        ip = model.personalize(self.data_new_ix, "lme_personalize")

        # check statsmodels consistency
        self.check_consistency_sm(model.parameters, ip, re_formula="~1+TIME_norm")

    def test_with_random_slope_age_indep(self):
        settings = AlgorithmSettings("lme_fit", force_independent_random_effects=True)

        self.assertDictEqual(
            settings.parameters,
            {
                **self.default_lme_fit_params,
                "force_independent_random_effects": True,
                "method": ["lbfgs", "bfgs"],  # powell method not supported
            },
        )
        model = model_factory("lme", with_random_slope_age=True)
        model.fit(self.data, "lme_fit", force_independent_random_effects=True)

        self.assertEqual(model.with_random_slope_age, True)
        self.assertAlmostEqual(model.parameters["cov_re"][0, 1], 0, places=5)

        ip = model.personalize(self.data_new_ix, "lme_personalize")
        free = MixedLMParams.from_components(fe_params=np.ones(2), cov_re=np.eye(2))

        self.check_consistency_sm(
            model.parameters, ip, re_formula="~1+TIME_norm", free=free
        )

    def test_deprecated_hyperparameter_in_algo(self):
        # Test deprecation behavior (test to be removed with this old behavior will be removed)

        ## 1: Overwrite LME hyperparameter from LME fit algo
        settings = AlgorithmSettings("lme_fit", with_random_slope_age=False)
        algo = LMEFitAlgorithm(settings)
        self.assertEqual(
            algo._model_hyperparams_to_set, {"with_random_slope_age": False}
        )

        model = model_factory("lme")
        self.assertTrue(model.with_random_slope_age)
        with self.assertWarns(FutureWarning):
            model.fit(self.data, "lme_fit", with_random_slope_age=False)

        self.assertFalse(model.with_random_slope_age)

        ## 2: No warning if hyperparameter set to None (--> default)
        settings = AlgorithmSettings("lme_fit", with_random_slope_age=None)
        algo = LMEFitAlgorithm(settings)
        self.assertEqual(
            algo._model_hyperparams_to_set, {"with_random_slope_age": None}
        )

        settings = AlgorithmSettings("lme_fit")
        algo = LMEFitAlgorithm(settings)
        self.assertEqual(
            algo._model_hyperparams_to_set, {"with_random_slope_age": None}
        )

        # no effect on model hyperparameter
        model = model_factory("lme", with_random_slope_age=False)
        self.assertFalse(model.with_random_slope_age)
        model.fit(self.data, "lme_fit")
        self.assertFalse(model.with_random_slope_age)
