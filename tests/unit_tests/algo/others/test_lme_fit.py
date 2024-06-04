import numpy as np
import pandas as pd

from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy.models.lme import LMEModel
from leaspy.algo.others.lme_fit import LMEFitAlgorithm
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

from tests import LeaspyTestCase


class LMEFitAlgorithmTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls):

        # for tmp handling
        super().setUpClass()

        # TODO? redo test with realistic values...?
        df = pd.DataFrame.from_records((np.arange(3, 3 + 10, 1),
                                        np.arange(15, 15 + 10, 1),
                                        np.arange(6, 6 + 10, 1)),
                                       index=['pat1', 'pat2', 'pat3'],
                                       ).T.stack()
        df = pd.DataFrame(df)
        df.index.names = ['TIME', 'ID']
        df = df.rename(columns={0: 'feat1'})
        df = df.swaplevel()
        # add a nan
        df.loc[('pat1', 0)] = np.nan
        cls.dataframe = df
        data = Data.from_dataframe(df, drop_full_nan=False)  # otherwise first visit of pat1 is dropped and then order of subjects is not as originally expected.........
        cls.dataset = Dataset(data)
        # models
        cls.model = LMEModel('lme', with_random_slope_age=False)
        cls.settings = AlgorithmSettings('lme_fit')
        cls.algo = LMEFitAlgorithm(cls.settings)

    def test_constructor(self):
        self.assertFalse(self.algo.deterministic)
        self.assertEqual(self.algo.family, 'fit')

    def test_get_reformated(self):
        ages = self.algo._get_reformated(self.dataset, 'timepoints')
        expected_ages = np.array(self.dataframe.sort_index(axis=0).index.get_level_values('TIME'))[1:]
        values = self.algo._get_reformated(self.dataset, 'values')
        expected_values = np.array(self.dataframe.sort_index(axis=0)['feat1'].values)[1:]
        self.assertTrue((ages == expected_ages).all())
        self.assertTrue((values == expected_values).all())

    def test_get_reformated_subjects(self):
        subjects = self.algo._get_reformated_subjects(self.dataset)
        expected_subjects = ['pat1'] * 9 + ['pat2'] * 10 + ['pat3'] * 10
        self.assertTrue((subjects == expected_subjects).all())

    def test_run(self):
        self.algo.run(self.model, self.dataset)

        ages_mean = 4.6551723
        ages_std = 2.7950184

        self.assertAlmostEqual(self.model.parameters["ages_mean"], ages_mean)
        self.assertAlmostEqual(self.model.parameters["ages_std"], ages_std)

        self.assertAlmostEqual(self.model.parameters["fe_params"][0], 8 + ages_mean*1., 0)
        self.assertAlmostEqual(self.model.parameters["fe_params"][1], 1. * ages_std, 0)
        self.assertAlmostEqual(self.model.parameters["noise_std"], 1.3220877080541017e-05)
        self.assertAlmostEqual(self.model.parameters["cov_re"][0][0], 2.8888, 3)
        self.assertAlmostEqual(self.model.parameters["cov_re_unscaled_inv"][0][0], 1/14e9, -9)
        self.assertAlmostEqual(self.model.parameters["bse_fe"][0], 1, 0)
        self.assertAlmostEqual(self.model.parameters["bse_fe"][1], 0, 0)
        self.assertAlmostEqual(self.model.parameters["bse_re"][0], 61806.329294854615, -3)
