import numpy as np
import pandas as pd
import torch

from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy.models.lme import LMEModel
from leaspy.algo.others.lme_personalize import LMEPersonalizeAlgorithm
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

from tests import LeaspyTestCase


class LMEPersonalizeAlgorithmTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls):

        # for tmp handling
        super().setUpClass()

        # Leaspy
        cls.model = LMEModel('lme', with_random_slope_age=False)
        # TODO? redo test with realistic values...?
        cls.parameters = {
                           "ages_mean": 0.,
                           "ages_std": 1.,
                           "fe_params": np.array([0.3333016, 1.]),
                           "cov_re": np.array([[0.4523892]]),
                           "cov_re_unscaled_inv": np.array([[1/1.41324825e+10]]),
                           "noise_std": 0.111 # fake value to be displayed...
                         }
        cls.model.features = ['feat1']
        cls.model.load_parameters(cls.parameters)
        cls.settings = AlgorithmSettings('lme_personalize')
        cls.algo = LMEPersonalizeAlgorithm(cls.settings)
        #cls.algo.features = ['A']
        cls.times = torch.tensor([0, 2, 4, 6])
        cls.values = np.array([
            [2.],
            [4.],
            [float('nan')],
            [8.]
        ])

    def test_constructor(self):
        self.assertEqual(self.algo.name, 'lme_personalize')
        self.assertTrue(self.algo.deterministic)
        self.assertEqual(self.algo.family, 'personalize')

    def test_remove_nans(self):
        new_values, new_times = self.algo._remove_nans(self.values, self.times)
        self.assertTrue((new_values == np.array([2., 4., 8.])).all())
        self.assertTrue((new_times.numpy() == np.array([0, 2, 6])).all())

    def test_get_individual_random_effect(self):
        ind_ip, res = self.algo._get_individual_random_effects_and_residuals(self.model, self.times, self.values)
        self.assertAlmostEqual(ind_ip['random_intercept'], 2, 0)

    def test_run(self):
        df = pd.DataFrame.from_records((np.arange(3, 3 + 10, 1),
                                        np.arange(15, 15 + 10, 1),
                                        np.arange(6, 6 + 10, 1)),
                                       index=['pat1', 'pat2', 'pat3'],
                                       ).T.stack()
        df = pd.DataFrame(df)
        df.index.names = ['TIME', 'ID']
        df = df.rename(columns={0: 'feat1'})
        df = df.swaplevel()
        data = Data.from_dataframe(df)
        self.dataset = Dataset(data)
        ip = self.algo.run(model=self.model, dataset=self.dataset)
        self.assertAlmostEqual(ip._individual_parameters['pat1']['random_intercept'], 3, 0)
        self.assertAlmostEqual(ip._individual_parameters['pat2']['random_intercept'], 15, 0)
        self.assertAlmostEqual(ip._individual_parameters['pat3']['random_intercept'], 6, 0)
