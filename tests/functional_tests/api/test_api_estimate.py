import numpy as np

from tests import LeaspyTestCase

class LeaspyEstimateTest_Mixin(LeaspyTestCase):

    def check_almost_equal_for_all_ind_tpts(self, a, b, tol=1e-5):
        return self.assertDictAlmostEqual(a, b, atol=tol)

    def batch_checks(self, ip, tpts, models, expected_ests):
        for model_name in models:
            with self.subTest(model_name=model_name):
                leaspy = self.get_hardcoded_model(model_name)
                estimations = leaspy.estimate(tpts, ip)
                self.check_almost_equal_for_all_ind_tpts(estimations, expected_ests)


class LeaspyEstimateTest(LeaspyEstimateTest_Mixin):

    def test_estimate_multivariate(self):

        ip = self.get_hardcoded_individual_params('ip_save.json')

        timepoints = {
            'idx1': [78, 81],
            'idx2': [71]
        }

        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ('logistic_scalar_noise', 'logistic_diag_noise_id', 'logistic_diag_noise')
        expected_ests = {
            'idx1': [[0.99641526, 0.34549406, 0.67467   , 0.98959327],
                     [0.9994672 , 0.5080943 , 0.8276345 , 0.99921334]],
            'idx2': [[0.13964376, 0.1367586 , 0.23170303, 0.01551363]]
        }

        self.batch_checks(ip, timepoints, models, expected_ests)

        # TODO logistic parallel?

        # TODO linear model?

    def test_estimate_univariate(self):

        ip = self.get_hardcoded_individual_params('ip_univariate_save.json')

        timepoints = {
            'idx1': [78, 81],
            'idx2': [71]
        }

        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ('univariate_logistic',)
        expected_ests = {
            'idx1': [
                [0.999607],
                [0.9999857]
            ],
            'idx2': [[0.03098414]]
        }

        self.batch_checks(ip, timepoints, models, expected_ests)
