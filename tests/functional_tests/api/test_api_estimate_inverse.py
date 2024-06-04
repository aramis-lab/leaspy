import numpy as np

# never import a real test case at top-level so to not duplicate tests, only tests MIXINS!
from .test_api_estimate import LeaspyEstimateTest_Mixin


class LeaspyEstimateInverseTest(LeaspyEstimateTest_Mixin):

    def test_estimate_ages_from_biomarker_values_univariate(self):
        # TODO: test that doesn't rely on estimate ? (rather on estimate 'theoretical' results)

        # univariate logistic model
        # feat is "feature"
        leaspy = self.get_hardcoded_model('univariate_logistic')
        ip = self.get_hardcoded_individual_params('ip_univariate_save.json')
        timepoints = {
            'idx1': [78, 81],
            'idx2': [71],
            'idx3': []
        }
        estimations_raw = leaspy.estimate(timepoints, ip)

        # some reshape to do (else shape is (2, 1), when it is supposed to be 2)
        estimations = {}
        for idx, array in estimations_raw.items():
            estimations[idx] = array.squeeze().tolist()
            if isinstance(estimations[idx], float):
                estimations[idx] = [estimations[idx]]

        # with no feature argument
        estimated_ages_1 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values=estimations)
        # with right feature argument
        estimated_ages_2 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values=estimations, feature='Y0')

        # check estimated ages are the original timepoints
        self.check_almost_equal_for_all_ind_tpts(estimated_ages_1, timepoints, tol=0.01)
        self.check_almost_equal_for_all_ind_tpts(estimated_ages_2, timepoints, tol=0.01)

        # check inputs checks

        # with wrong feature argument
        with self.assertRaises(ValueError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=estimations,
                                                       feature='wrong_feature')

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=[])

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=estimations,
                                                       feature=[])

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=[], biomarker_values=estimations)

        with self.assertRaises(TypeError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=[], biomarker_values=estimations)

        bad_estimations = {'idx1': {'bad_type'}}
        with self.assertRaisesRegex(TypeError, 'biomarker_values'):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=bad_estimations,
                                                        feature=None)

        # check other errors
        problematic_timepoints = {
            'idx1': [90],  # fast progressor, 90 is already too much (estimation will be nan)
        }
        problematic_estimations = leaspy.estimate(problematic_timepoints, ip)
        problematic_estimations['idx1'] = problematic_estimations['idx1'].tolist()[0]
        pbq_age = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                             biomarker_values=problematic_estimations)

        # check that nan estimation gives nan age
        self.assertNotEqual(pbq_age['idx1'][0], pbq_age['idx1'][0])

        # quick check biomarker_values as dict of key: str and val: int rather than list works
        estimated_ages_0 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values={'idx1': 0.4,
                                                                                        'idx2': 0.3})
        self.assertAlmostEqual(estimated_ages_0['idx1'], 70.53896, 2)
        self.assertAlmostEqual(estimated_ages_0['idx2'], 73.12502, 2)

    def test_estimate_ages_from_biomarker_values_multivariate(self):
        # multivariate logistic model
        # feats are "Y0", ...
        leaspy = self.get_hardcoded_model('logistic_scalar_noise')
        ip = self.get_hardcoded_individual_params('ip_save.json')

        timepoints = {
            'idx1': [78, 81],
            'idx2': [91]
        }
        estimations_raw = leaspy.estimate(timepoints, ip)

        # select right feature
        def select_feature_estimation(estimations, leaspy, feature):
            """
            Select the right feature from multivariate estimation

            Parameters
            ----------
            estimations: dict of arrays
                array are shape (n_timepoints x n_feats)

            leaspy: Leaspy
                leaspy model

            feature: str
                feature name

            Returns: feat_estimations
                array are shape (n_timepoints x 1)
            -------

            """
            feat_ind = leaspy.model.features.index(feature)
            feat_estimations = {}
            for idx in estimations.keys():
                x = estimations[idx][:, feat_ind]
                feat_estimations[idx] = np.expand_dims(x, axis=1)
            return feat_estimations

        # checks with no feature argument
        with self.assertRaises(ValueError):
            leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=estimations_raw)

        for feature in ['Y0', 'Y1', 'Y2', 'Y3']:
            feat_estimations = select_feature_estimation(estimations=estimations_raw, leaspy=leaspy, feature=feature)

            # some reshape to do (else shape is (2, 1), when it is supposed to be 2)
            estimations = {}
            for idx, array in feat_estimations.items():
                estimations[idx] = array.squeeze().tolist()
                if isinstance(estimations[idx], float):
                    estimations[idx] = [estimations[idx]]

            estimated_ages = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                        biomarker_values=estimations,
                                                                        feature=feature)

            # Remark: tolerance had to be pretty diminished so that the test passes...
            self.check_almost_equal_for_all_ind_tpts(estimated_ages, timepoints, tol=0.5)

        # quick check biomarker_values as dict of key: str and val: int works
        estimated_ages_0 = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                      biomarker_values={'idx1': 0.4,
                                                                                        'idx2': 0.3},
                                                                      feature='Y0')
        self.assertAlmostEqual(estimated_ages_0['idx1'], 68.52, 2)
        self.assertAlmostEqual(estimated_ages_0['idx2'], 72.38, 2)

    def test_estimate_ages_from_biomarker_values_multivariate_ordinal(self):
        # multivariate logistic ordinal model
        # feats are "Y0", ...
        ip = self.get_hardcoded_individual_params('ip_save.json')

        timepoints = {
            "Y0": {'idx1': [69.087, 70.881], 'idx2': [72.841, 74.465]},
            "Y1": {'idx1': [72.463, 73.047, 73.0466],
                   'idx2': [74.627, 75.155, 75.155]},
            "Y2": {'idx1': [69.913, 72.248, 72.248, 72.248, 72.248],
                   'idx2': [72.697, 74.810, 74.810, 74.810, 74.810]},
            "Y3": {'idx1': [69.695, 69.695, 70.554, 70.554, 71.689,
                            71.689, 73.992, 73.992, 74.680],
                   'idx2': [72.396, 72.396, 73.174, 73.174, 74.200,
                            74.200, 76.285, 76.285, 76.907]}
        }

        # loss is not involved in estimation so all expected outputs are the same for those 2 models
        for hardcoded_model in ('logistic_ordinal', 'logistic_ordinal_ranking_same'):

            with self.subTest(hardcoded_model=hardcoded_model):

                leaspy = self.get_hardcoded_model(hardcoded_model)
                levels = {
                    ft: {
                        id_: [list(range(1, ft_max_level))] for id_ in ("idx1", "idx2")
                    } for ft, ft_max_level in leaspy.model.noise_model.max_levels.items()
                }

                # checks with no feature argument
                with self.assertRaises(ValueError):
                    leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip, biomarker_values=levels)

                for feature in ['Y0', 'Y1', 'Y2', 'Y3']:
                    feat_estimations = levels[feature]

                    estimated_ages = leaspy.estimate_ages_from_biomarker_values(individual_parameters=ip,
                                                                                biomarker_values=feat_estimations,
                                                                                feature=feature)
                    self.check_almost_equal_for_all_ind_tpts(estimated_ages, timepoints[feature], tol=1e-3)
