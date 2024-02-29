import unittest
import torch

from leaspy import AlgorithmSettings, Leaspy
from leaspy.models.abstract_model import AbstractModel
from leaspy.models.factory import ModelFactory

from tests import LeaspyTestCase


class AbstractModelTest(LeaspyTestCase):

    model_names = (
        "linear",
        "univariate_logistic",
        "univariate_linear",
        "logistic",
        "logistic_parallel",
    )

    @LeaspyTestCase.allow_abstract_class_init(AbstractModel)
    def test_abstract_model_constructor(self):
        """
        Test initialization of abstract model class object.
        """
        model = AbstractModel("dummy_abstractmodel", obs_models="gaussian-scalar")
        self.assertFalse(model.is_initialized)
        self.assertEqual(model.name, "dummy_abstractmodel")
        # self.assertEqual(model.parameters, None)

        # Test the presence of all these essential methods
        main_methods = (
            # "compute_individual_attachment_tensorized",
            # "update_model_parameters_burn_in",
            # "update_model_parameters_normal",
            # "compute_regularity_realization",
            # "compute_regularity_variable",
            "compute_individual_ages_from_biomarker_values",
            "compute_individual_ages_from_biomarker_values_tensorized",
            "compute_individual_trajectory",
            "compute_jacobian_tensorized",
            "compute_mean_traj",
            "compute_mode_traj",
            "compute_prior_trajectory",
            "compute_sufficient_statistics",
            # "get_initial_model_parameters",
            "get_variables_specs",
            "initialize",
            "initialize_model_parameters",
            # "initialize_state",
            # "load_hyperparameters",
            "load_parameters",
            "move_to_device",
            "to_dict",
            "update_parameters",
            # "validate_compatibility_of_dataset",
        )

        present_attributes = [_ for _ in dir(model) if _[:2] != '__']  # Get the present method

        for attribute in main_methods:
            self.assertIn(attribute, present_attributes)
        # TODO: use python's hasattr and issubclass

    def test_all_model_run(self):
        """
        Check if the following models run with the following algorithms.
        """
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                extra_kws = {}
                if "univariate" not in model_name:
                    extra_kws["source_dimension"] = 2  # force so not to get a warning
                leaspy = Leaspy(model_name, **extra_kws)
                settings = AlgorithmSettings("mcmc_saem", n_iter=200, seed=0)
                data = self.get_suited_test_data_for_model(model_name)
                leaspy.fit(data, settings)
                #if model_name not in ['logistic', 'logistic_parallel']:
                #    # problem with nans with 'gradient_descent_personalize' in multivariate logistic models
                #    methods.append('gradient_descent_personalize')
                for method in ("mode_real", "mean_real", "scipy_minimize"):
                    extra_kws = dict()  # not for all algos
                    if method.endswith("real"):
                        extra_kws = dict(n_iter=100)
                    settings = AlgorithmSettings(method, seed=0, **extra_kws)
                    result = leaspy.personalize(data, settings)

    def test_all_model_run_crossentropy(self):
        """
        Check if the following models run with the following algorithms.
        """
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                extra_kws = {}
                if "univariate" not in model_name:
                    extra_kws["source_dimension"] = 2  # force so not to get a warning
                model = Leaspy(model_name, obs_models="bernoulli", **extra_kws)
                data = self.get_suited_test_data_for_model(f"{model_name}_binary")
                model.fit(data, AlgorithmSettings("mcmc_saem", n_iter=200, seed=0))
                model.personalize(data, AlgorithmSettings("scipy_minimize", seed=0))

    def test_audit_individual_parameters(self):
        # tuple: (valid, nb_inds, src_dim), ips_as_dict
        all_ips = [
            # 0 individual
            ((True, 0, 0), {'tau': [], 'xi': []}),
            ((True, 0, 5), {'tau': [], 'xi': [], 'sources': []}),  # src_dim undefined here...

            # 1 individual
            ((True, 1, 0), {'tau': 50, 'xi': 0}),
            ((False, 1, 1), {'tau': 50, 'xi': 0, 'sources': 0}),  # faulty (source should be vector)
            ((True, 1, 1), {'tau': 50, 'xi': 0, 'sources': [0]}),
            ((True, 1, 2), {'tau': 50, 'xi': 0, 'sources': [0, 0]}),

            # 2 individuals
            ((True, 2, 0), {'tau': [50, 60], 'xi': [0, 0.1]}),
            ((True, 2, 1), {'tau': [50, 60], 'xi': [0, 0.1], 'sources': [0, 0.1]}),  # accepted even if ambiguous
            ((True, 2, 1), {'tau': [50, 60], 'xi': [0, 0.1], 'sources': [[0], [0.1]]}),  # cleaner
            ((True, 2, 2), {'tau': [50, 60], 'xi': [0, 0.1], 'sources': [[0, -1], [0.1, 0]]}),

            # Faulty
            ((False, 1, 0), {'tau': 0, 'xi': 0, 'extra': 0}),
            ((False, 1, 0), {'tau': 0}),
            ((False, None, 0), {'tau': [50, 60], 'xi': [0]}),
        ]

        for src_compat, m in [
            (lambda src_dim: src_dim <= 0, ModelFactory.model('univariate_logistic')),
            (lambda src_dim: src_dim >= 0, ModelFactory.model('logistic'))
        ]:
            for (valid, n_inds, src_dim), ips in all_ips:
                if m.name == 'logistic':
                    m.source_dimension = src_dim
                if (not valid) or (not src_compat(src_dim)):
                    with self.assertRaises(ValueError, ):
                        ips_info = m._audit_individual_parameters(ips)
                    continue
                ips_info = m._audit_individual_parameters(ips)
                keys = set(ips_info.keys()).symmetric_difference(
                    {'nb_inds', 'tensorized_ips', 'tensorized_ips_gen'}
                )
                self.assertEqual(len(keys), 0)
                self.assertEqual(ips_info['nb_inds'], n_inds)
                list_t_ips = list(ips_info['tensorized_ips_gen'])
                self.assertEqual(len(list_t_ips), n_inds)
                t_ips = ips_info['tensorized_ips']
                self.assertIsInstance(t_ips, dict)
                keys_ips = set(t_ips.keys()).symmetric_difference(ips.keys())
                self.assertEqual(len(keys_ips), 0)

                for k, v in t_ips.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(v.dim(), 2)
                    self.assertEqual(
                        v.shape,
                        (n_inds, src_dim if (k == 'sources') and (n_inds > 0) else 1)
                    )
                if n_inds == 1:
                    t_ips0 = list_t_ips[0]
                    self.assertTrue(all(torch.equal(t_ips0[k], v) for k,v in t_ips.items())) # because only 1 individual
                elif n_inds > 1:
                    for t_ips_1i in list_t_ips:
                        for k,v in t_ips_1i.items():
                            self.assertIsInstance(v, torch.Tensor)
                            self.assertEqual(v.dim(), 2)
                            self.assertEqual(v.shape, (1, src_dim if (k == 'sources') else 1))

    def test_model_device_management_cpu_only(self):
        model_name = 'logistic'

        leaspy = Leaspy(model_name, source_dimension=1)
        settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)
        data = self.get_suited_test_data_for_model(model_name)
        leaspy.fit(data, settings)

        # model should be moved to the cpu at the end of the calibration
        self._check_model_device(leaspy.model, torch.device('cpu'))

        leaspy.model.move_to_device(torch.device('cpu'))
        self._check_model_device(leaspy.model, torch.device('cpu'))


    @unittest.skipIf(not torch.cuda.is_available(), 'Device management involving GPU '
                     'is not available without an available CUDA environment.')
    def test_model_device_management_with_gpu(self):
        model_name = 'logistic'

        leaspy = Leaspy(model_name, source_dimension=1)
        settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device='cuda')
        data = self.get_suited_test_data_for_model(model_name)
        leaspy.fit(data, settings)

        # model should be moved to the cpu at the end of the calibration
        self._check_model_device(leaspy.model, torch.device('cpu'))

        leaspy.model.move_to_device(torch.device('cuda'))
        self._check_model_device(leaspy.model, torch.device('cuda'))

        leaspy.model.move_to_device(torch.device('cpu'))
        self._check_model_device(leaspy.model, torch.device('cpu'))

    def _check_model_device(self, model, expected_device):
        if hasattr(model, 'parameters'):
            for param, tensor in model.parameters.items():
                self.assertEqual(tensor.device.type, expected_device.type)

        if hasattr(model, 'attributes'):
            for attribute_name in dir(model.attributes):
                tensor = getattr(model.attributes, attribute_name)
                if isinstance(tensor, torch.Tensor):
                    self.assertEqual(tensor.device.type, expected_device.type)

        if hasattr(model, 'MCMC_toolbox') and 'attributes' in model.MCMC_toolbox:
            for attribute_name in dir(model.MCMC_toolbox['attributes']):
                tensor = getattr(model.MCMC_toolbox['attributes'], attribute_name)
                if isinstance(tensor, torch.Tensor):
                    self.assertEqual(tensor.device.type, expected_device.type)

    # @LeaspyTestCase.allow_abstract_class_init(AbstractModel)
    # def test_compute_individual_trajectory(self):
    #     # TODO not sure it is the right place to test that
    #     # multivariate
    #     leaspy_object = self.get_hardcoded_model('logistic_scalar_noise')
    #     abstract_model = AbstractModel('logistic')
    #     abstract_model.load_parameters(leaspy_object.model.parameters)
    #
    #     ip = {
    #       "xi": 0.1,
    #       "tau": 70,
    #       "sources": [
    #         0.1,
    #         -0.3
    #       ]
    #     }
    #
    #     timepoints = [78, 81]
    #
    #     expected_estimation = torch.tensor([
    #         [[0.99641526, 0.34549406, 0.67467, 0.98959327],
    #          [0.9994672, 0.5080943, 0.8276345, 0.99921334]]
    #     ])
    #
    #     indiv_trajectory = abstract_model.compute_individual_trajectory(timepoints, ip)
    #     self.assertEqual(indiv_trajectory.shape, (1, len(timepoints), 2))
    #     self.assertTrue(torch.eq(indiv_trajectory, expected_estimation))
    #
    #     # TODO univariate ?
    #
