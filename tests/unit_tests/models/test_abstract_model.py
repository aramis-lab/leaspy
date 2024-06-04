import unittest
import torch

from leaspy import AlgorithmSettings, Leaspy
from leaspy.models.abstract_model import AbstractModel
from leaspy.models.factory import ModelFactory

from tests import LeaspyTestCase


class AbstractModelTest(LeaspyTestCase):

    @LeaspyTestCase.allow_abstract_class_init(AbstractModel)
    def test_abstract_model_constructor(self):
        """
        Test initialization of abstract model class object.
        """
        model = AbstractModel("dummy_abstractmodel", noise_model='gaussian-scalar')
        self.assertFalse(model.is_initialized)
        self.assertEqual(model.name, "dummy_abstractmodel")
        self.assertEqual(model.parameters, None)
        #self.assertIs(model.regularization_distribution_factory, torch.distributions.normal.Normal)  # removed

        # Test the presence of all these essential methods
        main_methods = (
            "load_parameters",
            "compute_individual_attachment_tensorized",
            "update_model_parameters_burn_in",
            "update_model_parameters_normal",
            "get_population_variable_names",
            "get_individual_variable_names",
            "compute_regularity_realization",
            "compute_regularity_variable",
            "compute_individual_ages_from_biomarker_values",
            "compute_individual_ages_from_biomarker_values_tensorized",
        )

        present_attributes = [_ for _ in dir(model) if _[:2] != '__']  # Get the present method

        for attribute in main_methods:
            self.assertIn(attribute, present_attributes)
        # TODO: use python's hasattr and issubclass

    @LeaspyTestCase.allow_abstract_class_init(AbstractModel)
    def test_load_parameters(self):
        """
        Test the method load_parameters.
        """
        leaspy_object = self.get_hardcoded_model('logistic_scalar_noise')

        abstract_model = AbstractModel("dummy_model", noise_model='gaussian-scalar')

        abstract_model.load_parameters(leaspy_object.model.parameters)

        expected_parameters = {
            'g': [0.5, 1.5, 1.0, 2.0],
            'v0': [-2.0, -3.5, -3.0, -2.5],
            'betas': [[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]],
            'tau_mean': 75.2,
            'tau_std': 7.1,
            'xi_mean': 0.0,
            'xi_std': 0.2,
            'sources_mean': 0.0,
            'sources_std': 1.0,
        }

        self.assertDictAlmostEqual(abstract_model.parameters, expected_parameters)

    def test_all_model_run(self):
        """
        Check if the following models run with the following algorithms.
        """
        for model_name in ('linear', 'univariate_logistic', 'univariate_linear', 'logistic', 'logistic_parallel'):
            with self.subTest(model_name=model_name):
                extra_kws = {}
                if 'univariate' not in model_name:
                    extra_kws['source_dimension'] = 2  # force so not to get a warning

                leaspy = Leaspy(model_name, **extra_kws)
                settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

                data = self.get_suited_test_data_for_model(model_name)

                leaspy.fit(data, settings)

                methods = ['mode_real', 'mean_real', 'scipy_minimize']
                #if model_name not in ['logistic', 'logistic_parallel']:
                #    # problem with nans with 'gradient_descent_personalize' in multivariate logistic models
                #    methods.append('gradient_descent_personalize')

                for method in methods:
                    extra_kws = dict() # not for all algos
                    if '_real' in method:
                        extra_kws = dict(n_iter=100)
                    settings = AlgorithmSettings(method, seed=0, **extra_kws)
                    result = leaspy.personalize(data, settings)

    def test_all_model_run_crossentropy(self):
        """
        Check if the following models run with the following algorithms.
        """
        for model_name in ('linear', 'univariate_logistic', 'univariate_linear', 'logistic', 'logistic_parallel'):
            with self.subTest(model_name=model_name):
                extra_kws = {}
                if 'univariate' not in model_name:
                    extra_kws['source_dimension'] = 2  # force so not to get a warning

                leaspy = Leaspy(model_name, noise_model='bernoulli', **extra_kws)
                settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

                data = self.get_suited_test_data_for_model(model_name + '_binary')

                leaspy.fit(data, settings)

                for method in ['scipy_minimize']:
                    extra_kws = dict() # not for all algos
                    if '_real' in method:
                        extra_kws = dict(n_iter=100)
                    settings = AlgorithmSettings(method, seed=0, **extra_kws)
                    result = leaspy.personalize(data, settings)

    def test_tensorize_2D(self):

        t5 = torch.tensor([[5]],dtype=torch.float32)

        for x, unsqueeze_dim, expected_out in zip([
            [1,2], [1,2], 5, 5, [5], [5]
        ], [0,-1,0,-1,0,-1], [
            torch.tensor([[1,2]],dtype=torch.float32),
            torch.tensor([[1],[2]],dtype=torch.float32),
            t5, t5, t5, t5
        ]):
            self.assertTrue(torch.equal(
                AbstractModel._tensorize_2D(x,unsqueeze_dim=unsqueeze_dim),
                expected_out
            ))

    def test_audit_individual_parameters(self):

        # tuple: (valid, nb_inds, src_dim), ips_as_dict
        all_ips = [
            # 0 individual
            ((True, 0, 0), {'tau':[],'xi':[]}),
            ((True, 0, 5), {'tau':[],'xi':[],'sources':[]}), # src_dim undefined here...

            # 1 individual
            ((True, 1, 0), {'tau':50,'xi':0,}),
            ((False, 1, 1), {'tau':50,'xi':0,'sources':0}), # faulty (source should be vector)
            ((True, 1, 1), {'tau':50,'xi':0,'sources':[0]}),
            ((True, 1, 2), {'tau':50,'xi':0,'sources':[0,0]}),

            # 2 individuals
            ((True, 2, 0), {'tau':[50,60],'xi':[0,0.1],}),
            ((True, 2, 1), {'tau':[50,60],'xi':[0,0.1],'sources':[0,0.1]}), # accepted even if ambiguous
            ((True, 2, 1), {'tau':[50,60],'xi':[0,0.1],'sources':[[0],[0.1]]}), # cleaner
            ((True, 2, 2), {'tau':[50,60],'xi':[0,0.1],'sources':[[0,-1],[0.1,0]]}),

            # Faulty
            ((False, 1, 0), {'tau':0,'xi':0,'extra':0}),
            ((False, 1, 0), {'tau':0,}),
            ((False, None, 0), {'tau':[50,60],'xi':[0]}),
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

                keys = set(ips_info.keys()).symmetric_difference({'nb_inds','tensorized_ips','tensorized_ips_gen'})
                self.assertEqual(len(keys), 0)

                self.assertEqual(ips_info['nb_inds'], n_inds)

                list_t_ips = list(ips_info['tensorized_ips_gen'])
                self.assertEqual(len(list_t_ips), n_inds)

                t_ips = ips_info['tensorized_ips']
                self.assertIsInstance(t_ips, dict)
                keys_ips = set(t_ips.keys()).symmetric_difference(ips.keys())
                self.assertEqual(len(keys_ips), 0)

                for k,v in t_ips.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(v.dim(), 2)
                    self.assertEqual(v.shape, (n_inds, src_dim if (k == 'sources') and (n_inds > 0) else 1))

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
