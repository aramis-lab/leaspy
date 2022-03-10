import torch

from leaspy.io.data.dataset import Dataset
from leaspy.algo.utils.samplers.gibbs_sampler import GibbsSampler

from tests import LeaspyTestCase


class SamplerTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling
        super().setUpClass()

        cls.leaspy = cls.get_hardcoded_model('logistic_scalar_noise')
        cls.data = cls.get_suited_test_data_for_model('logistic_scalar_noise')
        cls.dataset = Dataset(cls.data)

        # GibbsSampler scales so not to change old results
        cls.scale_ind = .1 / GibbsSampler.STD_SCALE_FACTOR_IND
        cls.scale_pop = 5e-3 / GibbsSampler.STD_SCALE_FACTOR_POP

    def test_realization(self):
        realizations = self.leaspy.model.initialize_realizations_for_model(2)
        tau_real = realizations['tau']
        self.assertIsInstance(str(tau_real), str)

    def test_sample(self):
        """
        Test if samples values are the one expected
        """
        # TODO change this instantiation
        n_patients = 17
        n_draw = 50
        temperature_inv = 1.0

        realizations = self.leaspy.model.initialize_realizations_for_model(n_patients)

        # test

        # Test with taus (individual parameter)
        var_name = 'tau'
        gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients, scale=self.scale_ind)
        random_draws = []
        for i in range(n_draw):
            gsampler.sample(self.dataset, self.leaspy.model, realizations, temperature_inv, attribute_type=None)
            random_draws.append(realizations[var_name].tensor_realizations.clone())

        stack_random_draws = torch.stack(random_draws)
        stack_random_draws_mean = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).mean(dim=0)
        stack_random_draws_std = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).std(dim=0)

        self.assertAlmostEqual(stack_random_draws_mean.mean(), 0.0160, delta=0.05)
        self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0861, delta=0.05)

        # Test with g (population parameter)
        var_name = 'g'
        gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients, scale=self.scale_pop)
        # a valid model MCMC toolbox is needed for sampling a population variable (update in-place)
        self.leaspy.model.initialize_MCMC_toolbox()
        random_draws = []
        for i in range(n_draw):
            gsampler.sample(self.dataset, self.leaspy.model, realizations, temperature_inv)  # attribute_type=None would not be used here
            random_draws.append(realizations[var_name].tensor_realizations.clone())

        stack_random_draws = torch.stack(random_draws)
        stack_random_draws_mean = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).mean(dim=0)
        stack_random_draws_std = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).std(dim=0)

        self.assertAlmostEqual(stack_random_draws_mean.mean(), 4.2792e-05, delta=0.05)
        self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0045, delta=0.05)

    def test_acceptation(self):
        n_patients = 17
        n_draw = 200
        # temperature_inv = 1.0

        # realizations = self.leaspy.model.initialize_realizations_for_model(n_patients)

        # Test with taus
        var_name = 'tau'
        gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients, scale=self.scale_ind)

        for i in range(n_draw):
            gsampler._update_acceptation_rate(torch.tensor([1.0]*10+[0.0]*7, dtype=torch.float32))

        self.assertAlmostEqual(gsampler.acceptation_history.mean(), 10/17, delta=0.05)

    def test_adaptative_proposition_variance(self):
        n_patients = 17
        n_draw = 200
        # temperature_inv = 1.0

        # realizations = self.leaspy.model.initialize_realizations_for_model(n_patients)

        # Test with taus
        var_name = 'tau'
        gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients, scale=self.scale_ind)

        for i in range(n_draw):
            gsampler._update_acceptation_rate(torch.tensor([1.0]*10+[0.0]*7, dtype=torch.float32))

        for i in range(1000):
            gsampler._update_std()

        self.assertAlmostEqual(gsampler.std[:10].mean(), 4.52, delta=0.05)
        self.assertAlmostEqual(gsampler.std[10:].mean(), 0.0015, delta=0.05)

        for i in range(n_draw):
            gsampler._update_acceptation_rate(torch.tensor([0.0]*10+[1.0]*7, dtype=torch.float32))

        for i in range(2000):
            gsampler._update_std()

        self.assertAlmostEqual(gsampler.std[:10].mean(), 9.8880e-04, delta=0.05)
        self.assertAlmostEqual(gsampler.std[10:].mean(), 3.0277, delta=0.05)
