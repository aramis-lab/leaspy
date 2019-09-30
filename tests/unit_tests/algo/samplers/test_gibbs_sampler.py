import unittest

from leaspy.algo.samplers.gibbs_sampler import GibbsSampler
from . import dummy_ind_variable_infos

from leaspy import Leaspy, Data
from leaspy.inputs.data.dataset import Dataset
import torch

from tests import test_data_dir

import os

class SamplerTest(unittest.TestCase):

    def test_sample(self):
        """
        Test if samples values are the one expected
        :return:
        """
        # TODO change this instanciation
        n_patients = 17
        n_draw = 50
        temperature_inv = 1.0

        path_model_sampler = os.path.join(test_data_dir, "model_parameters", "multivariate_model_sampler.json")
        path_data = os.path.join(test_data_dir, "inputs", "data_tiny.csv")

        data = Dataset(Data.from_csv_file(path_data))
        leaspy = Leaspy.load(path_model_sampler)
        realizations = leaspy.model.get_realization_object(n_patients)

        # Test with taus
        var_name = 'tau'
        gsampler = GibbsSampler(leaspy.model.random_variable_informations()[var_name], n_patients)
        random_draws = []
        for i in range(n_draw):
            gsampler.sample(data, leaspy.model, realizations, temperature_inv)
            random_draws.append(realizations[var_name].tensor_realizations.clone())

        stack_random_draws = torch.stack(random_draws)
        stack_random_draws_mean = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).mean(dim=0)
        stack_random_draws_std = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).std(dim=0)

        self.assertAlmostEqual(stack_random_draws_mean.mean(), 0.0160, delta=0.05)
        self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0861, delta=0.05)

        # Test with g
        var_name = 'g'
        gsampler = GibbsSampler(leaspy.model.random_variable_informations()[var_name], n_patients)
        random_draws = []
        for i in range(n_draw):
            gsampler.sample(data, leaspy.model, realizations, temperature_inv)
            random_draws.append(realizations[var_name].tensor_realizations.clone())

        stack_random_draws = torch.stack(random_draws)
        stack_random_draws_mean = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).mean(dim=0)
        stack_random_draws_std = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).std(dim=0)

        self.assertAlmostEqual(stack_random_draws_mean.mean(), 4.2792e-05, delta=0.05)
        self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0045, delta=0.05)





    def test_acceptation(self):
        n_patients = 17
        n_draw = 200
        temperature_inv = 1.0

        path_model_sampler = os.path.join(test_data_dir, "model_parameters", "multivariate_model_sampler.json")
        path_data = os.path.join(test_data_dir, "inputs", "data_tiny.csv")

        data = Dataset(Data.from_csv_file(path_data))
        leaspy = Leaspy.load(path_model_sampler)
        realizations = leaspy.model.get_realization_object(n_patients)

        # Test with taus
        var_name = 'tau'
        gsampler = GibbsSampler(leaspy.model.random_variable_informations()[var_name], n_patients)

        for i in range(n_draw):
            gsampler._update_acceptation_rate(torch.tensor([1.0]*10+[0.0]*7, dtype=torch.float32))

        self.assertAlmostEqual(gsampler.acceptation_temp.mean(), 10/17, delta=0.05)



    def test_adaptative_proposition_variance(self):
        n_patients = 17
        n_draw = 200
        temperature_inv = 1.0

        path_model_sampler = os.path.join(test_data_dir, "model_parameters", "multivariate_model_sampler.json")
        path_data = os.path.join(test_data_dir, "inputs", "data_tiny.csv")

        data = Dataset(Data.from_csv_file(path_data))
        leaspy = Leaspy.load(path_model_sampler)
        realizations = leaspy.model.get_realization_object(n_patients)

        # Test with taus
        var_name = 'tau'
        gsampler = GibbsSampler(leaspy.model.random_variable_informations()[var_name], n_patients)

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
