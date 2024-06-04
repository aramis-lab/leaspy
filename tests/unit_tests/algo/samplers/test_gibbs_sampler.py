from itertools import cycle

import torch

from leaspy.io.data.dataset import Dataset
from leaspy.samplers import sampler_factory, IndividualGibbsSampler, PopulationGibbsSampler
from leaspy.io.realizations import CollectionRealization, VariableType

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
        cls.scale_ind = .1 / IndividualGibbsSampler.STD_SCALE_FACTOR
        cls.scale_pop = 5e-3 / PopulationGibbsSampler.STD_SCALE_FACTOR

    def test_realization(self):
        realizations = CollectionRealization()
        realizations.initialize(self.leaspy.model, n_individuals=2)
        self.assertEqual(set(realizations.individual.names), {"tau", "xi", "sources"})
        tau_real = realizations["tau"].tensor
        self.assertIsInstance(tau_real, torch.Tensor)
        self.assertEqual(tau_real.shape, (2, 1))

        # check association between tensors
        realizations["tau"].set_tensor_realizations_element(torch.tensor(42.), (1, 0))
        self.assertEqual(tau_real[1, 0].item(), 42.)

        # test cloning
        r = realizations.clone()
        self.assertEqual(r.population.names, realizations.population.names)
        self.assertEqual(r.individual.names, realizations.individual.names)

        # check dissociation between tensors
        tau_real[1, 0] = 75.
        self.assertEqual(r[["tau"]].tensors[0][0, 0], tau_real[0, 0])
        self.assertNotEqual(r[["tau"]].tensors[0][1, 0], tau_real[1, 0])

    def test_sample(self):
        """
        Test if samples values are the one expected
        """
        # TODO change this instantiation
        n_patients = 17
        n_draw = 50
        temperature_inv = 1.0
        realizations = CollectionRealization()
        realizations.initialize(self.leaspy.model, n_individuals=n_patients)

        # Test with taus (individual parameter)
        var_name = "tau"
        for sampler_name in ("Gibbs",):
            rv_info = self.leaspy.model.get_individual_random_variable_information()[var_name]
            sampler = sampler_factory(
                sampler_name,
                VariableType.INDIVIDUAL,
                scale=self.scale_ind,
                n_patients=n_patients,

                name=var_name,
                shape=rv_info["shape"],
            )
            random_draws = []
            for i in range(n_draw):
                sampler.sample(self.dataset, self.leaspy.model, realizations, temperature_inv, attribute_type=None)
                random_draws.append(realizations[var_name].tensor.clone())

            stack_random_draws = torch.stack(random_draws)
            stack_random_draws_mean = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).mean(dim=0)
            stack_random_draws_std = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).std(dim=0)

            self.assertAlmostEqual(stack_random_draws_mean.mean(), 0.0160, delta=0.05)
            self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0861, delta=0.05)

        # Test with g (1D population parameter) and betas (2 dimensional population parameter)
        for var_name in ("g", "betas"):
            for sampler_name in ("Gibbs", "FastGibbs", "Metropolis-Hastings"):
                rv_info = self.leaspy.model.get_population_random_variable_information()[var_name]
                sampler = sampler_factory(
                    sampler_name,
                    VariableType.POPULATION,
                    scale=self.scale_pop,
                    name=var_name,
                    shape=rv_info["shape"],
                )
                # a valid model MCMC toolbox is needed for sampling a population variable (update in-place)
                self.leaspy.model.initialize_MCMC_toolbox()
                random_draws = []
                for i in range(n_draw):
                    # attribute_type=None would not be used here
                    sampler.sample(self.dataset, self.leaspy.model, realizations, temperature_inv)
                    random_draws.append(realizations[var_name].tensor.clone())

                stack_random_draws = torch.stack(random_draws)
                stack_random_draws_mean = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).mean(dim=0)
                stack_random_draws_std = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).std(dim=0)

                self.assertAlmostEqual(stack_random_draws_mean.mean(), 4.2792e-05, delta=0.05)
                self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0045, delta=0.05)

    def test_acceptation(self):
        n_patients = 17
        n_draw = 200

        # Test with tau (0D individual variable) and sources (1D individual variable of dim Ns, here 2)
        # --> we do not take care of dimension for individual parameter!
        for var_name in ("tau", "sources"):
            cst_acceptation = torch.tensor([1.0] * 10 + [0.0] * 7)
            for sampler_name in ("Gibbs",):
                rv_info = self.leaspy.model.get_individual_random_variable_information()[var_name]
                sampler = sampler_factory(
                    sampler_name,
                    VariableType.INDIVIDUAL,
                    scale=self.scale_ind,
                    n_patients=n_patients,
                    name=var_name,
                    shape=rv_info["shape"],
                )
                for i in range(n_draw):
                    sampler._update_acceptation_rate(cst_acceptation)

                acc_mean = sampler.acceptation_history.mean(dim=0)
                self.assertEqual(acc_mean.shape, cst_acceptation.shape)
                self.assertAllClose(acc_mean, cst_acceptation)

        # Test with g (1D population variable of dim N, here 4)
        # and betas (2D population variable of dim (N-1, Ns), here (3, 2))
        for var_name in ("g", "betas"):
            acceptation_for_draws = self._get_acceptation_for_draws(var_name)
            for sampler_name, (acceptation_it, expected_mean_acceptation) in acceptation_for_draws.items():
                rv_info = self.leaspy.model.get_population_random_variable_information()[var_name]
                sampler = sampler_factory(
                    sampler_name,
                    VariableType.POPULATION,
                    scale=self.scale_pop,
                    name=var_name,
                    shape=rv_info["shape"],
                )
                for i in range(n_draw):
                    sampler._update_acceptation_rate(next(acceptation_it))

                acc_mean = sampler.acceptation_history.mean(dim=0)
                self.assertEqual(acc_mean.shape, expected_mean_acceptation.shape)
                self.assertAllClose(acc_mean, expected_mean_acceptation, msg=(var_name, sampler_name))

    def _get_acceptation_for_draws(self, variable_name: str) -> dict:
        acceptation_for_draws = {
            "Metropolis-Hastings": (
                cycle(
                    [torch.tensor(1.)] * 3 + [torch.tensor(0.)] * 2
                ),
                torch.tensor(3 / 5),
            ),
        }
        if variable_name == "g":
            acceptation_for_draws.update(
                {
                    "Gibbs": (
                        cycle(
                            [torch.tensor([0., 0., 1., 1.])] * 3 + [torch.tensor([0., 1., 0., 1.])] * 2
                        ),
                        torch.tensor([0., 2 / 5, 3 / 5, 1.]),
                    ),
                    "FastGibbs": (
                        cycle(
                            [torch.tensor([0., 0., 1., 1.])] * 3 + [torch.tensor([0., 1., 0., 1.])] * 2
                        ),
                        torch.tensor([0., 2 / 5, 3 / 5, 1.]),
                    ),
                }
            )
        elif variable_name == "betas":
            acceptation_for_draws.update(
                {
                    "Gibbs": (
                        cycle(
                            [torch.tensor([[0., 0.], [0., 1.], [1., 1.]])] * 3 +
                            [torch.tensor([[0., 1.], [1., 0.], [0., 1.]])] * 2
                        ),
                        torch.tensor([[0., 2 / 5], [2 / 5, 3 / 5], [3 / 5, 1.]]),
                    ),
                    "FastGibbs": (
                        cycle(
                            [torch.tensor([0., 0., 1.])] * 3 + [torch.tensor([0., 1., 0.])] * 2
                        ),
                        torch.tensor([0., 2/5, 3/5]),
                    ),
                }
            )
        return acceptation_for_draws

    def test_adaptative_proposition_variance(self):
        n_patients = 17
        n_draw = 200
        # temperature_inv = 1.0

        # realizations = self.leaspy.model.initialize_realizations_for_model(n_patients)

        # Test with taus
        var_name = "tau"
        rv_info = self.leaspy.model.get_individual_random_variable_information()[var_name]
        sampler = sampler_factory(
            "Gibbs",
            VariableType.INDIVIDUAL,
            scale=self.scale_ind,
            n_patients=n_patients,
            name=var_name,
            shape=rv_info["shape"],
        )

        for i in range(n_draw):
            sampler._update_acceptation_rate(torch.tensor([1.0] * 10 + [0.0] * 7, dtype=torch.float32))

        for i in range(1000):
            sampler._update_std()

        self.assertAlmostEqual(sampler.std[:10].mean(), 4.52, delta=0.05)
        self.assertAlmostEqual(sampler.std[10:].mean(), 0.0015, delta=0.05)

        for i in range(n_draw):
            sampler._update_acceptation_rate(torch.tensor([0.0] * 10 + [1.0] * 7, dtype=torch.float32))

        for i in range(2000):
            sampler._update_std()

        self.assertAlmostEqual(sampler.std[:10].mean(), 9.8880e-04, delta=0.05)
        self.assertAlmostEqual(sampler.std[10:].mean(), 3.0277, delta=0.05)
