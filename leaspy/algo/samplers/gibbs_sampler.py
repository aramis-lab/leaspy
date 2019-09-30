import torch
from .abstract_sampler import AbstractSampler
import itertools


class GibbsSampler(AbstractSampler):

    def __init__(self, info, n_patients):
        super().__init__(info, n_patients)

        self.std = None

        if info["type"] == "population":
            # Proposition variance is adapted independantly on each dimension of the population variable
            self.std = 0.005 * torch.ones(size=self.shape) # TODO hyperparameter here
        elif info["type"] == "individual":
            # Proposition variance is adapted independantly on each patient, but is the same for multiple dimensions
            # TODO : gérer les shapes !!! Necessary for sources
            self.std = torch.tensor([0.1] * n_patients * int(self.shape[0]),
                                    dtype=torch.float32).reshape(n_patients,int(self.shape[0]))

        # Acceptation rate
        self.counter_acceptation = 0

        # Torch distribution
        self.distribution = torch.distributions.normal.Normal(loc=0.0, scale=self.std)

        self.previous_attachment = None
        self.previous_regularity = None

    def sample(self, data, model, realizations, temperature_inv): #TODO is data / model / realizations supposed to be in sampler ????
        """
        Sample either as population or individual.
        Modifies the realizations object.
        :param data:
        :param model:
        :param realizations:
        :param temperature_inv:
        :return:
        """
        if self.type == 'pop':
            self._sample_population_realizations(data, model, realizations, temperature_inv)
        else:
            self._sample_individual_realizations(data, model, realizations, temperature_inv)

    def _proposal(self, val):
        """
        Proposal value around the current value with sampler standard deviation.
        :param val:
        :return:
        """
        # return val+self.distribution.sample(sample_shape=val.shape)
        return val + self.distribution.sample()

    def _update_std(self):
        """
        Update standard deviation of sampler according to current frequency of acceptation.
        Adaptative std is known to improve sampling performances.
        Std is increased if frequency of acceptation > 40%, and decreased if <20%, so as
        to stay close to 30%.
        :return:
        """

        self.counter_acceptation += 1

        if self.counter_acceptation == self.temp_length:
            mean_acceptation = self.acceptation_temp.mean(0)

            idx_toolow = mean_acceptation < 0.2
            idx_toohigh = mean_acceptation > 0.4

            self.std[idx_toolow] *= 0.9
            self.std[idx_toohigh] *= 1.1

            # reset acceptation temp list
            self.counter_acceptation = 0

    def _set_std(self, std):
        self.std = std
        self.distribution = torch.distributions.normal.Normal(loc=0.0, scale=std)

    def _sample_population_realizations(self, data, model, realizations, temperature_inv):
        """
        For each dimension (1D or 2D) of the population variable, compute current attachment and regularity.
        Propose a new value for the given dimension of the given population variable,
        and compute new attachment and regularity.
        Do a MH step, keeping if better, or if worse with a probability.
        :param data:
        :param model:
        :param realizations:
        :param temperature_inv:
        :return:
        """
        realization = realizations[self.name]
        shape_current_variable = realization.shape
        index = [e for e in itertools.product(*[range(s) for s in shape_current_variable])]

        accepted_array = []

        for idx in index:
            # Compute the attachment and regularity
            # previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
            # previous_regularity = model.compute_regularity_realization(realization).sum()
            if self.previous_attachment is None:
                assert self.previous_regularity is None
                self.previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
                self.previous_regularity = model.compute_regularity_realization(realization).sum()

            # Keep previous realizations and sample new ones
            previous_reals_pop = realization.tensor_realizations.clone()
            new_val = self._proposal(realization.tensor_realizations[idx])[idx]
            realization.set_tensor_realizations_element(new_val, idx)

            # Update intermediary model variables if necessary
            model.update_MCMC_toolbox([self.name], realizations)

            # Compute the attachment and regularity
            new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
            new_regularity = model.compute_regularity_realization(realization).sum()

            alpha = torch.exp(-((new_regularity - self.previous_regularity) * temperature_inv +
                                (new_attachment - self.previous_attachment)))

            accepted = self._metropolis_step(alpha)
            accepted_array.append(accepted)

            # Revert if not accepted
            if not accepted:
                # Revert realizations
                realization.tensor_realizations = previous_reals_pop
                # Update intermediary model variables if necessary
                model.update_MCMC_toolbox([self.name], realizations)
                # force re-compute on next iteration
                self.previous_attachment = self.previous_regularity = None
            else:
                self.previous_attachment = new_attachment
                self.previous_regularity = new_regularity

        self._update_acceptation_rate(torch.Tensor([accepted_array]))
        self._update_std()

    def _sample_individual_realizations(self, data, model, realizations, temperature_inv):
        """
        For each indivual variable, compute current patient-batched attachment and regularity.
        Propose a new value for the individual variable,
        and compute new patient-batched attachment and regularity.
        Do a MH step, keeping if better, or if worse with a probability.
        :param data:
        :param model:
        :param realizations:
        :param temperature_inv:
        :return:
        """
        # Compute the attachment and regularity
        realization = realizations[self.name]

        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        previous_regularity = model.compute_regularity_realization(realization).sum(dim=1).reshape(data.n_individuals)

        # Keep previous realizations and sample new ones
        previous_reals = realization.tensor_realizations.clone()
        realization.tensor_realizations = self._proposal(realization.tensor_realizations)

        # Compute the attachment and regularity
        new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        new_regularity = model.compute_regularity_realization(realization).sum(dim=1).reshape(data.n_individuals)

        alpha = torch.exp(-((new_regularity - previous_regularity) * temperature_inv +
                            (new_attachment - previous_attachment)))

        accepted = self._group_metropolis_step(alpha)
        self._update_acceptation_rate(accepted)
        self._update_std()
        ##### PEUT ETRE PB DE SHAPE
        accepted = accepted.unsqueeze(1)
        realization.tensor_realizations = accepted*realization.tensor_realizations + (1-accepted)*previous_reals

