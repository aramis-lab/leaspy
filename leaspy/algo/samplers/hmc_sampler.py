import numpy as np
import torch
from .abstract_sampler import AbstractSampler


class HMCSampler(AbstractSampler):

    def __init__(self, info, n_patients, eps):
        super().__init__(info, n_patients)
        self.L = 10
        if self.name == 'tau':
            self.eps = eps * 10
        elif self.type == 'pop':
            if self.name == 'betas':
                self.eps = eps * 0.01
                self.L = 15
            else:
                self.eps = eps * 0.1
        else:
            self.eps = eps
        self.std = 0.1

    def sample(self, data, model, realizations, temperature_inv):
        if self.type == 'ind':
            if self.name == 'sources':
                self._sample_individual_realizations(data, model, realizations, 1.)
            else:
                self._sample_individual_realizations(data, model, realizations, temperature_inv)
        else:
            self._sample_pop_realizations(data, model, realizations, 1.)

    def _proposal(self, p, realizations, model, data, temperature_inv):
        for l in range(self.L + np.random.randint(-5, 5)):
            self._leapfrog_step(p, realizations, model, data, temperature_inv)
        return realizations[self.name].tensor_realizations

    def _sample_pop_realizations(self, data, model, realizations, temperature_inv):
        # this returns a tensor with values for each indiv
        realizations[self.name].to_torch_Variable()
        old_real = realizations[self.name].tensor_realizations.clone()
        p = self._initialize_momentum(old_real)

        old_H = self._compute_pop_hamiltonian(model, data, p, realizations, temperature_inv)

        realizations[self.name].tensor_realizations = self._proposal(p, realizations, model, data, temperature_inv)
        model.update_MCMC_toolbox([self.name], realizations)
        new_H = self._compute_pop_hamiltonian(model, data, p, realizations, temperature_inv)

        accepted = self._metropolis_step(torch.exp(old_H - new_H))
        self._update_acceptation_rate([accepted])
        with torch.no_grad():
            realizations[self.name].tensor_realizations = realizations[self.name].tensor_realizations * accepted + (
                    1. - accepted) * old_real
        realizations[self.name].to_torch_Tensor()
        model.update_MCMC_toolbox([self.name], realizations)

    def _sample_individual_realizations(self, data, model, realizations, temperature_inv):
        # this returns a tensor with values for each indiv
        realizations[self.name].to_torch_Variable()
        old_real = realizations[self.name].tensor_realizations.clone()
        p = self._initialize_momentum(old_real)
        old_H = self._compute_ind_hamiltonian(model, data, p, realizations, temperature_inv)
        realizations[self.name].tensor_realizations = self._proposal(p, realizations, model, data, temperature_inv)
        new_H = self._compute_ind_hamiltonian(model, data, p, realizations, temperature_inv)
        accepted = self._group_metropolis_step(torch.exp(old_H - new_H))
        self._update_acceptation_rate(accepted.detach().numpy())
        accepted = accepted.unsqueeze(1)
        with torch.no_grad():
            realizations[self.name].tensor_realizations = realizations[self.name].tensor_realizations * accepted + (
                    1. - accepted) * old_real
        realizations[self.name].to_torch_Tensor()

    def _compute_U(self, realizations, data, model, temperature_inv):
        U = torch.sum(model.compute_individual_attachment_tensorized_mcmc(data, realizations))
        U += torch.sum(model.compute_regularity_realization(realizations[self.name])) * temperature_inv
        return U

    def _update_p(self, p, realizations):
        a = realizations[self.name].tensor_realizations.grad
        if ((a != a).byte().any()):
            # realizations[self.name].tensor_realizations.grad.zero_()
            # return False
            a = torch.zeros(realizations[self.name].tensor_realizations.shape)
        p = p - self.eps / 2 * a
        realizations[self.name].tensor_realizations.grad.zero_()
        # print(self.name,'gardient is ok')
        return True

    def _leapfrog_step(self, p, realizations, model, data, temperature_inv):
        U = self._compute_U(realizations, data, model, temperature_inv)
        U.backward()
        if not self._update_p(p, realizations):
            self.eps = 0.1 * self.eps
            print(temperature_inv)
            print(self.eps)
            a = realizations[self.name].tensor_realizations.grad
            print((a != a).byte().any())
            print(np.mean(self.acceptation_temp))
            print(self.name, ' nan in grad')
            return
        with torch.no_grad():
            realizations[self.name].tensor_realizations.data = realizations[
                                                                   self.name].tensor_realizations.data + self.eps * p
            realizations[self.name].tensor_realizations.grad.zero_()
        U = self._compute_U(realizations, data, model, temperature_inv)
        U.backward()
        self._update_p(p, realizations)

        return

    def _initialize_momentum(self, old_real):
        p = torch.randn(old_real.shape)
        return p

    def _compute_ind_hamiltonian(self, model, data, p, realizations, temperature_inv):
        H = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        reg = model.compute_regularity_realization(realizations[self.name])
        H += torch.sum(reg.reshape(reg.shape[0], -1), dim=1) * temperature_inv
        momentum = p ** 2
        H += 0.5 * torch.sum(momentum.reshape(momentum.shape[0], -1), dim=1)
        return H

    def _compute_pop_hamiltonian(self, model, data, p, realizations, temperature_inv):
        H = torch.sum(model.compute_individual_attachment_tensorized_mcmc(data, realizations))
        reg = model.compute_regularity_realization(realizations[self.name])
        H += torch.sum(reg) * temperature_inv
        momentum = p ** 2
        H += 0.5 * torch.sum(momentum)
        return H
