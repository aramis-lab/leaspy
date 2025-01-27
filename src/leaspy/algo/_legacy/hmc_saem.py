# import torch
# from leaspy.algo.fit.abstract_mcmc import AbstractFitMCMC
# import numpy as np
#
#
# class HMC_SAEM(AbstractFitMCMC):
#
#     def __init__(self, settings):
#         super().__init__(settings)
#         self.name = "HMC SAEM"
#
#     def iteration(self, data, model, realizations):
#
#         # Sample step
#         # print(realizations.keys())
#         # realizations.reals_ind_variable_names
#         # realizations.reals_pop_variable_names
#         for key in realizations.reals_ind_variable_names:
#             realizations[key].set_autograd()
#         self._hmc_sample_individual_realizations(data, model, realizations)
#         self._sample_population_realizations(data, model, realizations)
#
#         # Maximization step
#         self._maximization_step(data, model, realizations)
#         model.update_MCMC_toolbox(['all'], realizations)
#
#         # Annealing
#         if self.algo_parameters['annealing']['do_annealing']:
#             self._update_temperature()
#
#         for key in realizations.reals_ind_variable_names:
#             realizations[key].unset_autograd()
#
#     def _metropolisacceptation_step(self, new_regularity, previous_regularity, new_attachment, previous_attachment,
#                                     key):
#
#         # Compute energy difference
#         alpha = np.exp(-((new_regularity - previous_regularity) * self.temperature_inv +
#                          (new_attachment - previous_attachment)).detach().numpy())
#
#         # Compute acceptation
#         accepted = self.samplers[key].acceptation(alpha)
#         return accepted
#
#     def _compute_U(self, realizations, data, model, p):
#         U = torch.sum(model.compute_individual_attachment_tensorized_mcmc(data, realizations))
#         if not self._is_burn_in():
#             for key in realizations.reals_ind_variable_names:
#                 U += torch.sum(model.compute_regularity_realization(realizations[key]))
#         return U
#
#     def _leapfrog_step(self, p, realizations, model, data):
#         U = self._compute_U(realizations, data, model, p)
#         if U != U:
#             print(p.keys(), U)
#         U.backward()
#         for key in p.keys():
#             a = realizations[key].tensor_realizations.grad
#             if ((a != a).byte().any()):
#                 for key in p.keys():
#                     realizations[key].tensor_realizations.grad.zero_()
#                 return
#             p[key] += - self.algo_parameters['eps'] / 2 * a
#             with torch.no_grad():
#                 if key == 'tau':
#                     realizations[key].tensor_realizations += self.algo_parameters['eps'] * p[key] * (
#                             10 ** 2)  ## for all i, m_i=1, to be changed some day
#                 else:
#                     realizations[key].tensor_realizations += self.algo_parameters['eps'] * p[key]
#                 realizations[key].tensor_realizations.grad.zero_()
#         U = self._compute_U(realizations, data, model, p)
#         U.backward()
#         for key in p.keys():
#             a = realizations[key].tensor_realizations.grad
#             if ((a != a).byte().any()):
#                 for key in p.keys():
#                     realizations[key].tensor_realizations.grad.zero_()
#                 return
#             p[key] += - self.algo_parameters['eps'] / 2 * realizations[key].tensor_realizations.grad
#             realizations[key].tensor_realizations.grad.zero_()
#         return
#
#     def _initialize_momentum(self, realizations, key):
#         if key == 'tau':
#             p = torch.randn(realizations.size(0), realizations.size(1), realizations.size(2)) / 10
#         else:
#             p = torch.randn(realizations.size(0), realizations.size(1), realizations.size(2))
#         return p
#
#     def _proposal_hmc(self, p, realisations, model, data):
#         self.algo_parameters['eps'] = np.random.random(1)[0] * 0.009 + 0.001
#         for l in range(self.algo_parameters['L'] + np.random.randint(-5, 5)):
#             self._leapfrog_step(p, realisations, model, data)
#         return
#
#     def _compute_hamiltonian(self, model, data, p, realizations):
#         H = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
#         for key in p.keys():
#             if not self._is_burn_in():
#                 H += torch.sum(model.compute_regularity_realization(realizations[key]), dim=2).squeeze(1)
#             if key == 'tau':
#                 H += 0.5 * torch.sum(p[key] ** 2, (1, 2)) * (10 ** 2)
#             else:
#                 H += 0.5 * torch.sum(p[key] ** 2, (1, 2))
#         return H
#
#     def _hmc_sample_individual_realizations(self, data, model, realizations):
#         # this returns a tensor with values for each indiv
#         for key in realizations.reals_ind_variable_names:
#             p = {}
#             old_real = realizations.copy()
#             p[key] = self._initialize_momentum(realizations[key].tensor_realizations, key)
#             old_H = self._compute_hamiltonian(model, data, p, realizations)
#             self._proposal_hmc(p, realizations, model, data)
#             new_H = self._compute_hamiltonian(model, data, p, realizations)
#             accepted = torch.tensor(1. * (torch.rand(new_H.size(0)) < torch.exp(old_H - new_H)), dtype=torch.float32)
#             self.samplers[key].acceptation_temp = accepted.detach().numpy()
#             accepted = accepted.unsqueeze(1).unsqueeze(2)
#             # print(key,torch.mean(accepted))
#             with torch.no_grad():
#                 realizations[key].tensor_realizations = realizations[key].tensor_realizations * accepted + (
#                         1. - accepted) * old_real[key].tensor_realizations
#
#     def _2_hmc_sample_individual_realizations(self, data, model, realizations):
#         # this returns a tensor with values for each indiv
#         p = {}
#         old_real = realizations.copy()
#         for key in realizations.reals_ind_variable_names:
#             p[key] = self._initialize_momentum(realizations[key].tensor_realizations, key)
#         old_H = self._compute_hamiltonian(model, data, p, realizations)
#         self._proposal_hmc(p, realizations, model, data)
#         new_H = self._compute_hamiltonian(model, data, p, realizations)
#         accepted = torch.tensor(1. * (torch.rand(new_H.size(0)) < torch.exp(old_H - new_H)), dtype=torch.float32)
#         self.samplers[key].acceptation_temp = accepted.detach().numpy()
#         accepted = accepted.unsqueeze(1).unsqueeze(2)
#         # print(torch.mean(accepted))
#         with torch.no_grad():
#             for key in realizations.reals_ind_variable_names:
#                 realizations[key].tensor_realizations = realizations[key].tensor_realizations * accepted + (
#                         1. - accepted) * old_real[key].tensor_realizations
#
#     def _sample_population_realizations(self, data, model, realizations):
#
#         for key in realizations.reals_pop_variable_names:
#             shape_current_variable = realizations[key].shape
#
#             # For all the dimensions
#             for dim_1 in range(shape_current_variable[0]):
#                 for dim_2 in range(shape_current_variable[1]):
#
#                     # Compute the attachment and regularity
#                     previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
#                     previous_regularity = model.compute_regularity_realization(realizations[key])
#
#                     # Keep previous realizations and sample new ones
#                     previous_reals_pop = realizations[key].tensor_realizations.clone()
#                     realizations[key].set_tensor_realizations_element(
#                         realizations[key].tensor_realizations[dim_1, dim_2] + self.samplers[key].sample(),
#                         (dim_1, dim_2))
#
#                     # Update intermediary model variables if necessary
#                     model.update_MCMC_toolbox([key], realizations)
#
#                     # Compute the attachment and regularity
#                     new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
#                     new_regularity = model.compute_regularity_realization(realizations[key])
#
#                     accepted = self._metropolisacceptation_step(new_regularity.sum(), previous_regularity.sum(),
#                                                                 new_attachment, previous_attachment,
#                                                                 key)
#
#                     # Revert if not accepted
#                     if not accepted:
#                         # Revert realizations
#                         realizations[key].tensor_realizations = previous_reals_pop
#                         # Update intermediary model variables if necessary
#                         model.update_MCMC_toolbox([key], realizations)
