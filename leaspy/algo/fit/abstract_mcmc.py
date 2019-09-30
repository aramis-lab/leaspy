import torch
from .abstract_fit_algo import AbstractFitAlgo
from ..samplers.hmc_sampler import HMCSampler
from ..samplers.gibbs_sampler import GibbsSampler

import numpy as np


class AbstractFitMCMC(AbstractFitAlgo):

    def __init__(self, settings):

        super().__init__()

        # Algorithm parameters
        self.algo_parameters = settings.parameters
        self.seed = settings.seed

        # Realizations and samplers
        self.realizations = None
        self.task = None
        self.samplers = None
        self.current_iteration = 0

        # Annealing
        self.temperature_inv = 1
        self.temperature = 1

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        """
        Initialize the samplers, annealing, MCMC toolbox and sufficient statistics.
        :param data:
        :param model:
        :param realizations:
        :return: realizations
        """
        # MCMC toolbox (cache variables for speed-ups + tricks)
        model.initialize_MCMC_toolbox()
        # Samplers
        self._initialize_samplers(model, data)
        self._initialize_sufficient_statistics(data, model, realizations)
        if self.algo_parameters['annealing']['do_annealing']:
            self._initialize_annealing()
        return realizations

    def _initialize_annealing(self):
        """
        Initialize annealing, setting initial temperature and number of iterations.
        :return:
        """
        if self.algo_parameters['annealing']['do_annealing']:
            if self.algo_parameters['annealing']['n_iter'] is None:
                self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['n_iter'] / 2)

        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1 / self.temperature

    def _initialize_samplers(self, model, data):
        """
        Instanciate samplers for Gibbs / HMC sampling as a dictionnary samplers {name: sampler}
        :param model:
        :param data:
        :return:
        """
        infos_variables = model.random_variable_informations()
        self.samplers = dict.fromkeys(infos_variables.keys())
        for variable, info in infos_variables.items():
            if info["type"] == "individual":
                if self.algo_parameters['sampler_ind'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
            else:
                if self.algo_parameters['sampler_pop'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])

    def _initialize_sufficient_statistics(self, data, model, realizations):
        suff_stats = model.compute_sufficient_statistics(data, realizations)
        self.sufficient_statistics = {k: torch.zeros(v.shape) for k, v in suff_stats.items()}

    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):
        """
        MCMC-SAEM iteration.
        1. Sample : MC sample successively of the populatin and individual variales
        2. Maximization step : update model parameters from current population/individual variables values.
        :param data:
        :param model:
        :param realizations:
        :return:
        """

        # Sample step
        for key in realizations.reals_pop_variable_names:
            self.samplers[key].sample(data, model, realizations, self.temperature_inv)
        for key in realizations.reals_ind_variable_names:
            self.samplers[key].sample(data, model, realizations, self.temperature_inv)

        # Maximization step
        self._maximization_step(data, model, realizations)
        model.update_MCMC_toolbox(['all'], realizations)

        # Update the likelihood with the new noise_var
        # TODO likelihood is computed 2 times, remove this one, and update it in maximization step ?
        # TODO or ar the update of all sufficient statistics ???
        # self.likelihood.update_likelihood(data, model, realizations)

        # Annealing
        if self.algo_parameters['annealing']['do_annealing']:
            self._update_temperature()

    def _update_temperature(self):
        """
        Update the temperature according to a plateau annealing scheme.
        :return:
        """
        if self.current_iteration <= self.algo_parameters['annealing']['n_iter']:
            # If we cross a plateau step
            if self.current_iteration % int(
                    self.algo_parameters['annealing']['n_iter'] / self.algo_parameters['annealing'][
                        'n_plateau']) == 0:
                # Decrease temperature linearly
                self.temperature -= self.algo_parameters['annealing']['initial_temperature'] / \
                                    self.algo_parameters['annealing']['n_plateau']
                self.temperature = max(self.temperature, 1)
                self.temperature_inv = 1 / self.temperature

    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Instance of {0} algo \n".format(self.name)
        out += "Iteration {0}\n".format(self.current_iteration)
        out += "=Samplers \n"
        for sampler_name, sampler in self.samplers.items():
            acceptation_rate = np.mean(sampler.acceptation_temp.detach().numpy())
            out += "    {} rate : {:.2f}%, std: {:.5f}\n".format(sampler_name, 100 * acceptation_rate,
                                                                 sampler.std.mean())

        if self.algo_parameters['annealing']['do_annealing']:
            out += "Annealing \n"
            out += "Temperature : {0}".format(self.temperature)
        return out

    #############
    ## HMC
    #############
