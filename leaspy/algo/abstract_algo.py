import torch
import numpy as np
from leaspy.utils.output.fit_output_manager import FitOutputManager


class AbstractAlgo():

    def __init__(self):
        self.algo_parameters = None
        self.name = None
        self.seed = None

    ###########################
    ## Initialization
    ###########################
    @staticmethod
    def _initialize_seed(seed):
        """
        Set both numpy and torch seeds.
        :param seed:
        :return:
        """
        if seed is not None: # TODO is numpy seed needed ?
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(" ==> Setting seed to {0}".format(seed))

    ###########################
    ## Getters / Setters
    ###########################

    def load_parameters(self, parameters):
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.algo_parameters[k] = v

    def set_output_manager(self, output_settings):
        if output_settings is not None:
            self.output_manager = FitOutputManager(output_settings)
        else:
            self.output_manager = None

    def run(self, data, model):
        """
        Initialize algorithm and loop for n_iter iterations on algorithm iter method.
        :param data:
        :param model:
        :return: realizations
        """
        raise NotImplementedError

    def iteration(self, data, model, realizations):
        """
        Iteration of given algorithm
        :param data:
        :param model:
        :param realizations:
        :return:
        """
        raise NotImplementedError