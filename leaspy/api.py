import copy
import json
import os
from pickle import UnpicklingError

from torch import load, save, tensor

from leaspy.algo.algo_factory import AlgoFactory
from leaspy.inputs.data.dataset import Dataset
from leaspy.models.model_factory import ModelFactory
from leaspy.inputs.settings.model_settings import ModelSettings
from leaspy.inputs.data.result import Result

from leaspy.algo import compatibility_algorithms


class Leaspy:
    def __init__(self, model_name):
        self.type = model_name
        self.model = ModelFactory.model(model_name)

    @classmethod
    def load(cls, path_to_model_settings):
        """
        Instanciate a Leaspy object from json model parameter file.
        :param path_to_model_settings:
        :return:
        """
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.initialize_MCMC_toolbox()
        leaspy.model.is_initialized = True
        return leaspy

    def save(self, path):
        """
        Save Leaspy object as json model parameter file.
        :param path:
        :return:
        """

        self.check_if_initialized()

        self.model.save(path)

    def fit(self, data, algorithm_settings):
        """
        Estimate model parameters for a given dataset.
        These model parameters correspond to the fixed-effects of the mixed effect model
        :param data:
        :param algorithm_settings:
        :return:
        """

        # Check algorithm compatibility
        Leaspy.check_if_algo_is_compatible(algorithm_settings, "fit")

        algorithm = AlgoFactory.algo(algorithm_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        if not self.model.is_initialized:
            self.model.initialize(dataset)
        algorithm.run(dataset, self.model)

    def personalize(self, data, settings):
        """
        From a model, estimate individual parameters for each ID of a given dataset.
        These individual parameters correspond to the random-effects of the mixed effect model.
        :param data:
        :param settings:
        :return: result object, aggregating individual parameters and input data
        """

        # Check algorithm compatibility
        Leaspy.check_if_algo_is_compatible(settings, "personalize")

        # Check if model has been initialized
        self.check_if_initialized()

        algorithm = AlgoFactory.algo(settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        individual_parameters, noise_std = algorithm.run(self.model, dataset)
        result = Result(data, individual_parameters, noise_std)

        return result

    def simulate(self, results, settings):
        """
        Generate synthetic patients data from a model.
        :param results:
        :param settings:
        :return: simulated_data: Data object generated via the population parameters.
        """

        # Check algorithm compatibility
        Leaspy.check_if_algo_is_compatible(settings, "simulate")

        # Check if model has been initialized
        self.check_if_initialized()

        algorithm = AlgoFactory.algo(settings)
        simulated_data = algorithm.run(self.model, results)
        return simulated_data

    @staticmethod
    def save_individual_parameters(path, individual_parameters, human_readable=True):
        """
        Save individual parameters coming from leaspy Result class object

        :param path: string - output path
        :param individual_parameters: dictionary of 2-dimensional torch.tensor (use result.individual_parameters)
        :param human_readable: boolean = True by default
            If set to True - save a json object
            If set to False - save a torch object (which cannot be read from a text editor)
        :return: None
        """
        # Test path's folder existence (if path contain a folder)
        if os.path.dirname(path) != '':
            if not os.path.isdir(os.path.dirname(path)):
                raise FileNotFoundError(
                    'Cannot save individual parameter at path %s - The folder does not exist!' % path)
                # Question : add 'make_dir = True' parameter to create the folder if it does not exist?

        dump = copy.deepcopy(individual_parameters)
        # Ex: individual_parameters = {'param1': torch.tensor([[1], [2], [3]]), ...}

        # Create a human readable file with json
        if human_readable:
            for key in dump.keys():
                if type(dump[key]) not in [list]:
                    # For multivariate parameter - like sources
                    # convert tensor([[1, 2], [2, 3]]) into [[1, 2], [2, 3]]
                    if dump[key].shape[1] == 2:
                        dump[key] = dump[key].tolist()
                    # for univariate parameters - like xi & tau
                    # convert tensor([[1], [2], [3]]) into [1, 2, 3] => use torch.tensor.view(-1)
                    elif dump[key].shape[1] == 1:
                        dump[key] = dump[key].view(-1).tolist()
            with open(path, 'w') as fp:
                json.dump(dump, fp)
        # Create a torch file
        else:
            save(dump, path)  # save function from torch

    @staticmethod
    def load_individual_parameters(path, verbose=True):
        """
        Load individual parameters from a json file or a torch file as a dictionary of torch.tensor
        :param path: string - file's path
        :param verbose: boolean = True by default
            Precise if the loaded file can be read as a torch file or need conversion
        :return: dictionary of torch.tensor - individual parameters
        """
        # Test if file is a torch file
        try:
            individual_parameters = load(path)  # load function from torch
            if verbose: print("Load from torch file")
        except UnpicklingError:
            # Else if it is a json file
            with open(path, 'r') as f:
                individual_parameters = json.load(f)
                if verbose: print("Load from json file ... conversion to torch file")
                for key in individual_parameters.keys():
                    # Convert every list in torch.tensor
                    individual_parameters[key] = tensor(individual_parameters[key])
                    # If tensor is 1-dimensional tensor([1, 2, 3]) => reshape it in tensor([[1], [2], [3]])
                    if individual_parameters[key].dim() == 1:
                        individual_parameters[key] = individual_parameters[key].view(-1, 1)
        return individual_parameters

    def check_if_initialized(self):
        """
        Check if model is initialized.
        :return:
        """
        if not self.model.is_initialized:
            raise ValueError("Model has not been initialized")

    @staticmethod
    def check_if_algo_is_compatible(settings, name):
        """
        Check compatibility of algorithms and API methods.
        :param settings:
        :param name:
        :return:
        """
        if settings.name not in compatibility_algorithms[name]:
            raise ValueError("Chosen algorithm is not compatible with method : {0} \n"
                             "please choose one in the following method list : {1}".format(name,
                                                                                           compatibility_algorithms[
                                                                                               name]))
