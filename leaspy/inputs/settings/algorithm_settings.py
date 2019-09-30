import os
import json
import warnings

from leaspy.inputs.settings.outputs_settings import OutputsSettings
from leaspy.inputs.settings import default_data_dir


class AlgorithmSettings:
    """
    Read a algo_parameters json file and create the corresponding algo
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.parameters = None
        self.seed = None
        self.initialization_method = None
        self.logs = None

        if name in ['mcmc_saem', 'scipy_minimize', 'simulation', 'mean_real', 'gradient_descent_personalize',
                    'mode_real']:
            self._load_default_values(os.path.join(default_data_dir, 'default_' + name + '.json'))
        else:
            raise ValueError('The algorithm name >>>{0}<<< you provided does not exist'.format(name))
        self._manage_kwargs(kwargs)

    @classmethod
    def load(cls, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        if 'name' not in settings.keys():
            raise ValueError("Your json file must contain a \'name\' ")
        algorithm_settings = cls(settings['name'])

        if 'parameters' in settings.keys():
            print("You overwrote the algorithm default parameters")
            algorithm_settings.parameters = cls._get_parameters(settings)

        if 'seed' in settings.keys():
            print("You overwrote the algorithm default seed")
            algorithm_settings.seed = cls._get_seed(settings)

        if 'initialization_method' in settings.keys():
            print("You overwrote the algorithm default initialization method")
            algorithm_settings.initialization_method = cls._get_initialization_method(settings)

        return algorithm_settings

    def _manage_kwargs(self, kwargs):
        if 'seed' in kwargs.keys():
            self.seed = self._get_seed(kwargs)
        if 'initialization_method' in kwargs.keys():
            self.initialization_method = self._get_initialization_method(kwargs)

        for k, v in kwargs.items():
            if k in ['seed', 'initialization_method']:
                continue

            if k in self.parameters.keys():
                self.parameters[k] = v
            else:
                warning_message = "The parameter key : >>>{0}<<< you provided is unknown".format(k)
                warnings.warn(warning_message)

        if self.name == 'mcmc_saem':
            if 'n_iter' in kwargs.keys() and ('n_burn_in_iter' is None or 'n_burn_in_iter' not in kwargs.keys()):
                self.parameters['n_burn_in_iter'] = int(0.9 * kwargs['n_iter'])

            # TODO : For Raphael : what does it mean? Because there are already default value.
            # TODO : Thus, either default value for annealing/iter in the json, either here. Not both.
            if 'n_iter' in kwargs.keys() and 'annealing' not in kwargs.keys():
                self.parameters['annealing']["n_iter"] = int(0.5 * kwargs['n_iter'])

    def _load_default_values(self, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        AlgorithmSettings._check_default_settings(settings)
        self.name = self._get_name(settings)
        self.parameters = self._get_parameters(settings)
        self.seed = self._get_seed(settings)

    def set_logs(self, path, **kwargs):
        settings = {
            'path': path,
            'console_print_periodicity': 50,
            'plot_periodicity': 100,
            'save_periodicity': 50
        }

        for k, v in kwargs.items():
            if k in ['console_print_periodicity', 'plot_periodicity', 'save_periodicity']:
                # Todo : Ca devrait planter si v n'est pas un int!!
                settings[k] = v
            else:
                warnings.warn("The kwargs {} you provided is not correct".format(v))

        self.logs = OutputsSettings(settings)

    @staticmethod
    def _check_default_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'name\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'seed' not in settings.keys():
            raise ValueError('The \'settings\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'initialization_method' not in settings.keys():
            raise ValueError(
                'The \'initialization_method\' key is missing in the algorithm settings (JSON file) you are loading')

    @staticmethod
    def _get_name(settings):
        return settings['name'].lower()

    @staticmethod
    def _get_parameters(settings):
        return settings['parameters']

    @staticmethod
    def _get_seed(settings):
        if settings['seed'] is None:
            return None
        try:
            return int(settings['seed'])
        except ValueError:
            print("The \'seed\' parameter you provided cannot be converted to int")

    @staticmethod
    def _get_initialization_method(settings):
        if settings['initialization_method'] is None:
            return None
        # TODO : There should be a list of possible initialization method. It can also be discussed depending on the algorithms name
        return settings['initialization_method']
