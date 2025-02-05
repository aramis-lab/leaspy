import os
import shutil
import warnings

from leaspy.exceptions import LeaspyAlgoInputError


class OutputsSettings:
    """
    Used to create the `logs` folder to monitor the convergence of the fit algorithm.

    Parameters
    ----------
    settings : dict[str, Any]
        Parameters of the object. It may be in:
            * path : str or None
                Where to store logs (relative or absolute path)
                If None, nothing will be saved (only console prints),
                unless save_periodicity is not None (default relative path './_outputs/' will be used).
            * print_periodicity : int >= 1 or None
                Flag to log into console convergence data every N iterations
                If None, no console prints.
            * save_periodicity : int >= 1, optional
                Flag to save convergence data every N iterations
                Default=50.
            * plot_periodicity : int >= 1 or None
                Flag to plot convergence data every N iterations
                If None, no plots will be saved.
                Note that you can not plot convergence data without saving data (and not more frequently than these saves!)
            * overwrite_logs_folder : bool
                Flag to remove all previous logs if existing (default False)

    Raises
    ------
    :exc:`.LeaspyAlgoInputError`
    """
    # TODO mettre les variables par défaut à None
    # TODO: Réfléchir aux cas d'usages : est-ce qu'on veut tout ou rien,
    # TODO: ou bien la possibilité d'avoir l'affichage console et/ou logs dans un fold
    # TODO: Aussi, bien définir la création du path

    DEFAULT_LOGS_DIR = '_outputs'  # logs

    def __init__(self, settings):
        self.print_periodicity = None
        self.plot_periodicity = None
        self.save_periodicity = 50
        self.nb_of_patients_to_plot = 5

        self.root_path = None
        self.parameter_convergence_path = None
        self.plot_path = None
        self.patients_plot_path = None

        self._set_print_periodicity(settings)
        self._set_save_periodicity(settings)
        self._set_plot_periodicity(settings)

        # only create folders if the user want to save data or plots and provided a valid path!
        self._create_root_folder(settings)

    def _set_param_as_int_or_ignore(self, settings, param: str):
        """Inplace set of parameter (as int) from settings."""
        if param not in settings:
            return

        val = settings[param]
        if val is not None:
            # try to cast as an integer.
            try:
                val = int(val)
                assert val >= 1
            except Exception:
                warnings.warn(f"The '{param}' parameter you provided is not castable to an int > 0. "
                              "Ignoring its value.", UserWarning)
                return

        # Update the attribute of self in-place
        setattr(self, param, val)

    def _set_print_periodicity(self, settings):
        self._set_param_as_int_or_ignore(settings, 'print_periodicity')

    def _set_save_periodicity(self, settings):
        self._set_param_as_int_or_ignore(settings, 'save_periodicity')

    def _set_plot_periodicity(self, settings):
        self._set_param_as_int_or_ignore(settings, 'plot_periodicity')

        if self.plot_periodicity is not None:
            if self.save_periodicity is None:
                raise LeaspyAlgoInputError('You can not define a `plot_periodicity` without defining `save_periodicity`. '
                                           'Note that the `plot_periodicity` should be a multiple of `save_periodicity`.')

            if self.plot_periodicity % self.save_periodicity != 0:
                raise LeaspyAlgoInputError('The `plot_periodicity` should be a multiple of `save_periodicity`.')


    def _create_root_folder(self, settings):

        # Get the path to put the outputs
        path = settings.get('path', None)

        if path is None and self.save_periodicity:
            warnings.warn("You did not provide a path for your logs outputs whereas you want to save convergence data. "
                          f"The default path '{self.DEFAULT_LOGS_DIR}' will be used (relative to the current working directory).")
            path = self.DEFAULT_LOGS_DIR

        if path == self.DEFAULT_LOGS_DIR:
            if os.path.exists(path):
                self._clean_folder(path)

        if path is None:
            # No folder will be created and no convergence data shall be saved
            return

        # store the absolute path in settings
        abs_path = os.path.abspath(path)
        settings['path'] = abs_path

        # Check if the folder does not exist: if not, create (and its parent)
        if not os.path.exists(abs_path):
            warnings.warn(f"The logs path you provided ({settings['path']}) does not exist. "
                          "Needed paths will be created (and their parents if needed).")
        elif settings.get('overwrite_logs_folder', False):
            warnings.warn(f"Overwriting '{path}' folder...")
            self._clean_folder(abs_path)

        all_ok = self._check_needed_folders_are_empty_or_create_them(abs_path)

        if not all_ok:
            raise LeaspyAlgoInputError(f"The logs folder '{path}' already exists and is not empty! "
                                       "Give another path or use keyword argument `overwrite_logs_folder=True`.")

    @staticmethod
    def _check_folder_is_empty_or_create_it(path_folder) -> bool:
        if os.path.exists(path_folder):
            if os.path.islink(path_folder) or not os.path.isdir(path_folder) or len(os.listdir(path_folder)) > 0:
                # path is a link, or not a directory, or a directory containing something
                return False
        else:
            os.makedirs(path_folder)

        return True

    @staticmethod
    def _clean_folder(path):
        shutil.rmtree(path)
        os.makedirs(path)

    def _check_needed_folders_are_empty_or_create_them(self, path) -> bool:
        self.root_path = path

        self.parameter_convergence_path = os.path.join(path, 'parameter_convergence')
        self.plot_path = os.path.join(path, 'plots')
        self.patients_plot_path = os.path.join(self.plot_path, 'patients')

        all_ok = self._check_folder_is_empty_or_create_it(self.parameter_convergence_path)
        all_ok &= self._check_folder_is_empty_or_create_it(self.plot_path)
        all_ok &= self._check_folder_is_empty_or_create_it(self.patients_plot_path)

        return all_ok
