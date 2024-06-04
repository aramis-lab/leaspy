import csv
import os
import time

from leaspy.io.logs.visualization.plotter import Plotter

from leaspy.io.data.data import Data
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations import CollectionRealization


class FitOutputManager:
    """
    Class used by :class:`.AbstractAlgo` (and its child classes) to display & save plots and statistics during algorithm execution.

    Parameters
    ----------
    outputs : :class:`~.io.settings.outputs_settings.OutputsSettings`
        Initialize the `FitOutputManager` class attributes, like the logs paths, the console print periodicity and so forth.

    Attributes
    ----------
    path_output : str
        Path of the folder containing all the outputs
    path_plot : str
        Path of the subfolder of path_output containing the logs plots
    path_plot_convergence_model_parameters_1 : str
        Path of the first plot of the convergence of the model's parameters (in the subfolder path_plot)
    path_plot_convergence_model_parameters_2 : str
        Path of the second plot of the convergence of the model's parameters (in the subfolder path_plot)
    path_plot_patients : str
        Path of the subfolder of path_plot containing the plot of the reconstruction of the patients' longitudinal
        trajectory by the model
    path_save_model_parameters_convergence : str
        Path of the subfolder of path_output containing the progression of the model's parameters convergence
    periodicity_plot : int (default 100)
        Set the frequency of the display of the plots
    periodicity_print : int
        Set the frequency of the display of the statistics
    periodicity_save : int
        Set the frequency of the saves of the model's parameters & the realizations
    plot_options : dict
        Contain all the additional information (for now contain only the number of displayed patients by the method
        plot_patient_reconstructions - which is 5 by default)
    plotter : :class:`~.utils.logs.visualization.plotter.Plotter`
        class object used to call visualization methods
    time : float
        Last timestamp (to display the duration between two visualization prints)
    """

    def __init__(self, outputs):

        self.periodicity_print = outputs.console_print_periodicity
        self.periodicity_save = outputs.save_periodicity
        self.periodicity_plot = outputs.plot_periodicity

        self.path_output = outputs.root_path
        self.path_plot = outputs.plot_path
        self.path_plot_patients = outputs.patients_plot_path
        self.path_save_model_parameters_convergence = outputs.parameter_convergence_path
        self.path_plot_convergence_model_parameters_1 = None
        self.path_plot_convergence_model_parameters_2 = None

        if outputs.patients_plot_path is not None:
            self.path_plot_convergence_model_parameters_1 = os.path.join(outputs.plot_path, "convergence_1.pdf")
            self.path_plot_convergence_model_parameters_2 = os.path.join(outputs.plot_path, "convergence_2.pdf")

        # Options
        # TODO : Maybe add to the outputs reader
        # <!> We would need the attributes of the model if we would want to reconstruct values without MCMC toolbox
        # This is the only location where we could need it during the calibration....
        self.plot_options = {'max_patient_number': 5, 'attribute_type': 'MCMC'}
        self.plotter = Plotter()
        self.plotter._show = False  # do not show any of the plots!

        self.time = time.time()

        self.save_last_n_realizations = outputs.save_last_n_realizations

    def iteration(
        self,
        algo,
        data: Data,
        model: AbstractModel,
        realizations: CollectionRealization,
    ):
        """
        Call methods to save state of the running computation, display statistics & plots if the current iteration
        is a multiple of `periodicity_print`, `periodicity_plot` or `periodicity_save`

        Parameters
        ----------
        algo : :class:`.AbstractAlgo`
            The running algorithm
        data : :class:`.Data`
            The data used by the computation
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Current state of the realizations
        """

        # <!> only `current_iteration` defined for AbstractFitAlgorithm... TODO -> generalize where possible?
        if not hasattr(algo, 'current_iteration'):
            # emit a warning?
            return

        iteration = algo.current_iteration

        if self.periodicity_print is not None:
            if iteration == 1 or iteration % self.periodicity_print == 0:
                # print first iteration
                print()
                self.print_algo_statistics(algo)
                print()
                self.print_model_statistics(model)
                print()
                self.print_time()

        if self.path_output is None:
            return

        if self.periodicity_save is not None:
            if iteration == 1 or iteration % self.periodicity_save == 0:
                # save first iteration
                self.save_model_parameters_convergence(iteration, model)
                # model.save(...)

        if self.periodicity_plot is not None:
            if iteration % self.periodicity_plot == 0:
                # do not plot first iteration (useless, no lines yet)
                self.plot_patient_reconstructions(iteration, data, model, realizations)
                self.plot_convergence_model_parameters(model)

        if (algo.algo_parameters['n_iter'] - iteration) < self.save_last_n_realizations:
            self.save_realizations(iteration, realizations)

    ########
    ## Printing methods
    ########

    def print_time(self):
        """
        Display the duration since the last print
        """
        current_time = time.time()
        print(f"Duration since last print: {current_time - self.time:.3f}s")
        self.time = current_time

    def print_model_statistics(self, model: AbstractModel):
        """
        Print model's statistics

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        """
        print(model)

    def print_algo_statistics(self, algo):
        """
        Print algorithm's statistics

        Parameters
        ----------
        algo : :class:`.AbstractAlgo`
            The running algorithm
        """
        print(algo)

    ########
    ## Saving methods
    ########

    def save_model_parameters_convergence(self, iteration: int, model: AbstractModel) -> None:
        """
        Save the current state of the model's parameters

        Parameters
        ----------
        iteration : int
            The current iteration
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        """
        model_parameters = model.parameters

        # TODO maybe better way ???
        model_parameters_save = model_parameters.copy()

        # TODO I Stopped here, 2d array saves should be fixed.

        # Transform the types
        for key, value in model_parameters.items():

            if value.ndim > 1:
                if key == "betas":
                    model_parameters_save.pop(key)
                    for column in range(value.shape[1]):
                        model_parameters_save[f"{key}_{column}"] = value[:, column].tolist()
                if key == "deltas":
                    model_parameters_save.pop(key)
                    for line in range(value.shape[0]):
                        model_parameters_save[f"{key}_{line}"] = value[line].tolist()
                # P0, V0
                elif value.shape[0] == 1 and len(value.shape) > 1:
                    model_parameters_save[key] = value[0].tolist()
            elif value.ndim == 1:
                model_parameters_save[key] = value.tolist()
            else:  # ndim == 0
                model_parameters_save[key] = [value.tolist()]

        # Save the dictionary
        for key, value in model_parameters_save.items():
            path = os.path.join(self.path_save_model_parameters_convergence, key + ".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                writer.writerow([iteration] + value)

    def save_realizations(self, iteration: int, realizations: CollectionRealization) -> None:
        """
        Save the current realizations.
        The path is given by the attribute path_save_model_parameters_convergence.

        Parameters
        ----------
        iteration : int
            The current iteration
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Current state of the realizations
        """
        # TODO: not generic at all
        for name in ("xi", "tau"):
            value = realizations[name].tensor.squeeze(1).detach().tolist()
            path = os.path.join(self.path_save_model_parameters_convergence, name + ".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                # writer.writerow([iteration]+list(model_parameters.values()))
                writer.writerow([iteration] + value)
        if "sources" in realizations.individual.names:
            for i in range(realizations["sources"].tensor.shape[1]):
                value = realizations["sources"].tensor[:, i].detach().tolist()
                path = os.path.join(self.path_save_model_parameters_convergence, 'sources' + str(i) + ".csv")
                with open(path, 'a', newline='') as filename:
                    writer = csv.writer(filename)
                    # writer.writerow([iteration]+list(model_parameters.values()))
                    writer.writerow([iteration] + value)

    ########
    ## Plotting methods
    ########

    def plot_convergence_model_parameters(self, model: AbstractModel):
        """
        Plot the convergence of the model parameters (calling the `Plotter`)

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        """
        self.plotter.plot_convergence_model_parameters(self.path_save_model_parameters_convergence,
                                                       self.path_plot_convergence_model_parameters_1,
                                                       self.path_plot_convergence_model_parameters_2,
                                                       model)

    #def plot_model_average_trajectory(self, model):
    #    raise NotImplementedError

    def plot_patient_reconstructions(
        self,
        iteration: int,
        data: Data,
        model: AbstractModel,
        realizations: CollectionRealization,
    ) -> None:
        """
        Plot on the same graph for several patients their real longitudinal values
        and their reconstructions by the model.

        Parameters
        ----------
        iteration : int
            The current iteration
        data : :class:`.Data`
            The data used by the computation
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Current state of the realizations
        """
        path_iteration = os.path.join(self.path_plot_patients, f'plot_patients_{iteration}.pdf')
        self.plotter.plot_patient_reconstructions(
            path_iteration, data, model, realizations.individual.tensors_dict, **self.plot_options
        )
