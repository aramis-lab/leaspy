import csv
import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations import CollectionRealization
from leaspy.variables.state import State


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
    path_plot_convergence_model_parameters : str
        Path of the first plot of the convergence of the model's parameters (in the subfolder path_plot)
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
    """

    def __init__(self, outputs):
        self.periodicity_print = outputs.console_print_periodicity
        self.periodicity_save = outputs.save_periodicity
        self.periodicity_plot = outputs.plot_periodicity

        self.path_output = outputs.root_path
        self.path_plot = outputs.plot_path
        self.path_plot_patients = outputs.patients_plot_path
        self.path_save_model_parameters_convergence = outputs.parameter_convergence_path

        if outputs.patients_plot_path is not None:
            self.path_plot_convergence_model_parameters = os.path.join(
                outputs.plot_path, "convergence_parameters.pdf"
            )

        self.time = time.time()

    def iteration(
        self,
        algo,
        model: AbstractModel,
        state: State,
    ) -> None:
        """
        Call methods to save state of the running computation, display statistics & plots if the current iteration
        is a multiple of `periodicity_print`, `periodicity_plot` or `periodicity_save`

        Parameters
        ----------
        algo : :class:`.AbstractAlgo`
            The running algorithm
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        data : :class:`.Dataset`
            The data used by the computation
        """

        # <!> only `current_iteration` defined for AbstractFitAlgorithm... TODO -> generalize where possible?
        if not hasattr(algo, "current_iteration"):
            # emit a warning?
            return

        iteration = algo.current_iteration

        if self.periodicity_print is not None:
            if iteration == 0 or iteration % self.periodicity_print == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)
                self.print_time()

        if self.path_output is None:
            return

        if self.periodicity_save is not None:
            if iteration == 0 or iteration % self.periodicity_save == 0:
                self.save_model_parameters_convergence(iteration, model)

        if self.periodicity_plot is not None:
            if iteration % self.periodicity_plot == 0:
                self.save_plot_convergence_model_parameters(model)

    def print_time(self):
        """
        Prints the duration since the last periodic point
        """
        current_time = time.time()
        print(f"Duration since last print: {current_time - self.time:.3f}s")
        self.time = current_time

    def print_model_statistics(self, model: AbstractModel):
        """
        Prints model's statistics

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        """
        print(model)

    def print_algo_statistics(self, algo):
        """
        Prints algorithm's statistics

        Parameters
        ----------
        algo : :class:`.AbstractAlgo`
            The running algorithm
        """
        print(algo)

    def save_model_parameters_convergence(
        self, iteration: int, model: AbstractModel
    ) -> None:
        """
        Saves the current state of the model's parameters

        Parameters
        ----------
        iteration : int
            The current iteration
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        """
        model.state.save(
            self.path_save_model_parameters_convergence,
            iteration=iteration,
        )

    def save_plot_convergence_model_parameters(self, model: AbstractModel):
        """
        Saves figures of the model parameters' convergence in one pdf file

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The model used by the computation
        """
        width = 10
        height_per_row = 3.5

        to_skip = ["betas", "sources", "space_shifts", "mixing_matrix", "xi", "tau"]
        if getattr(model, "is_ordinal", False):
            to_skip.append("deltas")
        params_to_plot = [p for p in model.state._tracked_variables if p not in to_skip]

        n_plots = len(params_to_plot)
        n_rows = math.ceil(n_plots / 2)
        _, ax = plt.subplots(n_rows, 2, figsize=(width, n_rows * height_per_row))

        for i, key in enumerate(params_to_plot):
            import_path = os.path.join(
                self.path_save_model_parameters_convergence, key + ".csv"
            )
            df_convergence = pd.read_csv(import_path, index_col=0, header=None)
            df_convergence.index.rename("iter", inplace=True)

            x_position = i // 2
            y_position = i % 2
            df_convergence.plot(ax=ax[x_position][y_position], legend=False)
            ax[x_position][y_position].set_title(key)

        plt.tight_layout()
        plt.savefig(self.path_plot_convergence_model_parameters)
        plt.close()
